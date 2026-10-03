"""Mirrors every AI-generated image into Dropbox, off the request's hot path.

Generated images already land in local media storage (see
``views._ai_flux_response``). This module additionally copies them to Dropbox
under ``/vidhyora/<email>/`` so there is a durable, per-account archive — local
media is ephemeral on most PaaS hosts, so a redeploy would otherwise take every
image the users ever generated with it.

The visual evidence attached to a user's bug report (``views.ai_report_submit``)
is archived the same way, under ``/vidhyora/<email>/reports/``, so whoever
reviews the report can still see what the user was actually looking at. That is
filed beside the person's own images rather than in a separate reports tree,
because "whose images are these" is the question the folder layout answers.

Speed is the binding constraint here, so the whole module is fire-and-forget:

* ``enqueue`` performs no network, no disk and no database work. It puts a
  tuple on an in-memory queue and returns immediately, so the chat response
  reaches the browser without ever waiting on Dropbox. Uploading inline would
  add a full round trip (typically several hundred ms, and up to the timeout
  when Dropbox is slow) to every single image reply.
* One daemon worker thread drains the queue and does the uploading.
* Every failure is logged and swallowed. Dropbox being down, throttled or
  misconfigured must never turn a successfully generated image into an error
  the user sees — they still get their image from local storage either way.

The queue is bounded on purpose. If uploads back up past ``MAX_QUEUED``, new
items are dropped with a log line rather than growing until the process runs
out of memory. A dropped item costs an archive copy, not a user's image.

Because the worker is a daemon thread, anything still queued is lost if the
process exits (a redeploy, a gunicorn worker recycle). That is the accepted
trade for never blocking a reply; the queue normally drains within seconds.
"""

import base64
import binascii
import logging
import os
import queue
import re
import secrets
import threading
import time

from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

# Everything this app writes lives under one root folder, so the images can
# never collide with the db.sqlite3 backups dropbox_backup.py keeps in
# '/EduTrellis Store'.
ROOT_FOLDER = '/vidhyora'
# Guests have no email to file under. They still get archived, just together.
GUEST_FOLDER = 'guests'
# Report evidence sits under the reporter's own folder, not a separate tree.
REPORTS_SUBFOLDER = 'reports'

MAX_QUEUED = 200
# Generous, because this is off the critical path — a slow upload delays only
# the next queued image, never a user. Still bounded so a hung connection
# cannot wedge the worker forever.
UPLOAD_TIMEOUT_SECONDS = 60

ALLOWED_EXTENSIONS = frozenset({'png', 'jpg', 'jpeg', 'webp', 'gif'})

# Only the header is matched, so working out the file type costs nothing even
# for a multi-megabyte data URI — the base64 body is decoded later, on the
# worker thread, where the time cannot reach the user.
_DATA_URI_HEADER_RE = re.compile(r'^data:image/([a-z0-9.+-]+);base64,', re.IGNORECASE)

# Dropbox rejects these outright in a path component.
_UNSAFE_PATH_CHARS_RE = re.compile(r'[\\/:?*<>"|]+')
_CONTROL_CHARS_RE = re.compile(r'[\x00-\x1f\x7f]+')

_queue = None
_worker = None
_worker_lock = threading.Lock()

# Touched only by the worker thread, so it needs no lock of its own.
_client = None
_client_credentials = None

_CREDENTIAL_SETTINGS = ('DROPBOX_APP_KEY', 'DROPBOX_APP_SECRET', 'DROPBOX_REFRESH_TOKEN')
_CREDENTIAL_TTL_SECONDS = 30
_saved_credentials = (0.0, ('', '', ''))


def _setting(name):
    return (getattr(settings, name, '') or '').strip()


def _dashboard_credentials():
    """The App Key/Secret/Refresh Token saved on the dashboard's Backup &
    Restore page, re-read at most every 30 seconds (or right after a change
    through reset_credentials) so this module never queries per image."""
    global _saved_credentials
    stamp, value = _saved_credentials
    if stamp and time.monotonic() - stamp < _CREDENTIAL_TTL_SECONDS:
        return value
    value = ('', '', '')
    try:
        from .models import DropboxSettings
        row = DropboxSettings.objects.filter(pk=1).values_list('app_key', 'app_secret', 'refresh_token').first()
        if row:
            value = tuple((part or '').strip() for part in row)
    except Exception:
        logger.debug('Could not read the saved Dropbox credentials', exc_info=True)
    _saved_credentials = (time.monotonic(), value)
    return value


def _credentials():
    """(app key, app secret, refresh token): what the dashboard saved wins,
    and the server settings fill whatever it left blank."""
    saved = _dashboard_credentials()
    return tuple(saved[i] or _setting(name) for i, name in enumerate(_CREDENTIAL_SETTINGS))


def reset_credentials():
    """Called when the dashboard changes the Dropbox account: the next upload
    reads the new credentials and builds a fresh client."""
    global _client, _client_credentials, _saved_credentials
    _saved_credentials = (0.0, ('', '', ''))
    _client = None
    _client_credentials = None


def is_enabled():
    """False when archiving is switched off outright — chiefly during tests.

    Kept separate from is_configured() because "no credentials" and "asked not
    to archive" are different situations: only the first is worth warning about.
    """
    return bool(getattr(settings, 'DROPBOX_IMAGE_ARCHIVE_ENABLED', True))


def is_configured():
    """True when all three Dropbox credentials are present."""
    return all(_credentials())


def folder_for(email):
    """Map an account email to a safe single Dropbox folder name.

    Emails are lowercased so 'User@x.com' and 'user@x.com' cannot end up as two
    separate archives for one person.
    """
    name = (email or '').strip().lower()
    name = _CONTROL_CHARS_RE.sub('', name)
    name = _UNSAFE_PATH_CHARS_RE.sub('_', name)
    # Truncate first, then strip: cutting at 120 chars could otherwise leave a
    # trailing dot, which Dropbox refuses as a folder name.
    name = name[:120].strip(' .')
    return name or GUEST_FOLDER


def _filename(extension, local_name=''):
    """A sortable, unique Dropbox filename for one generated image.

    The local storage basename is kept as the suffix so a file in Dropbox can
    be matched back to the one in local media (and to the AIMessage row that
    points at it) when something needs debugging.
    """
    extension = (extension or '').lower().lstrip('.')
    if extension == 'jpeg':
        extension = 'jpg'
    if extension not in ALLOWED_EXTENSIONS:
        extension = 'png'

    stamp = timezone.localtime().strftime('%Y%m%d-%H%M%S')
    stem = os.path.splitext(os.path.basename((local_name or '').replace('\\', '/')))[0]
    stem = _UNSAFE_PATH_CHARS_RE.sub('_', _CONTROL_CHARS_RE.sub('', stem))[:64].strip(' .')
    if not stem:
        stem = secrets.token_hex(8)
    return f'{stamp}-{stem}.{extension}'


def _extension_from_data_uri(data_uri):
    match = _DATA_URI_HEADER_RE.match((data_uri or '')[:64])
    return match.group(1).lower() if match else 'png'


def _resolve_content(source):
    """Turn a queue item into bytes, or None when there is nothing to upload.

    Raw bytes pass straight through. A data: URI is base64-decoded here, on the
    worker thread rather than in the request, because report evidence can be
    several megabytes and decoding it is real work.
    """
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    match = _DATA_URI_HEADER_RE.match((source or '')[:64])
    if not match:
        return None
    try:
        # validate=False: browsers produce data URIs with newlines in them, and
        # a wrapped-but-valid image should still be archived.
        return base64.b64decode(source[match.end():], validate=False)
    except (ValueError, binascii.Error):
        return None


def _build_client():
    try:
        import dropbox
    except ImportError:  # pragma: no cover - the package ships in requirements
        logger.warning('Dropbox image archive is off: the dropbox package is not installed')
        return None
    if not is_enabled():
        # Second line of defence: the worker thread outlives a single request,
        # so re-check here rather than trusting the check made at enqueue time.
        return None
    if not is_configured():
        logger.warning('Dropbox image archive is off: credentials are not set')
        return None
    # A refresh token (not a short-lived access token) so the client renews
    # itself indefinitely without anyone re-authorising the app by hand.
    app_key, app_secret, refresh_token = _credentials()
    return dropbox.Dropbox(
        oauth2_refresh_token=refresh_token,
        app_key=app_key,
        app_secret=app_secret,
        timeout=UPLOAD_TIMEOUT_SECONDS,
    )


def _upload(source, folder, filename):
    global _client, _client_credentials
    content = _resolve_content(source)
    if not content:
        logger.warning('Skipped a Dropbox archive upload with no usable image data (%s)', filename)
        return None
    credentials = _credentials()
    if _client is None or _client_credentials != credentials:
        _client = _build_client()
        _client_credentials = credentials
    if _client is None:
        return None

    import dropbox

    path = f'{ROOT_FOLDER}/{folder}/{filename}'
    # files_upload creates missing parent folders itself, so '/vidhyora' and
    # the per-email folder need no separate round trips to set up.
    # autorename guards the (vanishingly unlikely) name collision; mute stops
    # Dropbox pinging the owner's desktop and phone for every single image.
    _client.files_upload(
        content,
        path,
        mode=dropbox.files.WriteMode.add,
        autorename=True,
        mute=True,
    )
    return path


def _run():
    global _client
    while True:
        source, folder, filename = _queue.get()
        try:
            path = _upload(source, folder, filename)
            if path:
                logger.info('Archived generated image to Dropbox: %s', path)
        except Exception as exc:
            # Expired credentials, rate limits, insufficient space and network
            # blips all land here. Drop the cached client so the next item
            # rebuilds it (cheap — construction makes no network call) rather
            # than reusing one that may be in a bad state.
            _client = None
            logger.warning(
                'Could not archive generated image to Dropbox (%s/%s): %s',
                folder, filename, exc,
            )
        finally:
            _queue.task_done()


def _ensure_worker():
    global _queue, _worker
    with _worker_lock:
        if _queue is None:
            _queue = queue.Queue(maxsize=MAX_QUEUED)
        # Restart defensively: _run swallows everything, so a dead thread
        # should be impossible, but silently never archiving again would be a
        # bad way to find out otherwise.
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(
                target=_run, name='dropbox-image-archive', daemon=True,
            )
            _worker.start()
        return _queue


def _submit(source, email, extension, name, subfolder=''):
    """Shared, never-raising tail of the enqueue functions below.

    Both callers sit directly in a request the user is waiting on, so nothing
    here is allowed to fail that request — the image or report has already been
    saved locally by the time this runs.
    """
    try:
        if not source or not is_enabled() or not is_configured():
            return False
        folder = folder_for(email)
        if subfolder:
            folder = f'{folder}/{subfolder}'
        work_queue = _ensure_worker()
        work_queue.put_nowait((source, folder, _filename(extension, name)))
        return True
    except queue.Full:
        logger.warning('Dropbox archive queue is full; skipping %s', name or 'an image')
        return False
    except Exception:
        logger.exception('Could not queue an image for the Dropbox archive')
        return False


def enqueue(content, extension, email, local_name=''):
    """Queue one generated image for background upload. Never raises.

    Returns True when the image was queued. Callers are expected to ignore the
    result — it exists for tests and for the log line, not for the user, whose
    image is already saved locally regardless.
    """
    return _submit(content, email, extension, local_name)


def enqueue_report_images(report_id, email, user_image='', reply_image=''):
    """Queue a bug report's image evidence. Never raises; returns how many.

    ``user_image`` is what the person attached, ``reply_image`` what the AI sent
    back — both as the data URIs ``views._snapshot_ai_report_image`` produces.
    Either can instead be a bare media URL (when the original file was already
    gone by the time the report was filed); there are no bytes behind those, so
    they are skipped rather than uploaded as a broken file.

    A reported AI image was usually archived once already at generation time.
    Storing it again under a report-numbered name is deliberate: it keeps the
    evidence for one report together and findable by report number, which is
    how anyone reviewing it will actually look for it.
    """
    queued = 0
    for label, value in (('user', user_image), ('reply', reply_image)):
        if not isinstance(value, str) or not value.startswith('data:image/'):
            continue
        if _submit(
            value, email, _extension_from_data_uri(value),
            f'report-{report_id}-{label}', subfolder=REPORTS_SUBFOLDER,
        ):
            queued += 1
    return queued


def flush(timeout=10):
    """Block until queued uploads finish. For tests and shutdown hooks only.

    Returns True if the queue drained within the timeout.
    """
    if _queue is None:
        return True
    deadline = threading.Event()
    waiter = threading.Thread(target=lambda: (_queue.join(), deadline.set()), daemon=True)
    waiter.start()
    return deadline.wait(timeout)
