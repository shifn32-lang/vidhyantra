"""Backs up/restores db.sqlite3 to/from a Dropbox app folder, using the
store owner's Dropbox App Key/Secret + a long-lived OAuth2 refresh token
(configured from the store dashboard). Degrades gracefully if the
`dropbox` package isn't installed or credentials aren't set yet.
"""
import datetime
import hashlib
import io
import json
import logging
import re
import time
import sqlite3
import tempfile
import threading
import zipfile
from contextlib import closing
from pathlib import Path
from urllib.parse import urlencode

from django.conf import settings as dj_settings
from django.core.cache import cache

try:
    import dropbox
except ImportError:  # pragma: no cover - optional dependency until configured
    dropbox = None

BACKUP_ROOT = '/EduTrellis Store'
BACKUP_FOLDER = f'{BACKUP_ROOT}/backups'
LATEST_NAME = 'db_latest.sqlite3'
LATEST_BUNDLE_NAME = 'backup_latest.zip'
logger = logging.getLogger(__name__)


def _image_files():
    """Include uploaded images referenced by any application image field."""
    from django.apps import apps
    from django.db.models import ImageField
    seen = set()
    for model in apps.get_app_config('myapp').get_models():
        for field in model._meta.fields:
            if isinstance(field, ImageField):
                for name in model.objects.exclude(**{field.name: ''}).values_list(field.name, flat=True):
                    if name and name not in seen:
                        seen.add(name)
                        yield name, field.storage


def _upload(dbx, content, remote_path, mode):
    chunk_size = 8 * 1024 * 1024
    if len(content) <= chunk_size:
        return dbx.files_upload(content, remote_path, mode=mode)
    session = dbx.files_upload_session_start(content[:chunk_size])
    offset = chunk_size
    logger.info('Dropbox upload: %d/%d bytes', offset, len(content))
    while len(content) - offset > chunk_size:
        cursor = dropbox.files.UploadSessionCursor(session.session_id, offset)
        dbx.files_upload_session_append_v2(content[offset:offset + chunk_size], cursor)
        offset += chunk_size
        logger.info('Dropbox upload: %d/%d bytes', offset, len(content))
    return dbx.files_upload_session_finish(
        content[offset:], dropbox.files.UploadSessionCursor(session.session_id, offset),
        dropbox.files.CommitInfo(path=remote_path, mode=mode),
    )


class BackupError(Exception):
    """Raised for any Dropbox/backup failure with a message safe to show the admin.

    `code` names the kind of failure so callers can react to it: 'bad_token',
    'bad_code', 'bad_client', 'scope', 'network', or '' for anything else."""

    def __init__(self, message, code=''):
        super().__init__(message)
        self.code = code


def db_path():
    return Path(dj_settings.DATABASES['default']['NAME'])


def _client(settings_obj):
    if dropbox is None:
        raise BackupError("The 'dropbox' Python package isn't installed on this server.")
    if not settings_obj.is_configured:
        raise BackupError('Dropbox is not connected yet. Add the App Key, App Secret and Dropbox code on this page first.', 'not_connected')
    return dropbox.Dropbox(
        oauth2_refresh_token=settings_obj.effective_refresh_token,
        app_key=settings_obj.effective_app_key,
        app_secret=settings_obj.effective_app_secret,
        timeout=30,
        max_retries_on_error=1,
        max_retries_on_rate_limit=0,
    )


ACCOUNT_CACHE_SECONDS = 300


def credentials_fingerprint(settings_obj):
    """Identifies which Dropbox app/account a set of credentials points at."""
    raw = '|'.join((
        settings_obj.effective_app_key, settings_obj.effective_app_secret,
        settings_obj.effective_refresh_token,
    ))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


GET_CODE_HELP = (
    'Click “Get Dropbox code”, press Allow in the Dropbox window, then copy the code '
    'it shows and paste it into the third box.'
)
CLIENT_HELP = (
    "Dropbox doesn't recognise this App Key and App Secret. Copy both again from the Settings tab of your "
    'app at dropbox.com/developers/apps, with no spaces before or after.'
)
ACCESS_TOKEN_HELP = 'That is a short-lived access token, which stops working after about four hours. ' + GET_CODE_HELP
TOKEN_URL = 'https://api.dropboxapi.com/oauth2/token'


def classify_error(exc):
    """Turns whatever Dropbox or the network raised into (kind, plain-English message)."""
    text = str(exc)
    lowered = text.lower()
    if 'missing_scope' in lowered:
        scope = re.search(r"required_scope='([\w.]+)'", text)
        needed = f' ({scope.group(1)})' if scope else ''
        return 'scope', (
            f"Your Dropbox app doesn't have a permission it needs{needed}. Open your app at "
            'dropbox.com/developers/apps, go to the Permissions tab, tick files.content.write, '
            'files.content.read and account_info.read, press Submit. Then connect again: ' + GET_CODE_HELP[0].lower() + GET_CODE_HELP[1:])
    if 'invalid_grant' in lowered or 'invalid_access_token' in lowered or 'expired' in lowered:
        return 'bad_token', (
            'Dropbox no longer accepts the saved connection key (the Refresh Token). It may be incomplete, '
            'revoked, or made for a different Dropbox app. ' + GET_CODE_HELP)
    if 'invalid_client' in lowered or '401 client error' in lowered or 'app key' in lowered or 'app secret' in lowered:
        return 'bad_client', CLIENT_HELP
    if 'insufficient_space' in lowered:
        return 'full', 'Your Dropbox is full. Free up some space or upgrade the Dropbox plan, then try again.'
    if 'too_many_requests' in lowered or 'rate_limit' in lowered or '429' in lowered:
        return 'busy', 'Dropbox asked us to slow down. Wait a minute and try again.'
    if 'timed out' in lowered or 'timeout' in lowered or 'connection' in lowered or 'max retries' in lowered or 'name resolution' in lowered:
        return 'network', "Could not reach Dropbox. Check the server's internet connection and try again."
    return '', f"Dropbox sent back an error we don't recognise. Details for support: {text[:200]}"


def _connection_error(exc, intro=''):
    kind, message = classify_error(exc)
    return BackupError(f'{intro} {message}'.strip(), kind)


def looks_like_access_token(value):
    return (value or '').startswith('sl.')


def authorize_url(app_key):
    """The Dropbox page where the owner presses Allow and is shown a one-time code."""
    if not app_key:
        return ''
    return 'https://www.dropbox.com/oauth2/authorize?' + urlencode({
        'client_id': app_key, 'response_type': 'code', 'token_access_type': 'offline',
    })


def exchange_access_code(app_key, app_secret, code):
    """Turns the one-time code Dropbox shows after "Allow" into a Refresh Token.

    A code works once and only for a few minutes, so this runs the moment it is
    pasted. The permanent Refresh Token it returns is what gets saved."""
    try:
        import requests
        response = requests.post(TOKEN_URL, data={
            'grant_type': 'authorization_code', 'code': code,
            'client_id': app_key, 'client_secret': app_secret,
        }, timeout=20)
    except Exception as exc:
        raise _connection_error(exc, 'Could not check the code with Dropbox.')
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    if response.ok and body.get('refresh_token'):
        return body['refresh_token']
    if response.ok:
        raise BackupError(
            'Dropbox accepted the code but did not give a permanent connection. Use the “Get Dropbox code” '
            'button, which asks Dropbox for permanent access, instead of a code from somewhere else.', 'bad_code')
    error = body.get('error')
    if error == 'invalid_client' or response.status_code == 401:
        raise BackupError(CLIENT_HELP, 'bad_client')
    if error == 'invalid_grant':
        raise BackupError(
            'Dropbox did not accept that code. A code works only once, stops working after a few minutes, '
            'and must come from the same App Key as the one above. ' + GET_CODE_HELP, 'bad_code')
    detail = str(body.get('error_description') or error or response.status_code)[:120]
    raise BackupError(f'Dropbox refused the code ({detail}). ' + GET_CODE_HELP, 'bad_code')


def account_info(settings_obj, *, refresh=False):
    """Who the credentials belong to: name, email, folder and space used.

    Cached for a few minutes (keyed by the credentials) so opening the backup
    page doesn't make extra Dropbox calls every time. Raises BackupError when
    Dropbox rejects the credentials. An app without the account_info.read
    permission still counts as connected, just without the name and email.
    """
    cache_key = f'dropbox-account-{credentials_fingerprint(settings_obj)}'
    if not refresh:
        cached = cache.get(cache_key)
        if cached:
            return cached
    dbx = _client(settings_obj)
    info = {
        'name': '', 'email': '', 'used': None, 'allocated': None,
        'limited': False, 'folder': BACKUP_FOLDER,
    }
    try:
        account = dbx.users_get_current_account()
        info['name'] = account.name.display_name
        info['email'] = account.email
        try:
            usage = dbx.users_get_space_usage()
            info['used'] = usage.used
            allocation = usage.allocation
            if allocation.is_individual():
                info['allocated'] = allocation.get_individual().allocated
            elif allocation.is_team():
                info['allocated'] = allocation.get_team().allocated
        except Exception:
            logger.info('Dropbox space usage unavailable', exc_info=True)
    except Exception as exc:
        if 'missing_scope' not in str(exc):
            raise _connection_error(exc)
        try:
            dbx.check_user(query='ping')
        except Exception as ping_exc:
            raise _connection_error(ping_exc)
        info['limited'] = True
    cache.set(cache_key, info, ACCOUNT_CACHE_SECONDS)
    return info


def forget_account(settings_obj):
    cache.delete(f'dropbox-account-{credentials_fingerprint(settings_obj)}')
    forget_listing(settings_obj)


def _ensure_folder(dbx, path):
    try:
        dbx.files_create_folder_v2(path)
    except Exception as exc:
        if 'conflict' not in str(exc).lower():  # folder already exists — fine
            raise _connection_error(exc, f"Could not create the Dropbox folder '{path}'.")


UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024


def _snapshot(path, target):
    """A compact copy of the database: VACUUM INTO leaves out the free pages a
    busy database accumulates (often half the file). The plain SQLite backup
    copy is the fallback for a SQLite too old to have it."""
    try:
        with closing(sqlite3.connect(str(path))) as source:
            source.execute('VACUUM INTO ?', (str(target),))
    except sqlite3.Error:
        Path(target).unlink(missing_ok=True)
        with closing(sqlite3.connect(str(path))) as source, closing(sqlite3.connect(str(target))) as copy:
            source.backup(copy)


def _upload_file(dbx, file_path, remote_path, mode, progress=None):
    """Upload a file in pieces read from disk, so a large backup never has to
    sit in memory. Each piece is tried up to three times."""
    chunk_size = UPLOAD_CHUNK_BYTES
    total = Path(file_path).stat().st_size
    with open(file_path, 'rb') as handle:
        if total <= chunk_size:
            return dbx.files_upload(handle.read(), remote_path, mode=mode)

        def attempt(action):
            for number in range(3):
                try:
                    return action()
                except Exception:
                    if number == 2:
                        raise
                    time.sleep(1 + number)

        first = handle.read(chunk_size)
        session = attempt(lambda: dbx.files_upload_session_start(first))
        offset = len(first)
        while total - offset > chunk_size:
            piece = handle.read(chunk_size)
            cursor = dropbox.files.UploadSessionCursor(session.session_id, offset)
            attempt(lambda: dbx.files_upload_session_append_v2(piece, cursor))
            offset += len(piece)
            if progress:
                progress(f'Uploading to Dropbox: {offset // (1024 * 1024)} of {total // (1024 * 1024)} MB')
        last = handle.read()
        cursor = dropbox.files.UploadSessionCursor(session.session_id, offset)
        commit = dropbox.files.CommitInfo(path=remote_path, mode=mode)
        return attempt(lambda: dbx.files_upload_session_finish(last, cursor, commit))


def create_backup(settings_obj, *, missing_images=None, progress=None):
    """Upload a database-and-images ZIP and refresh the latest copies."""
    started = time.monotonic()
    say = progress or (lambda text: None)
    try:
        dbx = _client(settings_obj)
        _ensure_folder(dbx, BACKUP_ROOT)
        _ensure_folder(dbx, BACKUP_FOLDER)

        path = db_path()
        if not path.exists():
            raise BackupError('Local database file was not found.')
        skipped = []
        with tempfile.TemporaryDirectory() as folder:
            say('Preparing the database…')
            snapshot = Path(folder) / 'db.sqlite3'
            _snapshot(path, snapshot)
            logger.info('Dropbox database snapshot ready: %d bytes', snapshot.stat().st_size)
            say('Packing the database and images…')
            bundle_path = Path(folder) / 'bundle.zip'
            with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                archive.write(snapshot, 'db.sqlite3')
                for name, storage in _image_files():
                    try:
                        with storage.open(name, 'rb') as image:
                            image_data = image.read()
                    except FileNotFoundError:
                        skipped.append(name)
                        continue
                    archive.writestr('media/' + name.replace('\\', '/'), image_data, compress_type=zipfile.ZIP_STORED)
                archive.writestr('backup_manifest.json', json.dumps({'version': 1, 'missing_images': skipped}))

            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'backup_{stamp}.zip'
            size = bundle_path.stat().st_size
            logger.info('Dropbox archive ready: %d bytes; starting upload', size)
            say(f'Uploading to Dropbox: 0 of {size // (1024 * 1024)} MB')
            _upload_file(dbx, bundle_path, f'{BACKUP_FOLDER}/{filename}', dropbox.files.WriteMode.add, progress=say)
        # The timestamped bundle is the backup. A tiny pointer replaces two
        # redundant full uploads; restore always selects the actual bundle.
        try:
            dbx.files_upload(json.dumps({'filename': filename}).encode(),
                             f'{BACKUP_FOLDER}/latest.json', mode=dropbox.files.WriteMode.overwrite)
        except Exception:
            logger.warning('Backup saved, but latest pointer could not be updated', exc_info=True)
        forget_listing(settings_obj)
        logger.info('Dropbox backup saved: bytes=%d elapsed=%.1fs missing=%d',
                    size, time.monotonic() - started, len(skipped))
        if missing_images is not None:
            missing_images.extend(skipped)
        return filename
    except BackupError:
        raise
    except Exception as exc:
        raise _connection_error(exc, 'The backup was not saved to Dropbox.')


# ---- running a backup in the background -----------------------------------
# A backup of a large database takes minutes, longer than a web request is
# allowed to run, so it runs in a thread and the page shows its progress. The
# status lives in a small file so every server process sees the same state.
STATUS_FILE = Path(tempfile.gettempdir()) / 'vidhyora_dropbox_backup.json'
JOB_STALE_SECONDS = 60 * 60
_job_lock = threading.Lock()


def _write_status(**fields):
    try:
        current = job_status() or {}
        current.update(fields)
        STATUS_FILE.write_text(json.dumps(current), encoding='utf-8')
    except OSError:
        logger.warning('Could not write the backup status file', exc_info=True)


def job_status():
    """{'state': 'running'|'done'|'error', 'message', 'seen', ...} or None."""
    try:
        status = json.loads(STATUS_FILE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if status.get('state') == 'running' and time.time() - status.get('started', 0) > JOB_STALE_SECONDS:
        status.update(state='error', message='The backup stopped before it finished. Please start it again.', seen=False)
    return status


def mark_status_seen():
    _write_status(seen=True)


def start_backup_job(settings_obj):
    """Start a backup in the background. False when one is already running."""
    with _job_lock:
        status = job_status()
        if status and status.get('state') == 'running':
            return False
        _write_status(state='running', started=time.time(), message='Starting…', seen=False, filename='', skipped=[])

    def run():
        try:
            skipped = []
            filename = create_backup(settings_obj, missing_images=skipped, progress=lambda text: _write_status(message=text))
            _write_status(state='done', message=f'Backup saved to Dropbox as "{filename}".', filename=filename, skipped=skipped[:20], seen=False)
        except BackupError as exc:
            _write_status(state='error', message=str(exc), seen=False)
        except Exception as exc:
            logger.exception('Background Dropbox backup failed')
            _write_status(state='error', message=f'The backup failed unexpectedly ({exc.__class__.__name__}). Please try again.', seen=False)
        finally:
            from django.db import connection
            connection.close()

    threading.Thread(target=run, name='dropbox-backup', daemon=True).start()
    return True


LISTING_CACHE_SECONDS = 120


def _listing_key(settings_obj):
    return f'dropbox-listing-{credentials_fingerprint(settings_obj)}'


def forget_listing(settings_obj):
    try:
        cache.delete(_listing_key(settings_obj))
    except AttributeError:
        pass


def list_backup_entries(settings_obj, *, refresh=False):
    """[(name, client_modified)] newest first; kept for two minutes so opening
    the page does not ask Dropbox every time."""
    key = _listing_key(settings_obj)
    if not refresh:
        cached = cache.get(key)
        if cached is not None:
            return cached
    entries = [(f.name, f.client_modified) for f in list_backups(settings_obj)]
    cache.set(key, entries, LISTING_CACHE_SECONDS)
    return entries


def list_backups(settings_obj):
    """Returns timestamped backup files (newest first), excluding db_latest.sqlite3."""
    try:
        dbx = _client(settings_obj)
        try:
            res = dbx.files_list_folder(BACKUP_FOLDER)
        except Exception as exc:
            if 'not_found' in str(exc).lower():
                return []
            raise

        entries = list(res.entries)
        while res.has_more:
            res = dbx.files_list_folder_continue(res.cursor)
            entries.extend(res.entries)

        files = [e for e in entries if isinstance(e, dropbox.files.FileMetadata) and e.name not in (LATEST_NAME, LATEST_BUNDLE_NAME) and e.name.endswith(('.sqlite3', '.zip'))]
        files.sort(key=lambda e: e.name, reverse=True)
        return files
    except BackupError:
        raise
    except Exception as exc:
        raise _connection_error(exc, 'Could not list the backups on Dropbox.')


def delete_all_backups(settings_obj):
    """Delete every file/folder inside the dedicated Dropbox backup folder.

    The backup folder itself and everything outside it are deliberately left
    untouched. This includes deleting db_latest.sqlite3, which list_backups()
    normally hides from the restore-file list.
    """
    try:
        dbx = _client(settings_obj)
        try:
            res = dbx.files_list_folder(BACKUP_FOLDER)
        except Exception as exc:
            if 'not_found' in str(exc).lower():
                return 0
            raise

        entries = list(res.entries)
        while res.has_more:
            res = dbx.files_list_folder_continue(res.cursor)
            entries.extend(res.entries)

        folder_prefix = BACKUP_FOLDER.lower().rstrip('/') + '/'
        deleted = 0
        for entry in entries:
            path = getattr(entry, 'path_lower', '') or ''
            # Refuse to pass any unexpected path to Dropbox's destructive API,
            # even if a malformed/mock response somehow puts it in the listing.
            if not path.startswith(folder_prefix):
                continue
            dbx.files_delete_v2(path)
            deleted += 1
        forget_listing(settings_obj)
        return deleted
    except BackupError:
        raise
    except Exception as exc:
        raise _connection_error(exc, 'Could not delete the backups on Dropbox.')


def restore_backup(settings_obj, filename):
    """Restore an image bundle or a legacy database-only backup."""
    if not filename or '/' in filename or '\\' in filename:
        raise BackupError('Invalid backup filename.')

    try:
        dbx = _client(settings_obj)
        _, resp = dbx.files_download(f'{BACKUP_FOLDER}/{filename}')
        content = resp.content
    except BackupError:
        raise
    except Exception as exc:
        raise _connection_error(exc, 'Could not download that backup from Dropbox.')

    media_files = []
    if filename.endswith('.zip'):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                root = Path(dj_settings.MEDIA_ROOT).resolve()
                for entry in archive.infolist():
                    if entry.filename in ('db.sqlite3', 'backup_manifest.json'):
                        continue
                    if not entry.filename.startswith('media/') or entry.is_dir():
                        raise BackupError('Invalid file in backup archive.')
                    relative = entry.filename[6:]
                    target = (root / relative).resolve()
                    if '\\' in relative or ':' in relative or not target.is_relative_to(root) or target == root:
                        raise BackupError('Unsafe image path in backup archive.')
                    media_files.append((target, archive.read(entry)))
                content = archive.read('db.sqlite3')
        except (zipfile.BadZipFile, KeyError, OSError) as exc:
            raise BackupError('The backup archive is damaged or incomplete.') from exc
    if not content.startswith(b'SQLite format 3\x00'):
        raise BackupError('The backup does not contain a valid SQLite database.')

    from django.db import connections
    connections.close_all()

    path = db_path()
    tmp_path = path.with_suffix(path.suffix + '.restoring')
    replaced_images = []
    try:
        for target, image_content in media_files:
            previous = target.read_bytes() if target.exists() else None
            target.parent.mkdir(parents=True, exist_ok=True)
            image_tmp = target.with_name(target.name + '.restoring')
            image_tmp.write_bytes(image_content)
            image_tmp.replace(target)
            replaced_images.append((target, previous))
        tmp_path.write_bytes(content)
        tmp_path.replace(path)
    except OSError as exc:
        for target, previous in reversed(replaced_images):
            if previous is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(previous)
        raise BackupError(
            f'Could not write the restored database (it may be locked by the running server): {exc}'
        )
