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
import time
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path

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
    """Raised for any Dropbox/backup failure with a message safe to show the admin."""


def db_path():
    return Path(dj_settings.DATABASES['default']['NAME'])


def _client(settings_obj):
    if dropbox is None:
        raise BackupError("The 'dropbox' Python package isn't installed on this server.")
    if not settings_obj.is_configured:
        raise BackupError('Dropbox is not configured yet — add your App Key, App Secret and Refresh Token first.')
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


def _describe_connection_error(exc):
    text = str(exc)
    lowered = text.lower()
    if 'invalid_grant' in lowered or 'invalid_access_token' in lowered or 'expired' in lowered:
        return ('Dropbox rejected the Refresh Token. Generate a new one for this app '
                'with token_access_type=offline and paste it here.')
    if 'invalid_client' in lowered or 'app key' in lowered or 'app secret' in lowered:
        return 'Dropbox rejected the App Key or App Secret. Check both against your app in the Dropbox App Console.'
    if 'timed out' in lowered or 'connection' in lowered or 'name resolution' in lowered:
        return 'Could not reach Dropbox. Check the server\'s internet connection and try again.'
    return f'Could not connect to Dropbox: {text[:200]}'


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
            raise BackupError(_describe_connection_error(exc))
        try:
            dbx.check_user(query='ping')
        except Exception as ping_exc:
            raise BackupError(_describe_connection_error(ping_exc))
        info['limited'] = True
    cache.set(cache_key, info, ACCOUNT_CACHE_SECONDS)
    return info


def forget_account(settings_obj):
    cache.delete(f'dropbox-account-{credentials_fingerprint(settings_obj)}')


def _ensure_folder(dbx, path):
    try:
        dbx.files_create_folder_v2(path)
    except Exception as exc:
        if 'conflict' not in str(exc).lower():  # folder already exists — fine
            raise BackupError(f"Could not create the Dropbox folder '{path}': {exc}")


def create_backup(settings_obj, *, missing_images=None):
    """Upload a database-and-images ZIP and refresh the latest copies."""
    started = time.monotonic()
    try:
        dbx = _client(settings_obj)
        _ensure_folder(dbx, BACKUP_ROOT)
        _ensure_folder(dbx, BACKUP_FOLDER)

        path = db_path()
        if not path.exists():
            raise BackupError('Local database file was not found.')
        # SQLite's backup API includes committed WAL data in a consistent copy.
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / 'db.sqlite3'
            with closing(sqlite3.connect(str(path))) as source, closing(sqlite3.connect(str(snapshot))) as target:
                source.backup(target)
            data = snapshot.read_bytes()
        bundle = io.BytesIO()
        logger.info('Dropbox database snapshot ready: %d bytes', len(data))
        skipped = []
        with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr('db.sqlite3', data)
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
        logger.info('Dropbox archive ready: %d bytes; starting upload', bundle.tell())
        _upload(dbx, bundle.getvalue(), f'{BACKUP_FOLDER}/{filename}', dropbox.files.WriteMode.add)
        # The timestamped bundle is the backup. A tiny pointer replaces two
        # redundant full uploads; restore always selects the actual bundle.
        try:
            dbx.files_upload(json.dumps({'filename': filename}).encode(),
                             f'{BACKUP_FOLDER}/latest.json', mode=dropbox.files.WriteMode.overwrite)
        except Exception:
            logger.warning('Backup saved, but latest pointer could not be updated', exc_info=True)
        logger.info('Dropbox backup saved: bytes=%d elapsed=%.1fs missing=%d',
                    bundle.tell(), time.monotonic() - started, len(skipped))
        if missing_images is not None:
            missing_images.extend(skipped)
        return filename
    except BackupError:
        raise
    except Exception as exc:
        raise BackupError(f'Backup to Dropbox failed: {exc}')


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
        raise BackupError(f'Could not list Dropbox backups: {exc}')


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
        return deleted
    except BackupError:
        raise
    except Exception as exc:
        raise BackupError(f'Could not delete Dropbox backups: {exc}')


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
        raise BackupError(f'Could not download that backup from Dropbox: {exc}')

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
