import io
import json
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import SimpleTestCase, override_settings
from myapp import dropbox_backup as backup


class BackupBundleTests(SimpleTestCase):
    def test_missing_avatar_is_reported_without_blocking_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'db.sqlite3'
            with closing(sqlite3.connect(path)) as connection:
                connection.execute('CREATE TABLE example (id INTEGER)')
                connection.commit()
            storage = Mock()
            storage.open.side_effect = [FileNotFoundError('avatars/ll.png'), io.BytesIO(b'icon')]
            dbx = Mock()
            missing = []
            with patch.object(backup, 'db_path', return_value=path), patch.object(backup, '_client', return_value=dbx), patch.object(backup, '_image_files', return_value=[('avatars/ll.png', storage), ('pwa/icon.png', storage)]):
                backup.create_backup(SimpleNamespace(), missing_images=missing)
            self.assertEqual(missing, ['avatars/ll.png'])
            with zipfile.ZipFile(io.BytesIO(dbx.files_upload.call_args_list[0].args[0])) as archive:
                self.assertEqual(archive.read('media/pwa/icon.png'), b'icon')
                self.assertNotIn('media/avatars/ll.png', archive.namelist())
                self.assertEqual(json.loads(archive.read('backup_manifest.json'))['missing_images'], missing)

    def test_legacy_database_backup_still_restores(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'db.sqlite3'
            with closing(sqlite3.connect(path)) as connection:
                connection.execute('CREATE TABLE legacy (id INTEGER)')
                connection.commit()
            content = path.read_bytes()
            path.write_bytes(b'changed')
            dbx = Mock()
            dbx.files_download.return_value = (None, SimpleNamespace(content=content))
            with patch.object(backup, '_client', return_value=dbx), patch.object(backup, 'db_path', return_value=path):
                backup.restore_backup(SimpleNamespace(), 'db_old.sqlite3')
            self.assertEqual(path.read_bytes(), content)

    def test_round_trip_restores_database_and_branding_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / 'db.sqlite3'
            with closing(sqlite3.connect(database)) as connection:
                connection.execute('CREATE TABLE example (value TEXT)')
                connection.execute("INSERT INTO example VALUES ('original')")
                connection.commit()
            storage = Mock()
            storage.open.side_effect = lambda name, mode: io.BytesIO(b'image-' + name.encode())
            dbx = Mock()
            with override_settings(MEDIA_ROOT=root / 'media'), patch.object(backup, 'db_path', return_value=database), patch.object(backup, '_client', return_value=dbx), patch.object(backup, '_image_files', return_value=[('pwa/icon.png', storage), ('branding/favicon.ico', storage), ('branding/social/share.png', storage)]):
                filename = backup.create_backup(SimpleNamespace())
                self.assertEqual(dbx.files_upload.call_count, 2)
                self.assertTrue(dbx.files_upload.call_args_list[1].args[1].endswith('/latest.json'))
                self.assertLess(len(dbx.files_upload.call_args_list[1].args[0]), 200)
                content = dbx.files_upload.call_args_list[0].args[0]
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    self.assertIn('media/pwa/icon.png', archive.namelist())
                    self.assertIn('media/branding/favicon.ico', archive.namelist())
                database.write_bytes(b'changed')
                dbx.files_download.return_value = (None, SimpleNamespace(content=content))
                backup.restore_backup(SimpleNamespace(), filename)
                with closing(sqlite3.connect(database)) as connection:
                    self.assertEqual(connection.execute('SELECT value FROM example').fetchone()[0], 'original')
                self.assertEqual((root / 'media/pwa/icon.png').read_bytes(), b'image-pwa/icon.png')

    def test_unsafe_archive_is_rejected_before_database_changes(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as archive:
            archive.writestr('db.sqlite3', b'SQLite format 3\x00')
            archive.writestr('media/../../outside.png', b'bad')
        dbx = Mock()
        dbx.files_download.return_value = (None, SimpleNamespace(content=data.getvalue()))
        with tempfile.TemporaryDirectory() as folder, override_settings(MEDIA_ROOT=folder), patch.object(backup, '_client', return_value=dbx):
            with self.assertRaises(backup.BackupError):
                backup.restore_backup(SimpleNamespace(), 'backup.zip')


class LargeBackupTests(SimpleTestCase):
    def test_a_large_file_is_uploaded_in_pieces_read_from_disk(self):
        data = bytes(range(256)) * 10
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'big.zip'
            path.write_bytes(data)
            dbx = Mock()
            dbx.files_upload_session_start.return_value = SimpleNamespace(session_id='s1')
            progress = []
            with patch.object(backup, 'UPLOAD_CHUNK_BYTES', 1000):
                backup._upload_file(dbx, path, '/x/big.zip', backup.dropbox.files.WriteMode.add, progress=progress.append)
        sent = dbx.files_upload_session_start.call_args.args[0]
        sent += b''.join(call.args[0] for call in dbx.files_upload_session_append_v2.call_args_list)
        sent += dbx.files_upload_session_finish.call_args.args[0]
        self.assertEqual(sent, data)
        dbx.files_upload.assert_not_called()
        self.assertTrue(progress)

    def test_a_failed_piece_is_retried(self):
        data = b'x' * 2500
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'big.zip'
            path.write_bytes(data)
            dbx = Mock()
            dbx.files_upload_session_start.return_value = SimpleNamespace(session_id='s1')
            dbx.files_upload_session_append_v2.side_effect = [OSError('reset'), None, None]
            with patch.object(backup, 'UPLOAD_CHUNK_BYTES', 1000), patch.object(backup.time, 'sleep'):
                backup._upload_file(dbx, path, '/x/big.zip', backup.dropbox.files.WriteMode.add)
        self.assertEqual(dbx.files_upload_session_append_v2.call_count, 2)

    def test_the_snapshot_leaves_out_free_space(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'db.sqlite3'
            with closing(sqlite3.connect(source)) as connection:
                connection.execute('CREATE TABLE blobs (data BLOB)')
                connection.executemany('INSERT INTO blobs VALUES (?)', [(b'x' * 50000,)] * 40)
                connection.commit()
                connection.execute('DELETE FROM blobs')
                connection.commit()
            copy = Path(folder) / 'copy.sqlite3'
            backup._snapshot(source, copy)
            self.assertLess(copy.stat().st_size, source.stat().st_size / 4)
            with closing(sqlite3.connect(copy)) as connection:
                self.assertEqual(connection.execute('SELECT count(*) FROM blobs').fetchone()[0], 0)

    def test_the_background_job_reports_progress_and_the_result(self):
        with tempfile.TemporaryDirectory() as folder:
            status_file = Path(folder) / 'status.json'

            def fake_create(settings_obj, *, missing_images=None, progress=None):
                progress('Uploading to Dropbox: 5 of 10 MB')
                missing_images.append('avatars/gone.png')
                return 'backup_x.zip'

            with patch.object(backup, 'STATUS_FILE', status_file), patch.object(backup, 'create_backup', side_effect=fake_create):
                self.assertTrue(backup.start_backup_job(SimpleNamespace()))
                for thread in __import__('threading').enumerate():
                    if thread.name == 'dropbox-backup':
                        thread.join(5)
                status = backup.job_status()
        self.assertEqual(status['state'], 'done')
        self.assertIn('backup_x.zip', status['message'])
        self.assertEqual(status['skipped'], ['avatars/gone.png'])

    def test_a_second_backup_cannot_start_while_one_is_running(self):
        with tempfile.TemporaryDirectory() as folder:
            status_file = Path(folder) / 'status.json'
            import time as _time
            status_file.write_text(json.dumps({'state': 'running', 'started': _time.time(), 'message': 'Uploading'}))
            with patch.object(backup, 'STATUS_FILE', status_file), patch.object(backup, 'create_backup') as create:
                self.assertFalse(backup.start_backup_job(SimpleNamespace()))
            create.assert_not_called()

    def test_a_backup_that_never_finished_is_not_shown_as_running_forever(self):
        with tempfile.TemporaryDirectory() as folder:
            status_file = Path(folder) / 'status.json'
            status_file.write_text(json.dumps({'state': 'running', 'started': 1, 'message': 'Uploading'}))
            with patch.object(backup, 'STATUS_FILE', status_file):
                self.assertEqual(backup.job_status()['state'], 'error')
