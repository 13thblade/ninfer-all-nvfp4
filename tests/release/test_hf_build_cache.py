"""Check checkpoint recovery and preservation without network access."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location('hf_cache', Path(__file__).resolve().parents[2] / '.github/scripts/hf-build-cache.py')
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


class Bucket:
    def __init__(self):
        self.files = {}
        self.interrupt = False

    def get_bucket_paths_info(self, bucket, paths):
        return [SimpleNamespace(path=p, size=len(self.files[p])) for p in paths if p in self.files]

    def batch_bucket_files(self, bucket, add):
        for local, remote in add:
            self.files[remote] = Path(local).read_bytes()
            if self.interrupt and remote.endswith('.tar'):
                self.files[remote] = b'partial upload'
                raise ConnectionError('Interrupted upload')

    def download_bucket_files(self, bucket, files, **kwargs):
        for remote, local in files:
            Path(local).write_bytes(self.files[remote])

    def list_bucket_tree(self, bucket, prefix, **kwargs):
        return [SimpleNamespace(path=p) for p in self.files if p.startswith(prefix)]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.bucket = Bucket()
        self.ref = 'refs/heads/build/windows-release'
        self.family = 'linux-cuda-13.4.3'

    def save(self, content):
        (self.source / 'object').write_text(content)
        CACHE.save(self.bucket, self.source, self.family, self.ref, verify=True)

    def restored(self):
        target = self.root / 'restored'
        self.assertTrue(CACHE.restore(self.bucket, target, self.family, self.ref, required=True))
        return (target / 'object').read_text()

    def test_completed_snapshot_roundtrip_and_bounded_slots(self):
        for content in ('first', 'second', 'third'):
            self.save(content)
        self.assertEqual(self.restored(), 'third')
        self.assertEqual(len(self.bucket.files), 4)

    def test_interrupted_replacement_retains_last_completed_snapshot(self):
        self.save('first')
        self.save('second')
        self.bucket.interrupt = True
        with self.assertRaises(ConnectionError):
            self.save('third')
        self.assertEqual(self.restored(), 'second')

    def test_corrupt_newest_snapshot_falls_back(self):
        self.save('first')
        self.save('second')
        base = CACHE.prefix(self.family, self.ref)
        self.bucket.files[f'{base}/1.tar'] = b'corrupt'
        self.assertEqual(self.restored(), 'first')

    def test_empty_cache_cannot_replace_completed_checkpoint(self):
        self.save('first')
        (self.source / 'object').unlink()
        with self.assertRaises(ValueError):
            CACHE.save(self.bucket, self.source, self.family, self.ref)
        self.assertEqual(self.restored(), 'first')

    def test_other_branch_cannot_supply_objects_but_master_can(self):
        self.save('private branch')
        target = self.root / 'other'
        self.assertFalse(CACHE.restore(self.bucket, target, self.family, 'refs/heads/other'))
        CACHE.save(self.bucket, self.source, self.family, 'refs/heads/master')
        self.assertTrue(CACHE.restore(self.bucket, target, self.family, 'refs/heads/other'))

    def test_migration_merges_all_preserved_objects(self):
        for index in (1, 2):
            source = self.root / f'seed{index}'
            source.mkdir()
            (source / f'object{index}').write_text(str(index))
            archive = self.root / f'seed{index}.tar'
            CACHE.pack(source, archive)
            key = f'objects-v1-linux-cuda-13.4.3-{index}'
            remote = f'github-actions/migration/{key}.tar'
            metadata = dict(schema=1, key=key, archive=remote, source_ref=self.ref,
                            sha256=CACHE.digest(archive), files=1)
            self.bucket.files[remote] = archive.read_bytes()
            self.bucket.files[remote + '.json'] = json.dumps(metadata).encode()
        target = self.root / 'restored'
        CACHE.restore(self.bucket, target, self.family, self.ref,
                      legacy_prefix='objects-v1-linux-cuda-13.4.3-', required=True)
        self.assertEqual({p.name for p in target.iterdir()}, {'object1', 'object2'})

    def test_unsafe_members_are_rejected_before_extracting_any_files(self):
        for name, kind in (('../escape', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                           ('C:/escape', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)):
            with self.subTest(name=name):
                archive = self.root / 'unsafe.tar'
                with tarfile.open(archive, 'w') as package:
                    good = tarfile.TarInfo('good')
                    good.size = 1
                    package.addfile(good, io.BytesIO(b'x'))
                    bad = tarfile.TarInfo(name)
                    bad.type = kind
                    package.addfile(bad)
                with self.assertRaises(ValueError):
                    CACHE.unpack(archive, self.root / 'destination')
                self.assertFalse((self.root / 'destination/good').exists())

    def test_pack_excludes_mutable_statistics_and_temporary_entries(self):
        (self.source / 'tmp').mkdir()
        for name in ('object', 'stats', 'lock', 'entry.tmp', 'tmp/uncommitted'):
            (self.source / name).write_text(name)
        archive = self.root / 'test.tar'
        self.assertEqual(CACHE.pack(self.source, archive), 1)
        with tarfile.open(archive) as package:
            self.assertEqual(package.getnames(), ['object'])
