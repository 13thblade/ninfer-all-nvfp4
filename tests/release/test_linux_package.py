"""Check archive contents and failure boundaries with fixture executables, without a GPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('linux_package', ROOT / 'scripts/package-linux-release.py')
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


class LinuxPackageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.cuda = self.root / 'cuda'
        self.build = self.repo / 'build'
        for directory in ('scripts', 'docs', 'docker', 'build/portable/backends/cuda13',):
            (self.repo / directory).mkdir(parents=True)
        for directory in ('bin', 'lib64/stubs'):
            (self.cuda / directory).mkdir(parents=True)
        for name in ('bin/nvcc', 'lib64/stubs/libcuda.so', 'EULA.txt'):
            (self.cuda / name).write_text('fixture')
        for name in ('portable/ninfer', 'portable/ninfer-serve', 'portable/ninfer_bench',
                     'portable/ninfer-calibrate', 'portable/ninfer-perplexity', 'portable/ninfer-ngram-profile',
                     'portable/backends/cuda13/libninfer-runtime.so',
                     'portable/backends/cuda13/architectures.txt'):
            (self.build / name).write_text('binary fixture')
        (self.build / 'portable/backends/cuda13/architectures.txt').write_text('75 120a\n')
        (self.build / 'CMakeCache.txt').write_text(
            f'CMAKE_CUDA_COMPILER:FILEPATH={self.cuda}/bin/nvcc\n'
            'CMAKE_CUDA_ARCHITECTURES:STRING=75;120a\nNINFER_SHARED_RUNTIME:BOOL=ON\n')
        for name in ('LICENSE', 'RELEASE_NOTES_0.0.md', 'docs/release-archive-linux.md',
                     'scripts/run.sh', 'scripts/download-model.sh'):
            (self.repo / name).write_text('fixture')
        (self.repo / 'VERSION').write_text('0.0-undefined')
        shutil.copy2(ROOT / 'docker/entrypoint.sh', self.repo / 'docker/entrypoint.sh')
        shutil.copy2(ROOT / 'docker/release.Dockerfile', self.repo / 'docker/release.Dockerfile')
        self.images = 'ELF file: sm_75.cubin\nELF file: sm_120a.cubin'
        self.help_failure = False
        self.original_read = Path.read_text
        def read_text(path, *args, **kwargs):
            if str(path) == '/etc/os-release':
                return 'ID=debian\nVERSION_ID="13"\n'
            return self.original_read(path, *args, **kwargs)
        self.stack = [patch.object(PACKAGE, '__file__', str(self.repo / 'scripts/package-linux-release.py')),
                      patch.dict(os.environ, NINFER_BUILD_ROOT=str(self.build), GITHUB_ACTIONS='true'),
                      patch.object(Path, 'read_text', read_text),
                      patch.object(PACKAGE, 'output', self.output),
                      patch.object(PACKAGE, 'run', self.run_command),
                      patch.object(PACKAGE, 'runtime_packages', return_value={'libc6:amd64'})]
        for item in self.stack:
            item.start()
            self.addCleanup(item.stop)

    def output(self, *args, **kwargs):
        if args[0] == 'git':
            return 'a' * 40 if 'rev-parse' in args else ''
        if args[0] == 'ldd':
            return 'libc.so.6 => /usr/lib/libc.so.6 (0x1)'
        if args[0] == 'getconf':
            return 'glibc 2.41'
        if args[0].endswith('/nvcc'):
            return 'Cuda compilation tools, release 13.4, V13.4.92'
        if args[0].endswith('/cuobjdump'):
            return self.images
        if args[-1] in ('--help', '--version'):
            if self.help_failure:
                raise RuntimeError('fixture startup failure')
            return 'fixture help'
        raise AssertionError(args)

    def run_command(self, *args, **kwargs):
        if args[0] in ('strip', 'patchelf'):
            return
        if args[-1] in ('--help', '--version'):
            if self.help_failure:
                raise RuntimeError('fixture startup failure')
            self.assertTrue(Path(args[0]).is_file())
            return
        raise AssertionError(args)

    def test_archive_roundtrip_and_runtime_docker_context(self):
        PACKAGE.main()
        dist = self.repo / 'dist'
        volume = next(dist.glob('*.tar.gz.001'))
        self.assertNotIn('undefined', volume.name)
        self.assertIn('-aaaaaaaaaaaa.tar.gz.001', volume.name)
        manifest = next(dist.glob('SHA256SUMS-*.txt')).read_text()
        self.assertEqual(manifest, f'{hashlib.sha256(volume.read_bytes()).hexdigest()}  {volume.name}\n')
        with tarfile.open(volume, 'r:gz') as archive:
            names = archive.getnames()
            self.assertFalse(any('libcuda.so' in name for name in names))
            prefix = names[0]
            self.assertTrue(archive.getmember(prefix + '/ninfer').isfile())
            self.assertTrue(archive.getmember(prefix + '/backends/cuda13/libninfer-runtime.so').isfile())
            self.assertIn(prefix + '/Dockerfile', names)
            info = json.load(archive.extractfile(prefix + '/BUILD-INFO.json'))
            self.assertEqual(info['architectures'], ['75', '120a'])

    def test_missing_device_image_cannot_produce_manifest(self):
        self.images = 'ELF file: sm_75.cubin'
        with self.assertRaisesRegex(RuntimeError, 'lacks sm_120a'):
            PACKAGE.main()
        self.assertFalse(list((self.repo / 'dist').glob('SHA256SUMS-*')))

    def test_commit_archive_does_not_require_release_notes(self):
        (self.repo / 'RELEASE_NOTES_0.0.md').unlink()
        PACKAGE.main()
        volume = next((self.repo / 'dist').glob('*.tar.gz.001'))
        with tarfile.open(volume, 'r:gz') as archive:
            names = archive.getnames()
            self.assertFalse(any('RELEASE_NOTES_' in name for name in names))
            self.assertTrue(any(name.endswith('/BUILD-INFO.json') for name in names))

    def test_failed_startup_removes_old_manifest(self):
        PACKAGE.main()
        self.help_failure = True
        with self.assertRaisesRegex(RuntimeError, 'startup failure'):
            PACKAGE.main()
        self.assertFalse(list((self.repo / 'dist').glob('SHA256SUMS-*')))


class DependencyTest(unittest.TestCase):
    def test_unresolved_library_is_rejected(self):
        with patch.object(PACKAGE, 'output', return_value='libavcodec.so.61 => not found'):
            with self.assertRaisesRegex(RuntimeError, 'Unresolved libraries'):
                PACKAGE.runtime_packages(Path('/binary'), {}, Path('/stub'), Path('/cuda'), Path('/bundle'))

    def test_unowned_library_is_rejected(self):
        with patch.object(PACKAGE, 'output', return_value='libavcodec.so.61 => /unowned/libavcodec.so.61 (0x1)'), \
             patch.object(PACKAGE.subprocess, 'run') as run:
            run.return_value.returncode = 1
            with self.assertRaisesRegex(RuntimeError, 'No Debian package owns'):
                PACKAGE.runtime_packages(Path('/binary'), {}, Path('/stub'), Path('/cuda'), Path('/bundle'))


if __name__ == '__main__':
    unittest.main()
