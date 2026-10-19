"""Exercise publication boundaries with real archives/manifests and a fake GitHub CLI."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / '.github/scripts/publish-windows-release.sh'
SHA = 'a' * 40
VARIANTS = ('cuda12.9-sm70-75-80-86-89', 'cuda13.2-sm75-80-86-89-120a')

GH = r'''#!/usr/bin/env python3.11
import hashlib, json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['GH_CALLS'], 'a') as log:
    log.write(json.dumps(args) + '\n')
mode = os.environ.get('MODE', '')
if args[0] == 'api':
    route = next(a for a in args[1:] if a.startswith('repos/'))
    if mode == 'api-failure':
        sys.exit(1)
    if route.endswith('/commits/master'):
        calls = pathlib.Path(os.environ['GH_CALLS']).read_text()
        stale = mode == 'superseded' or (mode == 'superseded-upload' and '"upload"' in calls)
        print('b' * 40 if stale else os.environ['GITHUB_SHA'])
    elif route.endswith('/releases?per_page=100'):
        if mode in ('draft', 'published'):
            print('1\t' + ('true' if mode == 'draft' else 'false'))
    elif '/releases/tags/' in route:
        if '--jq' in args:
            print(os.environ['GITHUB_SHA'])
            sys.exit(0)
        assets = []
        for p in pathlib.Path('.').iterdir():
            if '.7z.' in p.name or p.name.startswith('SHA256SUMS-'):
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                assets.append(dict(name=p.name, size=p.stat().st_size,
                                   digest='sha256:' + ('0'*64 if mode == 'bad-remote' else digest)))
        print(json.dumps({'assets': assets}))
    elif route.endswith('/releases/latest'):
        print('windows-' + os.environ['GITHUB_SHA'][:12])
    else:
        raise SystemExit('Unexpected API call: ' + route)
elif args[:2] not in (['release', 'create'], ['release', 'upload'], ['release', 'edit']):
    raise SystemExit('Unexpected gh call: ' + repr(args))
'''


class WindowsPublicationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dist = self.root / 'dist'
        self.dist.mkdir()
        bindir = self.root / 'bin'
        bindir.mkdir()
        (bindir / 'gh').write_text(GH)
        (bindir / 'gh').chmod(0o755)
        self.calls = self.root / 'calls.jsonl'
        self.env = dict(os.environ, PATH=f'{bindir}:{os.environ["PATH"]}',
                        GITHUB_SHA=SHA, GITHUB_REPOSITORY='test/ninfer',
                        GITHUB_REF='refs/heads/master', GH_CALLS=str(self.calls))
        for variant in VARIANTS:
            stem = f'ninfer-windows-x64-{variant}-0.12.0-rtx3090-{SHA[:12]}'
            volume = self.dist / f'{stem}.7z.001'
            volume.write_bytes(b'archive fixture ' + variant.encode())
            digest = hashlib.sha256(volume.read_bytes()).hexdigest()
            (self.dist / f'SHA256SUMS-{stem}.txt').write_text(f'{digest}  {volume.name}\n')

    def run_publish(self, mode='', success=True):
        result = subprocess.run(['bash', str(SCRIPT)], cwd=self.root,
                                env=dict(self.env, MODE=mode), text=True, capture_output=True)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return [json.loads(s) for s in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_complete_matrix_publishes_after_remote_verification(self):
        calls = self.run_publish()
        edit = next(i for i, c in enumerate(calls) if c[:2] == ['release', 'edit'])
        verification = next(i for i, c in enumerate(calls) if any('/releases/tags/' in a for a in c))
        self.assertLess(verification, edit)
        self.assertIn('--latest', calls[edit])
        self.assertTrue(any(c[:2] == ['release', 'create'] and '--draft' in c for c in calls))

    def test_corrupt_archive_never_contacts_github(self):
        next(self.dist.glob('*.7z.001')).write_bytes(b'corrupt')
        self.assertEqual(self.run_publish(success=False), [])

    def test_missing_variant_never_contacts_github(self):
        next(self.dist.glob('SHA256SUMS-*')).unlink()
        self.assertEqual(self.run_publish(success=False), [])

    def test_unlisted_volume_never_contacts_github(self):
        (self.dist / 'ninfer-windows-extra.7z.002').write_bytes(b'extra')
        self.assertEqual(self.run_publish(success=False), [])

    def test_superseded_source_cannot_publish(self):
        calls = self.run_publish('superseded')
        self.assertFalse(any(c[0] == 'release' for c in calls))

    def test_api_failure_does_not_create_release(self):
        calls = self.run_publish('api-failure', success=False)
        self.assertFalse(any(c[0] == 'release' for c in calls))

    def test_superseded_during_upload_keeps_draft(self):
        calls = self.run_publish('superseded-upload')
        self.assertTrue(any(c[:2] == ['release', 'upload'] for c in calls))
        self.assertFalse(any(c[:2] == ['release', 'edit'] for c in calls))

    def test_bad_remote_digest_leaves_draft(self):
        calls = self.run_publish('bad-remote', success=False)
        self.assertTrue(any(c[:2] == ['release', 'upload'] for c in calls))
        self.assertFalse(any(c[:2] == ['release', 'edit'] for c in calls))

    def test_draft_retry_reuses_release(self):
        calls = self.run_publish('draft')
        self.assertFalse(any(c[:2] == ['release', 'create'] for c in calls))
        self.assertTrue(any(c[:2] == ['release', 'edit'] for c in calls))

    def test_published_snapshot_is_not_modified(self):
        calls = self.run_publish('published')
        self.assertFalse(any(c[0] == 'release' for c in calls))

    def test_other_branch_cannot_publish(self):
        self.env['GITHUB_REF'] = 'refs/heads/feature'
        self.assertEqual(self.run_publish(success=False), [])


if __name__ == '__main__':
    unittest.main()
