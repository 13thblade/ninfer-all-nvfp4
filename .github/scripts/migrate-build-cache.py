"""Preserve an Actions cache in a private HF bucket and verify a full round trip."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import sys
import threading


def heartbeat():
    while not finished.wait(60):
        print("Cache migration is running; waiting for the transfer or verification.", flush=True)


from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--key', required=True)
    args = parser.parse_args()
    args.directory = args.directory.expanduser()
    if not args.directory.is_dir() or not any(args.directory.iterdir()):
        raise SystemExit('Refusing to archive a missing or empty Actions cache')
    if not all(c.isalnum() or c in '-_.' for c in args.key):
        raise SystemExit('Invalid cache key')
    api = HfApi()
    bucket = 'WaveCut/ninfer-cache'
    if api.bucket_info(bucket).private is not True:
        raise SystemExit('The build cache bucket must be private')
    remote = f'github-actions/migration/{args.key}.tar'
    with tempfile.TemporaryDirectory() as work:
        archive = Path(work) / 'cache.tar'
        count = 0
        with tarfile.open(archive, 'w') as tar:
            for path in sorted(args.directory.rglob('*')):
                if path.is_file() and not path.is_symlink():
                    tar.add(path, arcname=path.relative_to(args.directory).as_posix(), recursive=False)
                    count += 1
        with archive.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        print(json.dumps({'state':'uploading', 'key':args.key, 'files':count, 'bytes':archive.stat().st_size}), flush=True)
        api.batch_bucket_files(bucket, add=[(archive, remote)])
        restored = Path(work) / 'restored.tar'
        api.download_bucket_files(bucket, files=[(remote, restored)], raise_on_missing_files=True)
        with restored.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise SystemExit('Bucket round-trip checksum mismatch; keep the Actions cache')
        manifest = {'schema':1, 'key':args.key, 'archive':remote, 'sha256':digest,
                    'bytes':archive.stat().st_size, 'files':count,
                    'source_ref':os.environ['GITHUB_REF'], 'migration_run':os.environ['GITHUB_RUN_ID']}
        metadata = Path(work) / 'manifest.json'
        metadata.write_text(json.dumps(manifest, indent=2) + '\n')
        api.batch_bucket_files(bucket, add=[(metadata, remote + '.json')])
        Path('migration-result.json').write_text(metadata.read_text())
        print(json.dumps({'state':'round-trip-verified', **manifest}), flush=True)


if __name__ == "__main__":
    finished = threading.Event()
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        main()
    except Exception as error:
        print(f"::error::Cache migration failed ({type(error).__name__}); retain the Actions cache", flush=True)
        sys.exit(1)
    finally:
        finished.set()
