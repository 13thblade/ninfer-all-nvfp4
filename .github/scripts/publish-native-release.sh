#!/usr/bin/env bash
# Publish an immutable source snapshot only after all Windows and Linux packages pass their checks.
set -euo pipefail
: "${GITHUB_REPOSITORY:?}" "${GITHUB_SHA:?}" "${GITHUB_REF:?}"
[[ "$GITHUB_REF" == refs/heads/master ]] || { echo 'Publication requires master' >&2; exit 1; }
[[ "$GITHUB_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid source revision' >&2; exit 1; }
repo="$GITHUB_REPOSITORY"
tag="native-${GITHUB_SHA:0:12}"
cd dist
shopt -s nullglob
checksums=(SHA256SUMS-ninfer-*.txt)
[[ ${#checksums[@]} == 6 ]] || { echo 'Expected all six single-CUDA and universal checksum manifests' >&2; exit 1; }
# Validate the exact expected variants and this source revision, not merely four arbitrary files.
for variant in {windows,linux}-x64-cuda12.9-sm70-75-80-86-89 {windows,linux}-x64-cuda13.4-sm75-80-86-89-120a {windows,linux}-x64-cuda12.9-13.4; do
  manifests=(SHA256SUMS-ninfer-"$variant-${GITHUB_SHA:0:12}.txt")
  [[ ${#manifests[@]} == 1 && -f "${manifests[0]}" ]] || { echo "Missing package: $variant" >&2; exit 1; }
done
assets=(ninfer-windows-*.7z.* ninfer-linux-*.tar.gz.*)
[[ ${#assets[@]} -ge 4 ]] || { echo 'Missing archive volumes' >&2; exit 1; }
# Refuse extra, unverified volumes and manifests with paths outside this directory.
python3.11 - "${checksums[@]}" <<'PY'
import hashlib
import pathlib
import re
import sys
listed = set()
for filename in sys.argv[1:]:
    lines = pathlib.Path(filename).read_text().splitlines()
    if not lines:
        raise SystemExit(f'Empty manifest: {filename}')
    stem = filename.removeprefix('SHA256SUMS-').removesuffix('.txt')
    extension = '7z' if stem.startswith('ninfer-windows-') else 'tar.gz'
    for index, line in enumerate(lines, 1):
        match = re.fullmatch(r'([0-9a-f]{64})  (ninfer-[A-Za-z0-9._-]+\.(?:7z|tar\.gz)\.[0-9]{3,})', line)
        if not match or match[2] != f'{stem}.{extension}.{index:03d}':
            raise SystemExit(f'Invalid volume entry: {line}')
        path = pathlib.Path(match[2])
        if not 0 < path.stat().st_size < 2 * 1024**3:
            raise SystemExit(f'Invalid asset size: {path}')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != match[1]:
                raise SystemExit(f'Checksum mismatch: {path}')
        listed.add(path.name)
if listed != {p.name for pattern in ('*.7z.*', '*.tar.gz.*') for p in pathlib.Path('.').glob(pattern)}:
    raise SystemExit('Archive volumes differ from the checksum manifests')
PY
head="$(gh api "repos/$repo/commits/master" --jq .sha)"
if [[ "$head" != "$GITHUB_SHA" ]]; then
  echo 'Source was superseded on master; packages remain in Actions artifacts.'
  exit 0
fi
# A failed API request is an error, not evidence that the release does not exist.
releases="$(gh api --paginate "repos/$repo/releases?per_page=100" --jq ".[] | select(.tag_name == \"$tag\") | [.id, .draft] | @tsv")"
if [[ -n "$releases" ]]; then
  read -r release_id draft <<< "$releases"
  [[ "$(gh api "repos/$repo/releases/tags/$tag" --jq .target_commitish)" == "$GITHUB_SHA" ]] || {
    echo 'Existing release tag points to a different source revision' >&2; exit 1;
  }
  [[ "$draft" == true ]] || { echo "Release $tag is already published; leaving it unchanged."; exit 0; }
else
  notes="$(mktemp)"
  trap 'rm -f "$notes"' EXIT
  cat > "$notes" <<NOTES
Native Windows and Debian 13 x64 builds from $GITHUB_SHA.

- CUDA 12.9.2: sm_70, sm_75, sm_80, sm_86, sm_89.
- CUDA 13.4.3: sm_75, sm_80, sm_86, sm_89, sm_120a.

Download every volume and the checksum manifest for ONE package. The cuda12.9-13.4 package contains both backends and selects one for the GPU and driver. Single-CUDA packages contain one backend.
Windows: extract the .7z.001 file with 7-Zip. Runtime DLLs are included; install a compatible NVIDIA driver and the Microsoft Visual C++ x64 runtime.
Linux: concatenate the .tar.gz volumes in numeric order and extract the result. The archive README lists Debian runtime installation steps. The included Dockerfile packages these binaries without compiling them again.

CI checks every requested device image and executable --help. GPU inference and performance are not tested. BUILD-INFO.json records the source, compiler, CUDA version and architectures.

NOTES
  gh release create "$tag" -R "$repo" --target "$GITHUB_SHA" --draft \
    --title "Native builds ${GITHUB_SHA:0:12}" --notes-file "$notes"
fi
# A retry can repair only this commit's draft. Published releases are immutable here.
gh release upload "$tag" -R "$repo" "${assets[@]}" "${checksums[@]}" --clobber
# Check the remote asset digests before making the draft visible or moving latest.
gh api "repos/$repo/releases/tags/$tag" > assets.json
python3.11 - <<'PY'
import hashlib
import json
import pathlib
assets = {a['name']: a for a in json.loads(pathlib.Path('assets.json').read_text())['assets']}
paths = [*pathlib.Path('.').glob('*.7z.*'), *pathlib.Path('.').glob('*.tar.gz.*'), *pathlib.Path('.').glob('SHA256SUMS-*.txt')]
if set(assets) != {p.name for p in paths}:
    raise SystemExit('Remote assets differ from the prepared release')
for path in paths:
    with path.open('rb') as stream:
        digest = 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()
    asset = assets[path.name]
    if asset.get('digest') != digest or asset['size'] != path.stat().st_size:
        raise SystemExit(f'Remote asset verification failed: {path}')
PY
head="$(gh api "repos/$repo/commits/master" --jq .sha)"
if [[ "$head" != "$GITHUB_SHA" ]]; then
  echo "Source was superseded during upload; $tag remains a draft."
  exit 0
fi
gh release edit "$tag" -R "$repo" --draft=false --latest
[[ "$(gh api "repos/$repo/releases/latest" --jq .tag_name)" == "$tag" ]] || {
  echo 'Latest release verification failed' >&2; exit 1;
}
echo "Published https://github.com/$repo/releases/tag/$tag"
