# Cutting a release

The [Native release](../../.github/workflows/native-release.yml) workflow builds Windows x64
executables and Debian 13 x64 binaries. After a successful master push run,
[Container image](../../.github/workflows/docker.yml) packages both Linux archives without compilation.
Ordinary CI builds and tests individual architectures independently.

## Container publication

| Toolkit | Floating tag | Commit tag | NInfer architectures |
|---|---|---|---|
| CUDA 12.9.2 | `cuda12.9` | `cuda12.9-<commit12>` | `70;75;80;86;89` |
| CUDA 13.4.3 | `cuda13.4` | `cuda13.4-<commit12>` | `75;80;86;89;120a` |

Both images use Debian 13 and the binaries and CUDA runtime libraries from the same native run.
The handoff verifies volume checksums, packaged files, full source revision, CUDA version and the
complete architecture list. Each image keeps all its device images in one multicall executable.
`commit12` is the first 12 hexadecimal characters of the source SHA; `VERSION` is not a tag.

The matrix publishes commit tags first. Only after both jobs succeed does promotion update the
floating CUDA tags, checking the current master SHA before each update. Registry tag updates are
separate operations: cancellation, a newer commit or registry failure can leave them temporarily
pointing to different revisions. Use both commit tags when the same revision is required.
A failed matrix does not promote either tag. Each variant reuses its previous image's inline
BuildKit cache. Compiler caches remain in Native release and CI.

The pipeline no longer updates `latest`, `multi`, version-only or `sha-…` tags. Existing registry
images are not deleted. The old latest-based pruning policy is removed because it cannot retain
both CUDA families safely; registry storage grows with published commit images until a separate
retention policy is configured. Manual native test runs do not publish container images.

## Native publication

Each platform builds two variants from the same source commit:

| Toolkit | Device images |
| --- | --- |
| CUDA 12.9.2 | `70;75;80;86;89` |
| CUDA 13.4.3 | `75;80;86;89;120a` |

These are the latest CUDA 12 and 13 releases in the
[NVIDIA archive](https://developer.nvidia.com/cuda-toolkit-archive), checked on 20 October 2026.
Volta requires CUDA 12. This project's Blackwell kernels require CUDA 13.1 or newer.
Windows uses the official NVIDIA network installer, VS 2022, Ninja and pinned vcpkg libraries.
It imports `vcvars64.bat`; setup-msbuild alone does not provide the compiler environment.
Linux uses Debian 13. CUDA 13 uses NVIDIA's exact Debian 13 Toolkit metapackage. CUDA 12 uses
NVIDIA's official runfile installer, with only the Toolkit selected. Debian 13 rejects the SHA-1
binding in NVIDIA's Debian 12 APT key; the workflow does not relax signature verification.
Debian 13 supplies the FFmpeg version required by NInfer. CUDA 12.9 also needs matching
`noexcept(true)` declarations for `sinpi`, `cospi`, `sinpif` and `cospif` with glibc 2.41.
A version-guarded script patches those declarations and inline helper signatures, without changing
function bodies. `BUILD-INFO.json` records the patch. The pre-build probe compiles CUDA headers
through both host C++ and nvcc before testing cache reuse.
Compilation needs no GPU. Driver installation is not part of either build.

1. Compile release products. Linux uses the multicall executable to share device images between
   commands; `ninfer_bench` remains a separate executable.
2. Check every requested architecture with `cuobjdump`. Run executable `--help` checks.
   Windows clears developer directories from `PATH`. Linux checks the dynamic library closure
   and uses a temporary CUDA driver stub, which is never packaged.
3. Package checksums, source metadata, launchers and licences. Windows includes runtime DLLs.
   Linux includes the CUDA runtime libraries and their EULA, records the required system packages,
   and includes a Dockerfile that installs those packages and copies the prebuilt binaries.
4. Test the archives and split them into volumes no larger than 1900 MiB. Keep Actions artifacts
   for seven days. Names are `ninfer-<os>-x64-cuda<major.minor>-sm<arches>-<commit12>`;
   `VERSION` remains metadata inside the archive and does not affect its name.
5. After all four builds pass, validate all local manifests, create draft `native-<commit12>`,
   upload assets, and compare their remote SHA-256 digests and sizes.
6. Publish and mark latest only if the source is still the head of `master`. Superseded builds
   retain their Actions artifacts or the draft created before a newer commit arrived.

Published snapshots are immutable. A retry can repair only the same commit's unfinished draft.
Every push to `master` starts the current build, cancels the previous run and publishes on success. In **Actions → Native release
→ Run workflow**, publication defaults to off; enable `publish` only on `master` when intended.
Feature branches run only through manual dispatch and cannot publish. Build jobs have read
access; only the publication job receives `contents: write`. The master cancellation coordinator
also cancels older container publishers. Closing a PR cancels that PR's remaining runs.
A superseded upload can leave an unpublished draft; it cannot replace the current release.

## Build caches

Native Windows, Debian and ordinary CUDA CI use pinned ccache 4.14.2. It stores complete object
files rather than sccache's per-architecture CUDA intermediate stages. Each native variant has a
1.5 GiB local limit; each of the three ordinary CUDA CI variants has a 256 MiB limit. The seven
current compiler-cache snapshots therefore have a combined limit of 6.75 GiB before archive
metadata. Windows vcpkg archives also use repository cache storage. The repository remains at
its free 10 GiB limit. Older snapshots and other branches compete for that limit and can be evicted.

Cache keys separate the platform and CUDA version or CI architecture. Source commits are not
restore-prefix boundaries. Ccache checks compiler content, source, headers and compilation flags
before reusing an entry. Depend mode hashes raw source and compiler-reported header dependencies,
without falling back to device-only preprocessing. The pre-build probe checks unchanged C++ and
multiarch CUDA objects, then verifies that changing host-only CUDA code or a host-only header
causes a cache miss.
Compilation reports cache counters, available RAM and disk space every minute.

Successful compiler invocations commit their object-cache entries to local disk. After success,
failure or ordinary cancellation, cleanup stops compiler work, applies the cache size limit and
runs an explicit Actions cache save. This also saves completed work from a build that reached its
step timeout. The next run waits for older runs of the same workflow and branch to finish cleanup.
The save has a three-minute limit within GitHub's five-minute cancellation window. Runner loss,
force cancellation, a failed upload or a full cancellation window can still lose the new snapshot.
There is no promise that an executing compiler process itself can resume.

The first build after switching from sccache starts with a cold object cache. Earlier sccache entries
are incompatible; they remain subject to GitHub's normal eviction. A warm rerun must establish
actual hits, retained size and completion time before the new cache is considered verified.
GitHub's branch restrictions still apply: master cannot restore a feature branch's private cache.
Windows vcpkg archives are saved after configuration, including configuration failure.
Docker publication reuses prebuilt native archives and per-CUDA inline image-layer caches.
Local source Docker builds retain their BuildKit cache mounts.

## Verification status

The four-variant native run on 20 October reached the 300-minute compilation limit on every runner:
Windows CUDA 12.9 completed 531/789 Ninja steps, Windows CUDA 13.4 completed 559/816,
Linux CUDA 12.9 completed 542/793, and Linux CUDA 13.4 completed 564/820. No compiler failures
were recorded in the final sccache counters; no release archive was produced.
Each variant reported 4,304–4,594 successful cache writes, with 12, 12, 96 and 70 write errors respectively.
On 21 October, the repository cache limit was 10 GiB and only 2,242 entries for the test branch remained.
These counts show that successful writes alone do not establish retention or useful reuse on the next run.
The new telemetry must establish reuse and identify backend write errors before claiming a cache fix.
The full matrix and Docker packaging still need a successful run. Loader and device-image checks
alone do not establish GPU inference correctness or speed.

## Local packaging

Both packagers read CUDA and architecture labels from the configured build tree. Set
`NINFER_BUILD_ROOT` to that tree. Build release products with benchmarks enabled. Linux accepts
separate CLI/server binaries or `NINFER_MULTICALL=ON`; CI uses the latter. Run
`scripts/package-release.sh` on the Linux build host, or `scripts\package-release.ps1` on Windows.
The repository's `VERSION` and matching release notes are included as source metadata. Neither
packager uses `VERSION` in the archive name. Modified local sources add `-dirty`; CI rejects them.
Linux packaging needs Python 3.11, `strip`, `patchelf`, `ldd`, `dpkg-query`, `tar` and the build's CUDA Toolkit.
The runtime Dockerfile is included only when packaging a Debian 13 build.

## Gotchas

**Always pass `-R iamwavecut/ninfer-all` to `gh`.** The checkout also has the `upstream` and
`ashalliants` remotes, and `gh` without `-R` may resolve to another fork.

**Windows executables import cuBLAS, and the archive must carry it.** The cuBLAS prefill route makes
every executable import `cublas64_<major>.dll` (which imports `cublasLt64_<major>.dll`), so a machine without
the CUDA Toolkit cannot start any of them (`0xC0000135`), even when `--prefill-cublas` is never used.
`package-release.ps1` copies both DLLs from the build compiler's Toolkit and refuses to produce an archive whose
executables do not start with only Windows on `PATH`. Do not weaken that check.

**Redistributing NVIDIA's DLLs has conditions.** The CUDA Toolkit EULA (Attachment A) lists the
cuBLAS DLLs as redistributable. The grant requires that the application has material functionality
of its own, that the DLLs are used only by it, and that the distribution terms are consistent with
the EULA and protect NVIDIA's rights. The packager therefore ships the toolkit's `EULA.txt` as
`NVIDIA-CUDA-EULA.txt`, and the archive README names the two DLLs as NVIDIA's, outside the
project's Apache-2.0 licence. Re-read the EULA of the toolkit that builds a release.

**Linux links cuBLAS, FFmpeg and glibc dynamically.** The packager copies the CUDA runtime closure
into `lib/`, checks the executable loader path, and ships NVIDIA's EULA. It lists system libraries'
Debian packages in `packages/multi.txt`; the runtime Dockerfile installs those without adding a
CUDA APT repository. An unowned system library or unresolved dependency fails packaging.

**Asset sizes.** Every executable links every device image. Native packages use 1900 MiB archive
volumes to remain below GitHub's 2 GiB asset limit. This does not remove MSVC's linked-image limit
or the hosted runner's memory and build-time limits. Their effect on the full matrix is unverified.

**Smoke-test the launcher default on the real card.** A desktop holding 2-3 GiB of a 24 GB card
refuses the default DFlash2 profile, both on the engine's runtime reservation and, on Windows, on
pinned host state. `run.{sh,bat}` step down by themselves when startup is refused for lack of GPU
memory ([launcher profiles](launcher-profiles.md)). Start the launcher with nothing overridden and
confirm it serves a request.

## Checks before publishing

- `scripts/check-linux-scripts.sh` passes.
- Every package: files match the inner `SHA256SUMS.txt`, and the outer checksum manifest verifies.
- Windows: all three executables start with only Windows on `PATH`.
- Linux: a fresh unpack runs `./ninfer-serve --help` and `./run.sh --help`.
- The default launcher profile starts and serves a chat completion on a real 24 GB card.
