# Cutting a release

The [Native release](../../.github/workflows/native-release.yml) workflow builds Windows x64
executables and Debian 13 x64 binaries. The existing Docker workflows remain separate.

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
A push affecting release inputs on `master` publishes automatically. In **Actions → Native release
→ Run workflow**, publication defaults to off; enable `publish` only on `master` when intended.
The test branch `build/windows-release` builds on push and cannot publish. Build jobs have read
access; only the publication job receives `contents: write`.

## Build caches

Native Windows and Debian builds use sccache for C, C++ and CUDA. Its GitHub Actions backend stores
completed compilation results during the build, so successfully stored entries can survive a later
failure or cancellation. Compiler identity, inputs, headers and flags determine reuse; the source
commit is not a cache-wide invalidation key. Before the long build, a small C++/multiarch CUDA probe
must produce cache hits after restarting the cache server. Job logs include final hit/miss statistics.
A timeout ends compilation before the job deadline so diagnostics can still upload.

Windows vcpkg archives are saved immediately after configuration, including after configuration
failure. Run/attempt keys allow a later run to replace a partial dependency cache. The ordinary Linux CI and multiarch image workflow retain ccache directories after failed
or timed-out compile steps. Their save keys include run ID and attempt; restore prefixes allow the
next run to reuse the newest partial cache. Runtime Docker jobs consume built artifacts and do not
compile NInfer again. GitHub can evict caches under repository storage/retention policies; a cache
hit and a shorter build time must be established from actual run statistics.

## Verification status

The earlier uncached Windows run hit the 5h50 job limit; its CUDA 13 build reached 555/807 Ninja
steps without a recorded compiler error. It produced no verified release package. A later obsolete
run was cancelled before switching Toolkits and caching. The new native matrix still needs a full
successful run. Loader and device-image checks do not establish GPU inference correctness or speed.

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
