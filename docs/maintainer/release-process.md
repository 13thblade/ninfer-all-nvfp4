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
files. Trusted builds save compiler and Windows vcpkg caches in the private HF bucket
`WaveCut/ninfer-cache`, under `github-actions/`. Existing pod caches and model files use separate
prefixes and are not changed. `HF_BUILD_CACHE_TOKEN` must be a fine-grained token with read/write
access to this bucket only. The client is pinned to `huggingface_hub==1.17.0`.

Each native variant has an 8 GiB local compiler-cache limit; each trusted CUDA CI variant has a
4 GiB limit. These are capacity limits, not measured working-set sizes. Ccache can evict older
objects when a local cache reaches its limit. Each platform/toolkit or CI architecture has its own
family, with separate branch prefixes and two rotating snapshots. Readers try the current branch,
then master. Source commits are not cache-family boundaries. A trusted branch's caches must be
explicitly promoted if they are to seed master; this workflow does not do that automatically.

Pull-request jobs receive no HF credential. They use Actions Cache with a 256 MiB local limit per
architecture, including an explicit save after interruption. Python dependency caches also remain
in Actions Cache. The repository remains at the free 10 GiB limit; PR snapshots and dependency
caches can still be evicted by GitHub. Docker publication reuses prebuilt native archives and
per-CUDA inline image-layer caches. Local source Docker builds retain their BuildKit cache mounts.

Ccache checks compiler content, source, headers and compilation flags before reusing an entry.
Depend mode hashes raw source and compiler-reported header dependencies, without falling back to
device-only preprocessing. The pre-build probe checks unchanged C++ and multiarch CUDA objects,
then verifies that changing host-only CUDA code or a host-only header causes a cache miss.
Compilation reports cache counters, available RAM and disk space every minute.

Successful compiler invocations commit their entries to local disk. During trusted builds, the
supervisor uploads a snapshot every ten minutes after the previous upload completes. Windows
configuration also checkpoints completed vcpkg archives. Uploads run while compilation continues;
an entry that finishes after the snapshot starts is retained by a subsequent snapshot. Temporary
files and mutable cache statistics are excluded. An upload writes the inactive archive slot,
then publishes its manifest with size and SHA-256. Readers verify the archive and try the previous
slot if the latest archive is damaged. A partial upload does not publish a new manifest.

After success, failure or ordinary cancellation, cleanup stops compiler work and saves completed
objects before packaging. Native builds stop after 300 minutes of compilation, or earlier at a
deadline set 330 minutes from the first job step. Ordinary CUDA CI uses a 320-minute compilation
limit and the same job deadline. The job limit is 350 minutes. The supervisor exits with code 124
and leaves cleanup time; the hard step timeout is five minutes longer than its duration limit.
Explicit save steps also run after a hard compilation-step timeout. The next run waits for older
runs of the same workflow and branch to finish cleanup.

The final save has a three-minute limit within GitHub's five-minute cancellation window. Runner
loss, force cancellation or a failed upload can lose work since the last completed checkpoint.
That interval can exceed ten minutes by the upload duration or by a failed checkpoint. An executing
compiler process itself cannot resume. Periodic checkpoint failures produce a warning and retry;
final-save failures fail the step. Retention, transfer duration and useful reuse must be measured
on the complete CUDA working set before claiming that every matrix job will finish.

Migration preserves existing Actions caches under `github-actions/migration/`. Each archive is
uploaded and downloaded on a GitHub runner, and its manifest is published only after a matching
SHA-256. Restore can merge these preserved snapshots to seed the new cache family. Existing
single-architecture Ubuntu pod caches are not assumed compatible with Debian multiarch builds.

## Verification status

The four-variant native run on 20 October reached the 300-minute compilation limit on every runner:
Windows CUDA 12.9 completed 531/789 Ninja steps, Windows CUDA 13.4 completed 559/816,
Linux CUDA 12.9 completed 542/793, and Linux CUDA 13.4 completed 564/820. No compiler failures
were recorded in the final sccache counters; no release archive was produced.
Each variant reported 4,304–4,594 successful cache writes, with 12, 12, 96 and 70 write errors respectively.
On 21 October, the repository cache limit was 10 GiB and only 2,242 entries for the test branch remained.
These counts show that successful writes alone do not establish retention or useful reuse on the next run.
The new telemetry must establish useful reuse before claiming that the full matrix can finish.
[Compiler cache recovery](../../.github/workflows/cache-recovery.yml) tests the transport separately.
Its `timeout` mode caches a real C++ object, then reaches a one-minute GitHub step timeout.
Its `cancel` mode waits for an ordinary workflow cancellation after caching that object.
Run `restore` with the producer's `source_run` and `source_attempt` to require a cache hit on fresh
Windows and Linux runners. These small probes test cache recovery, not retention of the full CUDA
working set. Before the HF migration on 21 October 2026, both interruption modes saved their Actions caches on Windows 2022 and Ubuntu 24.04.
Separate restore runs on fresh runners each recorded one compiler cache hit and zero misses:
[after timeout](https://github.com/iamwavecut/ninfer-all/actions/runs/39569140529) and
[after cancellation](https://github.com/iamwavecut/ninfer-all/actions/runs/39569242802).
The producers were [39568919978](https://github.com/iamwavecut/ninfer-all/actions/runs/39568919978)
and [39568972360](https://github.com/iamwavecut/ninfer-all/actions/runs/39568972360).
The HF migration preserved 13 snapshots (4,084,951,040 archive bytes), including the final caches
from all four cancelled native variants and all three CUDA CI architectures. Every migration
uploaded and downloaded its archive on the runner and verified SHA-256 before publishing metadata.
The first native-Linux restore missed because its Debian archive compression differed from the
Ubuntu runner; restoring in the original Debian environment resolved it. The Actions source
caches were retained. Migration records remain under `github-actions/migration/`; the temporary
migration workflow was removed after verification.

The HF [timeout producer](https://github.com/iamwavecut/ninfer-all/actions/runs/39584240411)
saved both periodic and final snapshots. Its [fresh-runner restore](https://github.com/iamwavecut/ninfer-all/actions/runs/39584567971)
recorded one hit and zero misses on Windows and Linux. A second
[restore](https://github.com/iamwavecut/ninfer-all/actions/runs/39584614325) recorded the same result
while its [producer](https://github.com/iamwavecut/ninfer-all/actions/runs/39584404425) was still
running, demonstrating that recovery does not depend on the job reaching cleanup. These probes
use a ten-second checkpoint interval; production uses ten minutes. They do not measure multi-GiB
upload times or full CUDA working-set retention.

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
