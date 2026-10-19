# Cutting a release

The Docker workflow publishes the container image. The separate
[`Windows release`](../../.github/workflows/windows-release.yml) workflow publishes native x64
executables. Linux release archives remain a manual path.

## Windows publication

A push to `master` that changes release inputs starts two builds on `windows-2022` with VS 2022,
Ninja and the pinned vcpkg baseline. CUDA 12.9.1 builds `70;75;80;86;89`. CUDA 13.2.0 builds
`75;80;86;89;120a`. Volta requires CUDA 12, while this project's Blackwell kernels require CUDA
13.1 or newer. These are separate packages, each with multiple device images in every executable.

The workflow uses the pinned [CUDA Toolkit action](https://github.com/Jimver/cuda-toolkit).
[setup-msbuild](https://github.com/microsoft/setup-msbuild) exposes MSBuild but does not import
`cl.exe`, headers or libraries. The workflow imports `vcvars64.bat` instead. VS 2022 supports both
selected Toolkits and avoids CUDA 12's rejection of VS 2026.

1. Build `ninfer`, `ninfer-serve` and `ninfer_bench` with real vcpkg libraries and device code.
2. Package DLLs from the build's vcpkg tree and the same CUDA Toolkit that supplied `nvcc`.
   Include CUDA and dependency licences, launchers, source metadata and checksums.
3. Check each exe with `cuobjdump --list-elf` for every requested device image.
   Run each exe with `--help` and only Windows directories on `PATH`.
4. Create and test 1900 MiB 7-Zip volumes. Preserve both variants as Actions artifacts for seven days.
5. After **both** builds pass, validate local checksums and create a draft release named
   `windows-<12-character source commit>`. Upload and compare remote SHA-256 digests and sizes.
6. Check that the source is still the head of `master`. Publish the draft and mark it **latest**.
   A superseded build keeps its Actions artifacts, or its draft if upload already started.

A first successful publication creates the first release. Later publications create new immutable
snapshots. They do not overwrite assets in earlier releases. A retry can repair an unfinished draft
for the same commit. A published snapshot is a no-op. Existing Linux or manually published releases
retain their assets. A subsequent successful automatic Windows publication becomes latest.

For a manual run, choose **Actions → Windows release → Run workflow**. The default builds and
checks artifacts without publishing. Select `publish` on `master` to publish the checked result.
Runs on other branches cannot publish. Publication needs only the workflow's `contents: write`
token. Build jobs have read access.
The `build/windows-release` branch also starts builds on push, so the first native run does not
need a merge into `master` or an existing workflow on the default branch.

This workflow has not yet completed a native Windows run. The matrix, linking, dependency closure
and hosted-runner resource limits therefore remain unverified. `--help` does not test inference.
Test the package on a real GPU before claiming numerical correctness or performance for a release.

## Local packaging and Linux archives

Set `VERSION` and supply its `RELEASE_NOTES_<X.Y.Z>.md`. Both packagers require those files.
For Windows, build the three release products, then run `scripts\package-release.ps1` with
`NINFER_BUILD_ROOT` set to that build tree. It reads the Toolkit and architecture list from CMake.
It requires `7z` and the Toolkit's `cuobjdump`. The output is under `dist/`, with a checksum
manifest per CUDA/architecture/source variant.

For Linux, build the same commit with `-DCMAKE_CUDA_ARCHITECTURES=86`, `-DBUILD_TESTING=OFF` and
`-DNINFER_BUILD_BENCHMARKS=ON`, targets `ninfer ninfer-serve ninfer_bench`. Then run
`scripts/package-release.sh`. Verify the tarball and smoke-test it on a real GPU before an
explicitly authorized `gh release create` or `gh release upload` with `-R iamwavecut/ninfer-all`.

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

**Linux links cuBLAS, FFmpeg and glibc dynamically.** The Linux archive does not ship `libcublas`
(the Lt library alone is about 500 MB), so `release-archive-linux.md` states the requirements: the
glibc of the build host, the CUDA runtime libraries including cuBLAS, FFmpeg and libcurl. Check them
with `ldd` on the new binaries and update the archive README when they change.

**Asset sizes.** Every executable links every device image. Windows packages use 1900 MiB 7-Zip
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
