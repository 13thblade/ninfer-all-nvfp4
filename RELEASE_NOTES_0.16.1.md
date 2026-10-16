# NInfer-3090 v0.16.1 — community engine adoption, native Windows build

**This release brings the continued RTX 3090 work from ashalliants' fork back into Don-Chad's
repository, with a downloadable native Windows build.**

The source includes Qwen3.8-27B DFlash2, Qwen3.6, and Qwen3.8-Flash-Next Windows preview,
specialized SM86 kernels, ReplaySSM, quantized paged KV, prefix reuse, constrained output, and
OpenAI/Anthropic serving. The release contains Windows x64 binaries only. Linux source and the
community's existing Linux builds remain available separately.

## Upgrade from the previous Don-Chad release

- **Model container v3 is required.** Old v2 files are rejected. Upgrade a supported v2 file with
  `tools/upgrade_ninfer_v2_to_v3.py` or use the pinned model downloader. Preserve your original.
- **This is a broad engine adoption, not a drop-in patch to v0.6.1.** Recheck existing launch
  commands against the bundled `--help` and current serving reference. Memory, scheduling and
  metrics contracts have changed since that release.
- **The old custom Anthropic top-level `timings` extension is retired.** The standard Anthropic
  response remains supported; use current logs/metrics for engine observations.
- **CUDA 13.2 Windows build.** A compatible NVIDIA driver is required. Qualification used driver
  610.88. CUDA Toolkit installation is unnecessary: the archive includes the required cuBLAS DLLs.
- **Default 27B profile uses DFlash2 with one active lane**, quantized `rk4v4` KV, cuBLAS prefill
  and vision overlay. The launcher fits its context ceiling to free GPU memory. Several requests
  require a compatible multi-lane profile; this is not high-concurrency datacenter batching.
- The downloader may fetch an optional private graft using a local Hugging Face login.
  Set `NINFER_GRAFTS=off` to skip it. Public model inference does not require that graft.

## What is included

- `ninfer-serve.exe`: streaming OpenAI Chat Completions/Responses and Anthropic Messages server.
- `ninfer.exe`: native CLI; `ninfer_bench.exe`: Engine-route benchmark.
- `run.bat`, pinned model downloader, runtime DLLs, release notes and dependency licenses.
- `BUILD_INFO.json` records source commit, CUDA toolkit version and SM86 GPU target.
- Inner SHA-256 checksums and an outer archive checksum file.

Models are downloaded separately. The recommended Qwen3.8-27B artifact is about 19 GB.
The server executable is about 577 MB before compression; this is not a tiny single-file package.

## Windows fixes in this release

- Resolve MSVC nested generic-lambda constant-capture failures in GDN gating and INT8 small-T
  attention without changing the intended kernel mathematics.
- Use the CUDA 13 graph-edge query signature in the stage-link test while retaining CUDA 12 support.
- Calculator checks accept both Windows CRLF and Linux LF line endings.
- Package the cuBLAS version actually linked by the build, including CUDA 13's `bin/x64` layout,
  rather than assuming CUDA 12 or trusting the caller's `CUDA_PATH`.
- Retain contributor history, original attribution and third-party license notices in the archive.

## Measurements and validation

The same runtime source was qualified on one desktop-attached RTX 3090 with CUDA 13.2/MSVC 14.43.
Qwen3.8-27B, DFlash2 K=7, one lane, 16K context cap, `rk4v4` KV, thinking and prefix reuse off:
three warmups and six measured requests yielded **116.7 tokens/s median decode** and
**127 ms median first streamed content**, with 52 prompt tokens and 512 generated tokens per
request. Cold startup was about 42 seconds and excluded. This is not a matched comparison with
v0.6.1, another engine, or the fork's separate 140-token/s workload.

The source qualification passed the full native app/test build, 15 selected C++ host tests
(one additional S3 test skipped), 22 Python checks, calculator checks, and Linux script fixtures.
Release packaging additionally requires all bundled programs to start with only Windows on PATH,
fresh-extraction checksums, and a real-card request through the packaged launcher.

No Linux CUDA runtime qualification, full GPU numerical suite, sanitizer campaign or complete
dependency CVE audit is claimed. The targeted source security review found no obvious malicious
behavior in the inspected build/download, process, network and persistence surfaces.
Distinct-GPU execution and Flash-Next were not part of our local inference benchmark.

## Credits

Original engine: **Neroued**. RTX 3090 adaptation: **Don-Chad**. Substantial continued development
and integration: **ashalliants**, **WarlaxZ**, and their contributors. **mgscreativa** contributed
fixes and proposed the import in PR #30. Flash-Next also adapts attributed work from
**wallawalla47/Infernix**. Both histories, source notices and `CONTRIBUTORS.md` are preserved.

The fork import is pinned at `931bd49ba332f48fad4b5230eec5a95ec5701290`; the adoption merge is
`1418aca37b253549e29f0e77fdb72db9d2b32195`. Historical optional Git LFS graft objects could not be
uploaded to this public fork; the current source tree has no Git LFS dependency.
