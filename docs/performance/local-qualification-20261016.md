# Local RTX 3090 qualification — 16 October 2026

This smoke benchmark validates a source-built import candidate based on
`931bd49ba332f48fad4b5230eec5a95ec5701290` from ashalliants/ninfer-3090, with local MSVC/CUDA
compatibility fixes. It is not a comparison against the previous Don-Chad release or another engine.

## Setup

Native Windows, one RTX 3090 (24 GB), driver 610.88, CUDA 13.2, VS 2022 MSVC 14.43, Release SM86.
Qwen3.8-27B public artifact revision `d47f2732d369acaec76dc44228f67c72b081df2f`, SHA-256
`7f2a0086a071ee932c66106e1552d118a85cfec8ddef13c67a54cd64abb39706`, independently verified.

Configuration: DFlash2 K=7, draft head, FP16 recurrent state, cuBLAS prefill at chunk 4096,
`rk4v4` KV, 16,384-token context/KV capacity, one lane, no vision, no host cache,
no cross-request prefix reuse, thinking off, greedy sampling. About 1.7 GB of desktop GPU memory
was allocated before loading. Speech GPU services were stopped; only this model ran on the GPU.
The GPU power limit was not changed for this run.

## Workload and measurement

Three untimed warmups followed by six measured requests, each with a 512-output-token cap:

> Explain how hash tables handle collisions, resizing, and worst-case lookup costs. Include a
> short Python example and discuss when a balanced tree is a better choice. Write about 450 words.

The rendered prompt was 52 tokens. Each measured request returned 512 tokens, zero reasoning
tokens and zero cached tokens. All hit the requested output cap. All six greedy text outputs
were identical. A viewed excerpt coherently explained collisions and load factor; this is not
a full answer-quality evaluation or non-speculative parity check.

TTFT is client wall time from submitting the HTTP request to receiving its first non-empty
streamed content delta. Decode rate is the server's `timings.predicted_per_second`, using its
generated-token interval convention. Cold engine startup was about 42 seconds and is excluded.

| Sample | First streamed content (ms) | Decode (tokens/s) |
|---|---:|---:|
| 1 | 130.545 | 116.737 |
| 2 | 130.319 | 117.355 |
| 3 | 123.824 | 116.512 |
| 4 | 123.464 | 117.072 |
| 5 | 123.486 | 116.611 |
| 6 | 131.327 | 116.468 |
| **Median** | **127.072** | **116.674** |

355 of 1,091 proposed draft tokens were accepted per request (32.5%). That acceptance ratio
depends on the prompt and output. This test does not establish long-context speed, multi-user
throughput, model quality, Linux runtime correctness, or a 180-token/s result.

## Build and host checks

The full native app/test build linked successfully after fixes for nested generic-lambda capture
on MSVC and CUDA 13's graph-edge API. `ninfer-serve --help` exited successfully.
15 selected C++ host tests passed; one S3 integration test skipped because it had no configured
endpoint. 22 Python artifact/reporting checks passed. Calculator arithmetic, all 420 calculator
render combinations, Linux launcher/downloader fixtures, and `git diff --check` passed.

The calculator tests needed a CRLF-compatible script extraction regex on Windows. Linux checks
ran under WSL; they are not a Linux CUDA runtime test. No full GPU operator suite, sanitizer
campaign or dependency CVE audit was run. A targeted security review found no obvious malicious
behavior in inspected build/download, network, process and persistence surfaces.

The [README video](../demo.md) is a separate single request and reports its own measurements.
