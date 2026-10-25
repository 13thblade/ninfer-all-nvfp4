# DeepSeek-V4.1-Flash

NInfer runs DeepSeek's [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(`DeepseekV41ForCausalLM`) from IST Austria's
[GSQ-RCO GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) (IQ1_M)
and its Engram table GGUF (IQ4_NL). It is its own model family next to Qwen3.5, Flash-Next, Gemma 4
and Flint: its own executor (`src/models/deepseek_v41`) and Engine core
(`src/runtime/engine/deepseek_v41_core`), sharing the Qwen3.5 frontend for tokenization, chat
templates, output parsing and the serving protocols in a DeepSeek dialect.

## What runs

- **The model.** 40 blocks of hidden width 5120 whose residual stream is four copies mixed by
  hyper-connections (a Sinkhorn normalization of 20 iterations at every sublayer). Each block has
  one latent attention: a low-rank query (rank 1280) of 64 heads of 512 values over a single key
  and value latent (keys are values), an interleaved RoPE on its last 64 values, a learned sink
  logit per head and a grouped low-rank output (8 groups of rank 1024). Every block attends over a
  window of 128 positions; blocks 2 to 19 also attend over latents compressed two tokens to one,
  blocks 20 to 39 over one compressed latent per token, of which an indexer (32 heads of 128) keeps
  the best 512 for each query; block 20 first keeps each query's best 2048 blocks of eight latents,
  and the later blocks rank only inside those. The MoE routes each token to 6 of 384 experts
  (sqrt-softplus scores, SwiGLU of width 2304 with its gate clamped at 10 and its up projection
  within ±10) beside a shared expert. Blocks 1 and 14 add Engram rows, the 2- to 4-grams ending
  at each position hashed into a table and gated by the stream. A 129,280-token head.
- **The weights stay the GGUF's blocks.** Every matrix keeps its ggml format, imported row for row
  without requantization and multiplied by the GGUF kernels. The routed experts' 80 banks (each
  block's gate with up, and its down) are IQ1_S (55 banks), IQ1_M (12), IQ2_XXS (11) and IQ2_S
  (2); the attention's query and latent down-projections Q8_0; the rest of the attention, the
  shared experts, the Engram projections, the token table and the head Q5_K. The router, the
  compressors, the indexers, the hyper-connection matrices, the norms and the sinks are FP32.
- **The release's roundings.** The attention keeps what the release's `kernel.py` rounds: the window
  latents in FP8 (E4M3 per 32 values with a power-of-two scale), the indexer's queries and keys in
  FP4 (E2M1 per 32, power-of-two scale), the compressed latents in FP4 (per 16, E4M3 scale) and the
  attention output in FP8 before its projection, each scale computed in FP32 as the release does.
  The caches hold those values in BF16.
- **Requests.** One to eight concurrent sequences (`--max-concurrency`), prompt chunks of 2048
  tokens (1024 or 512 on a card without room for them), prefix reuse across turns and requests with
  host-memory images ([below](#prefix-reuse)), branched reads for
  [typed decisions](serving.md#typed-decisions), structured output, streaming, log probabilities,
  causal scoring (`ninfer-perplexity`; a window that does not start with `<｜begin▁of▁sentence｜>`
  gets it in front), and the model's DSML tool calls returned as OpenAI tool calls, under the
  tool's JSON Schema when it is `strict` and beside a response format:

  ```text
  <｜DSML｜ calls>
  <｜DSML｜ invoke name="get_weather">
  <｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>
  </｜DSML｜ invoke>
  </｜DSML｜ calls>
  ```

  A value with `string="true"` is the text as written, any other its JSON. A forced `tool_choice`
  ends the generation prompt with the opening of that call, so the model continues with its
  parameters.
- **Reasoning.** The chat template renders the conversation as the release's `encoding.py` encodes
  it. Thinking is on by default: the generation prompt opens the reasoning and the model closes it
  with `</think>`, and the reasoning arrives as `reasoning_content`. `reasoning_effort` `low`,
  `high` (the default) or `max` writes the release's effort line (50, 75 or 100) into the system
  block; `enable_thinking: false` or `reasoning_effort: "none"` closes the reasoning at once.
  A reached thinking budget, or the [reasoning-loop guard](serving.md#openai-chat-completions)
  under `conclude`, closes the reasoning with Qwen's early-close guidance and the `</think>` token,
  and the model answers from what it has thought.
- **Sampling defaults.** The release's: temperature 1.0, top-p 1.0 (top-k 20, the candidates the
  sampler keeps).
- **Speculative decoding by context lookup.** The GGUF carries none of the release's draft blocks.
  `--lookup-ngram N` proposes what followed the last N tokens' earlier occurrence in the sequence
  (up to `--draft-tokens`, 3 by default, at most 7), and the target verifies the proposal in one
  pass of decode rows; the output is the one plain decoding gives.
- **Expert residency.** The routed experts stay on the GPU, in pinned host memory or in the
  artifact's files ([below](#expert-residency)); host memory is the residency to use.
- **The Engram table.** Stored in the model's artifact or in a table artifact of its own, read row
  by row from its file or held in RAM ([below](#the-engram-table)).

Not built for this family: the quantized `--kv-dtype` formats (the caches are BF16), the release's
draft blocks, MTP, DFlash and n-gram copy drafts, a CPU share of the expert misses (the IQ1_S, IQ1_M
and IQ2 blocks have no CPU kernels), rotation scaling (`--rope-yarn`, `--rope-scaling-factor`: the
compressed blocks carry the release's own YaRN), images and video (the GGUF has no vision tower),
pipeline stages and tensor parallelism, and of the context cache the shared paged KV pool, the disk
tier and session slots.

## Converting

The converter reads the weights from the GGUF shards and the Engram GGUF, and the tokenizer and
configuration from the base release. The release ships no Jinja chat template; the converter stores
its own transcription of the release's encoder (`tools/convert/templates/deepseek_v41`).

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 config.json tokenizer.json tokenizer_config.json \
  --local-dir models/deepseek-v4.1/hf
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  --include 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-*.gguf' \
  --include 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf' --local-dir models/deepseek-v4.1

# The model, which names its Engram table by the SHA-256 of the table's rows.
python -m tools.convert --model models/deepseek-v4.1/hf --recipe deepseek_v41_gguf --components text \
  --source gguf=models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf \
  --source engram=models/deepseek-v4.1/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \
  --name DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M \
  --out models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-ninfer-v3.ninfer --device cpu
# The Engram table, one artifact for every DeepSeek-V4.1-Flash-1017 conversion.
python -m tools.convert --model models/deepseek-v4.1/hf --recipe deepseek_v41_gguf --components engram \
  --source engram=models/deepseek-v4.1/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \
  --name DeepSeek-V4.1-Flash-Engram-IQ4_NL \
  --out models/DeepSeek-V4.1-Flash-Engram-IQ4_NL-ninfer-v3.ninfer --device cpu
```

The model reads the Engram GGUF only for the table's digest. The model is one file of
121,003,660,288 bytes (112.7 GiB), the table one of 110,595,294,496 bytes (103.0 GiB); on 24 cores
of a Threadripper 3960X the conversions took 7 min 19 s and 3 min 29 s.

## Running

Both artifacts are published: the model as
[WaveCut/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-NInfer-v3](https://huggingface.co/WaveCut/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-NInfer-v3)
and its table as
[WaveCut/DeepSeek-V4.1-Flash-Engram-IQ4_NL-NInfer-v3](https://huggingface.co/WaveCut/DeepSeek-V4.1-Flash-Engram-IQ4_NL-NInfer-v3).

```bash
M=models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-ninfer-v3.ninfer
T=models/DeepSeek-V4.1-Flash-Engram-IQ4_NL-ninfer-v3.ninfer
# A chat server for two concurrent requests over 32K-token contexts, experts in pinned host memory.
ninfer-serve $M --ngram-table $T --model-id deepseek-v4.1-flash \
  --max-context 32768 --max-concurrency 2 --expert-residency host

# One answer from the CLI.
ninfer $M --ngram-table $T --expert-residency host --prompt "Explain how a heat pump works."
```

On a 24 GB RTX 3090 with host experts the GPU holds the 6.0 GiB of dense weights, two layer banks
(7.1 GiB), a 1.3 GiB workspace, the latent caches (83 MiB at 16,384 tokens of context) and an
expert cache that takes the rest (8.1 GiB); the host pins 106.7 GiB of experts. With disk experts
the host pins 138 MiB of transfer buffers and the experts' pages sit in the OS page cache.

## Expert residency

The routed experts are 106.7 GiB of the artifact, more than any one card holds, so the model runs
with `--expert-residency host` or `disk`; `device` needs a card with room for every expert. Both run
all expert arithmetic on the GPU with the resident kernels, through the transport Gemma 4 and Flint
use ([Gemma 4's expert residency](gemma4.md#expert-residency) describes the layer banks, the cache
and its rounds):

- **Host.** The experts are read from the artifact once at startup into pinned host memory, and a
  layer's misses are copied from there. The host needs that memory beside the Engram table's page
  cache (or its RAM copy) and the rest of the system: 106.7 GiB pinned, so a host with 128 GB
  keeps about 12 GiB for everything else (not tested; the measurements below ran with 251 GB or more).
  The startup reads the experts with direct I/O, past the page cache, on eight threads into
  ordinary pages and pins them once read.
- **Disk.** The experts stay in the artifact's files. A layer's misses are read with ordinary file
  reads through the OS page cache into bounded pinned transfer buffers, then copied to the device;
  NInfer maps no expert file into memory, so a miss never waits on a page fault inside a copy.
  Whatever the page cache holds is read at memory speed, the rest at the storage's.
- **Layer banks and cache.** A prompt chunk's experts missing from the cache land in a whole-layer
  bank laid out like the resident bank; two banks let the next layer arrive while the current one
  computes. `--expert-cache-mib` (default: what is free after startup) holds the experts decode
  routes most, filled at startup from a routing profile (`--expert-profile`, else the built-in
  DeepSeek-V4.1 profile, recorded on summaries of four corpora and on chat, code and reasoning
  requests).
- **Every miss is copied.** There is no CPU share: `--expert-dma-share` and `--expert-cpu-threads`
  are refused.

## The Engram table

The two Engram blocks read 24 rows per token each (the 2-, 3- and 4-grams ending at the position,
hashed per head), IQ4_NL rows of 256 values. Before each pass the executor reads the rows it
addresses and copies them to the device:

- **Where it is stored.** In its own artifact: the model names the table by digest, and
  `--ngram-table PATH` supplies it (NInfer refuses a table with another digest). A model run without
  its table is refused unless `--no-ngram-table` runs it without the Engram, which degrades the
  output: on a 1,112-token WikiText slice the perplexity rises from 3.07 to 9.68.
- **How it is read.** `--ngram-residency disk` (the default) reads each pass's rows from the file
  (`--ngram-io`, `--ngram-io-depth`, `--ngram-ram-mib` for a row cache), and `ram` holds the whole
  table in RAM (`--ngram-lock` pins it). The table is 103.0 GiB, so `ram` and host experts together
  need about 210 GiB of RAM.

## Prefix reuse

With the context cache on (the default), a prompt resumes from a state a sequence or the host store
keeps instead of prefilling those tokens again:

- **On its sequence.** A sequence keeps its state when its request ends, and a checkpoint of its
  window rings and compressor tails where the prompt's last user turn closes; the next turn of a
  conversation resumes from either.
- **From host memory.** Before another request takes a sequence, what it held goes to an image in
  pinned host memory: the window rings, the compressor tails and the compressed caches up to its
  length. A prompt on any sequence resumes from the longest image it starts with. `--host-kv-mib`
  (or `--host-cache-mib`) sets the store.
- **Shared prefixes.** A prompt chunk also ends where the frontend marks a prefix other requests
  share (the end of the leading system messages, of the tool declarations, a declared cache
  boundary), and the state there goes to an image too.

A window state cannot be cut back, so a prompt resumes only where a state was kept, not at an
arbitrary common prefix.

## Measurements

October 25, 2026, one host with two RTX 3090s (350 and 390 W, PCIe 4.0 x16), a 24-core
Threadripper 3960X and 251 GB of RAM, CUDA 12.8: one sequence, greedy decoding, the same prompt
tokens in both engines. The short prompt is a 14-token question (an answer of 82 tokens), the long
one an 8,086-token summary request (32 tokens). NInfer served the model with
`ninfer-serve --max-concurrency 1` on one GPU; [Strata](https://github.com/Niko1221/Strata) 0.1.50
(`strata-dsv41`, commit `41cf05a0`) ran the GGUF shards through its pack index with its defaults
for a 24 GB card (all 6 experts, its DSpark drafter left unloaded), on one GPU and on both:

| engine and placement | decode, short answer | decode after 8,086 tokens | prefill of 8,086 tokens | ready to serve |
|---|---:|---:|---:|---:|
| NInfer, experts in pinned RAM | 21.8 tok/s | 18.9 tok/s | 410 tok/s | 33 s |
| NInfer, experts in pinned RAM, `--lookup-ngram 4` | 22.1 tok/s | 18.0 tok/s | 416 tok/s | |
| NInfer, experts on disk, page cache warm | 7.4 tok/s | 7.4 tok/s | 117 tok/s | 13 s |
| Strata, 1 GPU | 3.9 tok/s | 3.9 tok/s | 380 tok/s | about 34 s |
| Strata, 2 GPUs | 6.6 tok/s | 8.0 tok/s | 333 tok/s | about 34 s |

- **Decode.** Strata computes the experts its VRAM cache misses on its CPU lane, 107 to 242 ms of
  each token on this CPU; NInfer copies every miss to the GPU (a layer's misses in one bank) and
  computes it there.
- **Prefill.** A 2048-token chunk routes to nearly every expert of every block, so each chunk
  copies nearly all the experts the cache lacks over PCIe (four sweeps of about 100 GiB for 8,086
  tokens with host experts) or reads them through the page cache with disk experts. Strata
  prefills in windows of 16,384.
- **Placements agree.** Host experts, disk experts and host experts with context lookup produced
  the same greedy tokens for both prompts.
- **Context lookup.** On the long prompt 1 of 6 drafts was accepted: a summary repeats little of
  its source.
- **Startup.** The ready times come from the current build, whose expert cache starts from the
  built-in routing profile. With host experts the weights took 26.0 s (4.3 GiB/s: eight threads
  reading past the page cache into ordinary pages, pinned once read), the server was ready 33 s
  after it started, and stopping it took 15 s; that run produced the host row's tokens for both
  prompts, at 22.2 and 19.3 tok/s of decode and 419 tok/s of prefill. With disk experts the server
  was ready in 13 s and produced the same tokens, at 8.7 and 7.4 tok/s of decode and 120 tok/s of
  prefill. Host and disk experts score the WikiText slice to the same perplexity, 3.067.

The same day, one rented host per card ran the native release build of `b3402a6fd` (the CUDA 13.4
Linux archive; on the hosts' Ubuntu 26.04 with FFmpeg 7.1 from Debian 13) with the same prompts and
server flags on one GPU: host experts first, then disk experts with the model file in the page
cache. The server sizes the device expert cache to the card; the model has 15,360 routed experts:

| card | host | experts | device cache | decode, short answer | decode after 8,086 tokens | prefill of 8,086 tokens | ready to serve |
|---|---|---|---:|---:|---:|---:|---:|
| RTX PRO 6000 (600 W) | EPYC 9B14, PCIe 5.0 x16 | pinned RAM | 11,401 (74%) | 48.0 tok/s | 45.3 tok/s | 1,310 tok/s | 64 s |
| | | disk, page cache warm | 11,431 | 37.4 tok/s | 33.4 tok/s | 431 tok/s | 29 s |
| RTX 5090 (575 W) | EPYC 7B13, PCIe 4.0 x16 | pinned RAM | 2,251 (15%) | 29.2 tok/s | 22.5 tok/s | 499 tok/s | 34 s |
| | | disk, page cache warm | 2,282 | 12.5 tok/s | 10.5 tok/s | 181 tok/s | 15 s |
| RTX 4090 (450 W) | EPYC 7K62, PCIe 4.0 x16 | pinned RAM | 1,139 (7%) | 25.9 tok/s | 22.2 tok/s | 490 tok/s | 41 s |
| | | disk, page cache warm | 1,169 | 8.5 tok/s | 7.3 tok/s | 123 tok/s | 14 s |

- **The cache sets decode.** The RTX PRO 6000 keeps three quarters of the experts on the card and
  copies the rest; the 24 and 32 GB cards keep 7 and 15% and copy every other routed expert over
  PCIe 4.0, so they decode at 22 to 29 tok/s with host experts, 7 to 13 with disk experts.
- **Placements agree on each card.** Host and disk experts produced the same tokens for both
  prompts on every card. The short answer is 72 tokens on the RTX PRO 6000, 81 on the RTX 5090, 78
  on the RTX 4090 and 82 on the RTX 3090: each card's device profile selects its own kernel routes,
  which round differently.
- **Memory.** The server peaked at 115.4 to 115.8 GiB of resident host memory with host experts
  and 8.9 to 9.2 GiB with disk experts, and it fills the card in both placements.
- **Startup.** With host experts the weights took 33.7 s on the RTX 4090 (3.34 GiB/s) and 53.6 s on
  the RTX PRO 6000, whose read paused for 22 s at 95% on a host with 898 GB of its 1.5 TB in use
  and its swap full.

Each row is one run; no spread is measured.

## Numerical checks

- **Kernels.** `ninfer_deepseek_v41_kernels_test` qualifies the executor's own kernels against FP64
  oracles at the release's shapes: the hyper-connections and their Sinkhorn normalization, the
  latent and indexer projections' norms, rotations and roundings, the compressors, the index scores
  and both selections, the sparse attention with its sinks and splits, the output's inverse
  rotation, routing, the clamped SwiGLU, the expert combination, the Engram rows and gate, and the
  activation guard of the integer matrix kernels.
- **Against an FP64 forward.** `tools/reference/deepseek_v41.py` runs the model in FP64 over the
  GGUF's exactly decoded blocks and the Engram GGUF's rows, with the release's attention roundings.
  Over the 1,112 scored tokens of the WikiText slice above, its negative log-likelihood sums to
  1238.7 against NInfer's 1246.3 (perplexity 3.046 against 3.067, 0.7% higher in NInfer, whose
  matrix kernels take BF16 activations rounded to ggml's q8_1 blocks, as llama.cpp's do). NInfer's
  greedy next token matched the reference's argmax at 21 of 22 positions spread over the slice (the
  exception has a top-two logit gap of 0.32) and at 6 of the 10 positions where the reference's top
  two logits are closest (gaps of 0.01 to 0.04). The FP64 forward took 2 h 53 min on the
  Threadripper 3960X.
