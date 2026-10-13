# TITAN RTX (`sm_75`)

One TITAN RTX (TU102, 72 SMs, 24 GiB, 280 W, TCC driver mode, PCIe Gen2 x1), Windows 11, driver
616.92, Release `75` build with CUDA 12.8, October 2026. Artifacts: the Qwen3.8-27B GSQ-RCO IQ3_S
release (10.95 GiB of weights) and Ternary Bonsai 2 27B (6.70 GiB). `ninfer_bench`,
`--kv-dtype rk8v4 --max-ctx 16384`, one warmup and three repetitions; `--ngram-draft-tokens 0` with
`--spec`. Decode tok/s, standard deviations 0.01 to 0.15. How the build runs on Turing is in
[Build system](../maintainer/build-system.md#turing-tier-75).

## Device profile

Measured before the Q8 K-split changes below, which shorten Bonsai's MTP rounds:

| Model | Profile | tg128 | pp512+tg128 | MTP3 (ms per round) | pp512 prefill | pp8192 prefill |
|---|---|---:|---:|---:|---:|---:|
| IQ3_S | compiled routes | 36.05 | 35.79 | 49.90 (43.48) | 752.7 | 750.4 |
| IQ3_S | built in | 36.13 | 35.86 | 54.59 (42.63) | 800.6 | 803.9 |
| Bonsai T2 | compiled routes | 59.12 | 59.00 | 62.10 (31.71) | 1226.0 | 1248.1 |
| Bonsai T2 | built in | 60.31 | 60.18 | 58.97 (31.01) | 1356.4 | 1397.9 |

The built-in entry is `ninfer-calibrate` on this card: 59 keys in 98 s, and a second calibration
agreed on 51 of them (the other eight are near-tie bands). It prefills 6.6 to 7.1% faster for IQ3_S
and 10.6 to 12.0% for Bonsai, decodes Bonsai 2.0% faster and IQ3_S within the spread between
sessions (about 1%), and shortens an MTP3 round by 2.0 to 2.2%. A route change also changes which
near-tied tokens win, and with them acceptance, so MTP tok/s across profiles is compared by round
time ([determinism](../performance.md#determinism)).

## Speculative backend

Built-in profile. The benchmark corpus continues a 512-token prompt by 128 tokens; the chat prompt
is "Explain in detail how a CPU cache works, with examples." through `ninfer`, greedy, 512 tokens,
thinking off.

| Model | Workload | plain | MTP2 | MTP3 | MTP4 | DFlash2 K=3 | K=4 | K=5 | K=7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IQ3_S | corpus | 35.55 | **57.18** | 54.13 | 49.80 | 50.15 | | 36.15 | 31.79 |
| IQ3_S | chat | 35.8 | 62.8 | 63.6 | 61.1 | **68.1** | 63.9 | 63.7 | |
| Bonsai T2 | corpus | 60.61 | **87.40** | 78.70 | 73.35 | 71.03 | | 69.77 | 66.01 |
| Bonsai T2 | chat | 60.6 | 102.5 | **103.8** | 99.3 | 97.5 | 97.5 | 103.3 | |

MTP2 leads on the corpus for both models. On the chat answer DFlash2 K=3 leads for IQ3_S, and
Bonsai's MTP3, DFlash2 K=5 and MTP2 are within 1.3% of each other. Bonsai's MTP head and the QKV
projection of its drafter are Q8 G32 (its other drafter projections Q4 G64), and IQ3_S's drafter
is Q8 G32: they run the Turing path of the Q8 K-split contraction (below); IQ3_S's MTP head runs
the GGUF kernels. Chat rows are one generation each, and the corpus rows of one build moved by up to
3% between sessions, as the card's clocks did. Before the Q8 changes below, the n-gram proposer at
its default did not help on the chat prompt (IQ3_S MTP2 63.5 against 63.6, Bonsai DFlash2 K=3 91.9
against 94.4).

## The Q8 K-split contraction on Turing

Bonsai's MTP head and IQ3_S's DFlash2 drafter are Q8 G32, and so is the QKV projection of Bonsai's
drafter; at the widths of a draft or a verify they run `ops/linear/q8/q8_ksplit_mma.cuh`. On a `75`
build that contraction:

- forms the codes as FP16 pairs natively and, when every live activation of a tile is an FP16 value,
  issues FP16 MMA with the activations converted once; other tiles take the BF16 MMA emulation. The
  operands and their order are the emulation's, so the outputs are the same bits. A column past the
  live extent (a draft of 6 columns in an 8-column tile) is never staged and holds whatever the
  shared memory held before; it no longer sends the tile to the emulation.
- loads the next K group's codes and scales, and an activation tile of up to eight columns, into
  registers during the current group's MMAs: Turing's copies are synchronous, and a stage issued
  between two barriers waited out its whole latency.
- runs the 8-column tiles of matrices with at least 6144 rows with four K warps and at most 128
  registers (`Q8KSplitDeviceSchedule`): there four K warps stream the codes at 420 to 600 GB/s
  against 230 to 330 with the eight or sixteen the routes give; at 5120 rows and in wider tiles the
  routes' schedules stay ahead.

Kernel times in the measured repetition (CUPTI), per call; "FP16" is the first point alone, "now"
all three (Bonsai's MTP head was not traced in between):

| Q8 layer | emulation, us | FP16, us | now, us |
|---|---:|---:|---:|
| Bonsai MTP, 34816 x 5120 (MLP gate/up), 1 column | 1604 | | 400 |
| Bonsai MTP, 5120 x 17408 (MLP down) | 884 | | 374 |
| Bonsai MTP, 5120 x 10240 (input projection) | 558 | | 220 |
| Bonsai MTP, 14336 x 5120 (QKV) | 518 | | 157 |
| Bonsai MTP, 5120 x 6144 (attention output) | 332 | | 180 |
| IQ3_S drafter, 34816 x 5120 (fused gate/up), 6 columns | 683 | 482 | 420 |
| IQ3_S drafter, 5120 x 17408 (MLP down) | 852 | 478 | 417 |
| IQ3_S drafter, 6144 x 5120 (QKV) | 289 | 214 | 95 |
| IQ3_S drafter, 5120 x 25600 (feature projection) | 1286 | 722 | 599 |
| IQ3_S drafter, 5120 x 4096 (attention output) | 209 | 121 | 114 |

A Bonsai MTP draft step went from 3.90 ms of Q8 kernels (emulation) to 1.33, and IQ3_S's drafter
from 11.5 (emulation) and 7.2 (FP16) to 5.8 ms a DFlash2 round. The drafter's 5120-row layers still
read 200 to 230 GB/s, and none of four, eight or sixteen K warps took them past 270 GB/s in the
microbenchmark. On the corpus, against binaries built from the same source with the FP16 path alone
(interleaved runs):

| Model | Configuration | before | after | ms per round |
|---|---|---:|---:|---|
| Bonsai T2 | MTP2 | 69.08 | 87.40 | 26.85 to 21.22 |
| Bonsai T2 | MTP3 | 57.75 | 78.66 | 31.66 to 23.25 |
| Bonsai T2 | MTP4 | 50.85 | 73.42 | 36.48 to 25.27 |
| Bonsai T2 | DFlash2 K=5 | 67.52 | 69.82 | 27.88 to 27.36 |
| IQ3_S | DFlash2 K=3 | 50.94 | 51.58 | 45.69 to 45.12 |
| IQ3_S | DFlash2 K=5 | 35.50 | 36.63 | 56.35 to 55.46 |
| Bonsai T2 | plain | 60.79 | 60.65 | |
| IQ3_S | MTP2 | 57.29 | 57.25 | 37.87 to 37.89 |

MTP acceptance was identical; the four-K-warp QKV changes the drafter's reduction order, and with
it the acceptance of Bonsai DFlash2 K=5 (18.2 to 18.8%) and IQ3_S K=5 (20.1 to 20.7%). Plain
Bonsai and IQ3_S MTP run no Q8 kernel; their pairs differ by 0.04 to 0.14 tok/s, the earlier
build measured first each time.

The fallback is the cost to watch. Once the prefetch hides the load latency, a tile that falls back
to the emulation is paid in full: with one activation in a thousand below FP16's range, the fused
gate/up kernel in a microbenchmark runs at 685 us against 406 with the FP16 path alone (331 against
447 with FP16 activations only). The model activations measured here fall back rarely enough that
every layer is faster.

## Where the time goes

CUPTI kernel traces of the measured repetition:

- IQ3_S decode is 85% GGUF vector kernels (dp4a), at about 82% of the card's measured 583 GB/s on
  the large matrices. Its IQ3 decoders are instruction-bound past one column: IQ3_S at four columns
  takes 1.4 to 1.6 times its one-column time, IQ4_XS 1.1, which is what an MTP or DFlash2 verify
  pays.
- Bonsai decode is 85% `t2_small_t_i8` (INT8 MMA as four m8n8k16) at 76 to 97% of bandwidth.
- Prefill at 8K: GGUF MMQ is 78% of IQ3_S's time, the T2 INT8 GEMM 67% of Bonsai's; the GDN
  two-stage recurrence, which runs the TF32/BF16 emulation, is 10% and 18%.
- At 64K context an IQ3_S MTP2 round takes 47.9 ms against 37.8 at 640 tokens; small-T INT8
  attention at 262,144 keys reads at 58% of bandwidth for one column and 39% for four.

## Perplexity

Held-out corpus, `--quick`, `rk8v4`, against the same source built for `86`:

| Model | TITAN RTX, `75` | RTX 3060, `86` | Change |
|---|---:|---:|---:|
| Bonsai T2 | 5.598608 | 5.597737 (one card) | +0.016% |
| IQ3_S | 4.678183 | 4.677568 (two cards, two pipeline stages) | +0.013% |

Every stream is within 0.13% of its `86` value.

Per KV coding on the TITAN, the change against `bf16` of the same model:

| KV | Bonsai T2 PPL | change | IQ3_S PPL | change |
|---|---:|---:|---:|---:|
| `bf16` | 5.592838 | | 4.678694 | |
| `int8` | 5.592925 | +0.002% | 4.678362 | -0.007% |
| `fp8` | 5.594014 | +0.021% | 4.675334 | -0.072% |
| `rk8v4` | 5.598608 | +0.103% | 4.678183 | -0.011% |
| `k8v4` | 5.599651 | +0.122% | 4.687137 | +0.180% |
| `rk4v4` | 5.605246 | +0.222% | 4.689339 | +0.228% |
| `nvfp4` | 5.607401 | +0.260% | 4.693836 | +0.324% |
| `rk4v4-e8` | 5.624493 | +0.566% | 4.705337 | +0.569% |
| `rk2v4-e8` | 5.820319 | +4.067% | 4.838462 | +3.415% |

The `bf16` rows ran before the BF16 attention below; with it they read 5.592963 (+0.002%) and
4.680221 (+0.033%, all but 0.002 of it from `wikipedia_zh`, a domain that spreads from 11.51 to
11.62 across the codings).

## KV codings and long context

Every coding runs on a `75` build; FP8, NVFP4 and K8V4 widen K and V into one shared tile where two
would pass 64 KiB ([Turing tier](../maintainer/build-system.md#turing-tier-75)). At the native
262,144-token window `rk8v4` leaves 8.6 to 10.2 GiB free after startup for Bonsai and 2.4 to 5.0
GiB for IQ3_S (plain, MTP, DFlash2); `bf16` holds Bonsai's full window plain and 237,568 tokens
with MTP or DFlash2, IQ3_S's 180,224 plain. With `--rope-yarn` and `rk2v4-e8`, Bonsai starts at
1,048,576 tokens (3.0 GiB free plain, 1.6 with MTP3) and IQ3_S at 786,432. A needle at half depth
of repository text, greedy:

| Model | Prompt tokens | Found | Prefill tok/s | Decode tok/s |
|---|---:|---|---:|---:|
| Bonsai T2 | 62,630 | yes | 1,020 | 43.2 |
| Bonsai T2 | 378,146 | yes | 425.9 | 19.1 |
| Bonsai T2 | 761,294 | yes | 251.9 | 11.4 |
| IQ3_S | 378,146 | yes | 344.2 | 15.3 |

Decode at depth: 256 new tokens after repository-text prompts of 16,048, 67,103, 129,081 and
258,867 tokens, greedy, `--max-context 262144`; prefill tok/s, then decode tok/s plain, with MTP3,
and with DFlash2 (K=5 for Bonsai, K=3 for IQ3_S):

| Model | KV | Depth | Prefill | Plain | MTP3 | DFlash2 |
|---|---|---:|---:|---:|---:|---:|
| Bonsai T2 | `rk8v4` | 16K | 1,270 | 56.2 | 88.2 | 78.9 |
| Bonsai T2 | `rk8v4` | 64K | 1,030 | 45.8 | 74.5 | 65.0 |
| Bonsai T2 | `rk8v4` | 128K | 787 | 36.7 | 54.6 | 48.1 |
| Bonsai T2 | `rk8v4` | 255K | 549 | 27.1 | 43.4 | 36.8 |
| Bonsai T2 | `int8` | 16K | 1,200 | 55.5 | | 78.6 |
| Bonsai T2 | `int8` | 64K | 869 | 44.5 | | 57.4 |
| Bonsai T2 | `int8` | 128K | 624 | 36.5 | | 53.1 |
| Bonsai T2 | `int8` | 255K | 409 | 26.5 | | 36.1 |
| Bonsai T2 | `bf16` | 16K | 1,080 | 49.4 | | 75.2 |
| Bonsai T2 | `bf16` | 64K | 611 | 31.4 | | 53.3 |
| Bonsai T2 | `bf16` | 128K | 399 | 22.2 | | 43.8 |
| Bonsai T2 | `bf16` | 255K | 233 | 13.7 | | |
| IQ3_S | `rk8v4` | 16K | 750 | 33.6 | 53.1 | 48.7 |
| IQ3_S | `rk8v4` | 64K | 652 | 28.9 | 45.9 | 40.8 |
| IQ3_S | `rk8v4` | 128K | 542 | 25.2 | 38.7 | 34.6 |
| IQ3_S | `rk8v4` | 255K | 422 | 20.3 | 32.7 | 30.3 |
| IQ3_S | `int8` | 16K | 733 | 33.5 | | 51.0 |
| IQ3_S | `int8` | 64K | 585 | 28.5 | | 39.3 |
| IQ3_S | `int8` | 128K | 461 | 25.1 | | 34.1 |
| IQ3_S | `int8` | 255K | 332 | 20.1 | | 28.4 |
| IQ3_S | `bf16` | 16K | 682 | 30.9 | | 47.4 |
| IQ3_S | `bf16` | 64K | 456 | 22.4 | | 35.0 |
| IQ3_S | `bf16` | 128K | 327 | 17.4 | | |

`int8` decodes within 1 to 3% of `rk8v4` and prefills 2 to 26% slower, the gap growing with depth
(the INT8 prompt kernel against `rk8v4`'s). Both hold the whole window: `int8` leaves 8.3 GiB free
for Bonsai and 4.0 for IQ3_S plain, 6.7 and 1.4 with DFlash2. `bf16` runs in its own windows
(Bonsai 262,144 tokens plain and 237,568 with DFlash2, where a 206,053-token prompt prefilled at
271 tok/s and decoded at 34.1; IQ3_S 180,224 and 122,880) and decodes 8 to 48% slower than `int8`
plain, the gap growing with depth; DFlash2 narrows it (Bonsai at 128K: 43.8 against 53.1).

## BF16 attention

`bf16` KV is the lossless coding, and the BF16 MMA emulation made it the slowest one on Turing.
The attention kernels now multiply FP16 roundings of Q, K and V and keep the emulation for tiles
with a value FP16 cannot hold, in out-of-line sweeps
([Turing tier](../maintainer/build-system.md#turing-tier-75)). Bonsai, a 64,512-token needle
prompt, decode over the short answer:

| Configuration | Before (exact emulation) | Probabilities in FP16 | Now |
|---|---|---|---|
| plain, prefill / decode tok/s | 131.9 / 6.5 | 140.3 / 10.6 | 638.7 / 32.4 |
| MTP3 decode | | 35.4 | 100.8 |
| DFlash2 K=5 decode | | 51.2 | 127.0 |

`int8` on the same prompt: 878.0 / 45.0.

## TCC

The card ran in TCC mode, which takes the WDDM scheduler out of every launch but reports no virtual
memory management (`CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` is 0 there, 1 on the
host's WDDM cards): the VMM-based features (overlay Vision residency, model suspend) are unavailable
on a TCC device, and their tests skip.
