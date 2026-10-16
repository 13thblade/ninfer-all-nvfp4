# W4(NVFP4) A16(BF16) support for sm_86/89/100/120 and A8(int8) support for sm_70/75/86/89/100/120 on ninfer

## Summary

On GPUs without an FP4 tensor core (RTX 3090/4090, A16, etc.) upstream NVFP4 only
wires a SIMT `linear_swiglu` path for `T ≤ 16`; wider shapes throw. This PR adds
int8 matmul kernels for the wide path so NVFP4 prefill at large `T` actually runs,
a token-tile ladder so the same kernels stop paying 128 columns of tensor-core work
at narrow widths, a llama.cpp-shaped MMVQ mainloop that owns the decode band
(`T ≤ 8`) for every int8 route, plus an A16 reference route covering the same
upstream wiring gap.

That reference has since been rebuilt for speed (item 7): A16 `T = 1` decode went
16.1 → 48.0 tok/s on the same harness, so the int8 routes now lead it by ~3% at
decode and ~1.19× at prefill, not by 2.3×.

## What's included

1. **NVFP4 int8-grouped mainloop + SwiGLU route** (primary delivery).
   Self-implemented per-(token, 32-col) int8 activation quantiser against
   groupwise e2m1 weights with per-16 E4M3 group scales, writing through the
   caller's epilogue adapter (plain store, residual add, or gate product). Covers
   `Linear`, `LinearAdd`, `attn_input_proj`, `gdn_input_proj`, and `linear_swiglu`.
   `linear_swiglu` is two launches over one row range -- the gate half writes an
   FP32 plane the up half reads back, which is what keeps "no BF16 rounding before
   the SiLU" true -- so it does cost one FP32 plane of workspace (the other ops
   need none).
2. **NVFP4 int8 MMQ mainloop (llama.cpp shape)** — bit-exact vs the grouped kernel,
   default-off reference. Now staged on the same 11-step token ladder as (1) rather
   than `J = 128` only.
3. **Token-tile ladder, shared by both int8 mainloops**:
   `J ∈ {8,16,24,32,40,48,64,80,96,112,128}`. The stage is chosen as the smallest
   tile covering the width and both the grid and the activation-plane stride follow
   it, so every weight element is read exactly once at any width; surplus columns of
   a padded tile are computed and dropped by the epilogue guard rather than launched.
   (Earlier both loops padded to 128, which multiplied the tensor-core work per
   decode step by up to 16×.)
4. **MMVQ-shaped decode mainloop (llama.cpp port) + `mmvq` route**: one output row
   per CTA, dp4a, four warps sharing the row's K, `NC` padded to a power of two
   `{1,2,4,8}` so the accumulator stays in registers. Widths above 8 stay on the
   route's own wide kernel.
5. **The narrow band is now a property of the int8 family, not of the route name**:
   `T ≤ 8` is answered by (4) under `grouped`, `mmq` *and* `mmvq`, so decode has one
   implementation and the two MMA mainloops differ only from nine columns up. This is
   not a fallback cascade -- no width ever drops to A16 -- but it does mean the two
   MMA loops can no longer be A/B'd against each other at decode widths.
6. **Wide A16 MMA route for `NVFP4 linear_swiglu`** — fills the upstream `T > 16`
   gap on A16-only cards.
7. **A16 decode rewrite** — the same route, no longer a slow reference. The E2M1 pair
   decode no longer calls the outlined vendor conversion; the per-16 E4M3 group scales
   are read from a 256-entry ue4m3 table; the decode band's `2^14` rescale is folded
   into the group coefficient instead of applied per product; the weight flow carries a
   streaming cache hint and the fused SwiGLU scale reads are coalesced; and the GEMV,
   SIMT and sliced-K verify instances are pinned at 4 CTAs/SM (3 for sliced-K). Wide
   `T` prefill is untouched by this item -- it is a decode-band change.

## Routing / enabling

- `NINFER_NVFP4_GEMM = a16 | grouped | mmq | mmvq`; unset → `a16`. An unrecognised
  value throws and names the valid set (a trailing space in `cmd.exe` is part of the
  value), because a silently-ignored route name has already been mistaken for a
  measurement of "no difference".
- `NINFER_LINEAR_ROUTES=unified` is still required for narrow widths: the shipping
  device profile bands narrow widths away from the unified table, and the own table
  registers no int8 route at all. Both switches are needed to measure decode; naming
  only the first is a loud error, not a quiet A16 run.
- `device_profiles.json`: 3090 Ti prefill defaults to the int8-grouped unified table.
- `LinearPolicy::AllowNvfp4Int8` lands on sm_86 prefill shapes.

## Validation

**Perplexity** (`ninfer-perplexity`, `perplexity-1m` `--quick`, **KV bf16**, ctx 4096,
stride 2048, artifact `qwen3.8-27b-quasar`, 261,167 scored tokens, full-precision
reporting):

| Route   | PPL         |
|---------|-------------|
| a16     | 4.316281465 |
| grouped | 4.316377685 |
| mmq     | 4.316377685 |

`grouped` and `mmq` agree bit for bit (`mean_nll` equal to 16 digits), and the int8
routes sit **0.0000962 above** the A16 reference (+0.0022%). This table is not comparable
to the one this README used to carry: both the KV tier and the A16 kernels changed since,
so only the delta on this one build means anything. `mmvq` is not listed separately:
`T ≤ 8` is the MMVQ mainloop under every int8 route name, so it would be a third copy of
the same run. Throughput is reported only in the table below.

**Decode / prefill.** Same artifact, same 4202-token prompt, `unified` route table,
greedy, `--max-new 512`, **KV bf16**, CUDA graphs on except where a column says
otherwise. Every row below is this build.

| Route   | Spec    | Prefill tok/s | Decode tok/s | Decode, no graph |
|---------|---------|---------------|--------------|------------------|
| a16     | off     | 1.15k         | 48.0         | 45.2             |
| a16     | MTP=3   | 1.15k         | **71.2**     | 68.8             |
| grouped | off     | **1.37k**     | 49.4         | 45.6             |
| mmq     | off     | 1.32k         | 49.3         | 45.4             |

MTP=3 is measured on the a16 route only: 52.3% acceptance, 2.57 tok/round, per-position
histogram 143,102,67. No ngram row is published because ngram never fired here -- with
`--ngram-draft-tokens 31` the rate, the length and the histogram come back identical to
MTP=3 alone, i.e. zero ngram-only rounds. Graph-off prefill, for reference: a16 1.10k,
grouped 1.24k, mmq 1.23k.

Takeaways (sm_86, QUASAR-NVFP4):
- **The int8 decode lead is now ~1.03×, not 2.3×** (49.4-49.3 vs 48.0). The reason is
  item (7): the A16 decode band was rewritten and `T = 1` went 16.1 → 48.0 (3.0×), with
  4.2K prefill going 0.77-0.78k → 1.15k (1.49×). `grouped` and `mmq` still agree to
  0.1 tok/s at decode, which remains the proof that the band is unified.
- **Part of an int8 decode step runs on A16 kernels, deliberately.** The fused GDN
  conv/snapshot route keeps the A16 schedule domain under `AllowNvfp4Int8`
  (`src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp` resolves `T == 1` to
  `DecodeFusedA16`), and 48 of this model's 64 layers are GDN. That is why item (7)
  lifted the int8 arms as well: their own mainloops are byte-identical between the two
  builds (the diff across the A16 work names no int8 file) and their prefill column did
  not move (1.37k on both), so the whole int8 decode gain is A16-side residual work.
  llama.cpp has no such leak, because it does not fuse: the delta net's projections are
  ordinary `ggml_mul_mat` (so MMVQ takes them at `ne11 <= 8`) and its `ssm_conv1d` weight
  is never quantised (`src/llama-quant.cpp`), so it runs as a small f32 conv instead.
- **Prefill still belongs to int8**: 1.19× (1.37k vs 1.15k), and that is the regime the
  wide-`T` path exists for. Prefill widths never enter the narrow band -- the band is
  `T <= 8` and prefill here runs at `T >= 128`.
- KV tier matters and is pinned to bf16. The int8-KV rows this table used to carry are
  gone rather than carried forward: they came from an earlier build, and the tier costs
  ~1% of decode to buy ~13% of prefill.
- Kernel-level routing was verified by profiling rather than inferred: under all three
  int8 route names the decode phase launches the MMVQ kernel the same number of times
  (4352 launches), the `grouped` kernel answers zero decode calls under `mmq`/`mmvq`
  and the `mmq` kernel answers zero under `grouped`; and the grouped-form activation
  quantiser runs 7584 times under `grouped` versus 7328 + 256 under the wide-route
  names, i.e. the narrow band moved and the wide band did not. That census was taken on
  the build before item (7) and still describes this one, for the reason given above:
  nothing on the int8 side changed.
- Any speculative number needs the flag state read first: `--spec` turns
  `--ngram-draft-tokens 15` on by default, so an unlabelled "MTP" row is
  really a mixed MTP+ngram round, and the per-position acceptance histogram is sized by
  `max(draft_window, ngram_draft_window)`. Pure-MTP arms need `--ngram-draft-tokens 0`.
- 64K context starts and completes without OOM.

## Known limitations

- **A16 is no longer the slow reference, and it is still the shipping default.** After
  item (7) it sits within 3% of the int8 routes at `T = 1` decode, so picking an int8
  route is now a prefill decision (1.19×) rather than a decode one. Neither route has an
  FP4 tensor core on these cards: A16 dequantises into BF16 products, the int8 routes run
  the s8 MMA / dp4a loops. This is also likely why upstream left A16 unwired at width.
- **No stream-K in either int8 mainloop**: the grid is row-tile × column-tile and one
  CTA walks the whole K, so at narrow widths a 136-CTA launch on 84 SMs leaves a
  quarter-idle tail wave that stream-K would flatten (the GGUF path in this engine
  does have it).
- **`--prefill-cublas` is a no-op on NVFP4**: the int8 promotion helper checks only
  `prefill_a8`, unlike the Q4/Q5 one, and no cuBLAS route is registered for NVFP4.
- **The MMQ mainloop throws rather than degrading when `K % 256 != 0`** (a weight plane
  cannot be padded the way an activation tile can), and it registers no MoE /
  expert-table variant, no `input_columns` gather, and no `dequantize_rows` embedding
  path -- all of which the GGUF route has.
- There is **no decode-phase policy for NVFP4** (no `AllowA8IntDecode` counterpart):
  decode reaches the int8 path through the narrow band above, not through a
  width-admission flag.
- `J = 8` in both MMA ladders is unreachable under `grouped`/`mmq` now that `T ≤ 8`
  belongs to the MMVQ kernel; the ladder's bottom rung only matters to a build that
  puts the narrow band back.
- The title states kernel availability per architecture; the verification basis differs by
  tier: model-level numbers are **sm_86 only** (RTX 3090 Ti, QUASAR-NVFP4 artifact), the
  build + op-suite legs cover **sm_75 / 86 / 89 / 120a** (`arch: ['75','86','89','120a']`,
  all green on this master), and **sm_70 / sm_100 have no leg here** -- for those two the
  claim is that the routes resolve and the kernels are in the tree, not that they were run.

## Test plan

- Real-artifact int8 route tests (`test_nvfp4*.cpp`) on registered + aligned geometries.
- Op-level FP64-oracle tests: integer activation relative-L2 ≤ 0.04, over the ladder
  widths `{1,2,3,31..127}` and the MMVQ half `{1..8}`.
- **Five-route replay of the whole NVFP4 op suite** (env unset / a16 / grouped / mmq /
  mmvq × 8 binaries = 40 runs): 35 pass, 5 skip (the TMA staging-race case has no TMA
  route in an sm_8x build), 0 failures. Wrong route values fail loudly.
- Envelope checks that a declared `[min,max]` workspace interval covers every width its
  route can reach, including the padded width each mainloop actually addresses.

## Next steps

1. `J < 128` and the decode path are **done** by (3)-(5): the ladder, the MMVQ mainloop
   and the narrow-band unification give `T = 1` decode 49.3-49.4 tok/s on this build with
   PPL +0.0022% over A16. What is left in this direction is the
   fused narrow-`T` GLU: the int8 `linear_swiglu` is still two launches over an FP32
   gate plane at every width, where a single-launch `[gate;up]` form (as the GGUF
   decode kernel has) would remove one launch and the plane round-trip.
2. **Improve int8-path prefill performance.** Now that the ladder is in, the remaining
   known costs are stream-K (flat tail waves at 9-128 columns), the per-(token,32-col)
   quantiser's tile streaming, and a cuBLAS escape for very wide prefills.
3. **A higher-performance A16 path -- decode band done (item 7), prefill band open.**
   `T = 1` decode went 16.1 → 48.0 tok/s and 4.2K prefill 0.77-0.78k → 1.15k, so A16 is a
   viable route rather than a control. Still open on it: the wide-`T` A16 prefill band
   (item (7) is decode-band only), and the fused GDN conv/snapshot route, which keeps the
   A16 schedule domain even when an int8 route is selected -- now the largest single
   A16-sized block inside an int8 decode step. Letting that route reach the int8 kernels
   (or splitting the conv off the projection, the way llama.cpp's delta net does) is the
   next lever for both routes at once.
4. **Give the int8 route a decode-phase policy** (`AllowNvfp4Int8Decode`-shaped) instead
   of relying on the narrow band, and register the expert-table variant so an NVFP4 MoE
   artifact can use these kernels at all.
