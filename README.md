# Add NVFP4 int8 kernels (wide + decode) and the A16 SwiGLU wiring gap fill for sm_86/89 on Qwen3.8 27B

## Summary

On GPUs without an FP4 tensor core (RTX 3090/4090, A16, etc.) upstream NVFP4 only
wires a SIMT `linear_swiglu` path for `T ≤ 16`; wider shapes throw. This PR adds
int8 matmul kernels for the wide path so NVFP4 prefill at large `T` actually runs,
a token-tile ladder so the same kernels stop paying 128 columns of tensor-core work
at narrow widths, a llama.cpp-shaped MMVQ mainloop that owns the decode band
(`T ≤ 8`) for every int8 route, plus an A16 reference route covering the same
upstream wiring gap.

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
   gap on A16-only cards. Reference path (see Limitations).

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

**Perplexity** (`ninfer-perplexity`, `perplexity-1m` `--quick`, kv int8, ctx 4096,
artifact `qwen3.8-27b-quasar`, full-precision reporting):

| Route   | PPL         |
|---------|-------------|
| a16     | 4.315016269 |
| grouped | 4.314882867 |
| mmq     | 4.314882867 |
| mmvq    | 4.314882867 |

`grouped`, `mmq` and `mmvq` agree bit for bit, and all three sit **0.000133 below**
the A16 reference rather than above it: the int8 kernels introduce no quality
regression. Throughput is reported only in the decode/prefill table below.

**Decode / prefill.** Same artifact, same 4202-token prompt, `unified` route table,
greedy, `--max-new 512`. Every row below is this build.

| KV    | Route   | Prefill tok/s | Decode tok/s |
|-------|---------|---------------|--------------|
| int8  | a16     | 738.7         | 16.2         |
| int8  | grouped | 1.19k         | **36.8**     |
| int8  | mmq     | 1.20k         | **36.9**     |
| bf16  | a16     | 771 – 783     | 16.1         |
| bf16  | grouped | 1.37k         | 36.4         |
| bf16  | mmq     | 1.33k         | 36.4         |
| bf16  | mmvq    | 1.34k         | 36.4         |

Takeaways (sm_86, QUASAR-NVFP4):
- **`T = 1` decode is ~2.3× the A16 reference** (36.4-36.9 vs 16.1-16.2). `grouped`,
  `mmq` and `mmvq` agree to 0.1 tok/s at decode, which is itself the proof that the
  band is unified: three route names, one narrow-band kernel.
- **Prefill widths never enter the narrow band**: the band is `T <= 8` and prefill here
  runs at `T >= 128`, so the ladder and the MMVQ mainloop are a decode-only change by
  construction; the prefill column above runs the same wide mainloop as before.
- Prefill lead over A16 stands at ~1.7-1.8× (1.33-1.37k vs 0.77-0.78k at matched
  bf16 KV), and remains the regime the wide-`T` path exists for.
- KV tier matters and is now pinned: bf16 KV is the reporting tier from here on
  (it costs ~1% at decode and gains ~13% at prefill versus int8 KV), so numbers from
  the two tiers are not mixed inside one comparison.
- Kernel-level routing was verified by profiling rather than inferred: under all three
  int8 route names the decode phase launches the MMVQ kernel the same number of times
  (4352 launches), the `grouped` kernel answers zero decode calls under `mmq`/`mmvq`
  and the `mmq` kernel answers zero under `grouped`; and the grouped-form activation
  quantiser runs 7584 times under `grouped` versus 7328 + 256 under the wide-route
  names, i.e. the narrow band moved and the wide band did not.
- Any speculative number needs the flag state read first: `--spec` turns
  `--ngram-draft-tokens 15` on by default, so an unlabelled "MTP" row is
  really a mixed MTP+ngram round, and the per-position acceptance histogram is sized by
  `max(draft_window, ngram_draft_window)`. Pure-MTP arms need `--ngram-draft-tokens 0`.
- 64K context starts and completes without OOM.

## Known limitations

- The **a16 route is slow** (no FP4 core on these cards) — it is included as a
  reference for the wiring gap, not as a performance path. This is also likely why
  upstream left it unwired. It remains the shipping default.
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
- Validated on **sm_86 (RTX 3090 Ti)** with the QUASAR-NVFP4 artifact; broader
  sm_89/90 coverage is pending.

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
   and the narrow-band unification give `T = 1` decode 36.4-36.9 tok/s with PPL unmoved.
   What is left in this direction is the
   fused narrow-`T` GLU: the int8 `linear_swiglu` is still two launches over an FP32
   gate plane at every width, where a single-launch `[gate;up]` form (as the GGUF
   decode kernel has) would remove one launch and the plane round-trip.
2. **Improve int8-path prefill performance.** Now that the ladder is in, the remaining
   known costs are stream-K (flat tail waves at 9-128 columns), the per-(token,32-col)
   quantiser's tile streaming, and a cuBLAS escape for very wide prefills.
3. **Implement a higher performance A16-path kernel (separate).** Replace the current reference A16 MMA
   route with a higher-performance A16 implementation so the wiring-gap fill is no longer just
   a slow reference -- keeping it viable where the int8 path is not selected.
4. **Give the int8 route a decode-phase policy** (`AllowNvfp4Int8Decode`-shaped) instead
   of relying on the narrow band, and register the expert-table variant so an NVFP4 MoE
   artifact can use these kernels at all.
