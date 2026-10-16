# W4(NVFP4) A16(BF16) support for sm_86/89/100/120 and A8(int8) support for sm_70/75/86/89/100/120 on ninfer

## Summary

NVFP4 weights run on GPUs that have no FP4 tensor core, through two selectable activation routes
covering all five NVFP4 op routes (`linear`, `linear_add`, `attn_input_proj`, `gdn_input_proj`,
`linear_swiglu`):

- **W4A16** -- BF16 activations, dequantising kernels instantiated from the decode band through
  prefill width, so a wide prompt span prefills instead of throwing.
- **W4A8** -- int8 activations: E2M1 codes driven through the s8 tensor cores (grouped and MMQ
  mainloops) or through `dp4a` (MMVQ mainloop), with per-(token, 32-column) activation quantization
  against the per-16 E4M3 group scales.

The default route is A16 and A8 is opt-in at runtime; nothing here changes behaviour for a card with
native FP4 tensor cores, which keeps upstream's W4A4 paths. The A16 route reproduces the same
Op-level mathematics upstream already defined at narrow width -- what its added kernels change is
which widths and schedules reach it. The int8 route is an approximation trade and is measured as
one: +0.0022% PPL over the A16 reference on this model, with `grouped` and `mmq` bit-identical to
each other.

## What's included

1. **Wide-width coverage.** `linear_swiglu` has a wide A16 MMA route and a wide int8 grouped route,
   and the fused attn/GDN input projections and `linear_add` carry both activation forms, so every
   one of the five op routes has a kernel at prefill width rather than only at decode width.
2. **Three int8 mainloops with one width rule.** A grouped s8-MMA loop against groupwise E2M1
   weights, an MMQ loop in the llama.cpp shape, and an MMVQ loop (one output row per CTA, four
   warps sharing the row's K, `NC` padded to `{1,2,4,8}` so the accumulator stays in registers).
   `T <= 8` is answered by MMVQ under *all three* route names: the narrow band is a property of the
   int8 family, not of the name, so decode has one implementation and the two MMA loops differ only
   from nine columns up. No width ever drops to A16 -- this is not a fallback cascade.
3. **A shared token-tile ladder.** `J in {8,16,24,32,40,48,64,80,96,112,128}`, chosen as the
   smallest tile covering the width, with the grid and the activation-plane stride following the
   stage, so each weight element is read exactly once at any width. Surplus columns of a padded tile
   are computed and dropped by the epilogue guard rather than launched.
4. **A16 decode schedules.** The E2M1 pair decode no longer calls the outlined vendor conversion;
   per-16 E4M3 group scales come from a 256-entry ue4m3 table; the band's `2^14` rescale is folded
   into the group coefficient instead of applied per product; the weight flow carries a streaming
   cache hint and the fused SwiGLU scale reads are coalesced; GEMV, SIMT and sliced-K verify
   instances are pinned at 4 CTAs/SM (3 for sliced-K).
5. **`linear_swiglu` keeps exactness at width by spending workspace.** Two launches over one row
   range: the gate half writes an FP32 plane the up half reads back, which is what keeps "no BF16
   rounding before the SiLU" true. One FP32 plane of workspace is the price; the other ops need
   none.
6. **The fused GDN conv/snapshot route keeps the A16 schedule domain even under `AllowNvfp4Int8`**
   (`src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp` resolves `T == 1` to
   `DecodeFusedA16`). Since 48 of this model's 64 layers are GDN, part of an int8 decode step runs on
   A16 kernels by design, and any A16-side schedule change moves the int8 arms too.

## Routing / enabling

**Which architectures compile these kernels.** `NINFER_SM8X_COMPAT` is defined for a
`CMAKE_CUDA_ARCHITECTURES` of `70`, `75`, `80`, `86` or `89`, and for any build that does not ask for
`NINFER_SM120_NATIVE=ON` (default `OFF`, so a `120a` build also lands on this path). Every kernel
above compiles on all of them: the int8 MMA loops carry three instruction forms -- the pre-sm_80
warp form `sm70::mma_m16n8k16_s8` on a Volta build, two `sm75::mma_s8_k16` halves on Turing, and
`mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` from Ampere up -- while MMVQ needs only `__dp4a`.
On a compat build that is not `120a`, upstream's Blackwell-only translation units
(`nvfp4/**/*_(w4a4|a4).cu`, `fp8/**/*_a8*.cu`) are dropped from the target and replaced by
`src/ops/nvfp4_sm86_stubs.cpp` / `fp8_sm86_stubs.cpp`, because `kind::mxf4nvf4.block_scale` and
`kind::f8f6f4` have no pre-Blackwell equivalent.

Verification basis, stated per tier because it is not uniform: the build plus the NVFP4 op suite run
as CI legs for **sm_75 / 86 / 89 / 120a**; model-level numbers below exist for **sm_86 only** (RTX
3090 Ti); **sm_70 and sm_100** compile on this path but have no leg on this repository, so for those
two the claim is compile-and-resolve, not run.

**`device_profiles.json` holds no tuned policy for these routes, on any architecture.**
`src/calibration/route_catalog.cpp` registers six `unified/*` keys and they are all q4/q5 -- there is
no NVFP4 key -- so `--device-profile calibrate` can neither measure nor emit `unified/nvfp4_*` on any
card. The NVFP4 bands that do sit in the repository file (RTX 3090, RTX 4090, RTX 5090) were
hand-written against the kernels of 2026-09-27, and because built-in rows carry no `calibration`
block they are exempt from the route-catalog digest check that retires a stale calibrated entry. To
use these routes on your own card, either copy the `nvidia-geforce-rtx-3090-ti-sm86` entry as the
reference shape -- four keys, `T <= 127` left to the compiled route and every wider width sent to
the unified table -- or skip the profile and set `NINFER_LINEAR_ROUTES=unified` yourself.

**A16 is the default; A8 needs three switches.**

1. `NINFER_NVFP4_GEMM = a16 | grouped | mmq | mmvq`; unset means `a16`. An unrecognised value throws
   and names the valid set (a trailing space in `cmd.exe` is part of the value), because a
   silently-ignored route name has already been mistaken for a measurement of "no difference".
2. `NINFER_LINEAR_ROUTES=unified`. The own table registers no int8 route at all, so naming only the
   first switch is a loud error, not a quiet A16 run; both are needed to measure decode.
3. The engine's `prefill_a8` admission left on (it is `true` by default; `--no-prefill-a8` clears
   it). This is what registers `LinearPolicy::AllowNvfp4Int8`, and it registers **per exact
   `(n, k)` shape**, not per format: the projections admitted here are 34816x5120, 5120x17408,
   14336x5120, 16384x5120 and 5120x6144. Any other geometry keeps `A16Only`, which is why a new
   model shape has to be added at this layer before the route name can reach it.

## Validation

**Perplexity** (`ninfer-perplexity`, `perplexity-1m` `--quick`, **KV bf16**, ctx 4096, stride 2048,
artifact `qwen3.8-27b-quasar`, 261,167 scored tokens, full-precision reporting). Card: RTX 3090 Ti,
sm_86.

| Route   | PPL         |
|---------|-------------|
| a16     | 4.316281465 |
| grouped | 4.316377685 |
| mmq     | 4.316377685 |

`grouped` and `mmq` agree bit for bit (`mean_nll` equal to 16 digits); the int8 routes sit 0.0000962
(+0.0022%) above the A16 reference. `mmvq` is not listed separately: `T <= 8` is the MMVQ mainloop
under every int8 route name, so it would be a third copy of the same run.

**Throughput.** Same artifact, same 4202-token prompt, greedy, `--max-new 512`, **KV bf16**, CUDA
graphs on except where a column says otherwise, `unified` route table. Each row names the build it
was taken on; readings from different builds are in one table but are not comparable with each other
beyond the labels given.

| Route | Spec   | Prefill tok/s | Decode tok/s | Decode, no graph | Build |
|-------|--------|---------------|--------------|------------------|-------|
| a16   | off    | 1.15k         | 48.0         | 45.2             | this build |
| a16   | MTP=3  | 1.15k         | **71.2**     | 68.8             | this build |
| grouped | off  | **1.37k**     | 49.4         | 45.6             | this build |
| mmq   | off    | 1.32k         | 49.3         | 45.4             | this build |
| a16   | off    | 771 - 783     | 16.1         | --               | wired, before the schedule work (`ba8e3d33…`) |
| a16   | off    | **throws**    | 16.1 [d]     | --               | upstream as shipped, no local wiring |

[d] No unpatched build was run on this machine. The reading is still attributable to upstream rather
than to this fork because at that stage the decode phase executed upstream's own kernels -- see gap 3.

MTP=3 is measured on the a16 route only: 52.3% acceptance, 2.57 tok/round, per-position histogram
143,102,67. No ngram row is published because ngram never fired here -- with
`--ngram-draft-tokens 31` the rate, the length and the histogram come back identical to MTP=3 alone,
i.e. zero ngram-only rounds. Graph-off prefill, for reference: a16 1.10k, grouped 1.24k, mmq 1.23k.
`--spec` turns `--ngram-draft-tokens 15` on by default, so an unlabelled "MTP" reading is a mixed
MTP+ngram round; pure-MTP arms need `--ngram-draft-tokens 0`. 64K context starts and completes
without OOM.

**What is missing in upstream's W4A16 implementation**, i.e. what those last two rows are measuring:

1. **No wide-width kernel, and no fallback to one.** The fused NVFP4 SwiGLU route is instantiated
   only up to `T <= 16`, and `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp:57-67`
   resolves the A16 domain as `T <= 16 -> SmallTFusedA16`, `T <= 64 -> Materialized`,
   `T > 64 -> throw`. A 4202-token prompt span therefore does not produce a slow prefill, it produces
   a throw: the missing number in the table is the measurement.
2. **No int8 activation route at all for NVFP4 on these architectures.** The s8-MMA and `dp4a`
   forms this branch adds are not present upstream for the NVFP4 formats; only the Blackwell W4A4
   path exists, and its translation units are compiled out of a pre-Blackwell build (see Routing).
3. **The narrow-width kernels are unoptimized, which is why "runs at `T <= 16`" is not the same as
   "usable".** At the stage named in the table, the decode step ran upstream's own legacy forms
   (`nvfp4_gemv_kernel`, `nvfp4_linear_swiglu_decode_kernel`; the shipping device profile bands
   narrow widths away from the unified table, which is what makes the `16.1` attributable to them).
   An nsys census of that build measured 63.5 ms of kernel time against 62.9 ms of wall clock per
   token -- all kernel time, no gaps left to overlap -- with those five NVFP4 kernels holding 94.9%
   of it. About 35 warp instructions per FP4 code over 27e9 codes is roughly 47 ms of that 63 ms:
   issue-bound, and 16.1 tok/s against the 16.1 GB a step moves is 259 GB/s effective, 26% of this
   card's 1008 GB/s peak [d: arithmetic on the two measured numbers].

## Next steps

Continue raising kernel performance on both routes, W4A16 and W4A8.
