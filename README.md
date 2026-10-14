# Add NVFP4 int8 kernels and wide A16 SwiGLU route for sm_86/89 on Qwen3.8 27B

## Summary

On GPUs without an FP4 tensor core (RTX 3090/4090, A16, etc.) upstream NVFP4 only
wires a SIMT `linear_swiglu` path for `T ≤ 16`; wider shapes throw. This PR adds
int8 matmul kernels for the wide path so NVFP4 prefill at large `T` actually runs,
plus an A16 reference route covering the same upstream wiring gap.

## What's included

1. **NVFP4 int8-grouped mainloop + fused SwiGLU route** (primary delivery).
   Self-implemented per-(token, 32-col) int8 activation quantiser against
   groupwise e2m1 weights with per-16 E4M3 group scales; fused epilogue, no
   workspace. Covers `Linear`, `LinearAdd`, `attn_input_proj`, `gdn_input_proj`,
   and `linear_swiglu`.
2. **NVFP4 int8 MMQ mainloop (llama.cpp port)** — bit-exact vs the grouped kernel,
   staged at `J = 128`. Reference/experimental, default-off.
3. **Wide A16 MMA route for `NVFP4 linear_swiglu`** — fills the upstream `T > 16`
   gap on A16-only cards. Reference path (see Limitations).

## Routing / enabling

- `NINFER_NVFP4_GEMM = a16 | grouped | mmq`; unset → `a16`.
- `device_profiles.json`: 3090 Ti prefill defaults to the int8-grouped unified table.
- `LinearPolicy::AllowNvfp4Int8` lands on sm_86 prefill shapes.

## Validation

**Perplexity** (`ninfer-perplexity`, `perplexity-1m` `--quick`, kv int8, ctx 4096 / stride 2048,
artifact `qwen3.8-27b-quasar`):

| Route   | PPL       | scoring tok/s |
|---------|-----------|---------------|
| a16     | 4.315224  | 399.5         |
| grouped | 4.315502  | 713.2         |
| mmq     | 4.315502  | 667.2         |

`grouped` and `mmq` are bit-exact identical; all three within **+0.0003** of the A16
reference ⇒ the int8 kernels introduce no measurable quality regression. int8 scores
**~1.8×** faster than the A16 reference.

**Decode / prefill** (4K prompt ≈ 4202 tok, `--prefill-chunk 1024`, `--max-context 65536`,
greedy, kv int8):

| Config                | Route  | Prefill tok/s | Decode tok/s | 64K free VRAM | Draft accept |
|-----------------------|--------|---------------|--------------|---------------|--------------|
| baseline (no spec)    | a16    | 735.8         | 16.1         | 4.00 GiB      | —            |
| MTP=3 (SIMT path)     | a16    | 743.4         | 28.7         | 3.40 GiB      | 47.2%        |
| MTP=3 + ngram-31 (MMA)| a16    | 735.9         | 27.6         | 3.36 GiB      | 43.9%        |
| baseline (no spec)    | grouped| 1.28k         | 9.9          | 4.00 GiB      | —            |
| MTP=3 + ngram-31 (MMA)| grouped| 1.26k         | 35.4         | 3.36 GiB      | 64.8%        |
| baseline (no spec)    | mmq    | 1.19k         | 9.0          | 4.00 GiB      | —            |
| MTP=3 + ngram-31 (MMA)| mmq    | 1.18k         | 32.5         | 3.36 GiB      | 64.8%        |

Takeaways (sm_86, QUASAR-NVFP4, 4K prompt / 64K ctx):
- **Prefill / throughput: int8 ≈ 1.7–1.8× the A16 reference** (1.28k vs 736 prefill; 713 vs 399 PPL
  scoring). This is the regime the wide-`T` NVFP4 path exists for, and where int8 matters most.
- **Decode:** at `T = 1` without speculation the A16 reference is still faster (16.1 vs 9.9 grouped);
  under MTP+ngram the int8 routes win (grouped 35.4 / mmq 32.5 vs A16 27.6) once the per-step
  quantisation overhead is amortised over the drafted batch. Pure `T = 1` decode remains an
  optimisation target.

  The `T = 1` gap is **not** a quality or algorithmic issue — it is simply that the
  small-`J` (`J < 128`) kernels and the dedicated SIMT decode mainloop are **not implemented
  yet** in this PR. At `T = 1` the int8 route currently falls back to an unoptimised/larger-`J`
  path, so the per-step quantisation setup cost is not amortised and it loses to the straight
  A16 reference; the MTP+ngram numbers show the kernel itself is fine once enough work is
  batched per step. Closing this gap is the first item in *Next steps*.
- Draft acceptance 64.8% (MTP+ngram, accept length 3.93) confirms the MMA-path logits track the
  target with no systematic drift. 64K context starts and completes without OOM.

## Known limitations

- The **a16 route is slow** (no FP4 core on these cards) — it is included as a
  reference for the wiring gap, not as a performance path. This is also likely why
  upstream left it unwired.
- The **mmq route is J=128-only staged**; it is a bit-exact reference for the
  grouped kernel and is default-off.
- Validated on **sm_86 (RTX 3090 Ti)** with the QUASAR-NVFP4 artifact; broader
  sm_89/90 coverage is pending.

## Test plan

- Real-artifact int8 route tests (`test_nvfp4*.cpp`) on registered + aligned geometries.
- Op-level FP64-oracle tests: integer activation relative-L2 ≤ 0.04.
- Four-route configs (env unset / a16 / grouped / mmq) all green; wrong values fail loudly.

## Next steps

1. **Implement the `J < 128` kernels including the decode path.** Add the small-`J`
   SIMT decode mainloop so that `T = 1` decode stops falling back to the unoptimised
   large-`J` path. This is expected to close the pure-decode gap against the A16 reference
   (16.1 vs 9.9 grouped) without changing PPL.
2. **Improve int8-path prefill performance.** Further optimise the grouped/MMQ prefill
   mainloop (e.g. better tile/streaming of the per-(token, 32-col) activation quantiser,
   pack/dequant overlap, larger K-blocking) to push the current ~1.7–1.8× lead over A16 wider.
3. **Implement a higher performance A16-path kernel (separate).** Replace the current reference A16 MMA
   route with a higher-performance A16 implementation so the wiring-gap fill is no longer just
   a slow reference — keeping it viable where the int8 path is not selected.
