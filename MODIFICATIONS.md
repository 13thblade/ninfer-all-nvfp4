# Modifications

This repository is a fork of NInfer-all with additional NVFP4 work. This file
records the changes relative to the fork base, as required by section 4(b) of
the Apache License 2.0 ("You must cause any modified files to carry prominent
notices stating that You changed the files"). It is an index of the changes;
the fork's git history remains the authoritative record.

## Fork base

| Item | Value |
| --- | --- |
| Upstream project | NInfer-all, https://github.com/iamwavecut/ninfer-all |
| Upstream copyright | The NInfer-all contributors (see CONTRIBUTORS.md) |
| Base commit | `78fd39aaeca75b454748d5a1281278112e34d5ce` (upstream master) |
| Upstream sync | six merges: `f2675499` took `63428ac2`, `63b68fdc` took `b3402a6f`, `121f8283` took `a53a4f33`, `78a5bbc9` took `3d882fc1`, `ec07d895` took `78fd39aa`, `aa55653e` took `3fda3165`. The fork was cut at `e1f2d961`; the commits between those bases are recorded by the merge commits themselves |
| Fork repository | https://github.com/13thblade/ninfer-all-nvfp4 |
| License | Apache License 2.0, see LICENSE (unchanged from upstream) |

## Fork commits

| Commit | Date | Subject |
| --- | --- | --- |
| `7b084383` | 2026-10-23 | feat(nvfp4): W4A16 (BF16) route for pre-Blackwell GPUs, and the KV dequant it shares a kernel with |
| `c45859ff` | 2026-10-24 | feat(nvfp4): W4A8-int route on the s8 tensor cores, all widths, and its Op-level gates |
| `40d05d15` | 2026-10-24 | docs(readme): restore the fork README and refresh its measured numbers |
| `6b64e762` | 2026-10-24 | docs(readme): take the prefill column as the peak of three cold starts |
| `0e2d49b4` | 2026-10-24 | Update README.md |
| `b0cc1d6b` | 2026-10-25 | docs(license): record the fork's modifications and where its notices live |
| `3350dc11` | 2026-10-25 | docs(readme): name the W4A16 route in the first change and record the W4A8-int kernel |
| `0b3b62b0` | 2026-10-25 | docs(readme): bold all four change lines and pluralize the W4A8-int kernels |
| `f2675499` | 2026-10-25 | Merge upstream master 63428ac2 into the fork |
| `8c04b9c6` | 2026-10-25 | docs(license): rebase the modification record onto upstream 63428ac2 |
| `c02c7692` | 2026-10-25 | docs(readme): drop the misread bands from the A16 domain sentence |
| `63b68fdc` | 2026-10-25 | Merge upstream master b3402a6f into the fork |
| `03b78d67` | 2026-10-26 | fix(deepseek_v41): take pi from a local constant, as the RoPE schedule does |
| `ae484558` | 2026-10-26 | docs(license): rebase the modification record onto upstream b3402a6f |
| `121f8283` | 2026-10-26 | Merge upstream master a53a4f33 into the fork |
| `78a5bbc9` | 2026-10-26 | Merge upstream master 3d882fc1 into the fork |
| `9b3614e7` | 2026-10-26 | docs(readme): point the throughput table at its source model cards |
| `ec07d895` | 2026-10-30 | Merge upstream master 78fd39aa into the fork |
| `aa55653e` | 2026-10-30 | Merge upstream master 3fda3165 into the fork |
| `b17f41aa` | 2026-10-31 | fix(nvfp4): route a tensor-parallel rank's FP32 partial away from the int8 family |
| `b8715ea2` | 2026-10-31 | feat(nvfp4-int8): let the fused SwiGLU and linear_add kernels take a rank's share |
| `7c25ab66` | 2026-10-31 | test: put the CUDA and vcpkg runtime DLLs on every test's PATH from CMake |
| `9ed54e1f` | 2026-10-31 | feat(nvfp4-int8): let the fused attention and GDN input projections take a rank's share |
| `5e6ef4cb` | 2026-10-31 | feat(nvfp4-a16): let the fused A16 routes take a tensor-parallel rank's share |
| `557c6e9c` | 2026-10-31 | test(nvfp4): cover a rank's share in the fused SwiGLU and linear_add Op tests |
| `1069d4d7` | 2026-10-31 | fix(nvfp4-a16): size the unified small-T SwiGLU's output stride from the weight |
| `7aed8158` | 2026-10-31 | feat(qwen3_5): let --nvfp4-w4a8-int reach a tensor-parallel rank's input projections |

`03b78d67` no longer shows in the file lists below: upstream `7cf5b2ae` fixed
the same defect and the merge `121f8283` took upstream's side, so the fork keeps
no change to that file.

The two nvfp4 commits above are master's own history. The branches carried for
review against upstream are rebased equivalents of them on top of the current
base: `pr/nvfp4-w4a16` = `18491bed` and `pr/nvfp4-w4a8-int` = `220dbfb1`.
Neither branch is in master's ancestry; their content is. `220dbfb1` carries the
same patch-id as `05c47ff8` across both later rebases, and `18491bed` differs
from `b76d912e` only by the `gguf_only` gate named below.

## Change summary

Relative to the fork base this fork:

- modifies 111 existing upstream files;
- adds 29 new files;
- deletes no upstream files.

The functional changes add the NVFP4 W4A16 (BF16) route for pre-Blackwell GPUs,
the NVFP4 W4A8-int route on the s8 tensor cores across all supported widths,
the KV dequant path they share, the route selection and command-line switches
(`--nvfp4-w4a16`, `--nvfp4-w4a8-int`), and the accompanying Op-level and
real-data tests, benchmarks, build entries and documentation. The README
contains the measured results and their basis.

The upstream sync merge `f2675499` touched three files where both sides grew
the `ffn()` parameter list. The fork's `input_norm` and upstream's `partial`
were both kept, in that order, in `src/models/qwen3_5/execution/ffn.h`; the two
guards stay as separate checks in `src/models/qwen3_5/execution/ffn.cpp`; and
`src/models/qwen3_5/execution/text.cpp` keeps the fork's fused int8 branch on
top of upstream's hoisted `layer_config()` buffer. A tensor-parallel share that
is handed a deferred norm is refused by an explicit throw in both the callee and
the caller, because only the composed path folds the norm. The same three files
were the only conflicts when `pr/nvfp4-w4a16` and `pr/nvfp4-w4a8-int` were
rebased onto `b3402a6f`, and they took the resolution above byte for byte; both
later rebases onto `a53a4f33` were clean with no conflict at all.

`src/models/deepseek_v41/executor.cpp` is listed as modified by no one but
upstream. When `d3b2475c` landed it, its YaRN frequency ramp used `M_PI` with
only `<cmath>` included: MSVC does not define `M_PI` there and CUDA 13.4 no
longer leaks it from its headers, so that single edge broke a Windows build. The
fork patched it with a local `constexpr double kPi` (`03b78d67`); upstream fixed
the same defect in `7cf5b2ae` with the C++20 `std::numbers::pi` and its include,
which is both the standard spelling and already the convention in this repo's
tests. The merge `121f8283` therefore takes upstream's side and retires the
fork's patch, so the file is no longer a fork modification.

The merge `78a5bbc9` conflicted once, on the calibration driver
`src/calibration/device_calibration.cu`, which is where the nvfp4 keys enter the
device route profile. Upstream wrapped the groupwise family into a block that
also runs its new `gguf_vector`, `gguf_moe_vector` and `gguf_moe_up`
calibrations and gated `linear_attention` on the run's `gguf_only` filter. The
resolution keeps upstream's block whole, re-adds the fork's nvfp4 line after it
and gates that line the same way as its siblings. One behaviour follows from
that: a calibration run restricted with `--only gguf_*` no longer also measures
the nvfp4 family, which the fork's line had previously done by oversight.

The merge `ec07d895` conflicted on six files, all where the tensor-parallel work
between `3d882fc1` and `78fd39aa` met the fork's route wiring. In
`src/models/qwen3_5/execution/ffn.h` the fork's `InputNorm` stays and upstream's
new `partial` parameter to `ffn_workspace_bytes` is taken alongside it; in
`ffn.cpp` the fork's deferred-norm guards stay and are followed by upstream's
`MoeShareParameters` branch, whose rotated-share case upstream now supports, so
the fork's `rotated_dense` rejection is dropped rather than carried forward. In
`parameters.cpp` the fork's two nvfp4 route calls on the MTP output projection
stay and `out.ffn` takes upstream's renamed `ffn(w.layer.ffn)`. In
`src/ops/linear/linear.cpp` the fork's exclusion of `AllowNvfp4Int8` from the
sm_70 FP16 route stays, while upstream's FP32 partial-sum `Destination` is
taken whole (the fork's `out.dtype == BF16` gate was dropped: it would have
refused the FP32 shares the new work needs). `bench/ops/linear_bench.cu` keeps
both usage lines, and `tests/ops/tests.cmake` keeps both new test registrations.

The W4A16 and W4A8-int routes take a tensor-parallel rank's share of a projection
(`--tensor-parallel`). Plain Linear already did -- `nvfp4_general_a16` reads both dimensions at
launch time and selects its writer by output dtype -- but the fused ops did not, because their
shapes were instances of the whole 27B parent. A share moves only the divided dimension (the dense
FFN's gate and up rows, the down projection's input columns, and the attention and GDN input
projections' head rows), so those routes now read that dimension from the weight or from the
output tensors and keep the undivided one as the template argument every K-specialised schedule is
built on. The A16 wide SwiGLU, its decode GEMV and its SIMT small-T band, `linear_add`'s per-K
ladders (which gain a runtime-K MMA for a share) and the two input projections' segment adapters
all follow the call.

One defect had to be fixed before the int8 family was correct there at all. A row-split
projection's partial sum is FP32, but `allows_nvfp4_int8` was tested before the output dtype, so
an int8 call took the FP32 partial and stored it through a BF16 writer -- two bytes per element
into a four-byte buffer, filling the front half and leaving the rest stale without raising
anything, which the all-reduce then read back as garbage. FP32 now takes the general A16 route,
which selects its writer by dtype, and the int8 entry asserts its own contract so a future misuse
fails loudly instead of writing half a buffer.

The two input projections' fused single-parent routes take a share too, but a rank reaches them as
parts rather than as a fused parent, so what runs under `--tensor-parallel` is the parts route: one
Linear per segment. The fused share handling is what a caller that binds a rank's fused parent
needs, and the Op-level tests do not reach it yet.

Under `--tensor-parallel` the `--nvfp4-w4a8-int` switch reaches the column-split attention and
GDN input projections only, and it had to learn a second carrier to do even that: those
projections arrive as `PartProjectionWeights` on a rank, where they are stored as their segments,
and the promotion only knew `SingleProjectionWeight`. A row-split projection keeps running A16
under the flag, because its partial sum is FP32 and the int8 family writes BF16.

Tensor parallelism is not bit-identical to one device by construction: the ranks' FP32 partials
are summed in rank order and rounded once, a different association of the same terms than a single
device's chain. What is held is what `core/tensor_parallel.h` states -- every replica byte for byte
identical, and one rounding where a single device rounds once -- and the README's measured
perplexity shows the difference that leaves.

The lists below are generated from `git diff --name-status upstream/master..HEAD`
with `upstream/master` at `3fda3165`. Files marked as modified are upstream
files changed by this fork; files marked as added are new in this fork.

## Modified files (111)

### Root, build and documentation (4)

- `Dockerfile`
- `README.md`
- `docker/entrypoint.sh`
- `docs/device-profiles.md`

### apps (5)

- `apps/calibrate/main.cpp`
- `apps/cli/main.cpp`
- `apps/cli/options.cpp`
- `apps/cli/options.h`
- `apps/perplexity/main.cpp`

### bench (2)

- `bench/ops/linear_bench.cu`
- `bench/ops/nvfp4_linear_swiglu_bench.cu`

### include (4)

- `include/ninfer/engine_options.h`
- `include/ninfer/ops/attn_input_proj.h`
- `include/ninfer/ops/linear.h`
- `include/ninfer/ops/linear_swiglu.h`

### src/calibration (3)

- `src/calibration/device_calibration.cu`
- `src/calibration/device_calibration.h`
- `src/calibration/route_catalog.cpp`

### src/models (5)

- `src/models/load_options.h`
- `src/models/qwen3_5/execution/ffn.cpp`
- `src/models/qwen3_5/execution/ffn.h`
- `src/models/qwen3_5/execution/parameters.cpp`
- `src/models/qwen3_5/execution/text.cpp`

### src/ops/attn_input_proj (9)

- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_a16.cu`
- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode.cu`
- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode_unified.cu`
- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh`
- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp`
- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h`
- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan_unified.cpp`
- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan_unified.h`
- `src/ops/attn_input_proj/sources.cmake`

### src/ops/gdn_input_proj (12)

- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a16.cu`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode.cu`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode_unified.cu`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan_unified.cpp`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan_unified.h`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output_unified.cuh`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_decode.cu`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_decode_unified.cu`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan_unified.cpp`
- `src/ops/gdn_input_proj/sources.cmake`

### src/ops/kv_cache (2)

- `src/ops/kv_cache/fp8_e4m3_row_codec.cuh`
- `src/ops/kv_cache/nvfp4_group16_codec.cuh`

### src/ops/linear (27)

- `src/ops/linear/common/output.cuh`
- `src/ops/linear/common/route_table.cpp`
- `src/ops/linear/linear.cpp`
- `src/ops/linear/nvfp4/nvfp4_a16_gemv.cuh`
- `src/ops/linear/nvfp4/nvfp4_a16_mma.cuh`
- `src/ops/linear/nvfp4/nvfp4_a16_simt.cuh`
- `src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh`
- `src/ops/linear/nvfp4/nvfp4_codec.cuh`
- `src/ops/linear/nvfp4/nvfp4_codec_unified.cuh`
- `src/ops/linear/nvfp4/nvfp4_dispatch.cpp`
- `src/ops/linear/nvfp4/nvfp4_dispatch_unified.cpp`
- `src/ops/linear/nvfp4/nvfp4_gemv.cuh`
- `src/ops/linear/nvfp4/nvfp4_instances.cuh`
- `src/ops/linear/nvfp4/nvfp4_shapes_unified.h`
- `src/ops/linear/nvfp4/nvfp4_simt.cuh`
- `src/ops/linear/nvfp4/shapes/n14336_k5120.cu`
- `src/ops/linear/nvfp4/shapes/n14336_k5120_unified.cu`
- `src/ops/linear/nvfp4/shapes/n16384_k5120.cu`
- `src/ops/linear/nvfp4/shapes/n16384_k5120_unified.cu`
- `src/ops/linear/nvfp4/shapes/n34816_k5120.cu`
- `src/ops/linear/nvfp4/shapes/n34816_k5120_unified.cu`
- `src/ops/linear/nvfp4/shapes/n5120_k17408.cu`
- `src/ops/linear/nvfp4/shapes/n5120_k17408_unified.cu`
- `src/ops/linear/nvfp4/shapes/n5120_k6144.cu`
- `src/ops/linear/nvfp4/shapes/n5120_k6144_unified.cu`
- `src/ops/linear/nvfp4/sources.cmake`
- `src/ops/linear/t2/t2_dispatch.cpp`

### src/ops/linear_add (7)

- `src/ops/linear_add/nvfp4/nvfp4_linear_add_a16.cu`
- `src/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu`
- `src/ops/linear_add/nvfp4/nvfp4_linear_add_decode_unified.cu`
- `src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp`
- `src/ops/linear_add/nvfp4/nvfp4_linear_add_plan_unified.cpp`
- `src/ops/linear_add/nvfp4/nvfp4_linear_add_plan_unified.h`
- `src/ops/linear_add/sources.cmake`

### src/ops/linear_swiglu (8)

- `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu`
- `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode_unified.cu`
- `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp`
- `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h`
- `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan_unified.cpp`
- `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan_unified.h`
- `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu`
- `src/ops/linear_swiglu/sources.cmake`

### src/ops/wrapper (4)

- `src/ops/wrapper/attn_input_proj.cpp`
- `src/ops/wrapper/gdn_input_proj.cpp`
- `src/ops/wrapper/linear_add.cpp`
- `src/ops/wrapper/linear_swiglu.cpp`

### src/serve (4)

- `src/serve/generation_service.cpp`
- `src/serve/request_log.cpp`
- `src/serve/serve_options.cpp`
- `src/serve/serve_options.h`

### tests (14)

- `tests/CMakeLists.txt`
- `tests/cmake/NinferTests.cmake`
- `tests/ops/linear/linear_test_common.cpp`
- `tests/ops/linear/linear_test_common.h`
- `tests/ops/linear/test_nvfp4_a16.cpp`
- `tests/ops/linear/tests.cmake`
- `tests/ops/linear_swiglu/linear_swiglu_test_common.cpp`
- `tests/ops/linear_swiglu/linear_swiglu_test_common.h`
- `tests/ops/linear_add/test_nvfp4.cpp`
- `tests/ops/linear_swiglu/test_nvfp4.cpp`
- `tests/ops/op_tester.h`
- `tests/ops/quantized_weight.h`
- `tests/ops/test_gdn_input_proj_conv_snapshot.cpp`
- `tests/ops/tests.cmake`

### tools (1)

- `tools/w4a8_inner_loop_probe.cu`

## Added files (29)

### root (1)

- `MODIFICATIONS.md`

### src/ops/attn_input_proj (1)

- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_int8.cu`

### src/ops/gdn_input_proj (1)

- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_int8.cu`

### src/ops/linear (18)

- `src/ops/linear/nvfp4/nvfp4_e4m3_scale.h`
- `src/ops/linear/nvfp4/nvfp4_int8_fused_norm_quant.cu`
- `src/ops/linear/nvfp4/nvfp4_int8_fused_norm_quant.cuh`
- `src/ops/linear/nvfp4/nvfp4_int8_fused_norm_quant.h`
- `src/ops/linear/nvfp4/nvfp4_int8_grouped.cu`
- `src/ops/linear/nvfp4/nvfp4_int8_grouped_host.h`
- `src/ops/linear/nvfp4/nvfp4_int8_grouped_mma.cuh`
- `src/ops/linear/nvfp4/nvfp4_int8_narrow_host.h`
- `src/ops/linear/nvfp4/nvfp4_int8_narrow_launch.cuh`
- `src/ops/linear/nvfp4/nvfp4_int8_narrow_mma.cuh`
- `src/ops/linear/nvfp4/nvfp4_int8_route.h`
- `src/ops/linear/nvfp4/nvfp4_mmq_host.h`
- `src/ops/linear/nvfp4/nvfp4_mmq_launch.cuh`
- `src/ops/linear/nvfp4/nvfp4_mmq_mma.cuh`
- `src/ops/linear/nvfp4/nvfp4_mmvq_host.h`
- `src/ops/linear/nvfp4/nvfp4_mmvq_launch.cuh`
- `src/ops/linear/nvfp4/nvfp4_mmvq_mma.cuh`
- `src/ops/linear/nvfp4/shapes/n8192_k4096_unified.cu`

### src/ops/linear_add (1)

- `src/ops/linear_add/nvfp4/nvfp4_linear_add_int8.cu`

### src/ops/linear_swiglu (3)

- `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a16_mma.cu`
- `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_int8_grouped.cu`
- `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_int8_grouped.cuh`

### tests (4)

- `tests/fixtures/nvfp4_int8_real/generate_fixture.py`
- `tests/ops/linear/test_nvfp4_int8.cpp`
- `tests/ops/linear/test_nvfp4_int8_fused_norm.cpp`
- `tests/ops/test_nvfp4_int8_real.cpp`
