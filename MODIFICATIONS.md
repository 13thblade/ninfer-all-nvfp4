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
| Base commit | `63428ac249da8a6c5d891b508d7e483d8c7def6c` (upstream master) |
| Upstream sync | `f2675499` merges upstream master into the fork. The fork was cut at `e1f2d961` and later took `9456cb7a`, `230ef895` and `63428ac2` |
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

## Change summary

Relative to the fork base this fork:

- modifies 106 existing upstream files;
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
the caller, because only the composed path folds the norm.

The lists below are generated from `git diff --name-status upstream/master..HEAD`
with `upstream/master` at `63428ac2`. Files marked as modified are upstream
files changed by this fork; files marked as added are new in this fork.

## Modified files (106)

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

### src/ops/attn_input_proj (8)

- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_a16.cu`
- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode.cu`
- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode_unified.cu`
- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp`
- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h`
- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan_unified.cpp`
- `src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan_unified.h`
- `src/ops/attn_input_proj/sources.cmake`

### src/ops/gdn_input_proj (11)

- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a16.cu`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode.cu`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode_unified.cu`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan_unified.cpp`
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan_unified.h`
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

### tests (11)

- `tests/ops/linear/linear_test_common.cpp`
- `tests/ops/linear/linear_test_common.h`
- `tests/ops/linear/test_nvfp4_a16.cpp`
- `tests/ops/linear/tests.cmake`
- `tests/ops/linear_swiglu/linear_swiglu_test_common.cpp`
- `tests/ops/linear_swiglu/linear_swiglu_test_common.h`
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
