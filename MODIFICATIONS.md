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
| Base commit | `a53a4f3342c06f7977a045d54b372a7c1253afb3` (upstream master) |
| Upstream sync | `f2675499` merged upstream `63428ac2`; `63b68fdc` merged `b3402a6f`; `121f8283` merged `a53a4f33`. The fork was cut at `e1f2d961` and has since taken `9456cb7a`, `230ef895`, `63428ac2`, `6826ac0e`, `d3b2475c`, `63b2b236`, `b3402a6f`, `7cf5b2ae`, `9d99efe7`, `710df200`, `8508dfd6`, `9b60d333`, `7c80f155`, `3f2b59ca`, `6ab3e660`, `e58e7279` and `a53a4f33` |
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

`03b78d67` no longer shows in the file lists below: upstream `7cf5b2ae` fixed
the same defect and the merge `121f8283` took upstream's side, so the fork keeps
no change to that file.

The two nvfp4 commits above are master's own history. The branches carried for
review against upstream are rebased equivalents of them on top of the current
base: `pr/nvfp4-w4a16` = `b76d912e` and `pr/nvfp4-w4a8-int` = `5dc9e11d`.
Neither rebase changed a patch-id: `b76d912e` still hashes to the same patch-id
as `7b084383`, and `5dc9e11d` to the same one as `05c47ff8`, which itself differs
from `c45859ff` only in the three `ffn()` files named below, and there by the
same bytes master already carried.

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

The lists below are generated from `git diff --name-status upstream/master..HEAD`
with `upstream/master` at `a53a4f33`. Files marked as modified are upstream
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
