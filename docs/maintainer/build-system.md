# Build system

NInfer builds from its source tree with CMake 3.28+ and C++/CUDA 20.
`CMAKE_CUDA_ARCHITECTURES` is `70`, `75`, `80`, `86` (the default), `89` or `120a`, one per build
tree; configure refuses anything else. CUDA 12.8 builds `75` and the `sm_8x` targets, a `70` build
needs CUDA 12.x ([Tesla V100](../build-linux.md#tesla-v100-sm_70)), and a `120a` build needs CUDA
13.1 or newer.

## Turing tier (`75`)

A `75` build (TITAN RTX, RTX 20 series) compiles the `mma.sync` compatibility path with
`NINFER_SM75=1` defined for host and device code. Turing lacks instructions the path is written
with, and the device helpers replace them without changing any Op's arithmetic contract (BF16-cache
attention, below, is the one exception):

- `cp_async*` in `src/ops/common/memory.cuh` copy synchronously through registers with the same
  cache policy and zero-fill contract; `cp_commit`/`cp_wait` are empty.
- `src/ops/common/mma.cuh` runs FP16 m16n8k16 as two m16n8k8 and INT8 m16n8k32 as four m8n8k16.
  BF16 and TF32 products run on FP16 m16n8k8 with FP32 accumulation, and in FP32 from the same
  operands where an operand is not exactly an FP16 value; no value is narrowed. The choice is made
  per output element, from its own row of A and column of B, so a result does not depend on which
  rows and columns happen to share its tile -- the batch-invariant replay contracts rely on that.
  `ninfer_mma_fragments_test` qualifies every helper against an FP64 oracle through the PTX
  fragment layouts, including that independence.
- Packed conversions (`cvt.rn.satfinite.f16x2`, `cvt.rna.tf32`, `cvt.rn.bf16x2`) and
  `__reduce_max_sync` have scalar or shuffle forms that give the same bits.
- A `70` build emulates more (no ldmatrix, no m16n8 shapes) in `ops/common/sm70_emulation.cuh`,
  whose `NINFER_SM70_DEVICE` stops below `sm_75`: Turing keeps the native ldmatrix and FP16 MMA and
  shares only Volta's scalar forms where both lack an instruction (the QSA indexer's FP32 dots, the
  GDN conversions).

`ops/common/device_limits.h` holds the 64 KiB per-block shared-memory limit kernel shapes assert
against, and `pipeline_stages()` caps cp.async rings at two stages; the rowsplit A8 GEMM runs one,
since a synchronous copy has nothing to overlap. Kernels that cannot fit take Turing shapes: the
BF16 prompt attention stages K and V through one buffer, the INT8-family prompt attention always
runs the fast kernel with one K/V stage (the standard INT8 prompt kernel is not built), small-T
INT8 attention has no 64-key arena tier, context-KV materialization uses 64-key blocks, and the
cooperative GDN gating projection launches the CTAs per SM a Turing SM can hold. The FP8, NVFP4
and K8V4 KV kernels widen K and V into one shared tile, V overwriting K once QK is done, wherever
two tiles would pass 64 KiB: the FP8 prompt kernel and the small-T shapes of two and three 16-row
tiles (`kCausalSmallTCompactKV`).
BF16-cache attention would spend most of its time in the BF16 emulation, so a `75` build rounds
the softmax probabilities to FP16, as the INT8-family kernels do on every architecture, and
multiplies Q, K and V as their FP16 roundings: values from 2^-17 to below 2^16 unchanged, a smaller
one moved by at most 2^-25 (`ops/softmax_attention/common/bf16_attention.cuh`). Tiles stage as FP16
and run on FP16 MMA; a tile with a value of 2^16 or more takes an emulation that forms the same
products for every other operand, so an output still depends on its own row and column alone.
The prompt and small-T BF16 kernels have Turing versions of their own (`prompt_bf16_sm75.cuh`,
`small_t_bf16_sm75.cuh`): a CTA sweeps its keys on FP16 MMA and, only when a tile does not fit
FP16, sweeps them again in an out-of-line function on the emulation; in one loop body with the fast
path, the emulation's code made it spill and slowed it several times over. The small-T kernel keeps
its probabilities in registers and its page IDs in the block table: 32 KiB per CTA, two CTAs per
SM.
A `75` build selects the legacy Linear route table only
(`kUnifiedLinearRoutes`); the unified templates compile but are never chosen, and their
calibration keys are absent from the route catalog.
The Q8 G32 A16 K-split contraction (`ops/linear/q8/q8_ksplit_mma.cuh`) forms its codes as FP16
pairs and issues FP16 MMA directly on tiles whose live activations are FP16 values, the operands
and order the emulation would use, so its outputs are the same bits at a fraction of the cost. It
loads the next K group's codes and scales, and an activation tile of up to eight columns, into
registers during the current group's MMAs, and `Q8KSplitDeviceSchedule`
(`ops/linear/q8/q8_ksplit_config.h`) runs the 8-column tiles of matrices with at least 6144 rows
with four K warps ([TITAN RTX measurements](../performance/titan-rtx.md)).
Product commands and prerequisites are in the README's [Running](../../README.md#running) and
[Building](../../README.md#building) sections; test and measurement workflows live in
[tests](../../tests/README.md) and [benchmarks](../../bench/README.md).

## Configuration

The root `CMakeLists.txt` owns language/toolchain constraints, build options and top-level
composition. Architecture selection and validation happen before `project()` detects CUDA.
The default configuration is Release. Ninja links and archives share the single-slot
`ninfer_link` pool; compilation uses the build command's parallelism.

| Cache option | Default | Scope |
|---|---|---|
| `NINFER_BUILD_APPS` | ON | CLI, HTTP server, perplexity evaluator and `ninfer-calibrate` |
| `BUILD_TESTING` | OFF | C++ tests and registered Python interoperability tests |
| `NINFER_BUILD_BENCHMARKS` | OFF | Op, model, Engine and context-cost benchmarks |

The opt-in kernel, build and platform options (`NINFER_SM120_NATIVE`, `NINFER_PDL`,
`NINFER_MULTICALL`, the SFU and Windows options, and the rest) are listed in the
[Linux build guide](../build-linux.md#build-options).

Apps or tests enable the internal product support components: media acquisition, prompt input,
logging and serving. This is one derived condition, not a separate user option. FFmpeg belongs
to model media decoding and remains required when apps are disabled. Curl and spdlog are needed
only for product support. A Python 3 interpreter is found when tests are configured. The larger Python
pytest suites and conversion/evaluation scripts run separately from CMake and CTest.

Python 3.11 describes the maintained environment, not a configuration version gate. The one CUDA
version gate is 13.1 for a `120a` build: CUDA 12.8 and 12.9 miscompile `sm_120a` kernels.
CMake also discovers FFmpeg without imposing library version floors. Actual language/API support
is exercised by compilation and tests. The libcurl 7.85 minimum has a concrete API basis:
media acquisition uses `CURLOPT_PROTOCOLS_STR` and `CURLOPT_REDIR_PROTOCOLS_STR`, introduced in
[that release](https://curl.se/libcurl/c/CURLOPT_PROTOCOLS_STR.html).

The checked-in `CMakePresets.json` provides:

- `release`: Release product binaries, tests and benchmarks disabled;
- `dev`: Release products, tests and benchmarks enabled;
- matching build presets and a `dev` test preset with failure output enabled.

Both configure presets use `build/`. Switching between them explicitly resets all three build
options; it does not create independent build trees. Build with `cmake --build build -j`.
For a separate configuration, override the binary directory with `-B build-<name>` and use that
directory in subsequent build/test commands.

Keep machine paths in the ignored `CMakeUserPresets.json`. For example, replacing the interpreter
placeholder with the interpreter from the selected environment:

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "local",
      "inherits": "dev",
      "cacheVariables": {
        "Python3_EXECUTABLE": "/path/to/python3"
      }
    }
  ]
}
```

Configure that environment with `cmake --preset local`. Compiler paths may also be supplied here;
changing compilers requires a fresh build directory. There is no tools option, installed SDK,
package export or configure-time dependency download.

## Targets and dependencies

`cmake/Dependencies.cmake` discovers system CUDA, Threads, FFmpeg and conditional curl, and
exposes the repository-pinned JSON/HTTP headers and conditional spdlog library. External include
requirements follow their consuming targets. `cmake/NinferTargets.cmake` provides private
project includes and the two CUDA archive policies.

`src/CMakeLists.txt` explicitly enters the component directories. Each component's
`CMakeLists.txt` owns its targets, dependency visibility and compile properties:

| Component | Targets |
|---|---|
| `core/` | `ninfer_core` |
| `artifact/` | `ninfer_artifact` |
| `text/` | `ninfer_text`, including bundled utf8proc C source |
| `media/` | `ninfer_media_decode` |
| `ops/` | `ninfer_ops`, `ninfer_nvfp4_non_rdc` |
| `models/` | `ninfer_model_loading`, `ninfer_model_runtime` |
| `runtime/` | `ninfer_runtime_support`, `ninfer_engine` and its `ninfer::engine` alias |
| `product/` | media acquisition, prompt input and logging libraries |
| `serve/` | `ninfer_serve` |

Component declaration order is not the execution dependency graph: CMake resolves named target
links during generation. Model runtime can refer to the runtime support target declared later;
support links core, while Engine links model runtime. The ownership and dependency contracts are
defined by [Engine architecture](engine-architecture.md) and [Op development](op-development.md).

## Explicit source ownership

Small components list their sources directly. Larger components include local `sources.cmake`
manifests which use `target_sources(<owner> PRIVATE ...)`. Included manifests anchor file paths
with `CMAKE_CURRENT_LIST_DIR`; `include()` does not change `CMAKE_CURRENT_SOURCE_DIR`.

- Every implementation under `src/` has one target owner and one explicit registration.
- Ops register simple horizontal implementations in `basic_sources.cmake` and larger semantic
  families in their own manifests. A family also registers its files under `wrapper/`.
- Linear includes one manifest for each of BF16, FP8, NVFP4, Q4, Q5, Q6 and Q8. Each manifest lists
  all its shape filenames and their actual `.cpp`/`.cu` extensions.
- Models separate loading, frontend, execution and Program source lists while retaining the
  distinct immutable-loading and mutable-runtime targets. Checkpoints/recipes do not define
  additional compile owners.
- New files require an explicit manifest edit. New families require an explicit parent include.
  There is no recursive glob, directory discovery or generated shape/dispatch registry.

Adding a source manifest does not require another library. Add a target only for a meaningful
dependency, reuse or compilation boundary. Shape semantics, supported domains and finite
dispatch remain in C++; CMake only selects translation units.

## CUDA compilation boundaries

`ninfer_core` and `ninfer_ops` compile whole-program: no relocatable device code (RDC) and no
device link. Every device symbol is defined and used in one translation unit. A change that needs a
device symbol across translation units has to bring RDC back to its target and measure the cost:
under RDC ptxas lowers a shuffle from a computed lane to an out-of-line call and stops pipelining
loads across loop back edges.

`ninfer_nvfp4_non_rdc` explicitly disables separable compilation and device-symbol resolution. It
contains the warp-specialized Linear, LinearSwiGLU and causal-attention NVFP4 sources that depend
on `setmaxnreg` register transfer, which RDC would discard. Their owning family manifests register
these sources into that target; host launchers connect them to the normal Ops. All three CUDA
archive targets retain `-lineinfo`.

Whole-program compilation is not a speed lever on every card. On the RTX 3090 (sm_86, CUDA 12.8, 27B
DFlash2 bundle, rk4v4 KV, three interleaved rounds) the base line measured the switch at
pp512/pp2048/pp8192 +0.4/-0.6/-1.4% and decode within +-0.3%, all inside run-to-run spread, while
the server binary grows from 498 to 637 MB; the sm_89 fork reported +2.7-4.9% prefill from it. What
it buys everywhere is the code generation described above; re-measure speed before attributing a
gain to it on a new architecture or toolchain.

Source-list maintenance must preserve language, architecture, RDC mode, device-link ownership
and numerical compiler options. Splitting a manifest does not reduce kernel instantiation work
or justify combining shape translation units or enabling unity builds.

## Consumers and verification

`apps/CMakeLists.txt` explicitly defines the four product executables, or with `NINFER_MULTICALL`
one multi-call executable that bundles them. Tests and benchmarks
include their domain registrations without creating new CMake subdirectories, preserving
`build/tests/`, `build/bench/` and CTest working directories. The helpers are local to those
consumers. Tests explicitly name their linked libraries; public-header and pure-host checks keep
their isolated include/link requirements.

Linear, LinearAdd and LinearSwiGLU reuse compiled test support libraries. Oracle options apply
to those libraries themselves as well as the test executables. Test variants, skip return codes,
interop commands and special linker options belong with the affected registrations.

For build-organization changes, compare configured target/source ownership, compile flags,
device links and CTest commands/properties; check the relevant option combinations, build the
affected targets and run their behavioral tests. A shape-source edit should rebuild its own
translation unit plus necessary downstream links. Build-time or inference-speed claims require
measurements at that scope.
