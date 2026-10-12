# NInfer ggml block-quant sources

These files are copied from [llama.cpp](https://github.com/ggml-org/llama.cpp),
commit [`a25c9865fe03c954c93fd755b5d79ae86ba99750`](https://github.com/ggml-org/llama.cpp/commit/a25c9865fe03c954c93fd755b5d79ae86ba99750),
`ggml/include/` and `ggml/src/`, under the MIT license; see [LICENSE](LICENSE). One local change,
marked `NInfer:` in the source: Q4_0's DP4A matrix product (`mmq-vec-dot.cuh`, the route of GPUs
without integer tensor cores such as Tesla V100) takes its -8 offset term from the quantized
activation instead of the block's FP16 sum, which overflows once 32 activations sum past 65504.

NInfer compiles ggml's integer tensor-core matrix multiplication (`ggml-cuda/mmq.cuh`) and its
dequantizers through the bridge in `src/ops/linear/gguf/`, as a separate archive, and keeps only the
headers that code includes. Products of up to eight tokens use NInfer's own kernel
(`ggml_bridge_vec.cuh`), which reads the grids and block structs from `ggml-common.h`.

The flash-attention kernel (`ggml-cuda/fattn-mma-f16.cuh` with `fattn-common.cuh` and
`cp-async.cuh`, same commit, unmodified) runs Gemma 4's prompt attention (head widths 256 and 512)
through `src/ops/linear/gguf/ggml_fattn.cu`, which launches it without ggml's backend context.
