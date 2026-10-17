# W4(NVFP4) A16(BF16) support for sm_86/89/100/120 and A8(int8) support for sm_70/75/86/89/100/120 on ninfer, delivered on the Qwen3.8 27B geometry

Upstream instantiates the fused NVFP4 SwiGLU route only to `T <= 16`, and resolves the A16 domain  
as `T <= 16 -> SmallTFusedA16`, `T <= 64 -> Materialized`, `T > 64 -> throw`, so a long prompt does  
not prefill slowly — it throws. This fork adds the missing wide-T kernel, then makes the route  
faster: **on the RTX 3090 Ti, at T1, decode is about 3.0x upstream's 16 tok/s** (50 against 16.1, same prompt basis),  
and prefill stops throwing — 1260 tok/s on the 4202-token prompt, about 80% of this card's dense  
BF16 tensor peak and close to the ceiling for a dequantising kernel with no native FP4 path, against  
the roughly 780 tok/s the first graft of the wide-T kernel managed.

Four changes, in order:

1. **A wide-T fused SwiGLU kernel**, so a wide span prefills instead of throwing.
2. **Kernel work across the route** — the wide-T MMA, the sliced-K MMA carrying verify widths, the  
   GEMV carrying single-token decode.
3. **The NVFP4 fused-route switches wired into `ninfer-calibrate`**, so legacy/unified selection is  
   measured on the machine rather than inherited from another card, and cached between starts.
4. Alongside them, **KV-cache performance for NVFP4 and k8v4** was improved as well.

## Throughput

Same artifact throughout (`Qwen3.8-27B-QUASAR-NVFP4-DFlash2`, proposal head) on an RTX 3090 Ti: greedy decoding, KV  
bf16, 32,768-token context. Prefill is measured on the 4K long prompt; decode is the mean over  
several realistic tasks.

| Build and speculative setting                 | Prefill tok/s | Decode tok/s | Acceptance | Tokens/round |
| --------------------------------------------- | ------------- | ------------ | ---------- | ------------ |
| Upstream as shipped, before the wide-T route  | throws        | throws       | —          | —            |
| Wide-T SwiGLU grafted, before any kernel work | 780           | 16           | —          | —            |
| This build, no speculation (T1)               | 1260          | 50           | —          | —            |
| This build, MTP K=3                           | 1250          | 93           | 59.7%      | 2.79         |
| This build, DFlash2 K=5                       | 1260          | 109          | 48.8%      | 3.42         |

One thing about the table: **upstream throws**, hence the empty row. The fused NVFP4 SwiGLU route is  
instantiated only to `T <= 16` and there is no wide fallback behind it — the kernel is absent from  
the build and there is no partial state to patch around, so that prompt never reaches decode.

## Next steps

W4A8 — int8 activations over the same NVFP4 weights — is still working on it. Several kernels still  
have only a W4A16 path, so it cannot run on sm_70/75, which have no BF16. On GPUs that support both  
BF16 and int8 (sm_86/89/100/120) it can be turned on with `NINFER_NVFP4_GEMM = grouped` or `mmq`; it  
is not yet well optimised and is only slightly faster than W4A16.

---

## 中文

上游只把 fused NVFP4 SwiGLU 通路实例化到 `T <= 16`，A16 域的解析是 `T <= 16 -> SmallTFusedA16`、`T <= 64 -> Materialized`、`T > 64 -> throw`：所以长 prompt 不是"prefill 很慢"，而是**直接抛异常**。本 fork 补上缺失的宽 T 内核，然后把整条通路做快：在 RTX 3090 Ti 上 **T1 的 decode 大约是上游 16 tok/s 的 3.0 倍**（50 对 16.1，同一 prompt 基准）；prefill 不再抛异常，同一段 4202 token prompt 上跑到 1260 tok/s，浮点利用率80%，接近无原生 FP4 通路的反量化内核理论上限，而最初把宽 T 内核接上去时只有约 780 tok/s。

按时间顺序，一共四处改动：

1. **新增宽 T fused SwiGLU 内核**，让宽的 prompt 段落能 prefill 而不是抛异常。
2. **整条通路的 kernel 性能工作** —— 宽 T MMA、承载 verify 宽度的 sliced-K MMA、承载单 token  
   decode 的 GEMV。
3. **把 NVFP4 fused 通路的开关接进 `ninfer-calibrate`**，legacy/unified 的选择从此在本机实测得出，  
   而不是从另一张卡照抄，并在启动之间缓存。
4. 顺带优化了KV NVFP4 和 k8v4 的性能。

### 吞吐

全程同一个制品（`Qwen3.8-27B-QUASAR-NVFP4-DFlash2`，带 proposal 头），RTX 3090 Ti，贪心解码，KV bf16，32,768 token 上下文，prefill 由 4K 的长 prompt 测出，decode 速度来自于多个真实场景下测得的均值。

| 构建与投机解码设置                    | Prefill tok/s | Decode tok/s | 接受率   | 每轮 token |
| ---------------------------- | ------------- | ------------ | ----- | -------- |
| 上游原样（宽 T 路由之前）               | throws        | throws       | —     | —        |
| 接上宽 T SwiGLU、尚未做任何 kernel 优化 | 780           | 16           | —     | —        |
| 本构建，无投机（T1）                  | 1260          | 50           | —     | —        |
| 本构建，MTP K=3                  | 1250          | 93           | 59.7% | 2.79     |
| 本构建，DFlash2 K=5              | 1260          | 109          | 48.8% | 3.42     |

这张表有一点要说明。**上游会抛异常**，所以那一行是空的：fused NVFP4 SwiGLU 通路只实例化到 `T <= 16`，后面没有宽内核兜底，这里没有"局部状态"可绕 —— 内核在构建里根本不存在，prompt 走不到 decode。

### 下一步

W4A8 —— 同一套 NVFP4 权重上跑 int8 激活 —— 仍在进行中，部分内核仍只有 W4A16 路径，所以在没有 BF16 的 sm_70/75 上不能运行，对于支持 BF16 和 int8 的 GPUs（sm_86/89/100/120），可以通过环境变量 NINFER_NVFP4_GEMM = grouped 或 mmq 开启，尚未经过充分优化，速度仅较 W4A16 稍快。
