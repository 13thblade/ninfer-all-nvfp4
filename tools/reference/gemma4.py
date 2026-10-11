"""FP64 reference forward of Gemma 4 26B-A4B (Gemma4ForCausalLM, text) over a llama.cpp GGUF.

An independent transcription of the text model in transformers' modeling_gemma4.py for golden
values: the embedding scaled by sqrt(H); per block the RMS-normed attention (sliding-window blocks
with their own heads, full-attention blocks whose keys double as values after an unweighted RMS
norm, proportional RoPE), the dense GeGLU MLP beside the routed experts (softmax over all experts,
top-k renormalized, per-expert output scale), the sandwich norms and the layer scalar; then the final
norm, the tied head and the logit soft cap. Every value is FP64 over the GGUF's exactly decoded
blocks, so the only difference from a real engine is the engine's own arithmetic.

    python -m tools.reference.gemma4 --gguf gemma-4-26B_q4_0-it.gguf --tokens ids.json \\
        [--layers 30] [--out reference.npz]

`ids.json` is the token ids to run (a JSON list, <bos> first). It prints the NLL of every target
(token i + 1 given tokens 0..i) and their total; `--out` keeps every block's output, the routed
experts and the logits. 26 tokens take about three minutes on CPU, mostly decoding expert blocks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tools.convert.sources.gguf import TYPE_BF16, TYPE_F16, TYPE_F32, GGUFFile

DIRECT_TYPES = (TYPE_F32, TYPE_F16, TYPE_BF16)


def _dequantize(gguf: GGUFFile, name: str, row_begin: int = 0, row_end: int | None = None):
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize

    blocks = gguf.read_blocks(name, row_begin, row_end)
    kind = GGMLQuantizationType(gguf.info(name).type_id)
    values = dequantize(np.ascontiguousarray(blocks), kind)
    return np.asarray(values, dtype=np.float64).reshape(blocks.shape[0], -1)


class Weights:
    def __init__(self, gguf: GGUFFile):
        self.gguf = gguf

    def has(self, name: str) -> bool:
        return name in self.gguf.tensors

    def matrix(self, name: str, row_begin: int = 0, row_end: int | None = None) -> np.ndarray:
        return _dequantize(self.gguf, name, row_begin, row_end)

    def vector(self, name: str) -> np.ndarray:
        return self.gguf.read_direct(name).astype(np.float64).reshape(-1)

    def rows(self, name: str) -> np.ndarray:
        """A whole matrix, stored as float words or as ggml blocks."""
        if self.gguf.info(name).type_id in DIRECT_TYPES:
            info = self.gguf.info(name)
            return self.gguf.read_direct(name).astype(np.float64).reshape(info.shape[0], -1)
        return self.matrix(name)


def rms(x: np.ndarray, eps: float, w: np.ndarray | None = None) -> np.ndarray:
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y * w if w is not None else y


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def forward(gguf: GGUFFile, ids: list[int], layers: int | None = None) -> dict[str, np.ndarray]:
    kv = gguf.kv
    w = Weights(gguf)
    hidden = int(kv["gemma4.embedding_length"])
    heads = int(kv["gemma4.attention.head_count"])
    kv_heads = kv["gemma4.attention.head_count_kv"]
    sliding = kv["gemma4.attention.sliding_window_pattern"]
    window = int(kv["gemma4.attention.sliding_window"])
    eps = float(kv["gemma4.attention.layer_norm_rms_epsilon"])
    top_k = int(kv["gemma4.expert_used_count"])
    width = int(kv["gemma4.expert_feed_forward_length"])
    cap = float(kv.get("gemma4.final_logit_softcapping", 0.0))
    blocks = int(kv["gemma4.block_count"]) if layers is None else layers
    n = len(ids)
    pos = np.arange(n)

    table = "token_embd.weight"
    x = np.stack([w.matrix(table, t, t + 1)[0] for t in ids]) * math.sqrt(hidden)
    out: dict[str, np.ndarray] = {"inp_scaled": x.copy()}
    for il in range(blocks):
        p = f"blk.{il}."
        full = not bool(sliding[il])
        d = int(kv["gemma4.attention.key_length" if full else "gemma4.attention.key_length_swa"])
        hkv = int(kv_heads[il] if isinstance(kv_heads, list) else kv_heads)
        half = d // 2
        a = rms(x, eps, w.vector(p + "attn_norm.weight"))
        q = (a @ w.matrix(p + "attn_q.weight").T).reshape(n, heads, d)
        k_raw = (a @ w.matrix(p + "attn_k.weight").T).reshape(n, hkv, d)
        v_raw = k_raw if full else (a @ w.matrix(p + "attn_v.weight").T).reshape(n, hkv, d)
        q = rms(q, eps, w.vector(p + "attn_q_norm.weight"))
        k = rms(k_raw, eps, w.vector(p + "attn_k_norm.weight"))
        v = rms(v_raw, eps)
        if full:
            base = float(kv["gemma4.rope.freq_base"])
            inv = base ** (-2.0 * np.arange(half) / d)
            if w.has("rope_freqs.weight"):
                inv = inv / w.vector("rope_freqs.weight")[:half]
            elif w.has(p + "rope_freqs.weight"):
                inv = inv / w.vector(p + "rope_freqs.weight")[:half]
        else:
            base = float(kv["gemma4.rope.freq_base_swa"])
            inv = base ** (-2.0 * np.arange(half) / d)
        angle = pos[:, None] * inv[None, :]
        cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]

        def rope(t: np.ndarray) -> np.ndarray:
            t1, t2 = t[..., :half], t[..., half:]
            return np.concatenate([t1 * cos - t2 * sin, t2 * cos + t1 * sin], -1)

        q, k = rope(q), rope(k)
        group = heads // hkv
        mask = np.tril(np.ones((n, n), dtype=bool))
        if not full:
            mask &= (pos[:, None] - pos[None, :]) < window
        attn = np.zeros((n, heads, d))
        for h in range(heads):
            s = q[:, h, :] @ k[:, h // group, :].T
            s = np.where(mask, s, -np.inf)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            attn[:, h, :] = s @ v[:, h // group, :]
        o = attn.reshape(n, heads * d) @ w.matrix(p + "attn_output.weight").T
        x = x + rms(o, eps, w.vector(p + "post_attention_norm.weight"))
        out[f"attn_out-{il}"] = x.copy()

        m = rms(x, eps, w.vector(p + "ffn_norm.weight"))
        gate = m @ w.matrix(p + "ffn_gate.weight").T
        up = m @ w.matrix(p + "ffn_up.weight").T
        dense = (gelu_tanh(gate) * up) @ w.matrix(p + "ffn_down.weight").T
        dense = rms(dense, eps, w.vector(p + "post_ffw_norm_1.weight"))

        router_in = rms(x, eps) / math.sqrt(hidden) * w.vector(p + "ffn_gate_inp.scale")
        logits = router_in @ w.rows(p + "ffn_gate_inp.weight").T
        prob = np.exp(logits - logits.max(-1, keepdims=True))
        prob /= prob.sum(-1, keepdims=True)
        chosen = np.argsort(-prob, axis=-1, kind="stable")[:, :top_k]
        scale = (
            w.vector(p + "ffn_down_exps.scale")
            if w.has(p + "ffn_down_exps.scale")
            else np.ones(prob.shape[1])
        )
        expert_in = rms(x, eps, w.vector(p + "pre_ffw_norm_2.weight"))
        moe = np.zeros_like(x)
        cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for t in range(n):
            weights = prob[t, chosen[t]]
            weights = weights / weights.sum()
            for j, e in enumerate(chosen[t]):
                e = int(e)
                if e not in cache:
                    cache[e] = (
                        w.matrix(p + "ffn_gate_up_exps.weight", e * 2 * width, (e + 1) * 2 * width),
                        w.matrix(p + "ffn_down_exps.weight", e * hidden, (e + 1) * hidden),
                    )
                gu = cache[e][0] @ expert_in[t]
                middle = gelu_tanh(gu[:width]) * gu[width:]
                moe[t] += weights[j] * (cache[e][1] @ middle) * scale[e]
        out[f"moe_ids-{il}"] = chosen
        out[f"moe_out-{il}"] = moe.copy()
        moe = rms(moe, eps, w.vector(p + "post_ffw_norm_2.weight"))
        layer_scale = (
            w.vector(p + "layer_output_scale.weight")
            if w.has(p + "layer_output_scale.weight")
            else np.ones(1)
        )
        x = (x + rms(dense + moe, eps, w.vector(p + "post_ffw_norm.weight"))) * layer_scale
        out[f"l_out-{il}"] = x.copy()

    if blocks == int(kv["gemma4.block_count"]):
        final = rms(x, eps, w.vector("output_norm.weight"))
        vocab = gguf.info(table).shape[0]
        logits = np.zeros((n, vocab))
        for first in range(0, vocab, 16384):
            last = min(vocab, first + 16384)
            logits[:, first:last] = final @ w.matrix(table, first, last).T
        if cap > 0:
            logits = cap * np.tanh(logits / cap)
        out["logits"] = logits
    return out


def target_nll(logits: np.ndarray, ids: list[int]) -> np.ndarray:
    top = logits.max(-1, keepdims=True)
    lse = top[:, 0] + np.log(np.exp(logits - top).sum(-1))
    n = len(ids)
    return lse[: n - 1] - logits[np.arange(n - 1), ids[1:]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path, help="JSON list of token ids")
    parser.add_argument("--layers", type=int, default=None, help="run only the first N blocks")
    parser.add_argument("--out", type=Path, default=None, help="write the tensors to this .npz")
    args = parser.parse_args()
    ids = [int(t) for t in json.loads(args.tokens.read_text())]
    gguf = GGUFFile(args.gguf)
    result = forward(gguf, ids, args.layers)
    if "logits" in result:
        nll = target_nll(result["logits"], ids)
        print(json.dumps({"tokens": len(ids), "total_nll": float(nll.sum()),
                          "nll": [round(float(v), 4) for v in nll]}))
    if args.out is not None:
        np.savez(args.out, **result)


if __name__ == "__main__":
    main()
