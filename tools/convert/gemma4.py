"""Gemma 4 (Gemma4ForConditionalGeneration) text model from a llama.cpp GGUF, kept in its blocks.

The QAT releases (`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`) store every matrix as ggml Q4_0, the
tied token table as Q6_K and every vector as F32. The recipe imports the blocks row for row: a row of
the artifact is byte for byte a row of the GGUF. Norm gammas are stored as Gemma 4 applies them
(`w`, not `1 + w`); the router's scale vector, the per-expert output scales and the layer scalars
stay F32 vectors that the runtime applies.

    python -m tools.convert --model HF_DIR --recipe gemma4_gguf \\
        --source gguf=gemma-4-26B_q4_0-it.gguf --out gemma-4-26B-A4B-it-qat.ninfer

`HF_DIR` holds the release's `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja` and `generation_config.json` (no weights are read from it).

`--components text,mtp --source mtp=ASSISTANT.gguf` adds the release's multi-token-prediction
assistant (Gemma4AssistantForCausalLM, llama.cpp architecture `gemma4-assistant`) as the `mtp`
component, its blocks imported the same way: a four-block drafter that reads the target's last
sliding-window and full-attention KV and its final hidden state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

from .gguf_blocks import block_source
from .methods import cast_direct, import_encoded
from .resources import token_domain
from .sources.gguf import GGUFFile
from .sources.logical import LogicalSource, array_source
from .ternary import rows

ARCHITECTURES = ("Gemma4ForConditionalGeneration", "Gemma4ForCausalLM")
TEXT_RESOURCES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)


def _int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: expected a positive integer")
    return value


def text_config(source: dict) -> dict:
    """The text component's config: Gemma 4's geometry, normalised for src/models/gemma4."""

    architectures = source.get("architectures")
    if not (isinstance(architectures, list) and architectures and architectures[0] in ARCHITECTURES):
        raise ValueError(f"unsupported Gemma 4 architecture {architectures!r}")
    raw = source.get("text_config", source)
    if raw.get("model_type") != "gemma4_text":
        raise ValueError("text.model_type must be gemma4_text")
    if raw.get("hidden_activation") != "gelu_pytorch_tanh":
        raise ValueError("Gemma 4 uses the tanh GELU")
    if raw.get("hidden_size_per_layer_input", 0) or raw.get("num_kv_shared_layers", 0):
        raise ValueError("per-layer embeddings and shared KV layers (E2B/E4B) are not implemented")
    if raw.get("attention_k_eq_v") is not True:
        raise ValueError("the implemented Gemma 4 shares K and V in its full-attention layers")
    out = {"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4_text"}
    for key in ("hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings",
                "num_attention_heads", "num_key_value_heads", "head_dim",
                "num_global_key_value_heads", "global_head_dim", "sliding_window",
                "intermediate_size"):
        out[key] = _int(raw.get(key), "text." + key)
    moe = bool(raw.get("enable_moe_block", False))
    out["enable_moe_block"] = moe
    if moe:
        out["num_experts"] = _int(raw.get("num_experts"), "text.num_experts")
        out["top_k_experts"] = _int(raw.get("top_k_experts"), "text.top_k_experts")
        out["moe_intermediate_size"] = _int(raw.get("moe_intermediate_size"),
                                            "text.moe_intermediate_size")
    layers = raw.get("layer_types")
    if (not isinstance(layers, list) or len(layers) != out["num_hidden_layers"]
            or any(k not in ("sliding_attention", "full_attention") for k in layers)):
        raise ValueError("text.layer_types must name every block")
    out["layer_types"] = list(layers)
    rope = raw.get("rope_parameters") or {}
    full, sliding = rope.get("full_attention", {}), rope.get("sliding_attention", {})
    if full.get("rope_type") != "proportional" or sliding.get("rope_type", "default") != "default":
        raise ValueError("Gemma 4 rotates full attention proportionally and sliding attention fully")
    out["rope_theta_full"] = float(full["rope_theta"])
    out["rope_theta_sliding"] = float(sliding["rope_theta"])
    out["partial_rotary_factor_full"] = float(full.get("partial_rotary_factor", 1.0))
    out["rms_norm_eps"] = float(raw.get("rms_norm_eps", 1e-6))
    out["final_logit_softcapping"] = float(raw.get("final_logit_softcapping") or 0.0)
    tied = raw.get("tie_word_embeddings", source.get("tie_word_embeddings", True))
    if tied is not True:
        raise ValueError("Gemma 4 ties its token table and output head")
    out["tie_word_embeddings"] = True
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in raw:
            out[key] = raw[key]
    return out


def _load_resources(root: Path, vocab: int, overrides: dict):
    references, payloads = {}, {}
    for role in TEXT_RESOURCES:
        path = Path(overrides[role]) if role in overrides else root / role
        data = path.read_bytes()
        if not data:
            raise ValueError(f"{path}: resource is empty")
        object_id = f"resource/text/{role}"
        references[role] = object_id
        payloads[object_id] = data
    tokenizer = json.loads(payloads["resource/text/tokenizer.json"])
    tokenizer_config = json.loads(payloads["resource/text/tokenizer_config.json"])
    count, special = token_domain(tokenizer, tokenizer_config, vocab)
    return {"text": references}, payloads, count, special


def _direct(gguf: GGUFFile, tensor: str, shape: tuple[int, ...]) -> LogicalSource:
    values = gguf.read_direct(tensor).astype(np.float32).reshape(shape)
    return array_source(torch.from_numpy(np.ascontiguousarray(values)), tensor)


def _matrix(gguf: GGUFFile, tensor: str, shape: tuple[int, int]) -> LogicalSource:
    info = gguf.info(tensor)
    if info.type_id not in GGUF_FORMATS_BY_TYPE:
        raise ValueError(f"{tensor}: expected ggml block rows, found {info.type_name}")
    stored = (int(np.prod(info.shape[:-1])), info.shape[-1])
    if stored != shape:
        raise ValueError(f"{tensor}: stored {info.shape}, expected rows x columns {shape}")
    return block_source(gguf, tensor, shape, rows())


def assistant_config(mtp: GGUFFile, target: dict) -> dict:
    """The `mtp` component's config from a gemma4-assistant GGUF, checked against the target whose
    KV and hidden state it reads."""

    kv = mtp.kv
    if kv.get("general.architecture") != "gemma4-assistant":
        raise ValueError("the MTP source must be a gemma4-assistant GGUF")
    a = "gemma4-assistant."
    layers = kv.get(a + "block_count")
    sliding = kv.get(a + "attention.sliding_window_pattern")
    if not isinstance(sliding, list) or len(sliding) != layers:
        raise ValueError("gemma4-assistant: sliding_window_pattern must name every block")
    heads_kv = kv.get(a + "attention.head_count_kv")
    if not isinstance(heads_kv, list) or len(heads_kv) != layers:
        raise ValueError("gemma4-assistant: head_count_kv must give every block")
    if kv.get(a + "attention.shared_kv_layers") != layers:
        raise ValueError("gemma4-assistant: every block must read the target's KV")
    vocab, hidden = mtp.info("token_embd.weight").shape
    out = {
        "architectures": ["Gemma4AssistantForCausalLM"],
        "hidden_size": int(kv[a + "embedding_length"]),
        "backbone_hidden_size": int(kv[a + "embedding_length_out"]),
        "intermediate_size": int(kv[a + "feed_forward_length"]),
        "num_hidden_layers": int(layers),
        "layer_types": ["sliding_attention" if s else "full_attention" for s in sliding],
        "num_attention_heads": int(kv[a + "attention.head_count"]),
        "num_key_value_heads": int(next(h for h, s in zip(heads_kv, sliding) if s)),
        "num_global_key_value_heads": int(next(h for h, s in zip(heads_kv, sliding) if not s)),
        "head_dim": int(kv[a + "attention.key_length_swa"]),
        "global_head_dim": int(kv[a + "attention.key_length"]),
        "sliding_window": int(kv[a + "attention.sliding_window"]),
        "rms_norm_eps": float(kv[a + "attention.layer_norm_rms_epsilon"]),
        "vocab_size": int(vocab),
    }
    if hidden != out["hidden_size"]:
        raise ValueError("gemma4-assistant: token table width differs from embedding_length")
    for key in ("vocab_size", "num_attention_heads", "num_key_value_heads",
                "num_global_key_value_heads", "head_dim", "global_head_dim", "sliding_window"):
        if out[key] != target[key]:
            raise ValueError(f"gemma4-assistant: {key} {out[key]} differs from the target's "
                             f"{target[key]}")
    if out["backbone_hidden_size"] != target["hidden_size"]:
        raise ValueError("gemma4-assistant: embedding_length_out must be the target hidden size")
    if not any(sliding) or all(sliding):
        raise ValueError("gemma4-assistant: needs sliding-window and full-attention blocks")
    return out


def build_model(base, gguf: GGUFFile, *, mtp: GGUFFile | None = None, resource_overrides=None):
    """The text component of a Gemma 4 GGUF, every parameter's source the GGUF, and with `mtp` the
    assistant's component from its own GGUF."""

    from .model import Model, Parameter

    config = text_config(base.config)
    h, vocab = config["hidden_size"], config["vocab_size"]
    references, payloads, count, special = _load_resources(
        base.root, vocab, dict(resource_overrides or {}))
    records = {"text": {"config": config, "resources": references["text"]}}
    if mtp is not None:
        records["mtp"] = {"config": assistant_config(mtp, config), "target": "text"}
    model = Model(records, resources=payloads, token_count=count, special_token_ids=special)

    def matrix(name, tensor, shape, inputs, source=gguf):
        model.add(Parameter(name, shape, _matrix(source, tensor, shape), inputs=tuple(inputs),
                            residency=name.split("/", 1)[0]))

    def vector(name, tensor, shape, source=gguf):
        model.add(Parameter(name, shape, _direct(source, tensor, shape), direct_format="fp32",
                            residency=name.split("/", 1)[0]))

    # The tied table: the head reads it as a projection, the embedding gathers its rows.
    matrix("text/output_head", "token_embd.weight", (vocab, h), ("text/final_hidden",))
    vector("text/final_norm", "output_norm.weight", (h,))
    for i, kind in enumerate(config["layer_types"]):
        g, p = f"blk.{i}.", f"text/layers/{i}/"
        full = kind == "full_attention"
        d = config["global_head_dim"] if full else config["head_dim"]
        kv = config["num_global_key_value_heads"] if full else config["num_key_value_heads"]
        q = config["num_attention_heads"] * d
        a = p + "attention/"
        matrix(a + "query", g + "attn_q.weight", (q, h), (p + "mixer_input",))
        matrix(a + "key", g + "attn_k.weight", (kv * d, h), (p + "mixer_input",))
        if not full:
            matrix(a + "value", g + "attn_v.weight", (kv * d, h), (p + "mixer_input",))
        matrix(a + "output", g + "attn_output.weight", (h, q), (p + "attention_output",))
        vector(a + "query_norm", g + "attn_q_norm.weight", (d,))
        vector(a + "key_norm", g + "attn_k_norm.weight", (d,))
        vector(p + "input_norm", g + "attn_norm.weight", (h,))
        vector(p + "post_attention_norm", g + "post_attention_norm.weight", (h,))
        vector(p + "ffn_norm", g + "ffn_norm.weight", (h,))
        vector(p + "post_ffn_norm", g + "post_ffw_norm.weight", (h,))
        vector(p + "layer_scalar", g + "layer_output_scale.weight", (1,))
        width = config["intermediate_size"]
        matrix(p + "mlp/gate", g + "ffn_gate.weight", (width, h), (p + "mlp_input",))
        matrix(p + "mlp/up", g + "ffn_up.weight", (width, h), (p + "mlp_input",))
        matrix(p + "mlp/down", g + "ffn_down.weight", (h, width), (p + "mlp_product",))
        if config["enable_moe_block"]:
            e, w = config["num_experts"], config["moe_intermediate_size"]
            m = p + "moe/"
            vector(p + "mlp/post_norm", g + "post_ffw_norm_1.weight", (h,))
            vector(m + "pre_norm", g + "pre_ffw_norm_2.weight", (h,))
            vector(m + "post_norm", g + "post_ffw_norm_2.weight", (h,))
            vector(m + "router", g + "ffn_gate_inp.weight", (e, h))
            vector(m + "router_scale", g + "ffn_gate_inp.scale", (h,))
            vector(m + "expert_scale", g + "ffn_down_exps.scale", (e,))
            matrix(m + "gate_up", g + "ffn_gate_up_exps.weight", (e * 2 * w, h), (m + "input",))
            matrix(m + "down", g + "ffn_down_exps.weight", (e * h, w), (m + "product",))
    if mtp is not None:
        assistant = records["mtp"]["config"]
        ah, b = assistant["hidden_size"], assistant["backbone_hidden_size"]
        # The tied table: the drafter's head reads it; its inputs are the target's embeddings.
        matrix("mtp/output_head", "token_embd.weight", (vocab, ah), ("mtp/final_hidden",), mtp)
        vector("mtp/final_norm", "output_norm.weight", (ah,), mtp)
        matrix("mtp/pre_projection", "nextn.pre_projection.weight", (ah, 2 * b),
               ("mtp/stem_input",), mtp)
        matrix("mtp/post_projection", "nextn.post_projection.weight", (b, ah),
               ("mtp/final_hidden",), mtp)
        for i, kind in enumerate(assistant["layer_types"]):
            g, p = f"blk.{i}.", f"mtp/layers/{i}/"
            d = assistant["global_head_dim" if kind == "full_attention" else "head_dim"]
            q = assistant["num_attention_heads"] * d
            a = p + "attention/"
            matrix(a + "query", g + "attn_q.weight", (q, ah), (p + "mixer_input",), mtp)
            matrix(a + "output", g + "attn_output.weight", (ah, q), (p + "attention_output",),
                   mtp)
            vector(a + "query_norm", g + "attn_q_norm.weight", (d,), mtp)
            vector(p + "input_norm", g + "attn_norm.weight", (ah,), mtp)
            vector(p + "post_attention_norm", g + "post_attention_norm.weight", (ah,), mtp)
            vector(p + "ffn_norm", g + "ffn_norm.weight", (ah,), mtp)
            vector(p + "post_ffn_norm", g + "post_ffw_norm.weight", (ah,), mtp)
            vector(p + "layer_scalar", g + "layer_output_scale.weight", (1,), mtp)
            width = assistant["intermediate_size"]
            matrix(p + "mlp/gate", g + "ffn_gate.weight", (width, ah), (p + "mlp_input",), mtp)
            matrix(p + "mlp/up", g + "ffn_up.weight", (width, ah), (p + "mlp_input",), mtp)
            matrix(p + "mlp/down", g + "ffn_down.weight", (ah, width), (p + "mlp_product",), mtp)
    return model


def gemma4_gguf(model, recipe, sources):
    """Every matrix keeps its GGUF blocks; vectors are FP32. Q, K and V of a layer share one parent
    when the GGUF stores them in one format (a mixed-format quantization such as Q4_K_M keeps V
    apart), and the dense FFN's gate and up rows share one."""

    if model.config.get("architectures") != ["Gemma4ForCausalLM"]:
        raise ValueError("the gemma4_gguf recipe requires a Gemma 4 model")
    formats = {}
    for name, parameter in model.parameters.items():
        if parameter.projection:
            fmt = formats[name] = parameter.source.read_encoded(0, 1).format
            recipe.assign(name, format=fmt, method=import_encoded, source=parameter.source,
                          activation_policy="AllowA8")
        else:
            recipe.assign(name, format="fp32", method=cast_direct, source=parameter.source)
    for i, kind in enumerate(model.config["layer_types"]):
        a = f"text/layers/{i}/attention/"
        group = [a + "query", a + "key"] + ([] if kind == "full_attention" else [a + "value"])
        if len({formats[name] for name in group}) == 1:
            recipe.group(group)
    mlps = [f"text/layers/{i}/mlp/" for i in range(len(model.config["layer_types"]))]
    if "mtp" in model.components:
        mlps += [f"mtp/layers/{i}/mlp/"
                 for i in range(model.components["mtp"]["config"]["num_hidden_layers"])]
    for p in mlps:
        if formats[p + "gate"] != formats[p + "up"]:
            raise ValueError(f"{p}: gate and up are stored in different formats")
        recipe.group([p + "gate", p + "up"])


RECIPES = {"gemma4_gguf": gemma4_gguf}
