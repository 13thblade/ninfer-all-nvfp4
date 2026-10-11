from __future__ import annotations

import json

import numpy as np
import pytest

from tools.convert import gemma4
from tools.convert.sources.gguf import TYPE_F32, GGUFFile, write_gguf

TYPE_Q4_0 = 2
TYPE_Q8_0 = 8
TYPE_Q6_K = 14

HIDDEN, VOCAB, HEADS, HEAD_DIM, GLOBAL_DIM = 256, 512, 2, 32, 64
WIDTH, EXPERTS, EXPERT_WIDTH = 64, 4, 32


def _hf_config(**overrides):
    text = dict(
        model_type="gemma4_text", hidden_activation="gelu_pytorch_tanh", attention_k_eq_v=True,
        hidden_size=HIDDEN, vocab_size=VOCAB, num_hidden_layers=2, max_position_embeddings=4096,
        num_attention_heads=HEADS, num_key_value_heads=1, head_dim=HEAD_DIM,
        num_global_key_value_heads=1, global_head_dim=GLOBAL_DIM, sliding_window=128,
        intermediate_size=WIDTH, enable_moe_block=True, num_experts=EXPERTS, top_k_experts=2,
        moe_intermediate_size=EXPERT_WIDTH,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "full_attention": {"rope_type": "proportional", "rope_theta": 1e6,
                               "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
        },
        rms_norm_eps=1e-6, final_logit_softcapping=30.0, tie_word_embeddings=True,
    )
    text.update(overrides)
    return {"architectures": ["Gemma4ForConditionalGeneration"], "text_config": text}


def _q4_0(generator, rows, columns):
    blocks = generator.integers(0, 256, size=(rows, columns // 32, 18), dtype=np.uint8)
    scales = generator.uniform(0.01, 0.02, size=(rows, columns // 32)).astype(np.float16)
    blocks[:, :, :2] = scales.view(np.uint8).reshape(rows, columns // 32, 2)
    return blocks.reshape(rows, -1)


def _q6_k(generator, rows, columns):
    blocks = generator.integers(0, 256, size=(rows, columns // 256, 210), dtype=np.uint8)
    scales = generator.uniform(0.01, 0.02, size=(rows, columns // 256)).astype(np.float16)
    blocks[:, :, 208:] = scales.view(np.uint8).reshape(rows, columns // 256, 2)
    return blocks.reshape(rows, -1)


def _q8_0(generator, rows, columns):
    blocks = generator.integers(0, 256, size=(rows, columns // 32, 34), dtype=np.uint8)
    scales = generator.uniform(0.01, 0.02, size=(rows, columns // 32)).astype(np.float16)
    blocks[:, :, :2] = scales.view(np.uint8).reshape(rows, columns // 32, 2)
    return blocks.reshape(rows, -1)


def _release(tmp_path, value_q8=False):
    """A two-block Gemma 4 release: the HF text files and a GGUF in llama.cpp's naming. With
    `value_q8` the sliding block's V is Q8_0 beside Q4_0 Q and K, as a mixed-format quantization
    stores it."""
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps(_hf_config()))
    vocab = {f"t{i}": i for i in range(VOCAB - 2)}
    (hf / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": [{"id": VOCAB - 2, "content": "<bos>", "special": True},
                         {"id": VOCAB - 1, "content": "<eos>", "special": True}]}))
    (hf / "tokenizer_config.json").write_text("{}")
    (hf / "chat_template.jinja").write_text("{{ bos_token }}")
    (hf / "generation_config.json").write_text('{"top_p": 0.95}')

    generator = np.random.default_rng(4)
    tensors, stored = [], {}

    def block(name, shape, kind, rows):
        tensors.append((name, shape, kind, rows.tobytes()))
        stored[name] = rows

    def vector(name, shape):
        values = generator.normal(size=shape).astype(np.float32)
        tensors.append((name, shape, TYPE_F32, values.tobytes()))
        stored[name] = values

    block("token_embd.weight", (VOCAB, HIDDEN), TYPE_Q6_K, _q6_k(generator, VOCAB, HIDDEN))
    vector("output_norm.weight", (HIDDEN,))
    for i, full in enumerate((False, True)):
        g = f"blk.{i}."
        d = GLOBAL_DIM if full else HEAD_DIM
        block(g + "attn_q.weight", (HEADS * d, HIDDEN), TYPE_Q4_0, _q4_0(generator, HEADS * d, HIDDEN))
        block(g + "attn_k.weight", (d, HIDDEN), TYPE_Q4_0, _q4_0(generator, d, HIDDEN))
        if not full and value_q8:
            block(g + "attn_v.weight", (d, HIDDEN), TYPE_Q8_0, _q8_0(generator, d, HIDDEN))
        elif not full:
            block(g + "attn_v.weight", (d, HIDDEN), TYPE_Q4_0, _q4_0(generator, d, HIDDEN))
        block(g + "attn_output.weight", (HIDDEN, HEADS * d), TYPE_Q4_0,
              _q4_0(generator, HIDDEN, HEADS * d))
        for name in ("attn_q_norm", "attn_k_norm"):
            vector(g + name + ".weight", (d,))
        for name in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm",
                     "post_ffw_norm_1", "pre_ffw_norm_2", "post_ffw_norm_2"):
            vector(g + name + ".weight", (HIDDEN,))
        vector(g + "layer_output_scale.weight", (1,))
        block(g + "ffn_gate.weight", (WIDTH, HIDDEN), TYPE_Q4_0, _q4_0(generator, WIDTH, HIDDEN))
        block(g + "ffn_up.weight", (WIDTH, HIDDEN), TYPE_Q4_0, _q4_0(generator, WIDTH, HIDDEN))
        block(g + "ffn_down.weight", (HIDDEN, WIDTH), TYPE_Q4_0, _q4_0(generator, HIDDEN, WIDTH))
        vector(g + "ffn_gate_inp.weight", (EXPERTS, HIDDEN))
        vector(g + "ffn_gate_inp.scale", (HIDDEN,))
        vector(g + "ffn_down_exps.scale", (EXPERTS,))
        block(g + "ffn_gate_up_exps.weight", (EXPERTS, 2 * EXPERT_WIDTH, HIDDEN), TYPE_Q4_0,
              _q4_0(generator, EXPERTS * 2 * EXPERT_WIDTH, HIDDEN))
        block(g + "ffn_down_exps.weight", (EXPERTS, HIDDEN, EXPERT_WIDTH), TYPE_Q4_0,
              _q4_0(generator, EXPERTS * HIDDEN, EXPERT_WIDTH))
    path = tmp_path / "gemma.gguf"
    write_gguf(path, {"general.architecture": "gemma4"}, tensors)
    return hf, path, stored


def test_text_config_takes_the_26b_geometry_and_refuses_other_variants():
    config = gemma4.text_config(_hf_config())
    assert config["architectures"] == ["Gemma4ForCausalLM"]
    assert config["layer_types"] == ["sliding_attention", "full_attention"]
    assert config["partial_rotary_factor_full"] == 0.25
    assert config["rope_theta_full"] == 1e6 and config["rope_theta_sliding"] == 1e4
    for bad in (dict(hidden_size_per_layer_input=256), dict(num_kv_shared_layers=4),
                dict(attention_k_eq_v=False), dict(hidden_activation="gelu"),
                dict(tie_word_embeddings=False), dict(layer_types=["full_attention"])):
        with pytest.raises(ValueError):
            gemma4.text_config(_hf_config(**bad))


def test_the_artifact_stores_the_gguf_rows_and_vectors_as_they_are(tmp_path):
    from tools.artifact.reader import Artifact
    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe
    from tools.convert.sources.safetensors import SafetensorsSource

    hf, path, stored = _release(tmp_path)
    output = tmp_path / "gemma.ninfer"
    with SafetensorsSource(hf) as base, GGUFFile(path) as gguf:
        model = gemma4.build_model(base, gguf)
        recipe = Recipe(model)
        gemma4.gemma4_gguf(model, recipe, {"base": base, "gguf": gguf})
        convert(model, recipe, output, device="cpu")
    with Artifact(output) as artifact:
        directory = artifact.directory
        config = directory.components["text"]["config"]
        assert config["architectures"] == ["Gemma4ForCausalLM"]
        bindings = directory.bindings

        def stored_bytes(name):
            binding = bindings[name]
            data = artifact.read_object(binding["object"])
            return data

        # The full-attention block has no value projection; the sliding one has all three.
        assert "text/layers/1/attention/value" not in bindings
        assert "text/layers/0/attention/value" in bindings
        # The token table keeps its Q6_K rows byte for byte.
        assert stored_bytes("text/output_head") == stored["token_embd.weight"].tobytes()
        # An expert bank keeps its rows expert-major, byte for byte.
        assert stored_bytes("text/layers/1/moe/down") == stored["blk.1.ffn_down_exps.weight"].tobytes()
        # Vectors are FP32 values.
        router = np.frombuffer(stored_bytes("text/layers/0/moe/router"), dtype=np.float32)
        assert np.array_equal(router, stored["blk.0.ffn_gate_inp.weight"].reshape(-1))
        scale = np.frombuffer(stored_bytes("text/layers/1/moe/expert_scale"), dtype=np.float32)
        assert np.array_equal(scale, stored["blk.1.ffn_down_exps.scale"])


ASSISTANT = 256  # the assistant's width


def _assistant(tmp_path):
    """A two-block gemma4-assistant GGUF for the release above (sliding, full)."""
    generator = np.random.default_rng(5)
    tensors, stored = [], {}

    def block(name, shape, rows):
        tensors.append((name, shape, TYPE_Q4_0, rows.tobytes()))
        stored[name] = rows

    def vector(name, shape):
        values = generator.normal(size=shape).astype(np.float32)
        tensors.append((name, shape, TYPE_F32, values.tobytes()))
        stored[name] = values

    block("token_embd.weight", (VOCAB, ASSISTANT), _q4_0(generator, VOCAB, ASSISTANT))
    vector("output_norm.weight", (ASSISTANT,))
    block("nextn.pre_projection.weight", (ASSISTANT, 2 * HIDDEN),
          _q4_0(generator, ASSISTANT, 2 * HIDDEN))
    block("nextn.post_projection.weight", (HIDDEN, ASSISTANT),
          _q4_0(generator, HIDDEN, ASSISTANT))
    for i, full in enumerate((False, True)):
        g = f"blk.{i}."
        d = GLOBAL_DIM if full else HEAD_DIM
        block(g + "attn_q.weight", (HEADS * d, ASSISTANT), _q4_0(generator, HEADS * d, ASSISTANT))
        block(g + "attn_output.weight", (ASSISTANT, HEADS * d),
              _q4_0(generator, ASSISTANT, HEADS * d))
        vector(g + "attn_q_norm.weight", (d,))
        for name in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"):
            vector(g + name + ".weight", (ASSISTANT,))
        vector(g + "layer_output_scale.weight", (1,))
        block(g + "ffn_gate.weight", (WIDTH, ASSISTANT), _q4_0(generator, WIDTH, ASSISTANT))
        block(g + "ffn_up.weight", (WIDTH, ASSISTANT), _q4_0(generator, WIDTH, ASSISTANT))
        block(g + "ffn_down.weight", (ASSISTANT, WIDTH), _q4_0(generator, ASSISTANT, WIDTH))
    a = "gemma4-assistant."
    kv = {"general.architecture": "gemma4-assistant", a + "block_count": 2,
          a + "embedding_length": ASSISTANT, a + "embedding_length_out": HIDDEN,
          a + "feed_forward_length": WIDTH, a + "attention.head_count": HEADS,
          a + "attention.head_count_kv": [1, 1], a + "attention.shared_kv_layers": 2,
          a + "attention.sliding_window_pattern": [True, False],
          a + "attention.key_length": GLOBAL_DIM, a + "attention.key_length_swa": HEAD_DIM,
          a + "attention.sliding_window": 128, a + "attention.layer_norm_rms_epsilon": 1e-6}
    path = tmp_path / "assistant.gguf"
    write_gguf(path, kv, tensors)
    return path, stored


def _convert(tmp_path, base_dir, gguf_path, mtp_path=None):
    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe
    from tools.convert.sources.safetensors import SafetensorsSource

    output = tmp_path / "gemma.ninfer"
    with SafetensorsSource(base_dir) as base, GGUFFile(gguf_path) as gguf:
        mtp = GGUFFile(mtp_path) if mtp_path is not None else None
        model = gemma4.build_model(base, gguf, mtp=mtp)
        recipe = Recipe(model)
        gemma4.gemma4_gguf(model, recipe, {"base": base, "gguf": gguf})
        convert(model, recipe, output, device="cpu")
    return output


def test_the_assistant_joins_as_the_mtp_component(tmp_path):
    from tools.artifact.reader import Artifact

    hf, path, _ = _release(tmp_path)
    mtp_path, stored = _assistant(tmp_path)
    output = _convert(tmp_path, hf, path, mtp_path)
    with Artifact(output) as artifact:
        directory = artifact.directory
        config = directory.components["mtp"]["config"]
        assert directory.components["mtp"]["target"] == "text"
        assert config["architectures"] == ["Gemma4AssistantForCausalLM"]
        assert config["hidden_size"] == ASSISTANT and config["backbone_hidden_size"] == HIDDEN
        assert config["layer_types"] == ["sliding_attention", "full_attention"]
        assert config["num_key_value_heads"] == 1 and config["num_global_key_value_heads"] == 1
        bindings = directory.bindings
        read = lambda name: artifact.read_object(bindings[name]["object"])
        # The assistant's blocks stay the GGUF's rows; it has queries only.
        assert read("mtp/output_head") == stored["token_embd.weight"].tobytes()
        assert read("mtp/pre_projection") == stored["nextn.pre_projection.weight"].tobytes()
        assert "mtp/layers/1/attention/query" in bindings
        assert not any(name.startswith("mtp/") and name.endswith(("/key", "/value"))
                       for name in bindings)
        scalar = np.frombuffer(read("mtp/layers/1/layer_scalar"), dtype=np.float32)
        assert np.array_equal(scalar, stored["blk.1.layer_output_scale.weight"])


def test_an_assistant_of_other_geometry_is_refused(tmp_path):
    _, path, _ = _release(tmp_path)
    mtp_path, _ = _assistant(tmp_path)
    with GGUFFile(mtp_path) as mtp:
        target = gemma4.text_config(_hf_config())
        assert gemma4.assistant_config(mtp, target)["vocab_size"] == VOCAB
        with pytest.raises(ValueError):
            gemma4.assistant_config(mtp, dict(target, head_dim=2 * HEAD_DIM))
        with pytest.raises(ValueError):
            gemma4.assistant_config(mtp, dict(target, hidden_size=2 * HIDDEN))


def test_a_mixed_format_block_keeps_each_projection_in_its_format(tmp_path):
    from tools.artifact.reader import Artifact

    hf, path, stored = _release(tmp_path, value_q8=True)
    output = _convert(tmp_path, hf, path)
    with Artifact(output) as artifact:
        bindings = artifact.directory.bindings
        a = "text/layers/0/attention/"
        # A block whose Q, K and V differ in format keeps each its own object; the full-attention
        # block's Q and K, both Q4_0, still share one parent.
        objects = {bindings[a + name]["object"] for name in ("query", "key", "value")}
        assert len(objects) == 3
        f = "text/layers/1/attention/"
        parent = lambda name: [part["object"] for part in bindings[f + name]["parts"]]
        assert parent("key") == parent("query")
        assert artifact.read_object(bindings[a + "value"]["object"]) == \
            stored["blk.0.attn_v.weight"].tobytes()
