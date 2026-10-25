from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.convert import deepseek_v41
from tools.convert.sources.gguf import TYPE_BF16, TYPE_F32, GGUFFile, write_gguf

TYPE_Q8_0, TYPE_Q5_K, TYPE_IQ4_NL, TYPE_I32, TYPE_I64 = 8, 13, 20, 26, 27

HIDDEN, VOCAB, HEADS, HEAD_DIM, ROPE = 256, 512, 2, 64, 16
Q_RANK, O_RANK, GROUPS = 64, 32, 2
INDEX_HEADS, INDEX_DIM = 2, 32
EXPERTS, WIDTH, HC = 4, 64, 4
RATIOS = [0, 2, 1]
ENGRAM_LAYERS, NGRAM, ENGRAM_HEADS, ENGRAM_DIM = [0], 3, 2, 32
PRIMES = [5, 7, 11, 13]
ENGRAM_ROWS = sum(PRIMES)
COMPRESSED_VOCAB = 100


def _hf_config(**overrides):
    text = dict(
        model_type="deepseek_v41_text", vocab_size=VOCAB, hidden_size=HIDDEN,
        moe_intermediate_size=WIDTH, num_hidden_layers=3, num_attention_heads=HEADS,
        num_key_value_heads=1, head_dim=HEAD_DIM, qk_rope_head_dim=ROPE, q_lora_rank=Q_RANK,
        o_lora_rank=O_RANK, o_groups=GROUPS, hidden_act="silu", swiglu_limit=10.0,
        rms_norm_eps=1e-20, attention_bias=False, tie_word_embeddings=False,
        max_position_embeddings=4096, rope_theta=10000,
        rope_scaling={"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
                      "original_max_position_embeddings": 1024},
        n_routed_experts=EXPERTS, n_shared_experts=1, num_experts_per_tok=2,
        scoring_func="sqrtsoftplus", topk_method="noaux_tc", norm_topk_prob=True,
        routed_scaling_factor=1.5, sliding_window=8, compress_ratios=RATIOS + [0],
        kv_source_layer_ids=[1, 2], index_source_layer_ids=[1, 2], candidate_source_layer_id=2,
        candidate_topk_blocks=4, candidate_block_size=8, index_n_heads=INDEX_HEADS,
        index_head_dim=INDEX_DIM, index_topk=16, compress_rope_theta=160000, hc_mult=HC,
        hc_sinkhorn_iters=20, hc_eps=1e-6, engram_layer_ids=ENGRAM_LAYERS,
        engram_num_embeddings=[ENGRAM_ROWS], engram_max_ngram_size=NGRAM,
        engram_n_heads=ENGRAM_HEADS, engram_head_dim=ENGRAM_DIM, engram_pad_token_id=2,
        engram_compressed_vocab_size=COMPRESSED_VOCAB)
    text.update(overrides)
    return {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41",
            "bos_token_id": 0, "eos_token_id": 1, "text_config": text}


def _gguf_kv(**overrides):
    f = "deepseek41."
    kv = {
        "general.architecture": "deepseek41", "split.count": 2, f + "block_count": 3,
        f + "embedding_length": HIDDEN, f + "attention.head_count": HEADS,
        f + "attention.head_count_kv": 1, f + "attention.key_length": HEAD_DIM,
        f + "attention.value_length": HEAD_DIM, f + "rope.dimension_count": ROPE,
        f + "attention.q_lora_rank": Q_RANK, f + "attention.output_group_count": GROUPS,
        f + "attention.output_lora_rank": O_RANK, f + "attention.sliding_window": 8,
        f + "expert_count": EXPERTS, f + "expert_used_count": 2,
        f + "expert_feed_forward_length": WIDTH, f + "expert_shared_count": 1,
        f + "expert_gating_func": 4, f + "expert_weights_norm": True,
        f + "expert_weights_scale": 1.5, f + "expert_scoring_func": "sqrtsoftplus",
        f + "attention.indexer.head_count": INDEX_HEADS,
        f + "attention.indexer.key_length": INDEX_DIM, f + "attention.indexer.top_k": 16,
        f + "attention.candidate_block_size": 8, f + "attention.candidate_topk_blocks": 4,
        f + "attention.candidate_source_layer_id": 2, f + "hyper_connection.count": HC,
        f + "hyper_connection.sinkhorn_iterations": 20, f + "hyper_connection.epsilon": 1e-6,
        f + "engram_head_dim": ENGRAM_DIM, f + "engram_n_heads": ENGRAM_HEADS,
        f + "engram_max_ngram_size": NGRAM, f + "engram_pad_token_id": 2,
        f + "rope.scaling.original_context_length": 1024, f + "rope.freq_base": 10000.0,
        f + "attention.compress_rope_freq_base": 160000.0, f + "rope.scaling.factor": 16.0,
        f + "rope.scaling.yarn_beta_fast": 32.0, f + "rope.scaling.yarn_beta_slow": 1.0,
        f + "attention.layer_norm_rms_epsilon": 1e-20,
        f + "attention.compress_ratios": RATIOS + [0], f + "kv_source_layer_ids": [1, 2],
        f + "index_source_layer_ids": [1, 2], f + "engram.layer_ids": ENGRAM_LAYERS,
        f + "engram.num_embeddings": [ENGRAM_ROWS], f + "swiglu_clamp_exp": [10.0] * 3,
        f + "swiglu_clamp_shexp": [10.0] * 3,
    }
    kv.update(overrides)
    return kv


def _blocks(generator, rows, columns, block_values, block_bytes, scale_bytes=2):
    count = columns // block_values
    blocks = generator.integers(0, 256, size=(rows, count, block_bytes), dtype=np.uint8)
    scales = generator.uniform(0.01, 0.02, size=(rows, count, scale_bytes // 2))
    blocks[:, :, :scale_bytes] = scales.astype(np.float16).view(np.uint8).reshape(
        rows, count, scale_bytes)
    return blocks.reshape(rows, -1)


def _release(tmp_path, kv=None, offsets=None):
    """A three-block DeepSeek-V4.1 release (window-only, ratio 2, ratio 1; Engram in block 0): the
    HF text files, a GGUF split in two shards and the Engram GGUF."""

    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps(_hf_config()))
    vocab = {f"t{i}": i for i in range(3, VOCAB)}
    added = [{"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True}
             for i, c in enumerate(["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
                                    "<｜▁pad▁｜>"])]
    vocab.update({token["content"]: token["id"] for token in added})
    (hf / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab, "merges": []}, "added_tokens": added}))
    # As the release's: no added_tokens_decoder.
    (hf / "tokenizer_config.json").write_text(json.dumps({"add_bos_token": False}))

    generator = np.random.default_rng(11)
    shards, stored = [[], []], {}

    def put(name, shape, kind, data, shard):
        shards[shard].append((name, shape, kind, data.tobytes()))
        stored[name] = data

    def q8(name, rows, columns, shard=0):
        put(name, (rows, columns), TYPE_Q8_0, _blocks(generator, rows, columns, 32, 34), shard)

    def vector(name, shape, kind=TYPE_F32, shard=0):
        values = generator.normal(size=shape).astype(np.float32)
        if kind == TYPE_BF16:
            words = (values.view(np.uint32) >> 16).astype(np.uint16)
            shards[shard].append((name, shape, kind, words.tobytes()))
            stored[name] = (words.astype(np.uint32) << 16).view(np.float32)
        else:
            put(name, shape, kind, values, shard)

    def integers(name, values, kind):
        put(name, values.shape, kind, values, 0)

    put("token_embd.weight", (VOCAB, HIDDEN), TYPE_Q5_K,
        _blocks(generator, VOCAB, HIDDEN, 256, 176, 4), 0)
    q8("output.weight", VOCAB, HIDDEN, 1)
    vector("output_norm.weight", (HIDDEN,), shard=1)
    columns = (NGRAM - 1) * ENGRAM_HEADS
    integers("engram_token_map.weight",
             generator.integers(0, COMPRESSED_VOCAB, size=(VOCAB,)).astype("<i4"), TYPE_I32)
    integers("engram_multipliers.weight", np.array([[1, 3, 5]], dtype="<i8"), TYPE_I64)
    integers("engram_primes.weight", np.array([PRIMES], dtype="<i8"), TYPE_I64)
    integers("engram_offsets.weight",
             np.array([offsets or [0, 5, 12, 23]], dtype="<i8"), TYPE_I64)
    mixes = (2 + HC) * HC
    for i, ratio in enumerate(RATIOS):
        g, shard = f"blk.{i}.", i % 2
        for site in ("attn", "ffn"):
            vector(g + f"hc_{site}_fn.weight", (mixes, HC * HIDDEN), shard=shard)
            vector(g + f"hc_{site}_base.weight", (mixes,), shard=shard)
            vector(g + f"hc_{site}_scale.weight", (3,), shard=shard)
        vector(g + "attn_norm.weight", (HIDDEN,), shard=shard)
        vector(g + "ffn_norm.weight", (HIDDEN,), shard=shard)
        q8(g + "attn_q_a.weight", Q_RANK, HIDDEN, shard)
        q8(g + "attn_kv.weight", HEAD_DIM, HIDDEN, shard)
        vector(g + "attn_q_a_norm.weight", (Q_RANK,), shard=shard)
        vector(g + "attn_kv_a_norm.weight", (HEAD_DIM,), shard=shard)
        q8(g + "attn_q_b.weight", HEADS * HEAD_DIM, Q_RANK, shard)
        q8(g + "attn_output_a.weight", GROUPS * O_RANK, HEADS * HEAD_DIM // GROUPS, shard)
        q8(g + "attn_output_b.weight", HIDDEN, GROUPS * O_RANK, shard)
        vector(g + "attn_sinks.weight", (HEADS,), shard=shard)
        if ratio:
            vector(g + "attn_compressor_kv.weight", (HEAD_DIM, HIDDEN), TYPE_BF16, shard)
            if ratio > 1:
                vector(g + "attn_compressor_gate.weight", (HEAD_DIM, HIDDEN), TYPE_BF16, shard)
            vector(g + "attn_compressor_norm.weight", (HEAD_DIM,), shard=shard)
            vector(g + "indexer.attn_k.weight", (INDEX_DIM, HEAD_DIM), TYPE_BF16, shard)
            vector(g + "indexer.k_norm.weight", (INDEX_DIM,), shard=shard)
            vector(g + "indexer.attn_q_b.weight", (INDEX_HEADS * INDEX_DIM, Q_RANK), TYPE_BF16,
                   shard)
            vector(g + "indexer.proj.weight", (INDEX_HEADS, HIDDEN), TYPE_BF16, shard)
        vector(g + "ffn_gate_inp.weight", (EXPERTS, HIDDEN), TYPE_BF16, shard)
        vector(g + "exp_probs_b.bias", (EXPERTS,), shard=shard)
        for bank, rows, k in (("gate", WIDTH, HIDDEN), ("up", WIDTH, HIDDEN),
                              ("down", HIDDEN, WIDTH)):
            put(g + f"ffn_{bank}_exps.weight", (EXPERTS, rows, k), TYPE_Q8_0,
                _blocks(generator, EXPERTS * rows, k, 32, 34), shard)
        q8(g + "ffn_gate_shexp.weight", WIDTH, HIDDEN, shard)
        q8(g + "ffn_up_shexp.weight", WIDTH, HIDDEN, shard)
        q8(g + "ffn_down_shexp.weight", HIDDEN, WIDTH, shard)
        if i in ENGRAM_LAYERS:
            q8(g + "engram_wkv.weight", (HC + 1) * HIDDEN, columns * ENGRAM_DIM, shard)
            vector(g + "engram_q.weight", (HC, HIDDEN), TYPE_BF16, shard)
            vector(g + "engram_k.weight", (HC, HIDDEN), TYPE_BF16, shard)
    first = tmp_path / "ds-00001-of-00002.gguf"
    write_gguf(first, kv or _gguf_kv(), shards[0])
    write_gguf(tmp_path / "ds-00002-of-00002.gguf", {"split.count": 2}, shards[1])
    table = tmp_path / "engram.gguf"
    rows = _blocks(generator, ENGRAM_ROWS, ENGRAM_DIM, 32, 18)
    write_gguf(table, {"general.architecture": "deepseek41-engram"},
               [("blk.0.engram_embd.weight", (ENGRAM_ROWS, ENGRAM_DIM), TYPE_IQ4_NL,
                 rows.tobytes())])
    stored["engram"] = rows
    return hf, first, table, stored


def _convert(tmp_path, hf, first, table, components=("text", "engram")):
    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe
    from tools.convert.sources.safetensors import SafetensorsSource

    output = tmp_path / ("-".join(components) + ".ninfer")
    with SafetensorsSource(hf) as base, GGUFFile(table) as engram:
        gguf = deepseek_v41.GGUFShards(first) if "text" in components else None
        model = deepseek_v41.build_model(base, gguf, components=components)
        recipe = Recipe(model)
        sources = {"base": base, "engram": engram} | ({"gguf": gguf} if gguf else {})
        deepseek_v41.deepseek_v41_gguf(model, recipe, sources)
        convert(model, recipe, output, device="cpu", rows_per_chunk=96)
        if gguf is not None:
            gguf.__exit__()
    return output


def test_text_config_refuses_other_variants():
    config = deepseek_v41.text_config(_hf_config())
    assert config["compress_ratios"] == RATIOS  # the drafter's entry is dropped
    for bad in (dict(num_key_value_heads=2), dict(scoring_func="sigmoid"),
                dict(compress_ratios=[0, 4, 1]), dict(hidden_act="gelu"),
                dict(norm_topk_prob=False), dict(tie_word_embeddings=True),
                dict(rope_scaling={"rope_type": "linear"})):
        with pytest.raises(ValueError):
            deepseek_v41.text_config(_hf_config(**bad))


@pytest.mark.parametrize("bad", [
    {"deepseek41.expert_gating_func": 2},
    {"deepseek41.attention.compress_ratios": [0, 1, 1, 0]},
    {"deepseek41.attention.candidate_source_layer_id": 1},
    {"deepseek41.expert_weights_scale": 2.5},
    {"deepseek41.swiglu_clamp_exp": [7.0] * 3},
    {"deepseek41.expert_scoring_func": "softmax"},
])
def test_the_gguf_metadata_must_match_the_config(tmp_path, bad):
    from tools.convert.sources.safetensors import SafetensorsSource

    hf, first, _, _ = _release(tmp_path, _gguf_kv(**bad))
    with SafetensorsSource(hf) as base, deepseek_v41.GGUFShards(first) as gguf:
        with pytest.raises(ValueError):
            deepseek_v41.build_model(base, gguf)


def test_engram_columns_must_stay_inside_their_table(tmp_path):
    from tools.convert.sources.safetensors import SafetensorsSource

    hf, first, _, _ = _release(tmp_path, offsets=[0, 5, 12, 24])
    with SafetensorsSource(hf) as base, deepseek_v41.GGUFShards(first) as gguf:
        with pytest.raises(ValueError, match="past its table"):
            deepseek_v41.build_model(base, gguf)


def test_the_artifact_keeps_the_gguf_rows_and_the_engram_table(tmp_path):
    from tools.artifact.reader import Artifact

    hf, first, table, stored = _release(tmp_path)
    output = _convert(tmp_path, hf, first, table)
    with Artifact(output) as artifact:
        directory = artifact.directory
        text = directory.components["text"]
        config = text["config"]
        assert config["architectures"] == ["DeepseekV41ForCausalLM"]
        assert config["engram"]["primes"] == [PRIMES]
        assert config["engram"]["multipliers"] == [[1, 3, 5]]
        bindings = directory.bindings

        def stored_bytes(name):
            return artifact.read_object(bindings[name]["object"])

        def floats(name):
            return np.frombuffer(stored_bytes(name), dtype=np.float32)

        # The shards' blocks row for row, the first shard's and the second's.
        assert stored_bytes("text/token_embedding") == stored["token_embd.weight"].tobytes()
        assert stored_bytes("text/output_head") == stored["output.weight"].tobytes()
        for layer in range(3):
            g, p = f"blk.{layer}.", f"text/layers/{layer}/"
            gate, up = stored[g + "ffn_gate_exps.weight"], stored[g + "ffn_up_exps.weight"]
            expected = np.concatenate([np.concatenate([gate[e * WIDTH:(e + 1) * WIDTH],
                                                       up[e * WIDTH:(e + 1) * WIDTH]])
                                       for e in range(EXPERTS)])
            gate_up = np.frombuffer(stored_bytes(p + "moe/gate_up"),
                                    dtype=np.uint8).reshape(-1, gate.shape[1])
            assert np.array_equal(gate_up, expected)
            assert stored_bytes(p + "moe/down") == stored[g + "ffn_down_exps.weight"].tobytes()
            # BF16 and F32 vectors become FP32 exactly.
            assert np.array_equal(floats(p + "moe/router"),
                                  stored[g + "ffn_gate_inp.weight"].reshape(-1))
            assert np.array_equal(floats(p + "hc_attention/fn"),
                                  stored[g + "hc_attn_fn.weight"].reshape(-1))
            assert (p + "attention/compressor/kv" in bindings) == (layer in (1, 2))
            assert (p + "attention/compressor/gate" in bindings) == (layer == 1)
        assert np.array_equal(floats("text/layers/2/attention/indexer/query"),
                              stored["blk.2.indexer.attn_q_b.weight"].reshape(-1))
        # The Engram gate is the exact FP32 product of its BF16 factors.
        assert np.array_equal(floats("text/layers/0/engram/gate"),
                              (stored["blk.0.engram_q.weight"] *
                               stored["blk.0.engram_k.weight"]).reshape(-1))
        assert "text/layers/1/engram/gate" not in bindings
        # The table keeps its IQ4_NL rows, named by their digest.
        table_bytes = stored["engram"].tobytes()
        assert stored_bytes("engram/table") == table_bytes
        engram = directory.components["engram"]["config"]
        assert engram["table_sha256"] == hashlib.sha256(table_bytes).hexdigest()
        assert engram["format"] == "gguf_iq4_nl" and engram["rows"] == ENGRAM_ROWS
        # The release names no added tokens in tokenizer_config.json; the artifact's does.
        resources = text["resources"]
        tokenizer_config = json.loads(artifact.read_object(resources["tokenizer_config.json"]))
        assert sorted(tokenizer_config["added_tokens_decoder"]) == ["0", "1", "2"]
        assert artifact.read_object(resources["engram_token_map"]) == \
            stored["engram_token_map.weight"].tobytes()
        assert artifact.read_object(resources["chat_template.jinja"]) == \
            (deepseek_v41.DEFAULT_RESOURCES / "chat_template.jinja").read_bytes()


def test_the_model_and_its_table_convert_apart(tmp_path):
    from tools.artifact.reader import Artifact

    hf, first, table, stored = _release(tmp_path)
    digest = hashlib.sha256(stored["engram"].tobytes()).hexdigest()
    with Artifact(_convert(tmp_path, hf, first, table, ("text",))) as artifact:
        directory = artifact.directory
        assert "engram/table" not in directory.bindings
        assert "text/token_embedding" in directory.bindings
        assert directory.components["engram"]["config"]["table_sha256"] == digest
    with Artifact(_convert(tmp_path, hf, first, table, ("engram",))) as artifact:
        directory = artifact.directory
        assert set(directory.bindings) == {"engram/table"}
        assert "text" not in directory.components
        assert directory.components["engram"]["config"]["table_sha256"] == digest
