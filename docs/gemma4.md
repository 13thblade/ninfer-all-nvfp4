# Gemma 4 26B-A4B

NInfer runs Google's Gemma 4 26B-A4B text model (`Gemma4ForCausalLM`, and the text component of
`Gemma4ForConditionalGeneration`) from the official quantization-aware-trained GGUF release
[`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf).
It is its own model family next to Qwen3.5 and Flash-Next: its own executor
(`src/models/gemma4`) and Engine core (`src/runtime/engine/gemma4_core`), sharing the Qwen3.5
frontend for tokenization, chat templates, output parsing and the serving protocols.

## What runs

- **The model.** 30 blocks of hidden width 2816: five sliding-window blocks (window 1024, head
  width 256, 16 query and 8 key/value heads) to every full-attention block (head width 512, 16 query
  heads over 2 key/value heads, keys used as values, proportional RoPE over a quarter of the
  width). Each block runs a dense GeGLU MLP (2112) beside 128 routed experts (8 per token, width
  704), with Gemma's sandwich norms and per-layer output scale, a tied 262,144-token head and the
  final logit soft cap.
- **The weights stay the release's blocks.** Every matrix is the GGUF's Q4_0 (the tied token table
  Q6_K), imported row for row without requantization (13.4 GiB) and multiplied by the GGUF kernels:
  vector kernels for up to eight tokens, integer matrix kernels for prompt chunks, the expert
  products over the routed experts only.
- **Requests.** One to eight concurrent sequences (`--max-concurrency`), prompt chunks of 2048
  tokens, prefix reuse across turns, structured output, tool calls in Gemma's
  `<|tool_call>call:NAME{...}<tool_call|>` form returned as OpenAI tool calls, streaming, reasoning
  in its own channel when thinking is on, and causal scoring (`ninfer-perplexity`).
- **Thinking.** Off unless asked for, as the release's chat template has it: a request turns it on
  with `chat_template_kwargs: {"enable_thinking": true}`, and the reasoning arrives as
  `reasoning_content`.
- **Sampling defaults.** The release's `generation_config.json` (temperature 1.0, top-p 0.95) with
  top-k held at the sampler's 20 candidates.

Not built for this family: images and video (the vision tower), MTP and other speculative
decoding, ngram copy drafts, pipeline stages over several GPUs, and KV formats other than F16.

## Converting

The converter reads the weights from the GGUF and the tokenizer, chat template and generation
config from the Hugging Face release:

```bash
mkdir -p models/gemma4/hf
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json; do
  curl -fsSL -o models/gemma4/hf/$f https://huggingface.co/google/gemma-4-26B-A4B-it/resolve/main/$f
done
hf download google/gemma-4-26B-A4B-it-qat-q4_0-gguf gemma-4-26B_q4_0-it.gguf --local-dir models/gemma4

python -m tools.convert --model models/gemma4/hf --recipe gemma4_gguf \
  --source gguf=models/gemma4/gemma-4-26B_q4_0-it.gguf --name Gemma-4-26B-A4B-it-QAT \
  --out models/gemma-4-26B-A4B-it-qat-q4_0.ninfer --device cpu
```

The conversion takes under three minutes and writes one 14.5 GB `.ninfer` file.

## Running

```bash
# A chat server for four concurrent requests over 40K-token contexts.
ninfer-serve models/gemma-4-26B-A4B-it-qat-q4_0.ninfer --model-id gemma4 \
  --max-context 40960 --max-concurrency 4

# One prompt from the CLI.
ninfer models/gemma-4-26B-A4B-it-qat-q4_0.ninfer --prompt "Explain RoPE in two sentences."
```

On a 24 GB card the weights take 13.4 GiB and the prompt-chunk workspace about 1.1 GiB. Every
sequence holds its whole context from the start: 200 MiB for the 1024-token windows of the 25
sliding blocks (and up to as much again for a prefix checkpoint) and 20 KiB per token for the five
full-attention blocks; prompt chunks share one 24 MiB staging ring. One sequence of 40,960 tokens
takes 1.0 GiB, four of 16,384 tokens 2.1 GiB and eight of 32,768 tokens 6.6 GiB. A configuration
that does not fit is refused at startup with the numbers.

## Measurements

An RTX 3090 (24 GB) held at a 300 W power limit, CUDA 12.8; three repetitions each; `ninfer_bench` over a 65,536-token WikiText corpus slice
(tokenized for Gemma) against `llama-bench` (llama.cpp `23b0202`, `-fa 1 -ngl 99`) on the same GGUF.
Prefill is tokens per second of a prompt of that length from an empty context; decode is 128
generated tokens after a prompt of that length (llama.cpp: `tg128 @ d<N>`).

| Test | NInfer | llama.cpp |
|---|---:|---:|
| prefill 2,048 | 6,220 | 4,735 |
| prefill 10,000 | 5,695 | 4,427 |
| prefill 40,000 | 4,446 | 3,624 |
| decode, empty context | 208.3 | 166.6 |
| decode after 10,000 | 176.7 | 151.3 |
| decode after 40,000 | 145.1 | 135.8 |

Eight 256-token chat generations sent at once to `ninfer-serve` finish at 458 tok/s together with
`--max-concurrency 8` (60 to 64 tok/s per stream) and at 415 tok/s with `--max-concurrency 4`
(106 to 114 tok/s per stream, four at a time).

## Numerical checks

- The GGUF vector and matrix products at Gemma's widths (2816, 2112 and 704, the last two no whole
  number of the matrix kernel's 256-value steps) and the routed-expert GeGLU products match an FP64
  product of the decoded blocks (`ninfer_linear_gguf_test`, `ninfer_gguf_moe_geglu_test`); the
  decode attention, its q8_1 output and the decode routing match FP64 oracles
  (`ninfer_gemma4_kernels_test`); Gemma's tool-call markup parses as streamed
  (`ninfer_gemma4_output_test`).
- End to end, [`tools/reference/gemma4.py`](../tools/reference/gemma4.py), an FP64 NumPy forward
  of the model over the exactly decoded GGUF, is the oracle. NInfer agrees with it within the
  activation-quantization noise llama.cpp shows against the same oracle: over a 147-token WikiText
  window NInfer's total NLL is 1229.1 nats against the oracle's 1220.5, and on the last 64 targets
  of a fresh 256-token window llama.cpp gives 557.9 against the oracle's 555.7. Block by
  block, both engines drift from the oracle alike (about 10% relative error in the last blocks'
  hidden states for a 26-token prompt), the cost of 8-bit activations in a model this deep.
- Compare perplexity with llama.cpp on whole fresh windows only: `llama-perplexity
  --ppl-stride` on this model does not reproduce its own fresh-window result for the windows after
  the first (a window scored alone gives 487.6 nats where the strided run reports 401.3), so its
  strided totals are not a reference here.
