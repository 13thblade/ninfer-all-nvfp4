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
  products over the routed experts only. A mixed-format GGUF of the same model (a Q4_K_M
  quantization with Q6_K values, Q5_0 or Q8_0 expert down projections) converts and runs the same
  way, each matrix in its own format.
- **The full-attention cache holds values only.** Those blocks' keys are their values times the
  key norm's gamma, rotated: the cache keeps V (10 KiB per token for the five blocks) and the
  attention derives each key from it as it reads it, with the same rotations the keys were stored
  with.
- **Speculative decoding with the release's assistant.** The multi-token-prediction drafter Google
  publishes beside the model (`Gemma4AssistantForCausalLM`, four blocks of width 1024 that read the
  target's own KV cache) drafts up to seven tokens a round (`--spec mtp --draft-tokens N`); the
  target verifies them in one pass and every token it keeps is its own sample, so the output is
  what plain decoding produces. See [below](#speculative-decoding-with-the-assistant).
- **Requests.** One to eight concurrent sequences (`--max-concurrency`), prompt chunks of 2048
  tokens, prefix reuse across turns, structured output, tool calls in Gemma's
  `<|tool_call>call:NAME{...}<tool_call|>` form returned as OpenAI tool calls, streaming, reasoning
  in its own channel when thinking is on, and causal scoring (`ninfer-perplexity`).
- **Thinking.** Off unless asked for, as the release's chat template has it: a request turns it on
  with `chat_template_kwargs: {"enable_thinking": true}`, and the reasoning arrives as
  `reasoning_content`.
- **Sampling defaults.** The release's `generation_config.json` (temperature 1.0, top-p 0.95) with
  top-k held at the sampler's 20 candidates.

Not built for this family: images and video (the vision tower), DFlash and ngram copy drafts,
pipeline stages over several GPUs, and KV formats other than F16.

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

The assistant joins as the artifact's `mtp` component from a llama.cpp `gemma4-assistant` GGUF.
The QAT model takes the QAT assistant (Google's model card: a QAT target needs a QAT drafter);
Unsloth publishes it quantized to Q4_0, the format its training targets, and the converter imports
its blocks as they are (252 MB):

```bash
hf download unsloth/gemma-4-26B-A4B-it-qat-GGUF mtp-gemma-4-26B-A4B-it.gguf --local-dir models/gemma4
python -m tools.convert --model models/gemma4/hf --recipe gemma4_gguf --components text,mtp \
  --source gguf=models/gemma4/gemma-4-26B_q4_0-it.gguf \
  --source mtp=models/gemma4/mtp-gemma-4-26B-A4B-it.gguf --name Gemma-4-26B-A4B-it-QAT \
  --out models/gemma-4-26B-A4B-it-qat-q4_0-mtp.ninfer --device cpu
```

An artifact with the assistant runs without it too unless `--spec mtp` asks for it.

## Running

```bash
# A chat server for four concurrent requests over 40K-token contexts.
ninfer-serve models/gemma-4-26B-A4B-it-qat-q4_0.ninfer --model-id gemma4 \
  --max-context 40960 --max-concurrency 4

# One prompt from the CLI.
ninfer models/gemma-4-26B-A4B-it-qat-q4_0.ninfer --prompt "Explain RoPE in two sentences."

# With the assistant drafting two tokens a round.
ninfer-serve models/gemma-4-26B-A4B-it-qat-q4_0-mtp.ninfer --model-id gemma4 \
  --max-context 40960 --max-concurrency 4 --spec mtp --draft-tokens 2
```

On a 24 GB card the weights take 13.4 GiB (the assistant 0.25 GiB more) and the prompt-chunk
workspace about 1.1 GiB. Every sequence holds its whole context from the start: 200 MiB for the
1024-token windows of the 25 sliding blocks (250 MiB with `--spec mtp`, whose rounds write their
drafts past the window; up to as much again for a prefix checkpoint) and 10 KiB per token for the
values of the five full-attention blocks. Shared by all sequences: a 24 MiB staging ring for prompt
chunks, the keys a chunk derives from the values (2 KiB per context token) and the rotation table
(0.5 KiB per context token). One sequence of 40,960 tokens takes 0.7 GiB, four of 16,384 tokens
1.5 GiB and eight of 32,768 tokens 4.2 GiB. A configuration that does not fit is refused at startup
with the numbers.

## Measurements

An RTX 3090 (24 GB) at its 350 W power limit, CUDA 12.8; three repetitions each; `ninfer_bench`
over a 65,536-token slice of the perplexity corpus (its first stream, NInfer source text, tokenized
for Gemma) against `llama-bench` (llama.cpp `23b0202`, `-fa 1 -ngl 99`) on the same GGUF. Prefill is
tokens per second of a prompt of that length from an empty context; decode is 128 generated tokens
after a prompt of that length (llama.cpp: `tg128 @ d<N>`).

| Test | NInfer | llama.cpp |
|---|---:|---:|
| prefill 2,048 | 6,566 | 4,925 |
| prefill 10,000 | 6,045 | 4,697 |
| prefill 40,000 | 4,698 | 3,767 |
| decode, empty context | 215.4 | 168.6 |
| decode after 10,000 | 190.4 | 151.7 |
| decode after 40,000 | 162.2 | 135.7 |

Deriving the full-attention keys from the values and the eight-warp attention kernel moved decode
after 40,000 tokens from 148.5 to 162.2 tok/s (after 10,000: 182.6 to 190.4; empty context 216.2 to
215.4, within its spread) and halved those blocks' cache. Perplexity over eleven 8,192-token
windows of a NInfer source stream: 24.8036 with stored keys, 24.7948 with derived ones (the keys
are rounded from the F16 values rather than from the unrounded projection).

Chat requests in parallel streams to `ninfer-serve --max-concurrency N` (eight prompts, 384-token
greedy answers, three requests per stream) finish at 208.4 tok/s together for one stream, 310.1 for
two, 354.7 for three and 397.2 for four.

## Speculative decoding with the assistant

The assistant drafts from the token the next position takes (the anchor) and the target's
final-normed hidden state before it: each step projects [the target's scaled embedding of the last
token; the hidden state] to width 1024, runs its four blocks at the anchor's position with queries
only, over the keys and values the target holds (its sliding-window blocks the target's last
sliding block's window, its full-attention block the target's last full-attention block), and takes
the argmax of its head as the draft; the post-projection of its final state is the next step's
hidden state. The chain runs from one CUDA graph per batch and width.

The target then runs the anchor and the drafts of every speculating request as decode rows of one
pass: a request decoding alone takes the configured drafts, two or three decoding at once take one
draft each, and four or more decode plainly (see the measurements below). The
rows' logits are bit for bit those of decoding the same tokens one at a time, and the acceptance
draws each position as the request samples (greedy or the request's seeded sampler, penalties and
grammars included), keeping drafts while they equal the draw; the first difference, or the token
after the last draft, is the target's own draw. The output is therefore the one plain decoding
gives. `--adaptive-mtp` lets the round width follow each request's draft survival, and
`--lookup-ngram N` proposes what followed the last N tokens' earlier occurrence instead of the
assistant's guess when there is one.

Ten chat requests one at a time to `ninfer-serve` (code in Python and C++, an explanation, a
summary of a 1,000-token source excerpt, a factual question, a translation, a math word problem, a
story, a Russian answer, and a 13,600-token source excerpt to describe), 384 generated tokens each,
on the same RTX 3090. Every configuration generates the same tokens (3,664 greedy, 3,672 sampled):
the drafts change how many tokens a pass commits, not which.

| Decode tok/s | greedy | 13,600-token prompt | sampled (T 1, top-k 64, top-p 0.95) |
|---|---:|---:|---:|
| plain decoding | 209.1 | 186.0 | 207.9 |
| `--draft-tokens 1` (acceptance 0.81) | 280.1 | 240.2 | |
| `--draft-tokens 2` (0.70; 0.63 sampled) | 290.7 | 241.9 | 270.6 |
| `--draft-tokens 3` (0.62; 0.55 sampled) | 293.7 | 229.2 | 268.0 |

llama.cpp `23b0202`'s `llama-server` with the same GGUF and drafter (`--spec-type draft-mtp`,
thinking off) decodes the same greedy requests at 131.4 tok/s plainly, 186.4 with one draft, 201.6
with two and 195.2 with three (acceptance 0.79, 0.70, 0.61).

Two drafts are the best default: within 1% of three on short prompts, ahead after long ones and
when sampling. `--adaptive-mtp` with three drafts kept all three on this set (293.7), and
`--lookup-ngram 8` found little to copy (292.9). On the corpus continuation of `ninfer_bench`
(`-pg 10000,256;40000,256`), where the source text repeats, two drafts decode at 236.1 tok/s after
10,000 tokens and 210.1 after 40,000 (plain: 190.4 and 162.2).

A round of three drafts costs about two decode steps of GPU time (an `nsys` trace at 512 tokens of
context: the verification pass 8.0 ms against 4.7 ms for one token, the three draft steps 1.3 ms).
Most of the verification's extra time is the routed experts: four tokens touch about three times
as many experts as one, and their weights are read once per pass.

With several requests decoding the verification's extra rows route to ever more experts, so the
round drafts shrink as requests join (same streams as above, `--spec mtp`, tok/s together):

| Streams | plain | with the assistant |
|---|---:|---:|
| 1 | 208.4 | 295.5 with three drafts, 291.8 with two |
| 2 | 310.1 | 370.0 with one draft each (332.5 with two, 330.0 with three) |
| 3 | 354.7 | 365.7 with one draft each |
| 4 | 397.2 | 387.6 with one draft each, so four or more decode plainly |

## Numerical checks

- The GGUF vector and matrix products at Gemma's widths (2816, 2112 and 704, the last two no whole
  number of the matrix kernel's 256-value steps) and the routed-expert GeGLU products match an FP64
  product of the decoded blocks (`ninfer_linear_gguf_test`, `ninfer_gguf_moe_geglu_test`); the
  decode attention (both head widths, stored and derived keys, several rows of one sequence at
  consecutive positions), its q8_1 output and the decode routing match FP64 oracles
  (`ninfer_gemma4_kernels_test`); Gemma's tool-call markup parses as streamed
  (`ninfer_gemma4_output_test`).
- The assistant: [`tools/reference/gemma4.py`](../tools/reference/gemma4.py) `--assistant` runs
  transformers' drafting algorithm in FP64 over the decoded GGUFs. On a 40-token prompt its three
  drafts after the greedy anchor (top-2 margins 15.4, 18.0 and 12.5 logits) are NInfer's
  (`ninfer_gemma4_assistant_real_test`, given an artifact). The same test verifies the anchor and
  the drafts in one pass and decodes them one at a time on another sequence: the logits are equal
  bit for bit.
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
