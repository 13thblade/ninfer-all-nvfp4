#!/usr/bin/env python3
"""The typed decision endpoints against a served model.

usage: decision_bench.py [--model PATH] [--server PATH] [--out DIR] [--lanes N]

Starts ninfer-serve on the model, then:
  1. answers: /v1/systemone and /v1/decisions on one short state, every question type;
  2. agreement: a choice question's probabilities against /v1/chat/completions' top log
     probabilities of the same prompt (greedy, no penalties, thinking off);
  3. latency: /v1/systemone with 1, 4 and 16 questions on a ~200-token and a ~10K-token state,
     a cold pass (the state prefilled) and warm passes (the state cached), per request wall time;
  4. concurrency: 16 questions while a 256-token chat completion streams, and the chat's decode
     rate with and without the decisions beside it.
Writes DIR/decisions.json and prints one line per measurement. Exits nonzero when an answer,
agreement or shape check fails.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("/workspace/ninfer-work")
PORT = int(os.environ.get("DECISION_BENCH_PORT", "18091"))
BASE = f"http://127.0.0.1:{PORT}"
RUNNING: list[subprocess.Popen] = []

SENTENCES = [
    "The customer wrote on Monday that the parcel arrived with a cracked screen.",
    "Support offered a replacement, but the customer asked for a refund instead.",
    "The order was paid by card and the invoice shows two identical charges.",
    "The warehouse confirmed that the package left on time with the usual courier.",
    "A follow-up message on Wednesday repeated the request and sounded impatient.",
    "The account has three earlier orders and no previous complaints.",
]


def call(path: str, body: dict | None = None, timeout: float = 600) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(BASE + path, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def call_status(path: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    request = urllib.request.Request(BASE + path, data=data,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def stop_servers(*_) -> None:
    for process in RUNNING:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=60)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    RUNNING.clear()


def handle_signal(number, _frame) -> None:
    stop_servers()
    raise SystemExit(128 + number)


def start_server(server: Path, model: Path, lanes: int, log_path: Path) -> subprocess.Popen:
    command = [str(server), str(model), "--host", "127.0.0.1", "--port", str(PORT),
               "--device", "0", "--max-context", "16384", "--kv-capacity", "65536",
               "--max-concurrency", str(lanes)]
    process = subprocess.Popen(command, stdout=log_path.open("w"), stderr=subprocess.STDOUT,
                               start_new_session=True)
    RUNNING.append(process)
    started = time.time()
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"server exited {process.returncode}; see {log_path}")
        try:
            call("/health", timeout=5)
            return process
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if time.time() - started > 900:
                raise RuntimeError("server start timed out") from None
            time.sleep(2)


def state_text(target_tokens: int) -> str:
    # About 15 tokens per sentence.
    lines = []
    while len(lines) * 15 < target_tokens:
        lines.append(f"Note {len(lines) + 1}: {SENTENCES[len(lines) % len(SENTENCES)]}")
    return "\n".join(lines)


def questions(count: int) -> dict:
    kinds = [
        ("choice", {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "payments and refunds", "shipping": "deliveries",
                                 "technical": "defects and repairs"}}),
        ("noul", {"type": "noul", "instructions": "Does the customer ask for a refund?"}),
        ("score", {"type": "score", "instructions": "How upset is the customer?",
                   "criteria": ["calm", "annoyed", "furious"]}),
        ("noul2", {"type": "noul", "instructions": "Is the parcel reported as damaged?",
                   "criteria": {"true": "the customer reports physical damage"}}),
    ]
    out = {}
    for index in range(count):
        name, body = kinds[index % len(kinds)]
        out[f"{name}_{index}"] = body
    return out


def check(condition: bool, label: str, failures: list[str]) -> None:
    if not condition:
        failures.append(label)
        print(f"CHECK FAILED: {label}", flush=True)


def answers_section(record: dict, failures: list[str]) -> None:
    state = state_text(200)
    body = {"state": state, "questions": questions(4)}
    status, response = call_status("/v1/systemone", body)
    record["systemone"] = {"status": status, "response": response}
    print(json.dumps({"systemone": response}, ensure_ascii=False), flush=True)
    check(status == 200, "systemone returns 200", failures)
    if status == 200:
        check(response["object"] == "structured_decision" and response["usage"]["output_tokens"] == 4,
              "systemone shape and usage", failures)
        for name, answer in response["answers"].items():
            diagnostics = response["diagnostics"][name]
            check(diagnostics["readout"] == "exact", f"{name} read from the exact readout", failures)
            check(0.0 < diagnostics["label_mass"] <= 1.0 + 1e-6, f"{name} label mass in (0, 1]",
                  failures)
            total = sum(answer["probabilities"].values())
            check(abs(total - 1.0) < 1e-6, f"{name} probabilities sum to 1 ({total})", failures)
            if answer["type"] == "choice":
                best = max(answer["probabilities"], key=answer["probabilities"].get)
                check(answer["choice"] == best, f"{name} choice is the most likely option", failures)
            elif answer["type"] == "score":
                expected = sum(float(k) * p for k, p in answer["probabilities"].items())
                check(abs(answer["score"] - expected) < 1e-6, f"{name} score is the expectation",
                      failures)
            else:
                check(abs(answer["noul"] - answer["probabilities"]["yes"]) < 1e-9,
                      f"{name} noul is P(yes)", failures)
    # The choice goes first: like System One's choice_0 it is then read alone, so both endpoints
    # must give it the same probabilities. A read sharing a prefill pass with others may round the
    # head's BF16 logits differently.
    openai = {"input": [{"role": "user", "content": [{"type": "input_text", "text": state}]}],
              "questions": [
                  {"type": "choice", "name": "team", "instructions": "Which team should handle this?",
                   "choices": [{"value": "billing", "description": "payments and refunds"},
                               {"value": "shipping", "description": "deliveries"},
                               {"value": "technical", "description": "defects and repairs"}]},
                  {"type": "predicate", "name": "refund", "instructions": "Does the customer ask for a refund?"},
                  {"type": "score", "instructions": "How upset is the customer?",
                   "levels": [{"label": "calm"}, {"label": "annoyed"}, {"label": "furious"}]}]}
    status, response = call_status("/v1/decisions", openai)
    record["decisions"] = {"status": status, "response": response}
    print(json.dumps({"decisions": response}, ensure_ascii=False), flush=True)
    check(status == 200, "decisions returns 200", failures)
    if status == 200:
        check(response["object"] == "decision" and len(response["answers"]) == 3 and
              response["answers"][2]["name"] is None, "decisions shape", failures)
        same = record["systemone"]["response"]["answers"]["choice_0"]["probabilities"]
        theirs = {entry["value"]: entry["probability"]
                  for entry in response["answers"][0]["probabilities"]}
        check(all(abs(same[k] - theirs[k]) < 1e-3 for k in same),
              "the same choice reads the same through both endpoints", failures)
    status, response = call_status("/v1/systemone", {"state": state, "questions": {
        "q": {"type": "choice", "instructions": "i", "criteria": {"only": None}}}})
    check(status == 400, "a one-option choice is refused", failures)
    status, response = call_status("/v1/systemone", {"state": state, "images": ["data:image/png;base64,AA=="],
                                                     "questions": {"q": {"type": "noul", "instructions": "i"}}})
    check(status == 501, "images are not implemented", failures)


def agreement_section(record: dict, failures: list[str]) -> None:
    state = state_text(200)
    question = {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "payments and refunds", "shipping": "deliveries",
                             "technical": "defects and repairs"}}
    decision = call("/v1/systemone", {"state": state, "questions": {"team": question}})
    # The prompt make_decision_read renders for this question: its system turn and user turn.
    system = ("You answer typed questions about a state with one label only: the letter of the best "
              "option, yes or no, or the number of the level. No words before or after the label.")
    text = (f"State:\n{state}\n\nQuestion: {question['instructions']}\nOptions:\n"
            "A. billing - payments and refunds\nB. shipping - deliveries\n"
            "C. technical - defects and repairs\nReply with a single letter (A, B or C) and nothing else.")
    chat = call("/v1/chat/completions", {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
        "max_tokens": 1, "temperature": 0,
        "presence_penalty": 0, "frequency_penalty": 0, "logprobs": True, "top_logprobs": 20,
        "graft": "", "chat_template_kwargs": {"enable_thinking": False}})
    top = chat["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    by_token = {entry["token"]: entry["logprob"] for entry in top}
    labels = {"A": "billing", "B": "shipping", "C": "technical"}
    chat_probabilities = {}
    for label, option in labels.items():
        if label in by_token:
            chat_probabilities[option] = math.exp(by_token[label])
    mass = sum(chat_probabilities.values())
    ours = decision["answers"]["team"]["probabilities"]
    diagnostics = decision["diagnostics"]["team"]
    record["agreement"] = {"decision": decision, "chat_top": top,
                           "chat_label_probabilities": chat_probabilities}
    print(json.dumps({"agreement": {"ours": ours, "label_mass": diagnostics["label_mass"],
                                    "chat_labels_in_top20": chat_probabilities,
                                    "chat_label_mass": mass}}), flush=True)
    check(len(chat_probabilities) >= 1, "the chat's top-20 shows at least one label", failures)
    if chat_probabilities:
        for option, probability in chat_probabilities.items():
            check(abs(probability - ours[option] * diagnostics["label_mass"]) < 2e-3,
                  f"{option}: readout {ours[option] * diagnostics['label_mass']:.5f} vs chat "
                  f"{probability:.5f}", failures)
        check(abs(mass - diagnostics["label_mass"]) < 5e-3 or len(chat_probabilities) < 3,
              "label mass agrees when every label is in the top 20", failures)


def timed(path: str, body: dict) -> tuple[float, dict]:
    begun = time.time()
    response = call(path, body)
    return time.time() - begun, response


def latency_section(record: dict, lanes: int) -> None:
    results = []
    for state_tokens in (200, 10000):
        for count in (1, 4, 16):
            state = state_text(state_tokens) + f"\nRequest marker {state_tokens}-{count}."
            body = {"state": state, "questions": questions(count)}
            cold, response = timed("/v1/systemone", body)
            warm = []
            for _ in range(3):
                wall, response = timed("/v1/systemone", body)
                warm.append(wall)
            masses = [d["label_mass"] for d in response["diagnostics"].values()]
            argmax = sum(1 for d in response["diagnostics"].values() if d["argmax_is_label"])
            row = {"state_tokens": response["usage"]["input_tokens"] // count, "questions": count,
                   "cold_s": round(cold, 3), "warm_s": [round(w, 3) for w in warm],
                   "warm_median_s": round(statistics.median(warm), 3),
                   "per_question_ms": round(1000 * statistics.median(warm) / count, 1),
                   "label_mass_min": round(min(masses), 4), "label_mass_mean": round(statistics.mean(masses), 4),
                   "argmax_is_label": f"{argmax}/{count}", "lanes": lanes}
            results.append(row)
            print(json.dumps({"latency": row}), flush=True)
    record["latency"] = results


def chat_stream(max_tokens: int, out: dict) -> None:
    body = {"messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper."}],
            "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    begun = time.time()
    response = call("/v1/chat/completions", body)
    out["wall"] = time.time() - begun
    out["timings"] = response.get("timings", {})


def concurrency_section(record: dict) -> None:
    state = state_text(2000) + "\nConcurrency marker."
    body = {"state": state, "questions": questions(16)}
    call("/v1/systemone", body)  # the state is cached
    alone_chat: dict = {}
    chat_stream(256, alone_chat)
    alone_decision, _ = timed("/v1/systemone", body)
    shared_chat: dict = {}
    thread = threading.Thread(target=chat_stream, args=(256, shared_chat))
    thread.start()
    time.sleep(0.5)
    shared_decision, _ = timed("/v1/systemone", body)
    thread.join()
    row = {"decision_alone_s": round(alone_decision, 3), "decision_beside_chat_s": round(shared_decision, 3),
           "chat_alone_tok_s": alone_chat["timings"].get("predicted_per_second"),
           "chat_alone_wall_s": round(alone_chat["wall"], 3),
           "chat_beside_decisions_tok_s": shared_chat["timings"].get("predicted_per_second"),
           "chat_beside_decisions_wall_s": round(shared_chat["wall"], 3)}
    record["concurrency"] = row
    print(json.dumps({"concurrency": row}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer"))
    parser.add_argument("--server", default=str(ROOT / "build/apps/ninfer-serve"))
    parser.add_argument("--out", default=os.environ.get("NINFER_JOB_DIR", "."))
    parser.add_argument("--lanes", type=int, default=4)
    arguments = parser.parse_args()
    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, handle_signal)
    record: dict = {"model": arguments.model, "lanes": arguments.lanes}
    failures: list[str] = []
    try:
        process = start_server(Path(arguments.server), Path(arguments.model), arguments.lanes,
                               out / "decisions.server.log")
        record["server_pid"] = process.pid
        answers_section(record, failures)
        agreement_section(record, failures)
        latency_section(record, arguments.lanes)
        concurrency_section(record)
    finally:
        stop_servers()
        (out / "decisions.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
    if failures:
        print(f"DECISION_BENCH_FAIL {len(failures)} check(s)", flush=True)
        return 1
    print("DECISION_BENCH_PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
