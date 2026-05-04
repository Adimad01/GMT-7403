#!/usr/bin/env python3
"""
Universal Reasoning Evaluation Runner (CoT, ToT, GoT).

Output schema:
  {"id", "question", "gold", "raw", "pred", "pred_norm", "correct", "gold_norm"}

Records and full reasoning logs are appended to disk ONE BY ONE (crash-safe).

Usage:
    python3 run_eval.py --strategy cot [options...]
    python3 run_eval.py --strategy tot [options...]
    python3 run_eval.py --strategy got [options...]
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from reasoning_prompt_eval import (
    build_cot_prompt,
    build_tot_prompt,
    build_got_prompt,
    extract_answer_from_cot,
    extract_json_like,
    serialize_graph_lines,
    normalize_direction,
)

VALID_DIRECTIONS = {
    "north", "south", "east", "west",
    "north-east", "north-west", "south-east", "south-west",
}

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    """Reads a JSONL file and returns a list of parsed record dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def append_record(path, record):
    """Appends a single JSON record as a new line to the given file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def append_log(path, text):
    """Appends a plain-text block to the given log file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# ── Extraction & Direction Helpers ────────────────────────────────────────────

def clean_direction(raw: Optional[str]) -> str:
    """Normalizes a raw answer string to a valid direction label or 'Invalid'."""
    if not raw:
        return "Invalid"
    d = normalize_direction(raw)
    return d if d in VALID_DIRECTIONS else "Invalid"

def extract_got_answer(text: str) -> Optional[str]:
    """Extracts the final answer from a Graph-of-Thoughts model response."""
    if not text:
        return None
    flat = re.sub(r"\s+", " ", text)
    m = re.search(r"Final\s+Answer\s*:\s*([^\n\.]+)", flat, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else None

# ── Model Call ────────────────────────────────────────────────────────────────

def call_model(client, model_name, prompt, max_tokens, temperature, retries=3, delay=2.0):
    """Sends a prompt to the model and returns the response text, retrying on failure."""
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [!] API error (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(delay)
    return ""

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_file(path, out_dir, strategy):
    """Computes accuracy and confusion matrix from a completed output file and saves metrics."""
    records = load_jsonl(path)
    correct  = sum(1 for r in records if r.get("correct"))
    total    = len(records)
    accuracy = correct / total if total else 0
    print(f"\n=== {strategy.upper()} Final Evaluation ===")
    print(f"Accuracy = {accuracy:.3f} ({correct}/{total})")

    metrics = {"accuracy": accuracy, "correct": correct, "total": total}
    with open(os.path.join(out_dir, f"metrics_eval_{strategy}_outputs.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    dirs = sorted(VALID_DIRECTIONS)
    cm = {g: {p: 0 for p in dirs} for g in dirs}
    for r in records:
        g, p = r.get("gold_norm"), r.get("pred_norm")
        if g in cm and p in cm[g]:
            cm[g][p] += 1
    with open(os.path.join(out_dir, f"confusion_eval_{strategy}_outputs.json"), "w") as f:
        json.dump(cm, f, indent=2)

    print("\n=== Current Results Summary ===")
    for mode in ["cot", "tot", "got"]:
        fname = f"metrics_eval_{mode}_outputs.json"
        p = os.path.join(out_dir, fname)
        if os.path.exists(p):
            m = json.load(open(p))
            print(f"  {mode.upper()}: {m['accuracy']:.3f} ({m['correct']}/{m['total']})")
        else:
            print(f"  {mode.upper()}: (not found)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Parses arguments, runs the selected reasoning strategy, and evaluates results."""
    parser = argparse.ArgumentParser(description="Universal Strategy Evaluation Runner")
    
    # 🔥 REQUIRED ARGUMENT: Strategy Selection
    parser.add_argument("--strategy", required=True, choices=["cot", "tot", "got"],
                        help="Choose the reasoning strategy to run.")
    
    parser.add_argument("--questions",   default="data/questions.jsonl")
    parser.add_argument("--answers",     default="evaluation_results/answers_eval_subset.jsonl")
    parser.add_argument("--kg",          default="kg_from_finetune.jsonl")
    parser.add_argument("--out-dir",     default="eval_outputs_real")
    parser.add_argument("--model-name",  default="gpt-oss")
    parser.add_argument("--base-url",    default="http://ollama.apps.crdig.ulaval.ca/v1")
    parser.add_argument("--api-key",     default="none")
    parser.add_argument("--max-tokens",  type=int,   default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit",       type=int,   default=0)
    parser.add_argument("--reset",       action="store_true")
    args = parser.parse_args()

    if not HAS_OPENAI:
        print("[ERROR] 'openai' package not found.  Run: pip install openai")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    strategy = args.strategy.lower()
    
    # Dynamically name output files based on strategy
    out_path = os.path.join(args.out_dir, f"eval_{strategy}_outputs.jsonl")
    log_path = os.path.join(args.out_dir, f"eval_{strategy}_reasoning_log.txt")

    print(f"🚀 RUNNING STRATEGY : {strategy.upper()}")
    print(f"Loading questions   : {args.questions}")
    questions_raw = load_jsonl(args.questions)

    print(f"Loading answers     : {args.answers}")
    answers_raw = load_jsonl(args.answers)
    gold_map = {str(a.get("id")): a.get("absoluteAnswer", "") for a in answers_raw}

    print(f"Loading KG          : {args.kg}")
    kg_triples = load_jsonl(args.kg)
    graph_text = serialize_graph_lines(kg_triples, max_lines=150)
    print(f"  -> {len(kg_triples)} triples")

    done_ids = set()
    n_correct = n_total = 0
    if args.reset:
        if os.path.exists(out_path): open(out_path, "w").close()
        if os.path.exists(log_path): open(log_path, "w").close()
        print(f"  -> Reset: starting {strategy.upper()} from scratch")
    elif os.path.exists(out_path):
        existing = load_jsonl(out_path)
        done_ids = {str(r["id"]) for r in existing}
        n_correct = sum(1 for r in existing if r.get("correct"))
        n_total = len(existing)
        if n_total > 0:
            print(f"  -> Auto-resuming {strategy.upper()}: {n_total} records already done")

    questions = [q for q in questions_raw if str(q.get("id")) in gold_map and str(q.get("id")) not in done_ids]
    if args.limit > 0:
        questions = questions[:args.limit]
    print(f"Questions to run    : {len(questions)}\n")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    for i, q in enumerate(questions, start=1):
        qid   = str(q.get("id"))
        qtext = q.get("question", "")
        gold  = gold_map.get(qid, "")

        # 1. Dynamically build prompt based on strategy
        if strategy == "cot":
            prompt = build_cot_prompt(qtext, graph_text)
        elif strategy == "tot":
            prompt = build_tot_prompt(qtext, graph_text)
        elif strategy == "got":
            prompt = build_got_prompt(qtext, graph_text)

        # 2. Call model
        raw_resp = call_model(client, args.model_name, prompt, args.max_tokens, args.temperature)

        # 3. Dynamically extract answer based on strategy
        raw_ans = None
        if strategy == "cot":
            raw_ans = extract_answer_from_cot(raw_resp)
        elif strategy == "tot":
            js = extract_json_like(raw_resp)
            if js and isinstance(js, dict):
                raw_ans = js.get("final_answer")
        elif strategy == "got":
            raw_ans = extract_got_answer(raw_resp)

        pred      = clean_direction(raw_ans)
        pred_norm = pred
        gold_norm = clean_direction(gold)
        correct   = (pred_norm != "Invalid" and gold_norm != "Invalid" and pred_norm == gold_norm)

        record = {
            "id": qid, "question": qtext, "gold": gold, "raw": raw_resp,
            "pred": pred, "pred_norm": pred_norm, "correct": correct, "gold_norm": gold_norm,
        }

        log_block = (
            f"==================================================\n"
            f"Strategy:    {strategy.upper()}\n"
            f"Question ID: {qid}\n"
            f"Question:    {qtext}\n"
            f"Expected:    {gold}\n"
            f"--- LLM RAW REASONING & OUTPUT ---\n"
            f"{raw_resp}\n"
            f"==================================================\n\n"
        )
        
        append_record(out_path, record)
        append_log(log_path, log_block)

        n_total += 1
        n_correct += int(correct)
        acc = n_correct / n_total

        status = "✔" if correct else "✘"
        print(f"[{i:>4}/{len(questions)}] {status}  id={qid:<6} gold={gold!r:<12} pred={pred!r:<12} acc={acc:.3f}")

    evaluate_file(out_path, args.out_dir, strategy)

if __name__ == "__main__":
    main()