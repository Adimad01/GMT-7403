"""
analyze_gptoss_directions_relatives.py — Relative Direction (Circular Path) Evaluation
========================================================================================
Evaluates Base GPT-OSS and Fine-tuned GPT-OSS (LoRA) on the relative spatial
reasoning task: predict which object a subject ends up at after a sequence of
clockwise / counter-clockwise moves on a circular path of 12 objects.

File selection — all from the Mar 28 original runs (clean, complete outputs):
  Base      zero-shot : results_openai_gpt_oss_20b_zeroshot_live.json
  Base      few-shot  : results_openai_gpt_oss_20b_fewshot_k3_live.json
  LoRA      zero-shot : results_openai_gpt_oss_20b_lora_zeroshot_live.json
  LoRA      few-shot  : results_openai_gpt_oss_20b_lora_fewshot_k3_live.json

The "improved" prompt variants from Mar 31 are NOT used:
  - Base improved : regressed from ~83/87% to ~60% (broken prompt)
  - LoRA improved_v2 : truncated outputs with inflated correct flags

The pre-computed `correct` field from each file is used directly (trusting the
original evaluation script). For error analysis, the model's answer is extracted
from model_answer / model_raw using the `final` channel tag when available.

Saves one txt results file per model:
  results_base_gptoss.txt
  results_finetuned_gptoss.txt

Run from evaluation_results/:
    python analyze_gptoss_directions_relatives.py
"""

import json
import re
from collections import Counter
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).parent

MODELS = {
    "Base GPT-OSS": {
        "configs": {
            "zero-shot": EVAL_DIR / "results_openai_gpt_oss_20b_zeroshot_live.json",
            "few-shot":  EVAL_DIR / "results_openai_gpt_oss_20b_fewshot_k3_live.json",
        },
        "out": EVAL_DIR / "results_base_gptoss.txt",
    },
    "Fine-tuned GPT-OSS (LoRA)": {
        "configs": {
            "zero-shot": EVAL_DIR / "results_openai_gpt_oss_20b_lora_zeroshot_live.json",
            "few-shot":  EVAL_DIR / "results_openai_gpt_oss_20b_lora_fewshot_k3_live.json",
        },
        "out": EVAL_DIR / "results_finetuned_gptoss.txt",
    },
}

TOP_N = 10  # most-missed answers to show


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_results(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        for k in ("results", "data", "records"):
            if k in raw and isinstance(raw[k], list):
                return raw[k]
        for v in raw.values():
            if isinstance(v, list):
                return v
    if isinstance(raw, list):
        return raw
    return []


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER EXTRACTION  (for error analysis only; accuracy uses the `correct` field)
# ─────────────────────────────────────────────────────────────────────────────
_FINAL_RE  = re.compile(r"<\|channel\|>final<\|message\|>(.+?)(?:<\||\Z)", re.DOTALL)
_BOLD_RE   = re.compile(r"\*\*(.+?)\*\*")


def _extract_pred(model_answer: str, model_raw: str) -> str | None:
    """
    Extract the model's predicted final answer.
    Priority:
      1. model_answer with no channel tag → clean answer, strip noise
      2. final-channel tag in model_raw
      3. last bold token in model_answer
    """
    ma = str(model_answer or "").strip()
    mr = str(model_raw or "")

    if not ma:
        return None

    # Case 1: clean model_answer (no channel tag)
    if "<|channel|>" not in ma:
        # Remove trailing "." and common sentence wrappers
        # Then try to isolate the actual object name
        # Bold pattern: **answer**
        bold = _BOLD_RE.findall(ma)
        if bold:
            return bold[-1].strip().rstrip(".")
        # Plain clean answer (short, no sentences)
        if len(ma) < 80 and "\n" not in ma:
            return ma.rstrip(".")
        return None  # too verbose to extract reliably

    # Case 2: final-channel tag in model_raw
    m = _FINAL_RE.search(mr)
    if m:
        return m.group(1).strip().rstrip(".")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(records: list[dict]) -> dict:
    correct  = sum(1 for r in records if r.get("correct") is True)
    total    = len(records)
    wrong    = total - correct

    # error analysis on wrong records
    missed_expected: Counter = Counter()
    wrong_preds: list[tuple[str, str | None]] = []

    for r in records:
        if r.get("correct") is True:
            continue
        exp  = str(r.get("expected_answer", "")).strip().rstrip(".")
        pred = _extract_pred(str(r.get("model_answer", "")),
                             str(r.get("model_raw", "")))
        missed_expected[exp] += 1
        wrong_preds.append((exp, pred))

    # count invalid (no extractable prediction) — only among wrong records
    invalid_count = sum(1 for _, p in wrong_preds if p is None)

    return {
        "total":           total,
        "correct":         correct,
        "wrong":           wrong,
        "invalid_in_wrong": invalid_count,
        "missed_expected": missed_expected,
        "wrong_preds":     wrong_preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_report(model_name: str, model_cfg: dict) -> str:
    SEP  = "═" * 78
    DASH = "─" * 78

    config_metrics = {}
    for shot_label, path in model_cfg["configs"].items():
        records = load_results(path)
        config_metrics[shot_label] = (compute_metrics(records), path.name)

    # best config = highest accuracy
    best_label = max(config_metrics,
                     key=lambda k: config_metrics[k][0]["correct"] /
                                   config_metrics[k][0]["total"])
    best_m, best_fname = config_metrics[best_label]

    lines = [
        SEP,
        f"  Relative Direction Reasoning — {model_name}",
        f"  Task    : Predict the final object after a sequence of",
        f"            clockwise / counter-clockwise moves on a 12-object circular path",
        f"  Metric  : Exact-match accuracy (pre-computed `correct` field per record)",
        SEP,
    ]

    # ── accuracy table ────────────────────────────────────────────────────────
    lines += [
        "",
        DASH,
        f"  {'Config':<12}  {'N':>5}  {'Correct':>8}  {'Wrong':>7}  {'Accuracy':>9}",
        "  " + "─" * 50,
    ]
    for shot_label, (m, fname) in config_metrics.items():
        acc     = m["correct"] / m["total"] * 100
        is_best = " ◀" if shot_label == best_label else ""
        lines.append(
            f"  {shot_label:<12}  {m['total']:>5}  {m['correct']:>8}  "
            f"{m['wrong']:>7}  {acc:>8.2f}%{is_best}"
        )
    lines += [
        "  " + "─" * 50,
        f"  Best config: {best_label}  "
        f"({best_m['correct']}/{best_m['total']} = "
        f"{best_m['correct']/best_m['total']*100:.2f}%)",
    ]

    # ── accuracy comparison bar ───────────────────────────────────────────────
    lines += ["", DASH, "  Accuracy Comparison (each █ ≈ 2.5%)"]
    for shot_label, (m, _) in config_metrics.items():
        acc = m["correct"] / m["total"] * 100
        bar = "█" * int(acc / 2.5)
        lines.append(f"  {shot_label:<12}  {acc:>6.2f}%  {bar}")

    # ── error breakdown (both configs) ───────────────────────────────────────
    lines += [
        "",
        DASH,
        "  Error Breakdown",
        "  " + "─" * 50,
        f"  {'Config':<12}  {'Correct':>8}  {'Wrong':>7}  {'No-extract':>11}",
        "  " + "─" * 50,
    ]
    for shot_label, (m, _) in config_metrics.items():
        acc = m["correct"] / m["total"] * 100
        lines.append(
            f"  {shot_label:<12}  {m['correct']:>7} ({acc:>5.1f}%)  "
            f"{m['wrong']:>6} ({m['wrong']/m['total']*100:>4.1f}%)  "
            f"{m['invalid_in_wrong']:>10}"
        )

    # ── most missed expected answers (best config) ────────────────────────────
    missed = best_m["missed_expected"]
    if missed:
        lines += [
            "",
            DASH,
            f"  Most Missed Expected Answers — Best Config ({best_label})",
            f"  (objects the model most often failed to predict correctly)",
            f"  {'Expected Answer':<38}  {'Times Missed':>12}",
            "  " + "─" * 54,
        ]
        for ans, cnt in missed.most_common(TOP_N):
            lines.append(f"  {ans:<38}  {cnt:>12}")

    # ── sample wrong predictions (best config) ───────────────────────────────
    extractable = [(exp, pred) for exp, pred in best_m["wrong_preds"]
                   if pred is not None]
    if extractable:
        lines += [
            "",
            DASH,
            f"  Sample Wrong Predictions — Best Config ({best_label})",
            f"  {'Expected':<35}  {'Model Predicted':<35}",
            "  " + "─" * 74,
        ]
        for exp, pred in extractable[:TOP_N]:
            lines.append(f"  {exp:<35}  {str(pred):<35}")

    # ── source files ──────────────────────────────────────────────────────────
    lines += [
        "",
        DASH,
        "  Source Files",
        "  " + "─" * 70,
    ]
    for shot_label, (_, fname) in config_metrics.items():
        lines.append(f"  {shot_label:<12}  {fname}")

    lines += ["", SEP]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    for model_name, cfg in MODELS.items():
        report = build_report(model_name, cfg)
        cfg["out"].write_text(report, encoding="utf-8")
        print(f"Saved → {cfg['out'].name}")
        print(report)


if __name__ == "__main__":
    main()
