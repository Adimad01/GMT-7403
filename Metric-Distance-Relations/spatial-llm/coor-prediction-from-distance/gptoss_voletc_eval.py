"""
gptoss_voletc_eval.py — Volet C: KG + Reasoning Strategies Evaluation (Metric)
================================================================================
Runs all three reasoning strategies (CoT, ToT, GoT) grounded on a data-driven
Geographic Knowledge Graph for metric distance estimation.

The KG provides: city coordinates, anchor distances, corpus signals.
The LLM estimates d(A, B) — the direct distance is NEVER given.

Output per strategy:
  - ONE txt log file  (e.g. voletc_base_cot.txt)   — full readable log
  - ONE checkpoint    (e.g. voletc_base_cot_ckpt.json) — resume support

Usage:
  python gptoss_voletc_eval.py --strategy cot --output-dir ./results
  python gptoss_voletc_eval.py --strategy all --output-dir ./results
  python gptoss_voletc_eval.py --strategy got --max-rows 5 --output-dir ./results
"""

import os
import sys
import json
import argparse
import tempfile
import numpy as np
from tqdm import tqdm
from datetime import datetime

from geographic_kg import MetricKnowledgeGraph
from reasoning_strategies import (
    get_strategy, extract_distance, STRATEGY_MAP,
    BASE_URL, MODEL_NAME, llm_generate,
)
import requests


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _save_json_atomic(path: str, data):
    """Write JSON atomically via temp file + rename (crash-safe)."""
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_checkpoint(ckpt_path: str) -> dict:
    """Load checkpoint. Returns empty state if none."""
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"processed_keys": [], "results": []}


# ---------------------------------------------------------------------------
# EVALUATION LOOP
# ---------------------------------------------------------------------------
def evaluate_strategy(strategy, test_pairs, kg: MetricKnowledgeGraph,
                      output_dir: str, model_tag: str,
                      top_k_anchors: int = 15):
    """
    Run a single reasoning strategy on all test pairs.

    Produces:
      1) voletc_{tag}_{strategy}.txt       — human-readable log
      2) voletc_{tag}_{strategy}_ckpt.json — checkpoint for resume
    """
    strategy_name = strategy.name.lower()
    log_path = os.path.join(output_dir, f"voletc_{model_tag}_{strategy_name}.txt")
    ckpt_path = os.path.join(output_dir, f"voletc_{model_tag}_{strategy_name}_ckpt.json")

    # ---- Resume from checkpoint ----
    ckpt = _load_checkpoint(ckpt_path)
    processed_keys = set(tuple(k) for k in ckpt["processed_keys"])
    results = list(ckpt["results"])

    if processed_keys:
        print(f"  ♻️  Resuming — {len(processed_keys)} pairs already done.")

    # Open log file
    log_f = open(log_path, "a", encoding="utf-8")

    # Write header on first run only
    if not processed_keys:
        header = (
            f"{'=' * 80}\n"
            f"  VOLET C — {strategy_name.upper()} — {model_tag}\n"
            f"  Model: {MODEL_NAME} @ {BASE_URL}\n"
            f"  Test pairs: {len(test_pairs)}\n"
            f"  Anchors per pair: top-{top_k_anchors}\n"
            f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'=' * 80}\n\n"
        )
        log_f.write(header)
        log_f.flush()

    desc = f"[{model_tag}/{strategy_name}]"

    try:
        for idx, pair in enumerate(tqdm(test_pairs, desc=desc)):
            a = pair["a_name"]
            b = pair["b_name"]
            true_dist = pair["true_distance"]
            key = (a, b)

            if key in processed_keys:
                continue

            # Build KG evidence block
            evidence = kg.build_evidence_block(a, b, top_k=top_k_anchors)

            # Logger callback → writes to the txt file
            def row_logger(msg, _f=log_f):
                _f.write(msg + "\n")
                _f.flush()

            # Run reasoning
            try:
                predicted, trace = strategy.reason(a, b, evidence, log_fn=row_logger)
            except Exception as e:
                predicted = None
                trace = {"error": str(e)[:500]}
                tqdm.write(f"  ⚠️  Pair {idx} ({a} ↔ {b}) error: {str(e)[:100]}")

            # Compute error
            error = abs(predicted - true_dist) if predicted is not None else None

            # Running stats
            errors_so_far = [r["error"] for r in results if r["error"] is not None]
            if error is not None:
                errors_so_far.append(error)
            running_mae = np.mean(errors_so_far) if errors_so_far else float("nan")
            total_so_far = len(results) + 1
            valid_so_far = len(errors_so_far)

            # Write to log
            block = (
                f"\n{'─' * 70}\n"
                f"  Pair {idx+1}/{len(test_pairs)}  |  {a}  ↔  {b}\n"
                f"  True distance:      {true_dist:.1f} km\n"
                f"  Predicted distance: {f'{predicted:.1f} km' if predicted else 'FAILED'}\n"
                f"  Error:              {f'{error:.1f} km' if error is not None else 'N/A'}\n"
                f"  Running MAE:        {running_mae:.1f} km ({valid_so_far}/{total_so_far} valid)\n"
                f"{'─' * 70}\n"
            )
            log_f.write(block)
            log_f.flush()

            # Terminal output
            symbol = "✅" if (error is not None and error < 100) else ("⚠️" if error is not None else "❌")
            tqdm.write(
                f"  {idx+1:>3d} | {a:<30s} ↔ {b:<30s} | "
                f"true={true_dist:>7.1f} pred={f'{predicted:>7.1f}' if predicted else '   FAIL'} "
                f"err={f'{error:>7.1f}' if error is not None else '    N/A'} {symbol}  "
                f"MAE={running_mae:.1f}"
            )

            # Store result
            results.append({
                "a_name": a,
                "b_name": b,
                "true_distance": true_dist,
                "predicted_distance": predicted,
                "error": error,
                "strategy": strategy_name,
            })
            processed_keys.add(key)

            # Checkpoint every 5 pairs
            if len(results) % 5 == 0:
                _save_json_atomic(ckpt_path, {
                    "processed_keys": [list(k) for k in processed_keys],
                    "results": results,
                })

    finally:
        # Always save checkpoint on exit
        _save_json_atomic(ckpt_path, {
            "processed_keys": [list(k) for k in processed_keys],
            "results": results,
        })

        # Write final summary
        if results:
            errors = [r["error"] for r in results if r["error"] is not None]
            mae = np.mean(errors) if errors else float("nan")
            rmse = np.sqrt(np.mean(np.array(errors) ** 2)) if errors else float("nan")
            median_err = np.median(errors) if errors else float("nan")
            valid = len(errors)

            summary = (
                f"\n{'=' * 80}\n"
                f"  FINAL SUMMARY — {strategy_name.upper()} ({model_tag})\n"
                f"  MAE:    {mae:.2f} km\n"
                f"  RMSE:   {rmse:.2f} km\n"
                f"  Median: {median_err:.2f} km\n"
                f"  Valid:  {valid}/{len(results)}\n"
                f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'=' * 80}\n"
            )
            log_f.write(summary)

        log_f.close()

    # Print summary
    if results:
        errors = [r["error"] for r in results if r["error"] is not None]
        mae = np.mean(errors) if errors else float("nan")
        rmse = np.sqrt(np.mean(np.array(errors) ** 2)) if errors else float("nan")
        valid = len(errors)
        print(f"\n  ✅ {desc} MAE: {mae:.2f} km | RMSE: {rmse:.2f} km | "
              f"Valid: {valid}/{len(results)}")
        print(f"     Log:        {log_path}")
        print(f"     Checkpoint: {ckpt_path}")

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Volet C: Evaluate reasoning strategies (CoT/ToT/GoT) "
                    "for metric distance estimation with KG grounding."
    )
    parser.add_argument("--strategy", required=True,
                        choices=list(STRATEGY_MAP.keys()) + ["all"],
                        help="Reasoning strategy to run (or 'all')")
    parser.add_argument("--dataset", default="cities_with_coords.json",
                        help="Path to cities_with_coords.json")
    parser.add_argument("--test-ids", default="100_target_ids.json",
                        help="Path to 100_target_ids.json")
    parser.add_argument("--output-dir", default="./results",
                        help="Directory for output files")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max tokens to generate per LLM call")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit test pairs (for debugging)")
    parser.add_argument("--top-k-anchors", type=int, default=15,
                        help="Number of anchor cities in the evidence table")
    parser.add_argument("--model-tag", default="base",
                        help="Tag for output filenames (e.g. 'base', 'finetuned')")
    args = parser.parse_args()

    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = args.dataset if os.path.isabs(args.dataset) else os.path.join(base_dir, args.dataset)
    test_ids_path = args.test_ids if os.path.isabs(args.test_ids) else os.path.join(base_dir, args.test_ids)

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
    if not os.path.exists(test_ids_path):
        print(f"❌ Test IDs not found: {test_ids_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize KG
    print(f"📂 Loading dataset: {dataset_path}")
    print(f"📂 Loading test IDs: {test_ids_path}")
    kg = MetricKnowledgeGraph(dataset_path, test_ids_path)

    # Get test pairs
    test_pairs = kg.iter_test_pairs()
    if args.max_rows:
        test_pairs = test_pairs[:args.max_rows]
        print(f"   Truncated to {len(test_pairs)} test pairs.")
    print(f"   {len(test_pairs)} test pairs to evaluate.")

    # Verify Ollama API
    print(f"🤖 Model: {MODEL_NAME} @ {BASE_URL}")
    try:
        test_resp = requests.post(
            f"{BASE_URL.rstrip('/')}/api/generate",
            json={"model": MODEL_NAME, "prompt": "Say OK", "stream": False,
                  "options": {"num_predict": 10}},
            timeout=30,
        )
        test_text = test_resp.json().get("response", "").strip()[:50]
        print(f"   ✅ Ollama API reachable — test: \"{test_text}\"")
    except Exception as e:
        print(f"   ⚠️  Could not reach Ollama API ({e}). Will try anyway...")

    # Strategies to run
    if args.strategy == "all":
        strategy_names = list(STRATEGY_MAP.keys())
    else:
        strategy_names = [args.strategy]

    # Run
    all_results = {}
    for sname in strategy_names:
        print(f"\n{'='*70}")
        print(f"  Strategy: {sname.upper()} | Model: {args.model_tag} ({MODEL_NAME})")
        print(f"  Anchors: top-{args.top_k_anchors} | Temp: {args.temperature}")
        print(f"{'='*70}")

        strategy = get_strategy(
            sname, kg,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        results = evaluate_strategy(
            strategy, test_pairs, kg,
            args.output_dir, args.model_tag,
            top_k_anchors=args.top_k_anchors,
        )
        all_results[sname] = results

    # Final summary table
    print(f"\n{'='*70}")
    print("  VOLET C — METRIC DISTANCE ESTIMATION — RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Strategy':<8} {'Model':<12} {'MAE':>10} {'RMSE':>10} {'Median':>10} {'Valid':>8}")
    print(f"  {'-'*60}")
    for sname, res in all_results.items():
        if res:
            errors = [r["error"] for r in res if r["error"] is not None]
            mae = np.mean(errors) if errors else float("nan")
            rmse = np.sqrt(np.mean(np.array(errors) ** 2)) if errors else float("nan")
            median = np.median(errors) if errors else float("nan")
            valid = len(errors)
            total = len(res)
            print(f"  {sname.upper():<8} {args.model_tag:<12} "
                  f"{mae:>8.1f}km {rmse:>8.1f}km {median:>8.1f}km "
                  f"{valid:>4d}/{total}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
