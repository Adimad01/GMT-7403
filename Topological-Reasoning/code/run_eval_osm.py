"""
gptoss_voletc_eval_neighborhood_details_spatial_relation.py
================================================================================
Runs CoT, ToT, and GoT reasoning strategies grounded on the
dynamic OpenStreetMap (OSM) Knowledge Graph.

*** CONFIGURED FOR 21 INSTANCES (16 per predicate) ***
"""

import os
import sys
import json
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from strategies_osm import (
    get_strategy,
    VALID_PREDICATES,
    STRATEGY_MAP,
    BASE_URL,
    MODEL_NAME,
    GeographicKnowledgeGraph
)

# =========================================================
# EXPERIMENT CONFIG
# =========================================================
EXPERIMENT_SUFFIX = "neighborhood_details_spatial_relation_16_sample"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _save_json_atomic(path: str, data):
    import tempfile
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
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"processed_indices": [], "results": []}


# ---------------------------------------------------------------------------
# EVALUATION LOOP
# ---------------------------------------------------------------------------
def evaluate_strategy(strategy, df: pd.DataFrame, output_dir: str,
                      model_tag: str, kg: GeographicKnowledgeGraph = None):

    strategy_name = strategy.name.lower()

    log_path = os.path.join(
        output_dir,
        f"voletc_{model_tag}_{strategy_name}_{EXPERIMENT_SUFFIX}.txt"
    )

    ckpt_path = os.path.join(
        output_dir,
        f"voletc_{model_tag}_{strategy_name}_{EXPERIMENT_SUFFIX}_ckpt.json"
    )

    ckpt = _load_checkpoint(ckpt_path)
    processed_indices = set(ckpt.get("processed_indices", []))
    results = list(ckpt.get("results", []))

    if processed_indices:
        print(f"♻️ Resuming — {len(processed_indices)} rows already done.")

    log_f = open(log_path, "a", encoding="utf-8")

    if not processed_indices:
        header = (
            f"{'=' * 90}\n"
            f"  VOLET C — {strategy_name.upper()} — {model_tag.upper()} [{EXPERIMENT_SUFFIX}]\n"
            f"  Model: {MODEL_NAME} @ {BASE_URL}\n"
            f"  KG: Dynamic OSM API\n"
            f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'=' * 90}\n\n"
        )
        log_f.write(header)
        log_f.flush()

    desc = f"[{model_tag}/{strategy_name}]"

    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):

            # Use the dataframe's actual index for tracking, not the loop counter
            real_idx = row.name

            if real_idx in processed_indices:
                continue

            entity = {
                "place_name_subject": str(row.get("place_name_subject", "")).strip(),
                "place_name_object": str(row.get("place_name_object", "")).strip(),
                "placetype_subject": str(row.get("placetype_subject", "")).strip(),
                "placetype_object": str(row.get("placetype_object", "")).strip(),
                "geometry_type_subject": str(row.get("geometry_type_subject", "unknown")).strip(),
                "geometry_type_object": str(row.get("geometry_type_object", "unknown")).strip(),
                "relation_predicate": str(row.get("relation_predicate", "")).strip(),
                "vernacular_relation": str(row.get("vernacular_relation", "")).strip(),
                "sentence": str(row.get("Sentence", "")).strip(),
            }

            expected = str(row.get("spatial_relation", "")).lower().strip()

            def row_logger(msg: str):
                log_f.write(msg + "\n")
                log_f.flush()

            row_logger(f"\n{'=' * 90}")
            row_logger(f"ROW {real_idx} | {entity['place_name_subject']} → {entity['place_name_object']}")
            row_logger(f"Expected: {expected}")
            row_logger(f"{'=' * 90}")

            try:
                predicted, trace = strategy.reason(entity, log_fn=row_logger)
            except Exception as e:
                predicted = "invalid"
                row_logger(f"ERROR: {str(e)}")

            if predicted not in VALID_PREDICATES:
                predicted = "invalid"

            is_match = (expected == predicted)

            correct_so_far = sum(1 for r in results if r.get("match", False)) + (1 if is_match else 0)
            total_so_far = len(results) + 1
            running_acc = (correct_so_far / total_so_far) * 100

            log_f.write(
                f"\nRESULT | Expected={expected} | Predicted={predicted} | "
                f"{'CORRECT' if is_match else 'WRONG'} | Acc={running_acc:.2f}%\n"
            )

            tqdm.write(f"{real_idx} {expected} → {predicted} | acc={running_acc:.1f}%")

            results.append({
                "index": real_idx,
                "expected": expected,
                "predicted": predicted,
                "match": is_match,
            })

            processed_indices.add(real_idx)

            if len(results) % 16 == 0:  # Save more frequently since it's a small dataset
                _save_json_atomic(ckpt_path, {
                    "processed_indices": sorted(list(processed_indices)),
                    "results": results,
                })

    finally:
        _save_json_atomic(ckpt_path, {
            "processed_indices": sorted(list(processed_indices)),
            "results": results,
        })

        if results:
            rdf = pd.DataFrame(results)
            acc = rdf["match"].mean() * 100
            log_f.write(
                f"\nFINAL ACCURACY: {acc:.2f}% ({rdf['match'].sum()}/{len(rdf)})\n"
            )

        log_f.close()

    if results:
        rdf = pd.DataFrame(results)
        acc = rdf["match"].mean() * 100
        print(f"\n✅ {desc} Finished — Accuracy: {acc:.2f}%")
        print(f"   Log: {log_path}")
        print(f"   CKPT: {ckpt_path}")

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--strategy", required=True, choices=list(STRATEGY_MAP.keys()) + ["all"])
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)

    # Tag to differentiate this specific 21-instance run
    parser.add_argument("--model-tag", default=f"dynamic_osm_{EXPERIMENT_SUFFIX}")

    args = parser.parse_args()

    print(f"🧪 EXPERIMENT: {EXPERIMENT_SUFFIX}")

    # 1. Load the full dataset
    df_full = pd.read_csv(args.dataset)
    
    # Clean the spatial_relation column to ensure exact string matching
    df_full['spatial_relation_clean'] = df_full['spatial_relation'].astype(str).str.lower().str.strip()
    
    # 2. Perform Stratified Sampling (16 rows per valid predicate)
    print("\n📊 Performing Stratified Sampling (16 per predicate)...")
    sampled_dfs = []
    
    for pred in VALID_PREDICATES:
        # Filter for this specific predicate
        pred_df = df_full[df_full['spatial_relation_clean'] == pred]
        
        if len(pred_df) >= 16:
            # Sample exactly 3, using random_state so it picks the SAME 16 every time you run the script
            sampled = pred_df.sample(n=16, random_state=42) 
            sampled_dfs.append(sampled)
            print(f"  ✓ {pred.ljust(10)}: Found {len(pred_df)}, sampled 16.")
        else:
            print(f"  ⚠️ {pred.ljust(10)}: Only found {len(pred_df)} instances! Taking all.")
            sampled_dfs.append(pred_df)

    # Combine back into a single dataframe (should be exactly 21 rows)
    df = pd.concat(sampled_dfs)
    
    print(f"\n🎯 Final dataset size for evaluation: {len(df)} rows\n")

    kg = GeographicKnowledgeGraph("results/osm_cache.json")

    strategies = list(STRATEGY_MAP.keys()) if args.strategy == "all" else [args.strategy]

    for strat in strategies:
        print(f"\n🚀 Running {strat.upper()}")

        strategy_obj = get_strategy(strat, kg)

        evaluate_strategy(
            strategy_obj,
            df,
            args.output_dir,
            args.model_tag,
            kg
        )


if __name__ == "__main__":
    main()