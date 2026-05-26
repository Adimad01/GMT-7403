"""
clean_and_rerun.py
==================
Loops over all 6 improved-version checkpoints, removes invalid/missing
predictions, and re-runs the affected experiments until every checkpoint
has exactly 96 valid predictions.
"""

import json
import os
import subprocess
import sys
import time

# ── Config ────────────────────────────────────────────────────────────────────
CODE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(CODE_DIR, "results")
DATASET      = os.path.join(CODE_DIR, "..", "dataset", "triplet_update_v3_30.csv")
PYTHON       = os.path.join(CODE_DIR, "venv", "bin", "python3")
VALID_PREDS  = {"disjoint", "touches", "crosses", "within", "contains", "overlaps", "equals"}
TARGET_ROWS  = 96
MAX_ROUNDS   = 10

EXPERIMENTS = [
    {
        "label":    "OSM/CoT",
        "runner":   "run_eval_osm.py",
        "strategy": "cot",
        "tag":      "dynamic_osm_improved_version",
        "ckpt":     "voletc_dynamic_osm_improved_version_cot_neighborhood_details_spatial_relation_16_sample_ckpt.json",
    },
    {
        "label":    "OSM/ToT",
        "runner":   "run_eval_osm.py",
        "strategy": "tot",
        "tag":      "dynamic_osm_improved_version",
        "ckpt":     "voletc_dynamic_osm_improved_version_tot_neighborhood_details_spatial_relation_16_sample_ckpt.json",
    },
    {
        "label":    "OSM/GoT",
        "runner":   "run_eval_osm.py",
        "strategy": "got",
        "tag":      "dynamic_osm_improved_version",
        "ckpt":     "voletc_dynamic_osm_improved_version_got_neighborhood_details_spatial_relation_16_sample_ckpt.json",
    },
    {
        "label":    "WD/CoT",
        "runner":   "run_eval_wikidata.py",
        "strategy": "cot",
        "tag":      "dynamic_wikidata_improved_version",
        "ckpt":     "voletc_dynamic_wikidata_improved_version_cot_neighborhood_details_spatial_relation_16_sample_wikidata_ckpt.json",
    },
    {
        "label":    "WD/ToT",
        "runner":   "run_eval_wikidata.py",
        "strategy": "tot",
        "tag":      "dynamic_wikidata_improved_version",
        "ckpt":     "voletc_dynamic_wikidata_improved_version_tot_neighborhood_details_spatial_relation_16_sample_wikidata_ckpt.json",
    },
    {
        "label":    "WD/GoT",
        "runner":   "run_eval_wikidata.py",
        "strategy": "got",
        "tag":      "dynamic_wikidata_improved_version",
        "ckpt":     "voletc_dynamic_wikidata_improved_version_got_neighborhood_details_spatial_relation_16_sample_wikidata_ckpt.json",
    },
]


def load_ckpt(path):
    if not os.path.exists(path):
        return {"processed_indices": [], "results": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_ckpt(path, data):
    import tempfile
    dir_name = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def check_invalids(exp):
    path = os.path.join(RESULTS_DIR, exp["ckpt"])
    data = load_ckpt(path)
    bad = [r for r in data["results"] if r.get("predicted") not in VALID_PREDS]
    return bad, data


def clean_invalids(exp):
    path = os.path.join(RESULTS_DIR, exp["ckpt"])
    data = load_ckpt(path)
    before = len(data["results"])
    data["results"] = [r for r in data["results"] if r.get("predicted") in VALID_PREDS]
    data["processed_indices"] = sorted([r["index"] for r in data["results"]])
    after = len(data["results"])
    save_ckpt(path, data)
    return before - after


def run_experiment(exp):
    cmd = [
        PYTHON,
        os.path.join(CODE_DIR, exp["runner"]),
        "--dataset",    DATASET,
        "--strategy",   exp["strategy"],
        "--output-dir", RESULTS_DIR,
        "--model-tag",  exp["tag"],
    ]
    print(f"  → Running: {' '.join(os.path.basename(c) if '/' in c else c for c in cmd)}")
    result = subprocess.run(cmd, cwd=CODE_DIR)
    return result.returncode == 0


def print_status(round_num):
    print(f"\n{'─'*60}")
    print(f"  Status after round {round_num}")
    print(f"{'─'*60}")
    all_clean = True
    for exp in EXPERIMENTS:
        path = os.path.join(RESULTS_DIR, exp["ckpt"])
        data = load_ckpt(path)
        total = len(data["results"])
        invalids = [r for r in data["results"] if r.get("predicted") not in VALID_PREDS]
        status = "✅" if total == TARGET_ROWS and len(invalids) == 0 else "⚠️ "
        if total != TARGET_ROWS or len(invalids) > 0:
            all_clean = False
        print(f"  {status} {exp['label']:10s}  done={total:3d}/{TARGET_ROWS}  invalid={len(invalids)}")
    print(f"{'─'*60}")
    return all_clean


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AUTO CLEAN & RERUN — IMPROVED VERSION")
    print(f"  Target: {TARGET_ROWS} valid predictions per experiment")
    print("=" * 60)

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_num}")
        print(f"{'='*60}")

        needs_rerun = []

        for exp in EXPERIMENTS:
            bad, _ = check_invalids(exp)
            path = os.path.join(RESULTS_DIR, exp["ckpt"])
            data = load_ckpt(path)
            total = len(data["results"])

            if len(bad) > 0 or total < TARGET_ROWS:
                removed = clean_invalids(exp)
                remaining = TARGET_ROWS - (total - removed)
                print(f"  {exp['label']:10s}  removed {removed} invalids → {remaining} rows to rerun")
                needs_rerun.append(exp)
            else:
                print(f"  {exp['label']:10s}  ✅ clean ({total}/{TARGET_ROWS})")

        if not needs_rerun:
            print("\n✅ All experiments are clean — no re-runs needed.")
            break

        print(f"\n  Re-running {len(needs_rerun)} experiment(s)...")
        for exp in needs_rerun:
            print(f"\n  [{exp['label']}]")
            success = run_experiment(exp)
            if not success:
                print(f"  ⚠️  {exp['label']} exited with error — will retry next round")

        all_clean = print_status(round_num)
        if all_clean:
            print("\n✅ All 6 experiments complete with 96 valid predictions each.")
            break
    else:
        print(f"\n⚠️  Reached max rounds ({MAX_ROUNDS}) — some invalids may persist.")
        print_status(MAX_ROUNDS)

    sys.exit(0)


if __name__ == "__main__":
    main()
