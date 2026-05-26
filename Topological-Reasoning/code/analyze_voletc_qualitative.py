"""
Qualitative per-predicate accuracy analysis for Volet C topological grounding.
For each (strategy x KG-source) combination, reports:
  - accuracy per DE-9IM predicate
  - which relations are well predicted   (>= 90%)
  - which relations are mid-range        (30% < acc < 90%)
  - which relations are poorly predicted (<= 30%)
"""

import json
from collections import defaultdict

BASE = "/Users/imadeddinelassakeur/Downloads/GMT/Topological-Reasoning/code/results/"

CONFIGS = {
    "OSM / CoT": "voletc_dynamic_osm_neighborhood_details_spatial_relation_16_sample_cot_neighborhood_details_spatial_relation_16_sample_ckpt.json",
    "OSM / ToT": "voletc_dynamic_osm_neighborhood_details_spatial_relation_16_sample_tot_neighborhood_details_spatial_relation_16_sample_ckpt.json",
    "OSM / GoT": "voletc_dynamic_osm_neighborhood_details_spatial_relation_16_sample_got_neighborhood_details_spatial_relation_16_sample_ckpt.json",
    "Wikidata / CoT": "voletc_dynamic_wikidata_neighborhood_details_spatial_relation_16_sample_wikidata_cot_neighborhood_details_spatial_relation_16_sample_wikidata_ckpt.json",
    "Wikidata / ToT": "voletc_dynamic_wikidata_neighborhood_details_spatial_relation_16_sample_wikidata_tot_neighborhood_details_spatial_relation_16_sample_wikidata_ckpt.json",
    "Wikidata / GoT": "voletc_dynamic_wikidata_neighborhood_details_spatial_relation_16_sample_wikidata_got_neighborhood_details_spatial_relation_16_sample_wikidata_ckpt.json",
}

PREDICATES = ["within", "contains", "disjoint", "touches", "crosses", "overlaps"]

WELL_THRESHOLD   = 90.0   # >= 90%  → well predicted
POOR_THRESHOLD   = 30.0   # <= 30%  → poorly predicted


def load_per_predicate_accuracy(fpath: str) -> dict[str, float]:
    with open(fpath) as f:
        data = json.load(f)

    counts: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in data["results"]:
        pred = r["expected"]
        counts[pred]["total"] += 1
        if r["match"]:
            counts[pred]["correct"] += 1

    return {
        pred: (counts[pred]["correct"] / counts[pred]["total"] * 100)
        for pred in counts
    }


def classify(acc: float) -> str:
    if acc >= WELL_THRESHOLD:
        return "WELL (≥90%)"
    elif acc <= POOR_THRESHOLD:
        return "POOR (≤30%)"
    else:
        return "MID"


def print_separator(char="─", width=72):
    print(char * width)


def main():
    all_results: dict[str, dict[str, float]] = {}

    for config_name, fname in CONFIGS.items():
        all_results[config_name] = load_per_predicate_accuracy(BASE + fname)

    # ── Per-configuration detailed table ────────────────────────────────────
    for config_name, accs in all_results.items():
        print()
        print_separator("═")
        print(f"  {config_name}")
        print_separator("═")

        total_correct = sum(
            round(accs[p] / 100 * 16) for p in PREDICATES if p in accs
        )
        total_samples = 16 * len(PREDICATES)
        global_acc = total_correct / total_samples * 100

        print(f"  Global accuracy: {global_acc:.1f}%  ({total_correct}/{total_samples})")
        print()

        well, mid, poor = [], [], []
        print(f"  {'Predicate':<12} {'Correct/16':>10} {'Accuracy':>10}  {'Category'}")
        print_separator()
        for pred in PREDICATES:
            acc = accs.get(pred, 0.0)
            correct = round(acc / 100 * 16)
            cat = classify(acc)
            print(f"  {pred:<12} {correct:>6}/16    {acc:>6.1f}%   {cat}")
            if cat.startswith("WELL"):
                well.append(pred)
            elif cat.startswith("POOR"):
                poor.append(pred)
            else:
                mid.append(pred)

        print_separator()
        print(f"  ✅  Well predicted  (≥90%): {well if well else 'none'}")
        print(f"  ⚠️   Mid range (30-90%)   : {mid if mid else 'none'}")
        print(f"  ❌  Poorly predicted (≤30%): {poor if poor else 'none'}")

    # ── Cross-configuration summary heatmap (text) ───────────────────────────
    print()
    print()
    print_separator("═")
    print("  SUMMARY: accuracy per predicate across all configurations (%)")
    print_separator("═")

    col_w = 14
    header = f"  {'Predicate':<12}" + "".join(f"{c:>{col_w}}" for c in CONFIGS)
    print(header)
    print_separator()

    for pred in PREDICATES:
        row = f"  {pred:<12}"
        for config_name in CONFIGS:
            acc = all_results[config_name].get(pred, 0.0)
            tag = "✅" if acc >= WELL_THRESHOLD else ("❌" if acc <= POOR_THRESHOLD else "⚠️ ")
            cell = f"{acc:.0f}%{tag}"
            row += f"{cell:>{col_w}}"
        print(row)

    # ── Strategy-level insight ───────────────────────────────────────────────
    print()
    print_separator("═")
    print("  INSIGHT: which predicates are consistently well/poorly predicted")
    print_separator("═")

    for pred in PREDICATES:
        vals = [all_results[c].get(pred, 0.0) for c in CONFIGS]
        mean_acc = sum(vals) / len(vals)
        always_well = all(v >= WELL_THRESHOLD for v in vals)
        always_poor = all(v <= POOR_THRESHOLD for v in vals)
        label = (
            "→ consistently WELL predicted across all configs"
            if always_well
            else "→ consistently POORLY predicted across all configs"
            if always_poor
            else f"→ variable  (mean {mean_acc:.0f}%)"
        )
        print(f"  {pred:<12}  {label}")


if __name__ == "__main__":
    main()
