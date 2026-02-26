import json
import argparse
from pathlib import Path


def read_result_temps_by_id(results_path: Path):
    temps_by_id = {}
    lines = 0
    if not results_path.exists():
        return temps_by_id, lines
    with results_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                req = obj.get('request', {})
                qid = req.get('id')
                temp = req.get('temperature')
                if qid is not None and temp is not None:
                    qid = str(qid)
                    temps_by_id.setdefault(qid, set()).add(float(temp))
                lines += 1
            except Exception:
                continue
    return temps_by_id, lines


def read_subset_ids(subset_path: Path):
    subset_ids = set()
    if not subset_path.exists():
        return subset_ids
    with subset_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get('id')
                if sid is not None:
                    subset_ids.add(str(sid))
            except Exception:
                continue
    return subset_ids


def main():
    parser = argparse.ArgumentParser(description="Check per-ID temperature coverage in results")
    parser.add_argument('--results', default='experiment_results_gptoss_fewshot.jsonl', help='Path to results JSONL')
    parser.add_argument('--subset', default='evaluation_results/answers_eval_subset.jsonl', help='Path to subset IDs JSONL')
    parser.add_argument('--subset-only', action='store_true', default=True, help='Restrict check to subset IDs (default True)')
    parser.add_argument('--expected-temps', nargs='*', type=float, default=[0.25, 0.5, 1.0], help='Temperatures expected per ID')
    parser.add_argument('--show-missing-detail', action='store_true', help='Print detailed missing temps per ID')
    args = parser.parse_args()

    results_path = Path(args.results)
    subset_path = Path(args.subset)

    temps_by_id, total_lines = read_result_temps_by_id(results_path)
    subset_ids = read_subset_ids(subset_path)

    considered_ids = set(temps_by_id.keys())
    if args.subset_only and subset_ids:
        considered_ids = considered_ids & subset_ids

    expected = set(args.expected_temps)
    missing_per_id = {}
    coverage_counts = {t: 0 for t in expected}
    full_coverage = 0

    for qid in sorted(considered_ids, key=lambda x: int(x)):
        present = temps_by_id.get(qid, set())
        # Track present count per temp
        for t in expected:
            if t in present:
                coverage_counts[t] += 1
        # Missing temps
        missing = sorted(expected - present)
        if not missing:
            full_coverage += 1
        else:
            missing_per_id[qid] = missing

    print(f"Result lines: {total_lines}")
    print(f"IDs in results: {len(temps_by_id)}")
    print(f"Subset IDs: {len(subset_ids)}")
    print(f"Considered IDs: {len(considered_ids)}")
    print(f"Expected temperatures: {sorted(expected)}")
    print(f"Full coverage IDs: {full_coverage}")
    print(f"IDs missing one or more temps: {len(missing_per_id)}")
    for t in sorted(expected):
        print(f"IDs with temp {t}: {coverage_counts[t]}")

    if args.show_missing_detail and missing_per_id:
        print("\nMissing temps per ID (first 50):")
        cnt = 0
        for qid, missing in sorted(missing_per_id.items(), key=lambda kv: int(kv[0])):
            print(f"ID {qid}: missing {missing}")
            cnt += 1
            if cnt >= 50:
                print("...")
                break


if __name__ == '__main__':
    main()
