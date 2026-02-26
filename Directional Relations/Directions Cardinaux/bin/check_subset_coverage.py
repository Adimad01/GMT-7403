import json
import argparse
from pathlib import Path


def read_result_ids(results_path: Path):
    res_ids = set()
    res_lines = 0
    unique_pairs = set()
    if not results_path.exists():
        return res_ids, res_lines, unique_pairs
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
                if qid is not None:
                    qid = str(qid)
                    res_ids.add(qid)
                    unique_pairs.add((float(temp) if temp is not None else None, qid))
                res_lines += 1
            except Exception:
                continue
    return res_ids, res_lines, unique_pairs


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
    parser = argparse.ArgumentParser(description="Check coverage of results against eval subset IDs")
    parser.add_argument('--results', default='experiment_results_gptoss_fewshot.jsonl', help='Path to results JSONL')
    parser.add_argument('--subset', default='evaluation_results/answers_eval_subset.jsonl', help='Path to subset IDs JSONL')
    parser.add_argument('--show-missing', action='store_true', help='Print missing IDs list')
    parser.add_argument('--show-extra', action='store_true', help='Print extra IDs list')
    parser.add_argument('--show-temps', action='store_true', help='Print unique temperatures present')
    args = parser.parse_args()

    results_path = Path(args.results)
    subset_path = Path(args.subset)

    res_ids, res_lines, unique_pairs = read_result_ids(results_path)
    subset_ids = read_subset_ids(subset_path)

    missing = sorted(int(x) for x in (subset_ids - res_ids))
    extra = sorted(int(x) for x in (res_ids - subset_ids))

    print(f"Subset IDs: {len(subset_ids)}")
    print(f"Result lines: {res_lines}")
    print(f"Unique (temp,id) pairs: {len(unique_pairs)}")
    print(f"Unique result IDs: {len(res_ids)}")
    print(f"Missing IDs count: {len(missing)}")
    print(f"Extra IDs count: {len(extra)}")

    if args.show_missing:
        print("Missing IDs:", missing)
    if args.show_extra:
        print("Extra IDs:", extra)
    if args.show_temps:
        temps = sorted({t for (t, _) in unique_pairs if t is not None})
        print("Temperatures present:", temps)


if __name__ == '__main__':
    main()
