import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


def read_result_file(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if ext == ".json":
        try:
            return pd.read_json(path)
        except Exception:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                return pd.DataFrame(payload)
            return pd.json_normalize(payload)
    # fallback: try pandas reader
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Unsupported file type or failed to read {path}: {e}")


def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return pd.DataFrame()
    return numeric.agg(["count", "mean", "median", "std", "min", "max"]).transpose()


def generate_report(results_dir: Path, out_prefix: str = None):
    results_dir = results_dir.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    patterns = ["*.jsonl", "*.ndjson", "*.json", "*.csv"]
    files = []
    for p in patterns:
        files.extend(sorted(results_dir.glob(p)))
    if not files:
        print(f"No result files found in {results_dir}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_prefix is None:
        out_prefix = f"report_{timestamp}"

    metrics_records = []
    detail_sections = []

    # Collect set of numeric column names across files to build a comparison table
    numeric_columns = set()

    for path in files:
        name = path.name
        try:
            df = read_result_file(path)
        except Exception as e:
            detail_sections.append(f"<h2>{name} — failed to read: {e}</h2>")
            continue

        n = len(df)
        numeric = df.select_dtypes(include=[np.number])
        numeric_columns.update(numeric.columns.tolist())

        rec = {"file": name, "n_rows": n}
        # mean for every numeric column
        for col in numeric.columns:
            try:
                rec[f"{col}::mean"] = float(df[col].mean()) if n > 0 else float("nan")
                rec[f"{col}::median"] = float(df[col].median()) if n > 0 else float("nan")
                rec[f"{col}::std"] = float(df[col].std()) if n > 1 else float("nan")
            except Exception:
                rec[f"{col}::mean"] = float("nan")
        metrics_records.append(rec)

        # detailed html: describe numeric, head, and top value_counts for non-numeric
        sec = [f"<h2>{name}</h2>"]
        if not df.empty:
            try:
                sec.append("<h3>Numeric summary</h3>")
                sec.append(summarize_numeric(df).to_html(classes='table table-striped', na_rep=''))
            except Exception as e:
                sec.append(f"<pre>Numeric summary failed: {e}</pre>")
            try:
                sec.append("<h3>Top rows</h3>")
                sec.append(df.head(10).to_html(classes='table table-sm', index=False, na_rep=''))
            except Exception as e:
                sec.append(f"<pre>Head render failed: {e}</pre>")
            # non-numeric columns value counts
            obj_cols = df.select_dtypes(exclude=[np.number]).columns
            if len(obj_cols) > 0:
                sec.append("<h3>Top value counts (non-numeric)</h3>")
                for c in obj_cols:
                    try:
                        vc = df[c].value_counts(dropna=False).head(10).rename_axis(c).reset_index(name='count')
                        sec.append(f"<h4>{c}</h4>")
                        sec.append(vc.to_html(index=False, classes='table table-sm'))
                    except Exception:
                        pass
        else:
            sec.append("<p><em>Empty dataframe</em></p>")

        detail_sections.append('\n'.join(sec))

    # Build comparison dataframe
    if metrics_records:
        metrics_df = pd.DataFrame(metrics_records).set_index('file')
    else:
        metrics_df = pd.DataFrame()

    # Save CSV comparison
    csv_path = results_dir / f"{out_prefix}_comparison.csv"
    metrics_df.to_csv(csv_path, index=True)

    # Build HTML
    html_parts = ["<html><head><meta charset='utf-8'><title>Evaluation report</title>",
                  "<style>body{font-family:Arial,Helvetica,sans-serif;padding:20px}table{border-collapse:collapse;margin-bottom:20px}table,th,td{border:1px solid #ddd;padding:6px}th{background:#f6f8fa}</style></head><body>"]
    html_parts.append(f"<h1>Evaluation report — {timestamp}</h1>")
    html_parts.append(f"<h2>Scanned directory: {results_dir}</h2>")
    html_parts.append("<h2>Summary comparison</h2>")
    if not metrics_df.empty:
        html_parts.append(metrics_df.to_html(classes='table table-bordered', na_rep=''))
    else:
        html_parts.append("<p><em>No numeric metrics collected.</em></p>")

    html_parts.append("<h2>Per-file details</h2>")
    html_parts.extend(detail_sections)
    html_parts.append("</body></html>")

    html_path = results_dir / f"{out_prefix}.html"
    html_path.write_text('\n'.join(html_parts), encoding='utf-8')

    print(f"Wrote comparison CSV: {csv_path}")
    print(f"Wrote HTML report: {html_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate an HTML comparison report over result files in a folder")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Path to results directory (defaults to ./results next to this script)")
    p.add_argument("--out-prefix", type=str, default=None, help="Prefix for output files")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    base = Path(__file__).parent
    results_dir = Path(args.results_dir) if args.results_dir else base / "results"
    try:
        generate_report(results_dir, out_prefix=args.out_prefix)
    except Exception as e:
        print(f"Failed to generate report: {e}")
