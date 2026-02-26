import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, asin


def read_any(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s == ".jsonl" or s == ".ndjson":
        return pd.read_json(path, lines=True)
    if s == ".json":
        # try lines first
        try:
            return pd.read_json(path, lines=True)
        except Exception:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                return pd.DataFrame(payload)
            return pd.json_normalize(payload)
    if s == ".csv":
        return pd.read_csv(path)
    # fallback
    try:
        return pd.read_json(path)
    except Exception:
        return pd.read_csv(path)


def haversine_meters(lat1, lon1, lat2, lon2):
    # all args can be arrays or scalars
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371000.0
    return R * c


def detect_coord_columns(df: pd.DataFrame):
    cols = [c.lower() for c in df.columns]
    # look for pairs like (pred_lat, pred_lon) and (true_lat,true_lon) or (latitude, longitude) with prefixes
    def find_pair(prefixes):
        for p in prefixes:
            lat = None
            lon = None
            for c in df.columns:
                lc = c.lower()
                if lc == f"{p}lat" or lc == f"{p}_lat" or lc == f"{p}latitude":
                    lat = c
                if lc == f"{p}lon" or lc == f"{p}_lon" or lc == f"{p}longitude":
                    lon = c
            if lat and lon:
                return lat, lon
        return None

    # common prefix pairs
    pred = find_pair(["pred", "predicted", "p_"]) or find_pair(["pred_"]) 
    true = find_pair(["true", "gt", "target", "t_"]) or find_pair(["true_"])

    # fallback to generic latitude/longitude pairs
    if not pred:
        # look for any lat/lon pair with same prefix (e.g., pred_lat, pred_lon) already covered; else try 'latitude' and 'longitude'
        if any(k in cols for k in ["latitude", "longitude"]):
            lat = next((c for c in df.columns if c.lower() == "latitude"), None)
            lon = next((c for c in df.columns if c.lower() == "longitude"), None)
            if lat and lon:
                true = (lat, lon) if not true else true
    # lastly look for columns named lat/lon with suffixes
    # We'll return dict of candidates
    return {"pred": pred, "true": true}


def summarize_file(path: Path):
    try:
        df = read_any(path)
    except Exception as e:
        return {"file": path.name, "error": str(e)}
    rec = {"file": path.name, "n_rows": len(df)}

    # numeric summary (count, mean, median, std)
    num = df.select_dtypes(include=[np.number])
    for c in num.columns:
        rec[f"{c}::mean"] = float(df[c].mean()) if len(df) > 0 else np.nan
        rec[f"{c}::median"] = float(df[c].median()) if len(df) > 0 else np.nan
        rec[f"{c}::std"] = float(df[c].std()) if len(df) > 1 else np.nan

    # detect coordinates
    coords = detect_coord_columns(df)
    if coords.get("pred") and coords.get("true"):
        pl, pol = coords["pred"]
        tl, tol = coords["true"]
        try:
            dists = haversine_meters(df[pl].astype(float), df[pol].astype(float), df[tl].astype(float), df[tol].astype(float))
            rec["distance::mean_m"] = float(np.nanmean(dists))
            rec["distance::median_m"] = float(np.nanmedian(dists))
            rec["distance::rmse_m"] = float(np.sqrt(np.nanmean(dists**2)))
            rec["distance::mae_m"] = float(np.nanmean(np.abs(dists)))
            # thresholds
            for t in (5, 10, 50, 100):
                rec[f"within_{t}m"] = int(np.sum(dists <= t))
                rec[f"within_{t}m_pct"] = float(100.0 * np.sum(dists <= t) / len(dists)) if len(dists) > 0 else np.nan
        except Exception:
            pass
    return rec


def generate(results_dir: Path, out_prefix: str = None):
    results_dir = results_dir.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    files = sorted([p for p in results_dir.iterdir() if p.suffix.lower() in ('.json', '.jsonl', '.ndjson', '.csv')])
    if not files:
        print("No result files found.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_prefix is None:
        out_prefix = f"spatial_awareness_report_{timestamp}"

    records = []
    details = []
    for p in files:
        rec = summarize_file(p)
        records.append(rec)
        # also capture some detail (head) for HTML
        try:
            df = read_any(p)
            head_html = df.head(10).to_html(index=False, classes='table table-sm', na_rep='')
            details.append(f"<h2>{p.name}</h2>\n{head_html}")
        except Exception as e:
            details.append(f"<h2>{p.name} — failed head: {e}</h2>")

    df_rec = pd.DataFrame(records).set_index('file')
    out_csv = results_dir / f"{out_prefix}_comparison.csv"
    df_rec.to_csv(out_csv)

    html = ["<html><head><meta charset='utf-8'><title>Spatial Awareness report</title>",
            "<style>body{font-family:Arial,Helvetica,sans-serif;padding:18px}table{border-collapse:collapse;margin-bottom:18px}table,th,td{border:1px solid #ddd;padding:6px}th{background:#f6f8fa}</style></head><body>"]
    html.append(f"<h1>Spatial Awareness report — {timestamp}</h1>")
    html.append(f"<h3>Scanned: {results_dir}</h3>")
    html.append("<h2>Summary comparison</h2>")
    html.append(df_rec.to_html(classes='table table-bordered', na_rep=''))
    html.append("<h2>Per-file sample (top 10 rows)</h2>")
    html.extend(details)
    html.append("</body></html>")

    out_html = results_dir / f"{out_prefix}.html"
    out_html.write_text('\n'.join(html), encoding='utf-8')

    print(f"Wrote: {out_csv}\nWrote: {out_html}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--out-prefix", type=str, default=None)
    args = parser.parse_args()
    base = Path(__file__).parent
    results_dir = Path(args.results_dir) if args.results_dir else base / "outputs_gpt_oss_30"
    generate(results_dir, out_prefix=args.out_prefix)
