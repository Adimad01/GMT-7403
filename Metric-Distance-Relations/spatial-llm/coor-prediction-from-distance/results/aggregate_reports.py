import re
import json
from pathlib import Path


RESULTS_DIR = Path(__file__).parent
OUT_CSV = RESULTS_DIR / "comparative_summary.csv"
OUT_MD = RESULTS_DIR / "comparative_summary.md"


def parse_report(path: Path):
    info = {
        "report_file": path.name,
        "file_name": None,
        "eval_time": None,
        "total_samples": None,
        "valid_predictions": None,
        "prediction_rate": None,
        "mean_error_km": None,
        "median_error_km": None,
        "phase": None,
        "shot": None,
    }

    text = path.read_text(encoding="utf-8", errors="ignore")
    # Extract lines
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("File Name"):
            # File Name        : gen_dis_p1_3-shot.json
            m = re.search(r":\s*(.+)$", line)
            if m:
                fname = m.group(1).strip()
                info["file_name"] = fname
                # Derive phase/shot from file name if possible
                m2 = re.search(r"gen_dis_(p\d+)_(3-shot|zero-shot)\.json", fname)
                if m2:
                    info["phase"] = m2.group(1)
                    info["shot"] = m2.group(2)
        elif line.startswith("Evaluation Time"):
            m = re.search(r":\s*(.+)$", line)
            if m:
                info["eval_time"] = m.group(1).strip()
        elif line.startswith("Total Samples"):
            m = re.search(r":\s*(\d+)$", line)
            if m:
                info["total_samples"] = int(m.group(1))
        elif line.startswith("Valid Predictions"):
            m = re.search(r":\s*(\d+)$", line)
            if m:
                info["valid_predictions"] = int(m.group(1))
        elif line.startswith("Prediction Rate"):
            m = re.search(r":\s*([0-9\.]+)\%", line)
            if m:
                info["prediction_rate"] = float(m.group(1))
        elif line.startswith("Mean Error Dist"):
            m = re.search(r":\s*([0-9\.]+)\s*km", line)
            if m:
                info["mean_error_km"] = float(m.group(1))
        elif line.startswith("Median Error Dist"):
            m = re.search(r":\s*([0-9\.]+)\s*km", line)
            if m:
                info["median_error_km"] = float(m.group(1))

    return info


def collect_reports():
    reports = []
    for path in RESULTS_DIR.glob("report_gen_dis_*.txt"):
        try:
            reports.append(parse_report(path))
        except Exception:
            continue
    # Sort by phase then shot for readability
    def sort_key(r):
        # Phase pX -> number
        p = r.get("phase") or "p0"
        try:
            pn = int(p[1:])
        except Exception:
            pn = 0
        shot = r.get("shot") or ""
        shot_order = 0 if shot == "zero-shot" else 1
        return (pn, shot_order)
    reports.sort(key=sort_key)
    return reports


def write_csv(rows):
    headers = [
        "phase", "shot", "report_file", "file_name", "eval_time",
        "total_samples", "valid_predictions", "prediction_rate",
        "mean_error_km", "median_error_km",
    ]
    with OUT_CSV.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            vals = [r.get(h, "") for h in headers]
            f.write(",".join(str(v) if v is not None else "" for v in vals) + "\n")


def write_md(rows):
    headers = [
        "phase", "shot", "file_name", "eval_time",
        "total_samples", "valid_predictions", "prediction_rate",
        "mean_error_km", "median_error_km",
    ]
    # Markdown table
    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "---|" * len(headers) + "\n")
        for r in rows:
            vals = [r.get(h, "") for h in headers]
            f.write("| " + " | ".join(str(v) if v is not None else "" for v in vals) + " |\n")


def main():
    rows = collect_reports()
    if not rows:
        print("No reports found.")
        return
    write_csv(rows)
    write_md(rows)
    print(f"✅ Wrote CSV: {OUT_CSV}")
    print(f"✅ Wrote MD : {OUT_MD}")


if __name__ == "__main__":
    main()
