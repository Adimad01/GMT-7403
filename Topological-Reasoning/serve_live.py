#!/usr/bin/env python3
"""
serve_live.py — run from Topological-Reasoning/

Serves the full Topological-Reasoning tree over HTTP and exposes a live
/api/results endpoint that reads checkpoint JSON files in real time.

Usage:
    cd "/Users/imadeddinelassakeur/Documents/PhD Journey/GMT/Topological-Reasoning"
    python serve_live.py          # default port 8080
    python serve_live.py 8081     # custom port

Open: http://localhost:8080/dataset/dataset_report.html
"""

import http.server, json, os, sys
from datetime import datetime, timezone

PORT        = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "code", "results")
SUFFIX      = "neighborhood_details_spatial_relation_16_sample"
TOTAL       = 96

EXPERIMENTS = [
    ("exp1_base_gpu",               "Config 01 — Base GPT-OSS-20B",            "#0071e3"),
    ("exp2_finetuned_topo_gpu",     "Config 02 — Topo-LoRA",                    "#30d158"),
    ("exp3_finetuned_kg_in_gpu",    "Config 03 — OSM-KG LoRA",                  "#ff9f0a"),
    ("exp4_finetuned_osm_kg_gpu",   "Config 04 — Wikidata-KG LoRA",             "#bf5af2"),
    ("exp5_finetuned_enriched_gpu", "Config 05 — Topo-LoRA + extended budget",  "#ff375f"),
]


def read_ckpt(tag, strat):
    path = os.path.join(RESULTS_DIR, f"voletc_{tag}_{strat}_{SUFFIX}_ckpt.json")
    if not os.path.exists(path):
        return {"n": 0, "total": TOTAL, "accuracy": None, "status": "pending"}
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    n = len(results)
    if n == 0:
        return {"n": 0, "total": TOTAL, "accuracy": None, "status": "pending"}
    acc = round(sum(1 for r in results if r.get("match")) / n * 100, 1)
    status = "complete" if n >= TOTAL else "running"
    return {"n": n, "total": TOTAL, "accuracy": acc, "status": status}


def build_results():
    exps = []
    for tag, label, color in EXPERIMENTS:
        cot = read_ckpt(tag, "cot")
        tot = read_ckpt(tag, "tot")
        got = read_ckpt(tag, "got")
        statuses = {s["status"] for s in (cot, tot, got)}
        if "running" in statuses:
            overall = "running"
        elif statuses == {"complete"}:
            overall = "complete"
        else:
            overall = "pending"
        best = max(
            (s["accuracy"] for s in (cot, tot, got) if s["accuracy"] is not None),
            default=None
        )
        exps.append({
            "id": tag, "label": label, "color": color, "status": overall,
            "best_accuracy": best,
            "strategies": {"cot": cot, "tot": tot, "got": got},
        })
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_expected": TOTAL,
        "experiments": exps,
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/api/results":
            body = json.dumps(build_results(), indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def log_message(self, fmt, *args):
        if args and "/api/results" in str(args[0]):
            return
        super().log_message(fmt, *args)


os.chdir(BASE_DIR)
print(f"\n  Serving  : http://localhost:{PORT}/dataset/dataset_report.html")
print(f"  API      : http://localhost:{PORT}/api/results")
print(f"  Root     : {BASE_DIR}\n")
with http.server.HTTPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
