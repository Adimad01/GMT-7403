"""
build_history_report.py  (run from the repo root)
================================================================================
Mines git history for every commit that touched the voletc_* result checkpoints
and renders history.html: one table per domain showing how accuracy evolved
run after run (zero-shot per experiment, overall zero-shot / few-shot, eval-set
size). Cells are color-coded by accuracy.

Runs are NOT always comparable to each other — the eval split, the dataset, the
prompts, and even the answer-extraction code changed between some commits. The
table shows n (eval rows) and the commit subject so those regime changes are
visible; read jumps across them as methodology changes, not model changes.

Usage:
  python build_history_report.py
  python build_history_report.py --out history.html
"""
import argparse
import collections
import json
import os
import re
import subprocess

DOMAINS = [
    ("Topological", "Topological-Reasoning/code/results"),
    ("Cardinal",    "Cardinal-Reasoning/code/results"),
    ("Relative",    "Relative-Reasoning/code/results"),
]
EXP_ORDER = ["exp1_base", "exp2_ft_nokg", "exp3_ft_osmkg",
             "exp4_base_kg_input", "exp5_ft_kg_input", "exp6_base_kg_rag"]
EXP_SHORT = {"exp1_base": "Exp1", "exp2_ft_nokg": "Exp2", "exp3_ft_osmkg": "Exp3",
             "exp4_base_kg_input": "Exp4", "exp5_ft_kg_input": "Exp5",
             "exp6_base_kg_rag": "Exp6"}
_CKPT_RE = re.compile(r"voletc_(?P<tag>.+?)_(?P<strat>cot|tot|got)_.*_ckpt\.json$")
_FS_RE = re.compile(r"_fs(\d+)$")


def _git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout


def commits_touching_results():
    """[(sha, date, subject)] oldest → newest, deduplicated by tree state later."""
    out = _git("log", "--reverse", "--date=short", "--format=%H%x09%ad%x09%s",
               "--", "*/code/results/voletc_*")
    rows = []
    for line in out.strip().splitlines():
        sha, date, subj = line.split("\t", 2)
        rows.append((sha, date, subj))
    return rows


def ckpts_at(sha: str, results_dir: str):
    """{(shots, exp, strat): (accuracy%, n_rows)} for one domain at one commit."""
    paths = [p for p in _git("ls-tree", "-r", "--name-only", sha,
                             "--", results_dir).splitlines()
             if _CKPT_RE.search(os.path.basename(p))]
    cfg = {}
    for p in paths:
        m = _CKPT_RE.search(os.path.basename(p))
        tag, strat = m.group("tag"), m.group("strat")
        fs = _FS_RE.search(tag)
        shots = int(fs.group(1)) if fs else 0
        exp = _FS_RE.sub("", tag)
        blob = _git("show", f"{sha}:{p}")
        try:
            rows = json.loads(blob).get("results", [])
        except Exception:
            continue
        if rows:
            acc = sum(bool(r.get("match")) for r in rows) / len(rows) * 100
            cfg[(shots, exp, strat)] = (acc, len(rows))
    return cfg


def acc_color(p):
    if p >= 90: return "#1a7f37", "#eaf8ee"
    if p >= 75: return "#2da44e", "#f0fbf2"
    if p >= 60: return "#bf8700", "#fff8e6"
    if p >= 45: return "#d1842c", "#fff3e3"
    return "#cf222e", "#ffeef0"


def cell(v):
    if v is None:
        return "<td class='muted'>—</td>"
    fg, bg = acc_color(v)
    return f"<td style='background:{bg};color:{fg};font-weight:600'>{v:.0f}%</td>"


def domain_history_html(name, results_dir, commits):
    rows_html, prev_fingerprint = [], None
    for sha, date, subj in commits:
        cfg = ckpts_at(sha, results_dir)
        if not cfg:
            continue
        # skip commits that didn't change THIS domain's numbers
        fingerprint = tuple(sorted((k, round(v[0], 2), v[1]) for k, v in cfg.items()))
        if fingerprint == prev_fingerprint:
            continue
        prev_fingerprint = fingerprint

        # zero-shot per-experiment mean over strategies present
        per_exp = {}
        for e in EXP_ORDER:
            accs = [cfg[(0, e, s)][0] for s in ("cot", "tot", "got") if (0, e, s) in cfg]
            per_exp[e] = sum(accs) / len(accs) if accs else None
        zs = [v[0] for (sh, _, _), v in cfg.items() if sh == 0]
        fs = [v[0] for (sh, _, _), v in cfg.items() if sh > 0]
        ns = collections.Counter(v[1] for v in cfg.values())
        n_common = ns.most_common(1)[0][0]
        n_cfg = len(cfg)

        cells = "".join(cell(per_exp[e]) for e in EXP_ORDER)
        zs_c = cell(sum(zs) / len(zs)) if zs else "<td class='muted'>—</td>"
        fs_c = cell(sum(fs) / len(fs)) if fs else "<td class='muted'>—</td>"
        rows_html.append(
            f"<tr><td class='meta'>{date}<br><code>{sha[:7]}</code></td>"
            f"<td class='subj'>{subj}</td>"
            f"<td>{n_common}</td><td>{n_cfg}</td>{cells}{zs_c}{fs_c}</tr>")
    if not rows_html:
        return ""
    head = ("<tr><th>run</th><th>commit</th><th>n</th><th>cfgs</th>"
            + "".join(f"<th>{EXP_SHORT[e]}</th>" for e in EXP_ORDER)
            + "<th>ZS avg</th><th>FS avg</th></tr>")
    return (f"<div class='sec'><h2>{name}</h2>"
            f"<table>{head}{''.join(rows_html)}</table></div>")


CSS = """
body{margin:0;background:#fbfbfd;color:#1d1d1f;font-family:-apple-system,BlinkMacSystemFont,
 'SF Pro Text','Segoe UI',Roboto,Helvetica,Arial,sans-serif;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:48px 24px 80px}
h1{font-size:36px;letter-spacing:-.02em;margin:0 0 6px}
.lead{color:#86868b;font-size:16px;max-width:820px;margin:0 0 10px}
.sec{margin-top:44px}
.sec h2{font-size:20px;margin:0 0 12px}
table{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e8e8ed;
 border-radius:12px;overflow:hidden;font-size:13px}
th{background:#f5f5f7;color:#86868b;font-weight:600;font-size:11px;letter-spacing:.03em;
 padding:8px 9px;text-align:center}
td{padding:8px 9px;text-align:center;border-top:1px solid #f0f0f4}
td.meta{white-space:nowrap;color:#86868b;font-size:11px}
td.subj{text-align:left;max-width:330px;font-size:12px}
code{background:#f0f0f4;border-radius:4px;padding:1px 5px;font-size:11px}
.muted{color:#c7c7cc}
.note{background:#fff8e6;border:1px solid #f0e0b0;border-radius:10px;padding:12px 16px;
 font-size:13px;color:#6b5900;margin:14px 0 0}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="history.html")
    args = ap.parse_args()

    commits = commits_touching_results()
    sections = [domain_history_html(n, d, commits) for n, d in DOMAINS]

    page = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Results History</title><style>{CSS}</style></head><body>"
            f"<div class='wrap'><h1>Results history</h1>"
            f"<p class='lead'>Every run that changed the result checkpoints, oldest first. "
            f"Per experiment: zero-shot accuracy averaged over CoT/ToT/GoT. "
            f"<b>n</b> = eval rows per config · <b>cfgs</b> = configs present · "
            f"ZS/FS avg = mean over all zero-/few-shot configs.</p>"
            f"<div class='note'>⚠ Runs are not always comparable: eval splits, datasets, "
            f"prompts and answer-extraction code changed between some commits (see the "
            f"commit subjects and the n column). Jumps across such commits reflect "
            f"methodology fixes, not model behavior.</div>"
            + "".join(sections)
            + "</div></body></html>")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
