"""
build_report.py
================================================================================
ONE unified, Apple-minimalist HTML report = analysis + examples in a single file:

  1. Accuracy matrices (zero-shot / few-shot / delta)
  2. Difficulty (per-predicate + per-ambiguity-level)
  3. Overall confusion matrix
  4. Per-experiment breakdown — narrative paragraph + own confusion matrix + bars
  5. Hardest cases
  6. Examples Explorer — interactive: pick experiment · strategy · mode and read
     EVERY example with its full reasoning trace (errors-only + search filters)

Reuses build_error_report.py (analysis) and build_examples_report.py (log parsing).

Usage:
  python build_report.py
  python build_report.py --results-dir results --title "Topological Reasoning" \\
                         --eval-csv ../dataset/topo_v2_eval.csv --out ../report.html
"""
import os
import glob
import json
import html
import argparse

import build_error_report as A
import build_examples_report as E


def build_examples_data(results_dir, eval_csv):
    levels = E.load_levels(eval_csv)
    data = {"levels": {str(k): v for k, v in levels.items()}, "exps": {}}
    have_shots, n_files = set(), 0
    for path in sorted(glob.glob(os.path.join(results_dir, "voletc_*.txt"))):
        base = os.path.basename(path)
        if "-checkpoint" in base:
            continue
        m = E._FNAME_RE.search(base)
        if not m:
            continue
        tag, strat = m.group("tag"), m.group("strat")
        fs = E._FS_RE.search(tag)
        shots = str(int(fs.group(1)) if fs else 0)
        exp = E._FS_RE.sub("", tag)
        rows = list(E.parse_log(path))
        if not rows:
            continue
        n_files += 1
        have_shots.add(shots)
        data["exps"].setdefault(exp, {}).setdefault(shots, {})[strat] = rows
    ordered = {e: data["exps"][e] for e in A.EXP_ORDER if e in data["exps"]}
    for e in data["exps"]:
        ordered.setdefault(e, data["exps"][e])
    data["exps"] = ordered
    return data, have_shots, n_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--eval-csv", default="../dataset/topo_v2_eval.csv")
    ap.add_argument("--title", default="Topological Reasoning")
    ap.add_argument("--out", default="../report.html")
    args = ap.parse_args()

    cfg = A.load_configs(args.results_dir)
    if not cfg:
        print(f"[ERROR] no checkpoints in {args.results_dir}/")
        return
    meta = A.load_eval_meta(args.eval_csv)
    data, have_shots, n_logs = build_examples_data(args.results_dir, args.eval_csv)
    has_few = any(sh == 5 for (sh, _, _) in cfg)

    P = []
    combined_css = A.CSS + E.CSS + """
nav.toc{position:sticky;top:0;z-index:20;background:rgba(251,251,253,.8);backdrop-filter:blur(12px);
 border-bottom:1px solid var(--line);margin:-72px -24px 0;padding:12px 24px;display:flex;gap:18px;flex-wrap:wrap}
nav.toc a{font-size:13px;color:var(--dim);text-decoration:none;font-weight:600}
nav.toc a:hover{color:#5856d6}
"""
    P.append(f"<!doctype html><html><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Report — {html.escape(args.title)}</title><style>{combined_css}</style></head>"
             f"<body><div class='wrap'>")

    P.append("<nav class='toc'>"
             "<a href='#acc'>Accuracy</a><a href='#diff'>Difficulty</a>"
             "<a href='#conf'>Confusion</a><a href='#breakdown'>Per-experiment</a>"
             "<a href='#hard'>Hardest</a><a href='#explorer'>Examples</a></nav>")

    P.append(f"<p class='eyebrow'>Full Report</p>"
             f"<h1>Results &amp; reasoning, <span class='grad'>in one place.</span></h1>"
             f"<p class='lead'>{html.escape(args.title)} — GPT-OSS-20B across 6 KG-integration "
             f"experiments × CoT / ToT / GoT on the OSM-grounded held-out set"
             f"{' (zero-shot vs few-shot)' if has_few else ''}. Scroll for the aggregate analysis, "
             f"then read every example with its full reasoning at the end.</p>")
    if not meta:
        P.append("<p class='lead small'>Tip: copy <code>topo_v2_eval.csv</code> next to this run to unlock "
                 "per-ambiguity-level accuracy and the failing sentences.</p>")

    # 1. accuracy
    P.append("<div class='sec' id='acc'><h2>Accuracy</h2><h3>The 6 × 3 scorecard.</h3>")
    if has_few:
        P.append("<div class='grid2'>"
                 f"<div class='card'><b>Zero-shot</b>{A.matrix_html(cfg,0)}</div>"
                 f"<div class='card'><b>Few-shot (5) ⚠</b>{A.matrix_html(cfg,5)}</div></div>"
                 f"<div class='card'><b>Few-shot − Zero-shot</b>{A.delta_table(cfg)}</div>")
    else:
        P.append(f"<div class='card'>{A.matrix_html(cfg,0)}</div>")
    P.append("</div>")

    # 2. difficulty
    P.append("<div class='sec' id='diff'><h2>Difficulty</h2><h3>Which relations are hard.</h3><div class='grid2'>")
    P.append(f"<div class='card'><b>Accuracy by predicate</b><div style='height:14px'></div>"
             f"{A.bars_html(A.per_predicate(cfg,0))}</div>")
    lvl = A.level_accuracy(cfg, 0, meta)
    if lvl:
        P.append(f"<div class='card'><b>Accuracy by ambiguity level</b><div style='height:14px'></div>"
                 f"{A.bars_html(lvl)}</div>")
    else:
        P.append("<div class='card muted'><b>Accuracy by ambiguity level</b>"
                 "<p class='small'>Needs topo_v2_eval.csv (not found).</p></div>")
    P.append("</div></div>")

    # 3. overall confusion
    P.append("<div class='sec' id='conf'><h2>Confusion</h2><h3>Where errors go (overall).</h3>"
             f"<div class='card'>{A.confusion_html(cfg,0)}</div></div>")

    # 4. per-experiment breakdown
    exps_present = [e for e in A.EXP_ORDER if any((0, e, s) in cfg for s in A.STRATS)]
    P.append("<div class='sec' id='breakdown'><h2>Per-experiment breakdown</h2>"
             "<h3>What goes wrong, experiment by experiment.</h3>")
    for e in exps_present:
        narrative, _ = A.exp_narrative(cfg, e, 0)
        conf, labels = A.confusion_counts(cfg, 0, e)
        cap = f"rows = expected · columns = predicted · CoT/ToT/GoT (zero-shot) · {A.EXP_LABELS[e]}"
        sacc = [(s.upper(), sum(r["match"] for r in cfg[(0, e, s)]) / len(cfg[(0, e, s)]) * 100)
                for s in A.STRATS if (0, e, s) in cfg]
        P.append("<div class='card'>"
                 f"<div class='exp-title'>{A.EXP_LABELS[e]}</div>"
                 f"<p class='para'>{narrative or 'No data.'}</p>"
                 "<div class='grid2' style='align-items:start'>"
                 f"<div>{A.bars_html(sacc)}</div>"
                 f"<div>{A.confusion_table_html(conf, labels, cap)}</div></div></div>")
    P.append("</div>")

    # 5. hardest cases
    hard = A.hardest_rows(cfg, 0, meta)
    P.append("<div class='sec' id='hard'><h2>Hardest cases</h2><h3>Rows most configurations miss.</h3>"
             "<div class='card'>")
    for h in hard:
        lvl_chip = f"<span class='chip lv'>{h['level']}</span>" if h["level"] else ""
        pair = f"{h['a']} → {h['b']}" if h["a"] else f"row {h['index']}"
        q = f"<div class='ex-q'>“{html.escape(h['corpus'])}”</div>" if h["corpus"] else ""
        P.append(f"<div class='exrow'><span class='chip miss'>missed {h['miss']}/{h['total']}</span>"
                 f"{lvl_chip}<span class='chip exp'>{h['expected']}</span>"
                 f"<span class='muted small'>{html.escape(pair)}</span>{q}"
                 f"<div class='ex-meta'>predicted instead: {h['wrong']}</div></div>")
    P.append("</div></div>")

    # 6. examples explorer
    shots_opts = "".join(
        f"<option value='{s}'>{'Zero-shot' if s=='0' else s+'-shot ⚠'}</option>"
        for s in sorted(have_shots))
    explorer_js = E.JS.replace("__EXPLABELS__", json.dumps(A.EXP_LABELS))
    P.append("<div class='sec' id='explorer'><h2>Examples Explorer</h2>"
             "<h3>Every example, every reasoning step.</h3>"
             "<p class='lead' style='font-size:16px'>Pick an experiment, strategy and mode; toggle "
             "errors-only or search the text. Each card shows expected vs predicted and the full reasoning.</p>"
             "<div class='controls'>"
             "<span><label>Experiment</label><select id='exp'></select></span>"
             "<span><label>Strategy</label><select id='strat'></select></span>"
             f"<span><label>Mode</label><select id='shots'>{shots_opts}</select></span>"
             "<span class='ck'><input type='checkbox' id='err'><label for='err' style='margin:0'>errors only</label></span>"
             "<input type='search' id='q' placeholder='search reasoning / entities / label…'>"
             "</div><p id='summary' class='summary'></p><div id='list'></div></div>")

    P.append(f"<div class='foot'><b>{html.escape(args.title)}</b> — unified report · "
             f"{len(cfg)} result configs · {n_logs} reasoning logs · PhD Research 2026</div>")
    P.append("</div>")
    P.append(f"<script>const DATA={json.dumps(data, ensure_ascii=False)};</script>")
    P.append(f"<script>{explorer_js}</script>")
    P.append("</body></html>")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("".join(P))
    mb = os.path.getsize(args.out) / 1e6
    print(f"[OK] wrote {args.out}  ({len(cfg)} configs, {n_logs} logs, {mb:.1f} MB, "
          f"eval-meta={'yes' if meta else 'no'})")


if __name__ == "__main__":
    main()
