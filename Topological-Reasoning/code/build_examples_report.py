"""
build_examples_report.py
================================================================================
Generates an interactive, Apple-minimalist HTML "examples explorer" so you can
read EVERY evaluated example together with the model's full reasoning trace.

Parses the per-row reasoning out of the voletc_*.txt logs and embeds it. In the
page you pick: experiment · strategy (CoT/ToT/GoT) · zero/few-shot, then read all
rows for that config — each card shows entities, expected vs predicted (green =
correct, red = wrong) and the complete reasoning. Filters: errors-only + search.

Usage:
  python build_examples_report.py
  python build_examples_report.py --results-dir results --title "Topological Reasoning" \\
                                  --out ../examples_report.html --eval-csv ../dataset/topo_v2_eval.csv
"""
import os
import re
import csv
import glob
import json
import html
import argparse

EXP_LABELS = {
    "exp1_base": "Exp 1 · base / no-KG",
    "exp2_ft_nokg": "Exp 2 · no-KG LoRA",
    "exp3_ft_osmkg": "Exp 3 · OSM-KG LoRA (KG@train)",
    "exp4_base_kg_input": "Exp 4 · base + KG@input",
    "exp5_ft_kg_input": "Exp 5 · no-KG LoRA + KG@input",
    "exp6_base_kg_rag": "Exp 6 · base + KG@inference (RAG)",
}
EXP_ORDER = list(EXP_LABELS)
STRATS = ["cot", "tot", "got"]

_FNAME_RE = re.compile(r"voletc_(?P<tag>.+?)_(?P<strat>cot|tot|got)_.*\.txt$")
_FS_RE = re.compile(r"_fs(\d+)$")
_ROW_RE = re.compile(r"^ROW\s+(\d+)\s*\|\s*(.*)$")
_RESULT_RE = re.compile(r"^RESULT\s*\|\s*Expected=(.*?)\s*\|\s*Predicted=(.*?)\s*\|", re.I)


def parse_log(path):
    """Yield dicts {index, ab, expected, predicted, match, reasoning} per row."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    # find row start positions
    starts = [i for i, ln in enumerate(lines) if _ROW_RE.match(ln)]
    for k, s in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        block = lines[s:end]
        m = _ROW_RE.match(block[0])
        idx, ab = int(m.group(1)), m.group(2).strip()
        expected = predicted = ""
        match = False
        res_at = len(block)
        for j, ln in enumerate(block):
            rm = _RESULT_RE.match(ln.strip())
            if rm:
                expected = rm.group(1).strip().lower()
                predicted = rm.group(2).strip().lower()
                match = "CORRECT" in ln.upper()
                res_at = j
                break
        if not expected:
            ex = next((ln for ln in block if ln.lower().startswith("expected:")), "")
            expected = ex.split(":", 1)[-1].strip().lower() if ex else ""
        # reasoning = everything between the row header block and the RESULT line,
        # skipping the leading "Expected:" + separator lines
        body = block[1:res_at]
        body = [ln.rstrip("\n") for ln in body
                if not ln.lower().startswith("expected:") and set(ln.strip()) != {"="}]
        reasoning = "\n".join(body).strip()
        reasoning = re.sub(r"\n{3,}", "\n\n", reasoning)
        if len(reasoning) > 24000:
            reasoning = reasoning[:24000] + "\n… [truncated]"
        yield {"index": idx, "ab": ab, "expected": expected,
               "predicted": predicted, "match": match, "reasoning": reasoning}


def load_levels(eval_csv):
    if not eval_csv or not os.path.exists(eval_csv):
        return {}
    out = {}
    for i, r in enumerate(csv.DictReader(open(eval_csv, newline="", encoding="utf-8"))):
        out[i] = (r.get("ambiguity_level") or "").strip()
    return out


CSS = """
:root{--ink:#1d1d1f;--dim:#86868b;--line:#e8e8ed;--bg:#fbfbfd;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 -webkit-font-smoothing:antialiased;line-height:1.5}
.wrap{max-width:980px;margin:0 auto;padding:64px 24px 96px}
.eyebrow{font-size:13px;font-weight:600;color:#5856d6;margin:0 0 6px}
h1{font-size:44px;line-height:1.07;letter-spacing:-.02em;font-weight:700;margin:0 0 12px}
h1 .grad{background:linear-gradient(90deg,#5856d6,#0a84ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.lead{font-size:18px;color:var(--dim);max-width:680px;margin:0 0 28px}
.controls{position:sticky;top:0;z-index:5;background:rgba(251,251,253,.85);backdrop-filter:blur(12px);
 border:1px solid var(--line);border-radius:16px;padding:16px 18px;margin-bottom:26px;
 display:flex;gap:14px;flex-wrap:wrap;align-items:center}
.controls label{font-size:12px;color:var(--dim);font-weight:600;margin-right:6px}
select,input[type=search]{font:inherit;font-size:14px;padding:7px 11px;border:1px solid var(--line);
 border-radius:10px;background:#fff;color:var(--ink)}
input[type=search]{min-width:200px;flex:1}
.ck{display:flex;align-items:center;gap:6px;font-size:13px;color:var(--ink)}
.summary{font-size:13px;color:var(--dim);margin:0 0 18px}
.summary b{color:var(--ink)}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin-bottom:14px}
.card.wrong{border-color:#ffd0d6;background:#fffafa}
.card.correct{border-color:#cdeccd}
.hd{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.chip{font-size:11px;font-weight:700;padding:3px 9px;border-radius:999px}
.chip.idx{background:#f0f0f4;color:#555}.chip.lv{background:#eef0ff;color:#3b39b3}
.chip.exp{background:#eef6ff;color:#0a5bd0}.chip.pred-ok{background:#eaf8ee;color:#1a7f37}
.chip.pred-no{background:#ffeef0;color:#cf222e}
.ab{font-size:15px;font-weight:600;margin:2px 0 6px}
.reason{white-space:pre-wrap;word-break:break-word;font-size:13px;line-height:1.6;
 color:#333;background:#fafafa;border:1px solid var(--line);border-radius:12px;padding:14px 16px;
 max-height:380px;overflow:auto}
details>summary{cursor:pointer;font-size:13px;color:#5856d6;font-weight:600;margin-top:4px}
.foot{margin-top:60px;color:var(--dim);font-size:13px;border-top:1px solid var(--line);padding-top:18px}
"""

JS = """
const $=s=>document.querySelector(s);
const EXPLABELS=__EXPLABELS__;
function opts(sel,arr,labels){sel.innerHTML=arr.map(v=>`<option value="${v}">${labels?labels[v]||v:v.toUpperCase()}</option>`).join('')}
const exps=Object.keys(DATA.exps);
opts($('#exp'),exps,EXPLABELS);
function curStrats(){const e=$('#exp').value,sh=$('#shots').value;return Object.keys((DATA.exps[e]||{})[sh]||{})}
function syncStrats(){const ss=curStrats();opts($('#strat'),ss.length?ss:['cot']);}
function render(){
 const e=$('#exp').value,sh=$('#shots').value,st=$('#strat').value;
 const rows=(((DATA.exps[e]||{})[sh]||{})[st])||[];
 const errOnly=$('#err').checked, q=$('#q').value.trim().toLowerCase();
 const cont=$('#list');cont.innerHTML='';
 let shown=0,correct=0;
 rows.forEach(r=>{
   if(r.match)correct++;
   if(errOnly&&r.match)return;
   if(q && !(r.ab.toLowerCase().includes(q)||r.reasoning.toLowerCase().includes(q)||r.expected.includes(q)||r.predicted.includes(q)))return;
   shown++;
   const lv=DATA.levels[r.index]?`<span class="chip lv">${DATA.levels[r.index]}</span>`:'';
   const pc=r.match?'pred-ok':'pred-no';
   const card=document.createElement('div');
   card.className='card '+(r.match?'correct':'wrong');
   card.innerHTML=`<div class="hd"><span class="chip idx">#${r.index}</span>${lv}
     <span class="chip exp">expected: ${r.expected}</span>
     <span class="chip ${pc}">predicted: ${r.predicted||'∅'} ${r.match?'✓':'✗'}</span></div>
     <div class="ab">${r.ab}</div>
     <details ${errOnly?'open':''}><summary>reasoning</summary>
     <div class="reason">${r.reasoning.replace(/</g,'&lt;')}</div></details>`;
   cont.appendChild(card);
 });
 const acc=rows.length?(correct/rows.length*100).toFixed(1):'0';
 $('#summary').innerHTML=`Showing <b>${shown}</b> of <b>${rows.length}</b> examples · accuracy <b>${acc}%</b> (${correct}/${rows.length})`;
}
['exp','shots','strat','err','q'].forEach(id=>{const el=$('#'+id);el.addEventListener(id==='q'?'input':'change',()=>{if(id==='exp'||id==='shots')syncStrats();render();});});
syncStrats();render();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--eval-csv", default="../dataset/topo_v2_eval.csv")
    ap.add_argument("--title", default="Topological Reasoning")
    ap.add_argument("--out", default="../examples_report.html")
    args = ap.parse_args()

    levels = load_levels(args.eval_csv)
    data = {"levels": {str(k): v for k, v in levels.items()}, "exps": {}}
    have_shots = set()
    n_files = 0
    for path in sorted(glob.glob(os.path.join(args.results_dir, "voletc_*.txt"))):
        base = os.path.basename(path)
        if "-checkpoint" in base:
            continue
        m = _FNAME_RE.search(base)
        if not m:
            continue
        tag, strat = m.group("tag"), m.group("strat")
        fs = _FS_RE.search(tag)
        shots = str(int(fs.group(1)) if fs else 0)
        exp = _FS_RE.sub("", tag)
        rows = list(parse_log(path))
        if not rows:
            continue
        n_files += 1
        have_shots.add(shots)
        data["exps"].setdefault(exp, {}).setdefault(shots, {})[strat] = rows

    if not data["exps"]:
        print(f"[ERROR] no parseable .txt logs in {args.results_dir}/")
        return

    # order experiments
    ordered = {e: data["exps"][e] for e in EXP_ORDER if e in data["exps"]}
    for e in data["exps"]:
        ordered.setdefault(e, data["exps"][e])
    data["exps"] = ordered

    shots_opts = "".join(
        f"<option value='{s}'>{'Zero-shot' if s=='0' else s+'-shot ⚠'}</option>"
        for s in sorted(have_shots))
    js = (JS.replace("__EXPLABELS__", json.dumps(EXP_LABELS)))

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Examples — {html.escape(args.title)}</title><style>{CSS}</style></head><body><div class="wrap">
<p class="eyebrow">Examples Explorer</p>
<h1>Every example, <span class="grad">every reasoning step.</span></h1>
<p class="lead">{html.escape(args.title)} — read the full chain-of-thought for each held-out
example. Pick an experiment, strategy and prompting mode; toggle errors-only or search the text.</p>
<div class="controls">
 <span><label>Experiment</label><select id="exp"></select></span>
 <span><label>Strategy</label><select id="strat"></select></span>
 <span><label>Mode</label><select id="shots">{shots_opts}</select></span>
 <span class="ck"><input type="checkbox" id="err"><label for="err" style="margin:0">errors only</label></span>
 <input type="search" id="q" placeholder="search reasoning / entities / label…">
</div>
<p id="summary" class="summary"></p>
<div id="list"></div>
<div class="foot"><b>{html.escape(args.title)}</b> — examples explorer · {n_files} reasoning logs · PhD Research 2026</div>
</div>
<script>const DATA={json.dumps(data, ensure_ascii=False)};</script>
<script>{js}</script>
</body></html>"""

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    mb = os.path.getsize(args.out) / 1e6
    print(f"[OK] wrote {args.out}  ({n_files} logs, {mb:.1f} MB, levels={'yes' if levels else 'no'})")


if __name__ == "__main__":
    main()
