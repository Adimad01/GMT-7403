"""
build_combined_report.py  (run from the repo root)
================================================================================
ONE report covering MULTIPLE domains (Topological · Cardinal · Relative) with a
tab per domain. Each tab contains that domain's full analysis (accuracy matrices,
difficulty, confusion, per-experiment narrative + confusion, hardest cases) and
its own interactive Examples Explorer (read every example with full reasoning).

Auto-discovers any domain whose results dir holds voletc_*_ckpt.json.

Usage:
  python build_combined_report.py
  python build_combined_report.py --out report.html
"""
import os
import sys
import glob
import json
import html
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Topological-Reasoning", "code"))
import build_error_report as A
import build_examples_report as E

# (Display name, key, results dir, eval csv, noun, label order)
DOMAINS = [
    ("Topological", "topo", "Topological-Reasoning/code/results",
     "Topological-Reasoning/dataset/topo_v2_eval.csv", "predicate",
     ["contains", "within", "touches", "crosses", "overlaps", "disjoint", "equals"]),
    ("Cardinal", "card", "Cardinal-Reasoning/code/results",
     "Cardinal-Reasoning/dataset/cardinal_direction_relations.csv", "direction",
     ["north_of", "northeast_of", "east_of", "southeast_of",
      "south_of", "southwest_of", "west_of", "northwest_of"]),
    ("Relative", "rel", "Relative-Reasoning/code/results",
     "Relative-Reasoning/dataset/relative_direction_relations.csv", "direction",
     ["in_front_of", "behind", "left_of", "right_of", "next_to"]),
]

# Per-domain dataset locations for the "Dataset statistics" section.
#   full         = the complete labelled dataset
#   eval         = CSV the eval split is drawn from (row order = index space)
#   eval_indices = JSON list of row indices held out for evaluation
#   cache        = OSM geocode cache (both entities resolved ⇒ geocodable)
DATASET_SPECS = {
    "topo": {"full": "Topological-Reasoning/dataset/topological_relations.csv",
             "eval": "Topological-Reasoning/dataset/topo_v2_eval.csv",
             "eval_indices": "Topological-Reasoning/dataset/topo_v2_eval_indices.json",
             "cache": "Topological-Reasoning/code/results/osm_cache.json"},
    "card": {"full": "Cardinal-Reasoning/dataset/cardinal_direction_relations.csv",
             "eval": "Cardinal-Reasoning/dataset/cardinal_direction_relations.csv",
             "eval_indices": "Cardinal-Reasoning/dataset/eval_40_balanced_indices.json",
             "cache": "Cardinal-Reasoning/code/results/osm_cache.json"},
    "rel":  {"full": "Relative-Reasoning/dataset/relative_direction_relations.csv",
             "eval": "Relative-Reasoning/dataset/relative_direction_relations.csv",
             "eval_indices": "Relative-Reasoning/dataset/eval_25_balanced_indices.json",
             "cache": "Relative-Reasoning/code/results/osm_cache.json"},
}


# ---------------------------------------------------------------------------
# DATASET STATISTICS
# ---------------------------------------------------------------------------
def _pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def _label_level_counts(rows):
    """(labels_present, levels_sorted, Counter[(label, level)]) — column-agnostic."""
    import re as _re
    import collections as _col
    if not rows:
        return [], [], _col.Counter()
    cols = rows[0].keys()
    lab_c = _pick_col(cols, ["relation_label", "spatial_relation"])
    lvl_c = _pick_col(cols, ["ambiguity_level"])
    cnt = _col.Counter()
    for r in rows:
        lab = (r.get(lab_c) or "").strip().lower()
        lvl = (r.get(lvl_c) or "—").strip() if lvl_c else "—"
        if lab:
            cnt[(lab, lvl)] += 1
    labels = sorted({k[0] for k in cnt})
    def _lvnum(s):
        m = _re.search(r"(\d+)", s)
        return int(m.group(1)) if m else 99
    levels = sorted({k[1] for k in cnt}, key=_lvnum)
    return labels, levels, cnt


def _geocodable_by_label(rows, cache):
    """Counter[label] of rows whose BOTH entities resolved in the OSM cache."""
    import collections as _col
    geo = _col.Counter()
    if not rows or not cache:
        return geo
    cols = rows[0].keys()
    lab_c = _pick_col(cols, ["relation_label", "spatial_relation"])
    src_c = _pick_col(cols, ["source_entity", "place_name_subject"])
    tgt_c = _pick_col(cols, ["target_entity", "place_name_object"])
    for r in rows:
        lab = (r.get(lab_c) or "").strip().lower()
        a = (r.get(src_c) or "").strip()
        b = (r.get(tgt_c) or "").strip()
        if lab and a and b and cache.get(a) is not None and cache.get(b) is not None:
            geo[lab] += 1
    return geo


def _stats_table(rows, order, geo=None):
    """label × level count matrix as HTML (optional OSM-geocodable column)."""
    labels, levels, cnt = _label_level_counts(rows)
    if not labels:
        return "<p class='muted small'>no data</p>"
    ordered = [l for l in order if l in labels] + [l for l in labels if l not in order]
    show_lvl = levels != ["—"]
    head = "".join(f"<th>{lv.replace('Level ', 'L')}</th>" for lv in levels) if show_lvl else ""
    geo_head = "<th>OSM ✓</th>" if geo is not None else ""
    body = ""
    col_tot = {lv: 0 for lv in levels}
    g_tot = 0
    for lab in ordered:
        cells, row_tot = "", 0
        for lv in levels:
            v = cnt.get((lab, lv), 0)
            row_tot += v
            col_tot[lv] += v
            if show_lvl:
                cells += f"<td>{v or '·'}</td>"
        g = ""
        if geo is not None:
            gv = geo.get(lab, 0)
            g_tot += gv
            color = "#1a7f37" if gv == row_tot else ("#bf8700" if gv else "#cf222e")
            g = f"<td style='color:{color};font-weight:600'>{gv}/{row_tot}</td>"
        body += (f"<tr><th class='rowh'>{lab}</th>{cells}"
                 f"<td style='font-weight:700'>{row_tot}</td>{g}</tr>")
    tot_cells = "".join(f"<td>{col_tot[lv]}</td>" for lv in levels) if show_lvl else ""
    g_all = ""
    if geo is not None:
        g_all = f"<td style='font-weight:700'>{g_tot}/{sum(col_tot.values())}</td>"
    body += (f"<tr style='border-top:2px solid var(--line)'><th class='rowh'>ALL</th>{tot_cells}"
             f"<td style='font-weight:700'>{sum(col_tot.values())}</td>{g_all}</tr>")
    return (f"<table class='mtx'><thead><tr><th class='rowh'></th>{head}"
            f"<th>Total</th>{geo_head}</tr></thead><tbody>{body}</tbody></table>")


def dataset_stats_html(spec, order, noun):
    """'Dataset statistics' section: full corpus + held-out eval, per label × level."""
    import csv as _csv
    def _load(path):
        if not path or not os.path.exists(path):
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return list(_csv.DictReader(f))
    full = _load(spec["full"])
    if not full:
        return ""
    eval_pool = _load(spec["eval"])
    idx_path = spec.get("eval_indices")
    if idx_path and os.path.exists(idx_path):
        keep = set(json.load(open(idx_path)))
        eval_rows = [r for i, r in enumerate(eval_pool) if i in keep]
    else:
        eval_rows = eval_pool
    cache = {}
    if spec.get("cache") and os.path.exists(spec["cache"]):
        try:
            cache = json.load(open(spec["cache"]))
        except Exception:
            cache = {}
    geo = _geocodable_by_label(eval_rows, cache) if cache else None

    same_pool = os.path.abspath(spec["full"]) == os.path.abspath(spec["eval"])
    train_note = (f" · train = the remaining {len(full) - len(eval_rows)} rows"
                  if same_pool else "")
    geo_note = (" · <b>OSM ✓</b> = eval rows whose two entities both resolve in the "
                "OSM geocode cache (get real KG evidence in Exp 4/5/6)") if geo is not None else ""
    return (
        "<div class='sec'><h2>Dataset</h2><h3>What the data looks like.</h3>"
        "<div class='grid2' style='align-items:start'>"
        f"<div class='card'><b>Full dataset · {len(full)} rows</b>"
        f"<div style='height:12px'></div>{_stats_table(full, order)}"
        f"<p class='muted small'>examples per {noun} × ambiguity level · "
        f"{os.path.basename(spec['full'])}</p></div>"
        f"<div class='card'><b>Held-out eval · {len(eval_rows)} rows</b>"
        f"<div style='height:12px'></div>{_stats_table(eval_rows, order, geo=geo)}"
        f"<p class='muted small'>the split every experiment is scored on"
        f"{train_note}{geo_note}</p></div>"
        "</div></div>")


def examples_data(results_dir, eval_csv):
    levels = E.load_levels(eval_csv)
    data = {"levels": {str(k): v for k, v in levels.items()}, "exps": {}}
    have_shots, n = set(), 0
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
        if rows:
            n += 1
            have_shots.add(shots)
            data["exps"].setdefault(exp, {}).setdefault(shots, {})[strat] = rows
    ordered = {e: data["exps"][e] for e in A.EXP_ORDER if e in data["exps"]}
    for e in data["exps"]:
        ordered.setdefault(e, data["exps"][e])
    data["exps"] = ordered
    return data, have_shots, n


def analysis_html(cfg, meta, noun="predicate"):
    has_few = any(sh == 5 for (sh, _, _) in cfg)
    H = []
    # accuracy
    H.append("<div class='sec'><h2>Accuracy</h2><h3>The 6 × 3 scorecard.</h3>")
    if has_few:
        H.append("<div class='grid2'>"
                 f"<div class='card'><b>Zero-shot</b>{A.matrix_html(cfg,0)}</div>"
                 f"<div class='card'><b>Few-shot (5) ⚠</b>{A.matrix_html(cfg,5)}</div></div>"
                 f"<div class='card'><b>Few-shot − Zero-shot</b>{A.delta_table(cfg)}</div>")
    else:
        H.append(f"<div class='card'>{A.matrix_html(cfg,0)}</div>")
    H.append("</div>")
    # difficulty — zero-shot AND few-shot side by side
    H.append(f"<div class='sec'><h2>Difficulty</h2><h3>Which {noun}s are hard.</h3><div class='grid2'>")
    H.append(f"<div class='card'><b>Accuracy by {noun} — zero-shot</b><div style='height:14px'></div>"
             f"{A.bars_html(A.per_predicate(cfg,0))}</div>")
    if has_few:
        H.append(f"<div class='card'><b>Accuracy by {noun} — few-shot (5) ⚠</b><div style='height:14px'></div>"
                 f"{A.bars_html(A.per_predicate(cfg,5))}</div>")
    lvl = A.level_accuracy(cfg, 0, meta)
    if lvl:
        H.append(f"<div class='card'><b>Accuracy by ambiguity level — zero-shot</b><div style='height:14px'></div>"
                 f"{A.bars_html(lvl)}</div>")
        lvl5 = A.level_accuracy(cfg, 5, meta) if has_few else []
        if lvl5:
            H.append(f"<div class='card'><b>Accuracy by ambiguity level — few-shot (5) ⚠</b>"
                     f"<div style='height:14px'></div>{A.bars_html(lvl5)}</div>")
    else:
        H.append("<div class='card muted'><b>Accuracy by ambiguity level</b>"
                 "<p class='small'>Needs the eval CSV (not found).</p></div>")
    H.append("</div></div>")
    # overall confusion — zero-shot AND few-shot
    H.append("<div class='sec'><h2>Confusion</h2><h3>Where errors go (overall).</h3>")
    if has_few:
        H.append("<div class='grid2' style='align-items:start'>"
                 f"<div class='card'><b>Zero-shot</b><div style='height:10px'></div>{A.confusion_html(cfg,0)}</div>"
                 f"<div class='card'><b>Few-shot (5) ⚠</b><div style='height:10px'></div>{A.confusion_html(cfg,5)}</div>"
                 "</div>")
    else:
        H.append(f"<div class='card'>{A.confusion_html(cfg,0)}</div>")
    H.append("</div>")
    # per-experiment breakdown
    exps_present = [e for e in A.EXP_ORDER if any((0, e, s) in cfg for s in A.STRATS)]
    H.append("<div class='sec'><h2>Per-experiment breakdown</h2>"
             "<h3>What goes wrong, experiment by experiment.</h3>")
    for e in exps_present:
        parts, _ = A.exp_narrative_parts(cfg, e, 0)
        bullets = "".join(f"<li>{p}</li>" for p in parts) or "<li>No data.</li>"
        conf, labels = A.confusion_counts(cfg, 0, e)
        cap = f"rows = expected · columns = predicted · CoT/ToT/GoT (zero-shot) · {A.EXP_LABELS[e]}"
        sacc = [(s.upper(), sum(r["match"] for r in cfg[(0, e, s)]) / len(cfg[(0, e, s)]) * 100)
                for s in A.STRATS if (0, e, s) in cfg]
        H.append("<div class='card'>"
                 f"<div class='exp-title'>{A.EXP_LABELS[e]}</div>"
                 f"<ul class='blist'>{bullets}</ul>"
                 "<div class='grid2' style='align-items:start'>"
                 f"<div>{A.bars_html(sacc)}</div>"
                 f"<div>{A.confusion_table_html(conf, labels, cap)}</div></div>"
                 "<div style='height:18px'></div>"
                 "<div class='cm-sub'>Confusion by strategy — zero-shot vs few-shot</div>"
                 f"{A.per_strategy_confusion_grid(cfg, e)}</div>")
    H.append("</div>")
    # hardest cases
    H.append("<div class='sec'><h2>Hardest cases</h2><h3>Rows most configurations miss.</h3><div class='card'>")
    for h in A.hardest_rows(cfg, 0, meta):
        lvl_chip = f"<span class='chip lv'>{h['level']}</span>" if h["level"] else ""
        pair = f"{h['a']} → {h['b']}" if h["a"] else f"row {h['index']}"
        q = f"<div class='ex-q'>“{html.escape(h['corpus'])}”</div>" if h["corpus"] else ""
        H.append(f"<div class='exrow'><span class='chip miss'>missed {h['miss']}/{h['total']}</span>"
                 f"{lvl_chip}<span class='chip exp'>{h['expected']}</span>"
                 f"<span class='muted small'>{html.escape(pair)}</span>{q}"
                 f"<div class='ex-meta'>predicted instead: {h['wrong']}</div></div>")
    H.append("</div></div>")
    return "".join(H), has_few


def explorer_html(key, have_shots):
    shots_opts = "".join(
        f"<option value='{s}'>{'Zero-shot' if s=='0' else s+'-shot ⚠'}</option>"
        for s in sorted(have_shots))
    return (
        "<div class='sec'><h2>Examples Explorer</h2><h3>Every example, every reasoning step.</h3>"
        "<p class='lead' style='font-size:16px'>Pick experiment · strategy · mode; toggle errors-only or "
        "search. Each card shows expected vs predicted and the full reasoning.</p>"
        "<div class='controls'>"
        f"<span><label>Experiment</label><select id='exp_{key}'></select></span>"
        f"<span><label>Strategy</label><select id='strat_{key}'></select></span>"
        f"<span><label>Mode</label><select id='shots_{key}'>{shots_opts}</select></span>"
        f"<span class='ck'><input type='checkbox' id='err_{key}'><label for='err_{key}' style='margin:0'>errors only</label></span>"
        f"<input type='search' id='q_{key}' placeholder='search reasoning / entities / label…'>"
        f"</div><p id='summary_{key}' class='summary'></p><div id='list_{key}'></div></div>")


JS = """
const EXPLABELS=__EXPLABELS__;
function opt(sel,arr,labels){sel.innerHTML=arr.map(v=>`<option value="${v}">${labels?labels[v]||v:v.toUpperCase()}</option>`).join('')}
function initExplorer(key){
 const D=DATA[key];const $=id=>document.getElementById(id+'_'+key);
 if(!$('exp'))return;
 opt($('exp'),Object.keys(D.exps),EXPLABELS);
 const curStrats=()=>Object.keys((D.exps[$('exp').value]||{})[$('shots').value]||{});
 const syncStrats=()=>{const ss=curStrats();opt($('strat'),ss.length?ss:['cot'])};
 function render(){
  const e=$('exp').value,sh=$('shots').value,st=$('strat').value;
  const rows=(((D.exps[e]||{})[sh]||{})[st])||[];
  const errOnly=$('err').checked,q=$('q').value.trim().toLowerCase();
  const cont=$('list');cont.innerHTML='';let shown=0,correct=0;
  rows.forEach(r=>{if(r.match)correct++;if(errOnly&&r.match)return;
   if(q&&!(r.ab.toLowerCase().includes(q)||r.reasoning.toLowerCase().includes(q)||r.expected.includes(q)||r.predicted.includes(q)))return;
   shown++;const lv=D.levels[r.index]?`<span class="chip lv">${D.levels[r.index]}</span>`:'';
   const pc=r.match?'pred-ok':'pred-no';const c=document.createElement('div');
   c.className='card '+(r.match?'correct':'wrong');
   c.innerHTML=`<div class="hd"><span class="chip idx">#${r.index}</span>${lv}<span class="chip exp">expected: ${r.expected}</span><span class="chip ${pc}">predicted: ${r.predicted||'∅'} ${r.match?'✓':'✗'}</span></div><div class="ab">${r.ab}</div><details ${errOnly?'open':''}><summary>reasoning</summary><div class="reason">${r.reasoning.replace(/</g,'&lt;')}</div></details>`;
   cont.appendChild(c);});
  const acc=rows.length?(correct/rows.length*100).toFixed(1):'0';
  $('summary').innerHTML=`Showing <b>${shown}</b> of <b>${rows.length}</b> · accuracy <b>${acc}%</b> (${correct}/${rows.length})`;
 }
 ['exp','shots','strat','err','q'].forEach(id=>{const el=$(id);if(!el)return;
  el.addEventListener(id==='q'?'input':'change',()=>{if(id==='exp'||id==='shots')syncStrats();render();});});
 syncStrats();render();
}
document.querySelectorAll('.domtabbtn').forEach(b=>b.addEventListener('click',()=>{
 document.querySelectorAll('.domtabbtn').forEach(x=>x.classList.remove('active'));
 document.querySelectorAll('.domtab').forEach(x=>x.style.display='none');
 b.classList.add('active');document.getElementById('dom-'+b.dataset.dom).style.display='block';
}));
Object.keys(DATA).forEach(initExplorer);
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report.html")
    args = ap.parse_args()

    DATA = {}
    tabs, panels = [], []
    first = True
    for name, key, rdir, ecsv, noun, order in DOMAINS:
        cfg = A.load_configs(rdir)
        if not cfg:
            continue
        # make the analysis label-aware for this domain
        A.PRED_ORDER = order
        A.NOUN = noun
        meta = A.load_eval_meta(ecsv)
        data, have_shots, n_logs = examples_data(rdir, ecsv)
        DATA[key] = data
        spec = DATASET_SPECS.get(key)
        stats = dataset_stats_html(spec, order, noun) if spec else ""
        anal, _ = analysis_html(cfg, meta, noun=noun)
        expl = explorer_html(key, have_shots) if data["exps"] else ""
        tabs.append(f"<button class='domtabbtn{' active' if first else ''}' data-dom='{key}'>{name}</button>")
        disp = "block" if first else "none"
        panels.append(f"<section class='domtab' id='dom-{key}' style='display:{disp}'>"
                      f"<p class='dom-h'>{name} Reasoning · {len(cfg)} configs · {n_logs} logs</p>"
                      f"{stats}{anal}{expl}</section>")
        first = False

    if not DATA:
        print("[ERROR] no domains with results found (run from repo root).")
        return

    css = A.CSS + E.CSS + """
.domtabs{position:sticky;top:0;z-index:30;background:rgba(251,251,253,.85);backdrop-filter:blur(12px);
 border-bottom:1px solid var(--line);margin:-72px -24px 28px;padding:14px 24px;display:flex;gap:8px;flex-wrap:wrap}
.domtabbtn{font:inherit;font-size:14px;font-weight:600;padding:8px 16px;border:1px solid var(--line);
 background:#fff;color:var(--dim);border-radius:999px;cursor:pointer}
.domtabbtn.active{background:#5856d6;color:#fff;border-color:#5856d6}
.dom-h{font-size:13px;color:var(--dim);font-weight:600;margin:0 0 18px}
.cm-sub{font-size:14px;font-weight:700;color:#1d1d1f;margin:0 0 10px;padding-top:6px;border-top:1px solid var(--line)}
"""
    page = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Spatial Reasoning — Full Report</title><style>{css}</style></head><body><div class='wrap'>"
            f"<div class='domtabs'>{''.join(tabs)}</div>"
            f"<p class='eyebrow'>Full Report</p>"
            f"<h1>Results &amp; reasoning, <span class='grad'>all domains.</span></h1>"
            f"<p class='lead'>GPT-OSS-20B across 6 KG-integration experiments × CoT / ToT / GoT, "
            f"zero-shot vs few-shot. Switch domains above; each tab has the analysis and a full "
            f"example-by-example reasoning explorer.</p>"
            f"{''.join(panels)}"
            f"<div class='foot'><b>Spatial Relations</b> — unified multi-domain report · PhD Research 2026</div>"
            f"</div>"
            f"<script>const DATA={json.dumps(DATA, ensure_ascii=False)};</script>"
            f"<script>{JS.replace('__EXPLABELS__', json.dumps(A.EXP_LABELS))}</script>"
            f"</body></html>")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    mb = os.path.getsize(args.out) / 1e6
    print(f"[OK] wrote {args.out}  (domains: {', '.join(DATA)} · {mb:.1f} MB)")


if __name__ == "__main__":
    main()
