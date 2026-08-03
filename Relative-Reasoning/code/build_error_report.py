"""
build_error_report.py
================================================================================
Generates a self-contained, Apple-minimalist HTML error-analysis report from the
voletc_*_ckpt.json result checkpoints:

  • accuracy matrices (zero-shot & few-shot, 6 experiments × CoT/ToT/GoT)
  • per-predicate accuracy (where the task itself is hard)
  • confusion matrix (where errors go: expected → predicted)
  • most-confused label pairs
  • hardest examples (rows missed by the most configurations)
  • zero-shot → few-shot deltas

If the eval CSV is supplied (and present), the report is enriched with
per-ambiguity-level accuracy and the actual corpus sentences of the failing rows.

Usage:
  python build_error_report.py
  python build_error_report.py --results-dir results --eval-csv ../dataset/topo_v2_eval.csv \\
                               --title "Topological Reasoning" --out ../error_report.html
"""
import os
import re
import csv
import glob
import json
import argparse
import collections

STRATS = ["cot", "tot", "got"]
# PRED_ORDER = preferred label ordering; NOUN = what a label is called in this domain.
PRED_ORDER = ["left_of", "right_of", "in_front_of", "behind", "next_to"]
NOUN = "direction"

EXP_LABELS = {
    "exp1_base": "Exp 1 · base / no-KG",
    "exp2_ft_nokg": "Exp 2 · no-KG LoRA",
    "exp3_ft_osmkg": "Exp 3 · OSM-KG LoRA (KG@train)",
    "exp4_base_kg_input": "Exp 4 · base + KG@input",
    "exp5_ft_kg_input": "Exp 5 · no-KG LoRA + KG@input",
    "exp6_base_kg_rag": "Exp 6 · base + KG@inference (RAG)",
    "exp7_base_graphrag": "Exp 7 · base + KG@inference (GraphRAG)",
}
EXP_ORDER = list(EXP_LABELS)

_CKPT_RE = re.compile(r"voletc_(?P<tag>.+?)_(?P<strat>cot|tot|got)_.*_ckpt\.json$")
_FS_RE = re.compile(r"_fs(\d+)$")


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
def load_configs(results_dir):
    """Return {(shots, exp, strat): [rows]} from all ckpt files."""
    cfg = {}
    for path in glob.glob(os.path.join(results_dir, "voletc_*_ckpt.json")):
        m = _CKPT_RE.search(os.path.basename(path))
        if not m:
            continue
        tag, strat = m.group("tag"), m.group("strat")
        fs = _FS_RE.search(tag)
        shots = int(fs.group(1)) if fs else 0
        exp = _FS_RE.sub("", tag)
        try:
            rows = json.load(open(path)).get("results", [])
        except Exception:
            continue
        if rows:
            cfg[(shots, exp, strat)] = rows
    return cfg


def load_eval_meta(eval_csv):
    """index -> {corpus, level, a, b} from the eval CSV (if available)."""
    if not eval_csv or not os.path.exists(eval_csv):
        return {}
    rows = list(csv.DictReader(open(eval_csv, newline="", encoding="utf-8")))
    def g(r, *ks):
        for k in ks:
            if r.get(k):
                return r[k].strip()
        return ""
    meta = {}
    for i, r in enumerate(rows):
        meta[i] = {
            "corpus": g(r, "relation_predicate", "corpus"),
            "level": g(r, "ambiguity_level"),
            "a": g(r, "place_name_subject", "source_entity"),
            "b": g(r, "place_name_object", "target_entity"),
        }
    return meta


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
def acc_color(p):
    # green→amber→red ramp by accuracy
    if p >= 90: return "#1a7f37", "#eaf8ee"
    if p >= 75: return "#2da44e", "#f0fbf2"
    if p >= 60: return "#bf8700", "#fff8e6"
    if p >= 45: return "#d1842c", "#fff3e3"
    return "#cf222e", "#ffeef0"


def matrix_html(cfg, shots):
    exps = [e for e in EXP_ORDER if any((shots, e, s) in cfg for s in STRATS)]
    if not exps:
        return "<p class='muted'>No runs.</p>"
    head = "".join(f"<th>{s.upper()}</th>" for s in STRATS)
    body = ""
    for e in exps:
        cells = ""
        for s in STRATS:
            rows = cfg.get((shots, e, s))
            if rows:
                acc = sum(r["match"] for r in rows) / len(rows) * 100
                fg, bg = acc_color(acc)
                cells += f"<td style='background:{bg};color:{fg};font-weight:600'>{acc:.1f}%</td>"
            else:
                cells += "<td class='muted'>—</td>"
        body += f"<tr><th class='rowh'>{EXP_LABELS[e]}</th>{cells}</tr>"
    n = len(next(iter(cfg.values())))
    return (f"<table class='mtx'><thead><tr><th class='rowh'></th>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table>"
            f"<p class='muted small'>n = {n} held-out rows · color = accuracy band</p>")


def bars_html(pairs, unit="%"):
    out = "<div class='bars'>"
    mx = max((v for _, v in pairs), default=1) or 1
    for name, v in pairs:
        fg, _ = acc_color(v) if unit == "%" else ("#5856d6", "")
        w = max(2, v / mx * 100)
        out += (f"<div class='barrow'><span class='blabel'>{name}</span>"
                f"<span class='btrack'><span class='bfill' style='width:{w:.0f}%;background:{fg}'></span></span>"
                f"<span class='bval'>{v:.1f}{unit}</span></div>")
    return out + "</div>"


def confusion_counts(cfg, shots, exp=None, strat=None):
    """expected→predicted counter over configs at `shots` (optionally one exp/strategy)."""
    conf = collections.Counter()
    labels = set()
    for (sh, e, s), rows in cfg.items():
        if sh != shots or (exp is not None and e != exp) or (strat is not None and s != strat):
            continue
        for r in rows:
            exp_l, pred = r["expected"], r.get("predicted") or "invalid"
            conf[(exp_l, pred)] += 1
            labels.add(exp_l)
            if pred in PRED_ORDER:
                labels.add(pred)
    return conf, labels


def confusion_table_html(conf, labels, caption="", compact=False):
    order = [p for p in PRED_ORDER if p in labels]
    extra = sorted(l for l in labels if l not in PRED_ORDER)
    cols = order + extra + (["invalid"] if any(p == "invalid" for (_, p) in conf) else [])
    rows_l = order + extra
    if not rows_l:
        return "<p class='muted small'>no data</p>"
    head = "".join(f"<th class='cm-c'>{c[:3]}</th>" for c in cols)
    body = ""
    for er in rows_l:
        row_total = sum(conf[(er, c)] for c in cols) or 1
        tds = ""
        for c in cols:
            v = conf[(er, c)]
            frac = v / row_total
            if v == 0:
                tds += "<td class='cm muted'>·</td>"
            elif er == c:
                tds += f"<td class='cm' style='background:rgba(45,164,78,{0.15+0.85*frac:.2f});color:#06381a'>{v}</td>"
            else:
                tds += f"<td class='cm' style='background:rgba(207,34,46,{0.12+0.7*frac:.2f});color:#5b0a12'>{v}</td>"
        body += f"<tr><th class='cm-r'>{er[:10]}</th>{tds}</tr>"
    cls = "cmtx compact" if compact else "cmtx"
    cap = f"<p class='muted small'>{caption}</p>" if caption else ""
    return f"<table class='{cls}'><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>{cap}"


def per_strategy_confusion_grid(cfg, exp):
    """For one experiment: a row per strategy, each with its zero-shot + few-shot
    confusion matrices side by side."""
    blocks = []
    for s in STRATS:
        cells = []
        for sh, lab in [(0, "zero-shot"), (5, "few-shot ⚠")]:
            if (sh, exp, s) in cfg:
                conf, labels = confusion_counts(cfg, sh, exp, s)
                rows = cfg[(sh, exp, s)]
                acc = sum(r["match"] for r in rows) / len(rows) * 100
                cells.append(f"<div class='cm-block'><div class='cm-cap'>{lab} · {acc:.0f}%</div>"
                             f"{confusion_table_html(conf, labels, compact=True)}</div>")
        if cells:
            blocks.append(f"<div class='cm-strat'><div class='cm-slabel'>{s.upper()}</div>"
                          f"<div class='cm-pair'>{''.join(cells)}</div></div>")
    if not blocks:
        return ""
    return ("<div class='cm-grid'>" + "".join(blocks) + "</div>"
            "<p class='muted small'>rows = expected · cols = predicted · green = correct · red = error · "
            "zero-shot vs few-shot side by side per strategy</p>")


def confusion_html(cfg, shots):
    conf, labels = confusion_counts(cfg, shots)
    cap = (f"rows = expected · columns = predicted · green diagonal = correct · red = misclassification "
           f"(aggregated over all {('zero-shot' if shots==0 else f'{shots}-shot')} runs)")
    return confusion_table_html(conf, labels, cap)


def exp_narrative_parts(cfg, exp, shots=0):
    """Auto-generate analysis sentences (one bullet each) for this experiment."""
    # per-strategy accuracy
    sacc = {}
    for s in STRATS:
        rows = cfg.get((shots, exp, s))
        if rows:
            sacc[s] = sum(r["match"] for r in rows) / len(rows) * 100
    if not sacc:
        return [], {}
    avg = sum(sacc.values()) / len(sacc)
    conf, _ = confusion_counts(cfg, shots, exp)
    errors = {(a, b): n for (a, b), n in conf.items() if a != b}
    tot_err = sum(errors.values())
    known = set(PRED_ORDER)
    invalid = sum(n for (a, b), n in conf.items() if b == "invalid")
    top = sorted(errors.items(), key=lambda kv: -kv[1])[:3]
    best = max(sacc, key=sacc.get); worst = min(sacc, key=sacc.get)

    parts = []
    sline = ", ".join(f"{s.upper()} {sacc[s]:.0f}%" for s in STRATS if s in sacc)
    parts.append(f"<b>{EXP_LABELS[exp]}</b> averages <b>{avg:.0f}%</b> ({sline}).")
    if len(sacc) > 1 and sacc[best] - sacc[worst] >= 8:
        parts.append(f"{best.upper()} is clearly its strongest strategy and {worst.upper()} the weakest "
                     f"(a {sacc[best]-sacc[worst]:.0f}-point spread), so the reasoning scaffold matters here.")
    if tot_err == 0:
        parts.append("It makes essentially no errors on this subset.")
    else:
        if top:
            pair_txt = ", ".join(f"{a}→{b} ({n})" for (a, b), n in top)
            parts.append(f"Its most frequent confusions are {pair_txt}.")
        # dominant error sink: the label most wrong answers collapse to
        sink = collections.Counter()
        for (a, b), n in errors.items():
            sink[b] += n
        sink_lbl, sink_n = sink.most_common(1)[0]
        if sink_n / tot_err >= 0.40:
            parts.append(f"Strikingly, <b>{sink_n/tot_err*100:.0f}% of all its errors collapse to "
                         f"<i>{sink_lbl}</i></b> — the model over-predicts that {NOUN} when unsure.")
        elif sink_n / tot_err >= 0.25:
            parts.append(f"About {sink_n/tot_err*100:.0f}% of errors fall into <i>{sink_lbl}</i>, "
                         f"a notable over-prediction of one {NOUN}.")
        # biggest two-way (symmetric) confusion
        seen, best_sym = set(), (None, 0)
        for (a, b) in errors:
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            two = errors.get((a, b), 0) + errors.get((b, a), 0)
            if two > best_sym[1]:
                best_sym = (key, two)
        if best_sym[0] and best_sym[1] >= 4:
            a, b = best_sym[0]
            parts.append(f"{a}↔{b} are mutually confused {best_sym[1]} times — a directional/symmetry weak spot.")
        if invalid >= 4:
            parts.append(f"It also emits {invalid} invalid/empty predictions, hinting at format or "
                         f"truncation issues.")
    # few-shot effect
    fs_deltas = []
    for s in STRATS:
        z, f = cfg.get((0, exp, s)), cfg.get((5, exp, s))
        if z and f:
            fs_deltas.append(sum(r["match"] for r in f) / len(f) * 100
                             - sum(r["match"] for r in z) / len(z) * 100)
    if fs_deltas:
        md = sum(fs_deltas) / len(fs_deltas)
        if md >= 5:
            parts.append(f"Label-conditioned few-shot lifts it by ~{md:.0f} pts on average (expected, since the "
                         f"demos reveal the class).")
        elif md <= -5:
            parts.append(f"Few-shot actually hurts it by ~{abs(md):.0f} pts — the demos appear to distract more "
                         f"than help.")
        else:
            parts.append("Few-shot barely moves it (~{:+.0f} pts), notable given the demos leak the label.".format(md))
    return parts, {"avg": avg, "sacc": sacc}


def exp_narrative(cfg, exp, shots=0):
    parts, stats = exp_narrative_parts(cfg, exp, shots)
    return (" ".join(parts) if parts else None), stats


def per_predicate(cfg, shots):
    """Accuracy per expected label, derived from the data (any label set)."""
    tot = collections.Counter()
    hit = collections.Counter()
    for (sh, e, s), rows in cfg.items():
        if sh != shots:
            continue
        for r in rows:
            tot[r["expected"]] += 1
            hit[r["expected"]] += bool(r["match"])
    labels = [p for p in PRED_ORDER if p in tot] + sorted(l for l in tot if l not in PRED_ORDER)
    return [(p, hit[p] / tot[p] * 100) for p in labels if tot[p]]


def hardest_rows(cfg, shots, meta, top=15):
    miss = collections.Counter()
    total = collections.Counter()
    expected = {}
    wrongpreds = collections.defaultdict(collections.Counter)
    for (sh, e, s), rows in cfg.items():
        if sh != shots:
            continue
        for r in rows:
            i = r["index"]
            total[i] += 1
            expected[i] = r["expected"]
            if not r["match"]:
                miss[i] += 1
                wrongpreds[i][r.get("predicted") or "invalid"] += 1
    ranked = sorted(miss.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    out = []
    for i, m in ranked:
        wp = ", ".join(f"{k}×{v}" for k, v in wrongpreds[i].most_common(3))
        info = meta.get(i, {})
        out.append({
            "index": i, "expected": expected[i], "miss": m, "total": total[i],
            "wrong": wp, "corpus": info.get("corpus", ""), "level": info.get("level", ""),
            "a": info.get("a", ""), "b": info.get("b", ""),
        })
    return out


def level_accuracy(cfg, shots, meta):
    if not meta:
        return []
    by = collections.defaultdict(lambda: [0, 0])
    for (sh, e, s), rows in cfg.items():
        if sh != shots:
            continue
        for r in rows:
            lv = meta.get(r["index"], {}).get("level", "")
            if lv:
                by[lv][1] += 1
                by[lv][0] += bool(r["match"])
    def lvnum(s):
        m = re.search(r"(\d+)", s); return int(m.group(1)) if m else 99
    return [(lv, by[lv][0] / by[lv][1] * 100) for lv in sorted(by, key=lvnum)]


# ---------------------------------------------------------------------------
# PAGE
# ---------------------------------------------------------------------------
CSS = """
:root{--ink:#1d1d1f;--dim:#86868b;--line:#e8e8ed;--bg:#fbfbfd;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 -webkit-font-smoothing:antialiased;line-height:1.5}
.wrap{max-width:1040px;margin:0 auto;padding:72px 24px 96px}
.eyebrow{font-size:13px;font-weight:600;letter-spacing:.02em;color:#5856d6;margin:0 0 6px}
h1{font-size:48px;line-height:1.06;letter-spacing:-.02em;font-weight:700;margin:0 0 14px}
h1 .grad{background:linear-gradient(90deg,#5856d6,#0a84ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.lead{font-size:19px;color:var(--dim);max-width:680px;margin:0 0 8px}
.sec{margin-top:72px}
.sec h2{font-size:13px;font-weight:600;letter-spacing:.02em;color:#5856d6;margin:0 0 6px}
.sec h3{font-size:28px;letter-spacing:-.01em;font-weight:700;margin:0 0 22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:26px 28px;margin-bottom:18px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.muted{color:var(--dim)}.small{font-size:12px}
table{border-collapse:collapse;width:100%}
.mtx th,.mtx td{padding:10px 12px;text-align:center;font-size:14px;border-bottom:1px solid var(--line)}
.mtx .rowh{text-align:left;font-weight:600;color:var(--ink);white-space:nowrap}
.mtx thead th{color:var(--dim);font-weight:600;font-size:12px;letter-spacing:.03em}
.cmtx{font-size:13px}.cmtx th{color:var(--dim);font-weight:600;padding:6px}
.cm{width:46px;height:38px;text-align:center;border-radius:7px;font-weight:600}
.cm-r{text-align:right;padding-right:10px;color:var(--ink);font-weight:600;white-space:nowrap}
.cm-c{font-size:11px}
.bars{display:flex;flex-direction:column;gap:11px}
.barrow{display:grid;grid-template-columns:130px 1fr 56px;align-items:center;gap:12px}
.blabel{font-size:13px;color:var(--ink)}
.btrack{height:9px;background:#f0f0f4;border-radius:6px;overflow:hidden}
.bfill{display:block;height:100%;border-radius:6px}
.bval{font-size:13px;font-weight:600;text-align:right;color:var(--dim)}
.exrow{padding:14px 0;border-bottom:1px solid var(--line)}
.exrow:last-child{border:0}
.chip{display:inline-block;font-size:11px;font-weight:600;padding:2px 9px;border-radius:999px;margin-right:6px}
.chip.exp{background:#eef0ff;color:#3b39b3}.chip.lv{background:#f0f0f4;color:#555}
.chip.miss{background:#ffeef0;color:#cf222e}
.ex-q{font-size:14px;margin:7px 0 4px}.ex-meta{font-size:12px;color:var(--dim)}
.foot{margin-top:80px;color:var(--dim);font-size:13px;border-top:1px solid var(--line);padding-top:20px}
.delta-pos{color:#1a7f37;font-weight:600}.delta-neg{color:#cf222e;font-weight:600}
.exp-title{font-size:18px;font-weight:700;margin-bottom:8px}
.para{font-size:15px;line-height:1.65;color:#333;margin:0 0 18px}
.para i{color:#cf222e;font-style:normal;font-weight:600}
.blist{margin:0 0 18px;padding-left:20px;font-size:15px;line-height:1.6;color:#333}
.blist li{margin:5px 0}
.blist i{color:#cf222e;font-style:normal;font-weight:600}
.cm-grid{display:flex;flex-direction:column;gap:16px;margin-top:6px}
.cm-strat{display:flex;align-items:flex-start;gap:14px}
.cm-slabel{font-weight:700;font-size:13px;color:#5856d6;width:42px;padding-top:18px;flex:none}
.cm-pair{display:flex;gap:22px;flex-wrap:wrap}
.cm-block{}
.cm-cap{font-size:12px;color:var(--dim);font-weight:600;margin-bottom:6px}
.cmtx.compact .cm{width:30px;height:26px;font-size:11px;border-radius:5px}
.cmtx.compact th{font-size:10px;padding:3px}
.cmtx.compact .cm-r{padding-right:7px}
"""


def delta_table(cfg):
    exps = [e for e in EXP_ORDER if any((0, e, s) in cfg for s in STRATS)]
    if not exps:
        return ""
    head = "".join(f"<th>{s.upper()}</th>" for s in STRATS)
    body = ""
    for e in exps:
        cells = ""
        for s in STRATS:
            z = cfg.get((0, e, s)); f = cfg.get((5, e, s))
            if z and f:
                za = sum(r["match"] for r in z) / len(z) * 100
                fa = sum(r["match"] for r in f) / len(f) * 100
                d = fa - za
                cls = "delta-pos" if d > 0 else ("delta-neg" if d < 0 else "muted")
                sign = "+" if d > 0 else ""
                cells += f"<td class='{cls}'>{sign}{d:.1f}</td>"
            else:
                cells += "<td class='muted'>—</td>"
        body += f"<tr><th class='rowh'>{EXP_LABELS[e]}</th>{cells}</tr>"
    return (f"<table class='mtx'><thead><tr><th class='rowh'></th>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table><p class='muted small'>few-shot minus zero-shot "
            f"(percentage points). ⚠ few-shot is label-conditioned (demos reveal the class).</p>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--eval-csv", default="../dataset/relative_direction_relations.csv")
    ap.add_argument("--title", default="Relative Reasoning")
    ap.add_argument("--out", default="../relative_error_report.html")
    args = ap.parse_args()

    cfg = load_configs(args.results_dir)
    if not cfg:
        print(f"[ERROR] no checkpoints in {args.results_dir}/")
        return
    meta = load_eval_meta(args.eval_csv)
    has_few = any(sh == 5 for (sh, _, _) in cfg)

    P = []
    P.append(f"<!doctype html><html><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Error Analysis — {args.title}</title><style>{CSS}</style></head><body><div class='wrap'>")
    P.append(f"<p class='eyebrow'>Error Analysis</p>"
             f"<h1>Where the model <span class='grad'>gets it wrong.</span></h1>"
             f"<p class='lead'>{args.title} — GPT-OSS-20B across 6 KG-integration experiments × "
             f"CoT / ToT / GoT, on the OSM-grounded held-out set"
             f"{' (zero-shot vs few-shot)' if has_few else ''}.</p>")
    if not meta:
        P.append("<p class='lead small'>Tip: pass <code>--eval-csv ../dataset/relative_direction_relations.csv</code> "
                 "to unlock per-ambiguity-level accuracy and the failing sentences.</p>")

    # accuracy matrices
    P.append("<div class='sec'><h2>Accuracy</h2><h3>The 6 × 3 scorecard.</h3>")
    if has_few:
        P.append("<div class='grid2'>"
                 f"<div class='card'><b>Zero-shot</b>{matrix_html(cfg,0)}</div>"
                 f"<div class='card'><b>Few-shot (5) ⚠</b>{matrix_html(cfg,5)}</div></div>")
        P.append(f"<div class='card'><b>Few-shot − Zero-shot</b>{delta_table(cfg)}</div>")
    else:
        P.append(f"<div class='card'>{matrix_html(cfg,0)}</div>")
    P.append("</div>")

    # per-predicate + level
    P.append("<div class='sec'><h2>Difficulty</h2><h3>Which relations are hard.</h3><div class='grid2'>")
    P.append(f"<div class='card'><b>Accuracy by predicate</b><div style='height:14px'></div>"
             f"{bars_html(per_predicate(cfg,0))}</div>")
    lvl = level_accuracy(cfg, 0, meta)
    if lvl:
        P.append(f"<div class='card'><b>Accuracy by ambiguity level</b><div style='height:14px'></div>"
                 f"{bars_html(lvl)}</div>")
    else:
        P.append("<div class='card muted'><b>Accuracy by ambiguity level</b>"
                 "<p class='small'>Needs topo_v2_eval.csv (not found).</p></div>")
    P.append("</div></div>")

    # confusion (overall)
    P.append("<div class='sec'><h2>Confusion</h2><h3>Where errors go (overall).</h3>"
             f"<div class='card'>{confusion_html(cfg,0)}</div></div>")

    # per-experiment breakdown: narrative + per-strategy acc + own confusion matrix
    exps_present = [e for e in EXP_ORDER if any((0, e, s) in cfg for s in STRATS)]
    P.append("<div class='sec'><h2>Per-experiment breakdown</h2>"
             "<h3>What goes wrong, experiment by experiment.</h3>")
    for e in exps_present:
        parts, _ = exp_narrative_parts(cfg, e, 0)
        bullets = ''.join(f'<li>{p}</li>' for p in parts) or '<li>No data.</li>'
        conf, labels = confusion_counts(cfg, 0, e)
        cap = (f"rows = expected · columns = predicted · aggregated over CoT/ToT/GoT "
               f"(zero-shot) for {EXP_LABELS[e]}")
        sacc = []
        for s in STRATS:
            rows = cfg.get((0, e, s))
            if rows:
                sacc.append((s.upper(), sum(r["match"] for r in rows) / len(rows) * 100))
        P.append(
            "<div class='card'>"
            f"<div class='exp-title'>{EXP_LABELS[e]}</div>"
            f"<ul class='blist'>{bullets}</ul>"
            "<div class='grid2' style='align-items:start'>"
            f"<div>{bars_html(sacc)}</div>"
            f"<div>{confusion_table_html(conf, labels, cap)}</div>"
            "</div></div>"
        )
    P.append("</div>")

    # hardest examples
    hard = hardest_rows(cfg, 0, meta)
    P.append("<div class='sec'><h2>Hardest cases</h2><h3>Rows most configurations miss.</h3><div class='card'>")
    for h in hard:
        lvl_chip = f"<span class='chip lv'>{h['level']}</span>" if h["level"] else ""
        pair = f"{h['a']} → {h['b']}" if h["a"] else f"row {h['index']}"
        q = f"<div class='ex-q'>“{h['corpus']}”</div>" if h["corpus"] else ""
        P.append(f"<div class='exrow'>"
                 f"<span class='chip miss'>missed {h['miss']}/{h['total']}</span>{lvl_chip}"
                 f"<span class='chip exp'>{h['expected']}</span>"
                 f"<span class='muted small'>{pair}</span>{q}"
                 f"<div class='ex-meta'>predicted instead: {h['wrong']}</div></div>")
    P.append("</div></div>")

    P.append(f"<div class='foot'><b>{args.title}</b> — error analysis · generated from "
             f"{len(cfg)} result checkpoints · PhD Research 2026</div>")
    P.append("</div></body></html>")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("".join(P))
    print(f"[OK] wrote {args.out}  ({len(cfg)} configs, eval-meta={'yes' if meta else 'no'})")


if __name__ == "__main__":
    main()
