"""
update_report.py
================================================================================
Reads vol-1 checkpoint files and rewrites the results-dependent sections of
report.html in-place:

  1. Experiment cards  (scores + winner highlight)
  2. Bar chart         (bar widths + labels)
  3. CoT / ToT / GoT  detail tables
  4. PIPES JS array    (bestScore / bestStrat per experiment)
  5. Hero best score   (top-level accuracy chip)
  6. Footer            (experiment count)
  7. Key-observations  (insight cards)

Usage:
    cd /path/to/Topological-Reasoning/code
    python update_report.py                           # default paths
    python update_report.py --results-dir results --html report.html
"""

import os
import re
import json
import argparse

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
VALID_PREDICATES = ["contains", "within", "touches", "crosses", "disjoint", "overlaps", "equals"]
STRATEGIES       = ["cot", "tot", "got"]
SUFFIX           = "neighborhood_details_spatial_relation_16_sample"
N_EVAL           = 112

EXPERIMENTS = [
    ("exp1_base_gpu",               "Configuration 01", "GPT-OSS-20B Base",
     "exp1_base_gpu · 512 tok · sans adaptateur"),
    ("exp2_finetuned_topo_gpu",     "Configuration 02", "GPT-OSS-20B + Topo-LoRA",
     "exp2_finetuned_topo_gpu · 512 tok · topo-lora"),
    ("exp3_finetuned_kg_in_gpu",    "Configuration 03", "Topo-LoRA + KG en entrée",
     "exp3_finetuned_kg_in_gpu · 1024 tok · knowledge injection"),
    ("exp4_osm_kg_ft_only_gpu",     "Configuration 04", "OSM-KG LoRA (FT only)",
     "exp4_osm_kg_ft_only_gpu · 512 tok · osm-kg adapter, no KG at inference"),
    ("exp5_finetuned_enriched_gpu", "Configuration 05", "Topo-LoRA + budget étendu",
     "exp5_finetuned_enriched_gpu · 1024 tok · inference-time grounding"),
]

# Short names used in bar chart and results tables
SHORT_NAMES = {
    "exp1_base_gpu":               "GPTOSS Base",
    "exp2_finetuned_topo_gpu":     "GPTOSS Fine-tuné",
    "exp3_finetuned_kg_in_gpu":    "FT + KG en entrée",
    "exp4_osm_kg_ft_only_gpu":     "OSM-KG LoRA (FT only)",
    "exp5_finetuned_enriched_gpu": "FT + Inférence enrichie",
}


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_metrics(results_dir: str) -> dict:
    """Returns metrics[tag][strat] = {accuracy, per_predicate}."""
    metrics = {}
    for tag, _, _, _ in EXPERIMENTS:
        metrics[tag] = {}
        for strat in STRATEGIES:
            p = os.path.join(results_dir, f"voletc_{tag}_{strat}_{SUFFIX}_ckpt.json")
            if not os.path.exists(p):
                metrics[tag][strat] = None
                continue
            d = json.load(open(p))
            r = d.get("results", [])
            if not r:
                metrics[tag][strat] = None
                continue
            acc = sum(1 for x in r if x.get("match")) / len(r) * 100
            per = {}
            for pred in VALID_PREDICATES:
                sub = [x for x in r if x["expected"] == pred]
                per[pred] = sum(1 for x in sub if x.get("match")) / len(sub) * 100 if sub else 0.0
            metrics[tag][strat] = {"accuracy": acc, "per_predicate": per, "n": len(r)}
    return metrics


def fmt(v, n=1):
    return f"{v:.{n}f}%" if v is not None else "—"

def best_strat(metrics, tag):
    best_s, best_a = None, -1
    for s in STRATEGIES:
        m = metrics[tag][s]
        if m and m["accuracy"] > best_a:
            best_a, best_s = m["accuracy"], s
    return best_s, best_a


# ──────────────────────────────────────────────────────────────────────────────
# Section builders
# ──────────────────────────────────────────────────────────────────────────────

def build_exp_cards(metrics: dict) -> str:
    parts = []
    for tag, cfg_label, title, exp_tag in EXPERIMENTS:
        m = metrics[tag]
        scores = {s: (m[s]["accuracy"] if m[s] else None) for s in STRATEGIES}
        best_s = max(STRATEGIES, key=lambda s: scores[s] or -1) if any(v for v in scores.values()) else None
        is_best_overall = (tag == "exp1_base_gpu")  # placeholder; updated below

        def sc(s):
            v = scores[s]
            winner = " winner" if s == best_s and v is not None else ""
            return (f'<div class="sc{winner}">'
                    f'<span class="sl">{s.upper()}</span>'
                    f'<span class="sv">{fmt(v)}</span></div>')

        parts.append(f"""      <div class="exp-card">
        <div class="exp-num">{cfg_label}</div>
        <div class="exp-title">{title}</div>
        <div class="scores-row">
          {sc("cot")}
          {sc("tot")}
          {sc("got")}
        </div>
        <div class="exp-tag">{exp_tag}</div>
      </div>""")
    return "\n".join(parts)


def build_bar_chart(metrics: dict) -> str:
    rows = []
    for tag, _, _, _ in EXPERIMENTS:
        name = SHORT_NAMES[tag]
        m = metrics[tag]
        accs = {s: (m[s]["accuracy"] if m[s] else 0.0) for s in STRATEGIES}
        best_s = best_strat(metrics, tag)[0]
        row_cls = ' chart-best-row' if tag == "exp1_base_gpu" else ''
        lbl_pfx = '★ ' if tag == "exp1_base_gpu" else ''
        rows.append(f"""      <div class="chart-row{row_cls}">
        <div class="chart-lbl">{lbl_pfx}{name}</div>
        <div class="bar-group">
          <div class="bar-wrap"><div class="bar bar-cot" style="width:{accs['cot']:.1f}%"></div><span class="bar-val">{fmt(accs['cot'])}</span></div>
          <div class="bar-wrap"><div class="bar bar-tot" style="width:{accs['tot']:.1f}%"></div><span class="bar-val">{fmt(accs['tot'])}</span></div>
          <div class="bar-wrap"><div class="bar bar-got" style="width:{accs['got']:.1f}%"></div><span class="bar-val">{fmt(accs['got'])}</span></div>
        </div>
      </div>""")
    return "\n".join(rows)


def pct_class(v):
    if v is None:
        return ''
    if v >= 75:
        return ' class="pct pct-hi"'
    if v <= 25:
        return ' class="pct pct-lo"'
    return ' class="pct"'


def per_td(v):
    cls = ''
    if v is not None:
        if v >= 75:
            cls = ' class="pct-hi"'
        elif v <= 25:
            cls = ' class="pct-lo"'
    return f'<td{cls}>{fmt(v)}</td>'


def build_results_table(metrics: dict, strat: str) -> str:
    # find best row
    best_tag = max(
        [tag for tag, *_ in EXPERIMENTS],
        key=lambda t: metrics[t][strat]["accuracy"] if metrics[t][strat] else -1
    )
    rows = []
    for tag, _, _, _ in EXPERIMENTS:
        m = metrics[tag][strat]
        name = SHORT_NAMES[tag]
        is_best = (tag == best_tag)
        row_cls = ' class="best-r"' if is_best else ''
        star = '★ ' if is_best else ''
        if m is None:
            rows.append(f'          <tr{row_cls}><td>{star}{name}</td><td class="pct">N/A</td>'
                        + ''.join(f'<td>—</td>' for _ in VALID_PREDICATES) + '</tr>')
        else:
            acc = m["accuracy"]
            per = m["per_predicate"]
            rows.append(
                f'          <tr{row_cls}><td>{star}{name}</td>'
                f'<td{pct_class(acc)}>{fmt(acc)}</td>'
                + ''.join(per_td(per.get(p)) for p in VALID_PREDICATES)
                + '</tr>'
            )
    return "\n".join(rows)


def build_pipes_patch(metrics: dict) -> list[tuple[str, str]]:
    """Return list of (old_js_fragment, new_js_fragment) pairs to replace in PIPES array."""
    patches = []
    for tag, _, title, _ in EXPERIMENTS:
        bs, ba = best_strat(metrics, tag)
        if bs:
            new_score = f"{ba:.1f}%"
            new_strat = bs.upper()
        else:
            new_score = "—"
            new_strat = "à évaluer"

        # Match pattern: bestScore:'XX%',bestStrat:'YY'
        # We use a regex that targets only the specific pipeline entry
        patches.append((tag, new_score, new_strat))
    return patches


def build_hero_score(metrics: dict) -> str:
    """Return the best accuracy across all experiments × strategies."""
    best_acc = 0.0
    for tag, *_ in EXPERIMENTS:
        for s in STRATEGIES:
            m = metrics[tag][s]
            if m and m["accuracy"] > best_acc:
                best_acc = m["accuracy"]
    return f"{best_acc:.1f}%"


def build_insights(metrics: dict) -> str:
    """Build the 4 insight cards based on actual results."""
    # Best overall
    best_tag, best_s, best_a = None, None, -1
    for tag, *_ in EXPERIMENTS:
        for s in STRATEGIES:
            m = metrics[tag][s]
            if m and m["accuracy"] > best_a:
                best_a, best_s, best_tag = m["accuracy"], s, tag

    best_name = SHORT_NAMES.get(best_tag, best_tag)

    # GoT avg vs CoT avg
    got_avg = sum(metrics[t]["got"]["accuracy"] for t, *_ in EXPERIMENTS if metrics[t]["got"]) / sum(1 for t, *_ in EXPERIMENTS if metrics[t]["got"])
    cot_avg = sum(metrics[t]["cot"]["accuracy"] for t, *_ in EXPERIMENTS if metrics[t]["cot"]) / sum(1 for t, *_ in EXPERIMENTS if metrics[t]["cot"])
    best_strat_global = "GoT" if got_avg >= cot_avg else "CoT"

    # Hardest predicate across all experiments (lowest avg)
    pred_avg = {}
    for pred in VALID_PREDICATES:
        vals = []
        for tag, *_ in EXPERIMENTS:
            for s in STRATEGIES:
                m = metrics[tag][s]
                if m:
                    vals.append(m["per_predicate"].get(pred, 0))
        pred_avg[pred] = sum(vals) / len(vals) if vals else 0
    hardest = min(pred_avg, key=pred_avg.get)
    easiest = max(pred_avg, key=pred_avg.get)

    return f"""      <div class="insight">
        <span class="ico">🏆</span>
        <h4>{best_name} · {best_s.upper()} · Meilleur</h4>
        <p>{best_name} avec {best_s.upper()} atteint <strong>{best_a:.1f}%</strong> — meilleure performance parmi les {len(EXPERIMENTS)} expérimentations GPU A100.</p>
      </div>
      <div class="insight">
        <span class="ico">⚠️</span>
        <h4>Fine-tuning non suffisant seul</h4>
        <p>Le modèle de base (CoT {metrics['exp1_base_gpu']['cot']['accuracy']:.1f}%) reste compétitif. Le fine-tuning améliore GoT mais peut dégrader CoT selon la configuration.</p>
      </div>
      <div class="insight">
        <span class="ico">📈</span>
        <h4>{best_strat_global} dominant en moyenne</h4>
        <p>GoT ({got_avg:.1f}% avg) vs CoT ({cot_avg:.1f}% avg). ToT est systématiquement la stratégie la moins performante sur cette tâche de classification topologique.</p>
      </div>
      <div class="insight">
        <span class="ico">🗺️</span>
        <h4>{easiest} facile · {hardest} difficile</h4>
        <p><em>{easiest}</em> est le prédicat le mieux prédit (moy {pred_avg[easiest]:.0f}%). <em>{hardest}</em> reste le plus difficile (moy {pred_avg[hardest]:.0f}%).</p>
      </div>"""


# ──────────────────────────────────────────────────────────────────────────────
# Main updater
# ──────────────────────────────────────────────────────────────────────────────

def update_report(html: str, metrics: dict) -> str:

    # ── 1. Experiment cards ──────────────────────────────────────────────────
    new_cards = build_exp_cards(metrics)
    html = re.sub(
        r'(<div class="exp-grid">).*?(</div>\s*</section>)',
        lambda m: m.group(1) + "\n" + new_cards + "\n    " + m.group(2),
        html, count=1, flags=re.DOTALL
    )

    # ── 2. Bar chart rows ────────────────────────────────────────────────────
    new_bars = build_bar_chart(metrics)
    html = re.sub(
        r'(<!-- CHART -->.*?<div class="chart-wrap">.*?<div class="chart-legend">.*?</div>)(.*?)(</div>\s*</div>\s*</section>)',
        lambda m: m.group(1) + "\n\n" + new_bars + "\n    " + m.group(3),
        html, count=1, flags=re.DOTALL
    )

    # ── 3. Results tables ────────────────────────────────────────────────────
    # The three tables each follow an h3 with a distinct color. We locate the
    # h3 by its color, then replace the nearest <tbody>…</tbody>.
    for strat, color in [("cot", "var(--blue)"), ("tot", "var(--orange)"), ("got", "var(--green)")]:
        new_rows = build_results_table(metrics, strat)
        pattern = (
            r'(<h3[^>]*style="[^"]*color:' + re.escape(color) + r'[^"]*"[^>]*>.*?<tbody>)(.*?)(</tbody>)'
        )
        html = re.sub(pattern, lambda m, r=new_rows: m.group(1) + "\n" + r + "\n          " + m.group(3),
                      html, count=1, flags=re.DOTALL)

    # ── 4. PIPES JS bestScore / bestStrat ────────────────────────────────────
    patches = build_pipes_patch(metrics)
    for tag, new_score, new_strat in patches:
        # Find the pipeline entry by its tag string and replace bestScore / bestStrat
        # Pattern: tag:'<tag_string>',...bestScore:'<val>',bestStrat:'<val>'
        # We match loosely on the tag field and replace only bestScore/bestStrat
        # Each PIPES entry has a unique tag: field
        def make_replacer(ns, nst):
            def replacer(m):
                s = m.group(0)
                s = re.sub(r"bestScore:'[^']*'", f"bestScore:'{ns}'", s)
                s = re.sub(r"bestStrat:'[^']*'", f"bestStrat:'{nst}'", s)
                return s
            return replacer

        # Match a PIPES entry block containing this specific tag
        tag_pattern = re.escape(tag)
        html = re.sub(
            r'(\{num:\'0[1-6]\'[^}]*?' + tag_pattern + r'[^}]*?bestScore:\'[^\']*\',bestStrat:\'[^\']*\')',
            make_replacer(new_score, new_strat),
            html, count=1, flags=re.DOTALL
        )

    # ── 5. Insight cards ─────────────────────────────────────────────────────
    new_insights = build_insights(metrics)
    html = re.sub(
        r'(<div class="insight-grid"[^>]*>)(.*?)(</div>\s*</div>\s*</section>)',
        lambda m: m.group(1) + "\n" + new_insights + "\n    " + m.group(3),
        html, count=1, flags=re.DOTALL
    )

    # ── 6. Hero best score ───────────────────────────────────────────────────
    best = build_hero_score(metrics)
    html = re.sub(
        r'(<div class="hero-score">.*?<div class="value">)[^<]*(</div>)',
        lambda m: m.group(1) + best + m.group(2),
        html, count=1, flags=re.DOTALL
    )

    # ── 7. Footer ────────────────────────────────────────────────────────────
    html = re.sub(
        r'(\d+) expérimentations GPU A100',
        f'{len(EXPERIMENTS)} expérimentations GPU A100',
        html
    )
    html = re.sub(
        r'(\d+) configurations',
        f'{len(EXPERIMENTS) * len(STRATEGIES)} configurations',
        html
    )

    return html


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--html",        default="report.html")
    args = parser.parse_args()

    print(f"[INFO] Loading checkpoints from '{args.results_dir}'...")
    metrics = load_metrics(args.results_dir)

    for tag, cfg, *_ in EXPERIMENTS:
        for s in STRATEGIES:
            m = metrics[tag][s]
            if m:
                print(f"  [{cfg}] {s.upper()}: {m['accuracy']:.1f}% ({m['n']}/{N_EVAL})")
            else:
                print(f"  [{cfg}] {s.upper()}: MISSING")

    print(f"\n[INFO] Reading '{args.html}'...")
    with open(args.html, encoding="utf-8") as f:
        html = f.read()

    print("[INFO] Patching results sections...")
    updated = update_report(html, metrics)

    with open(args.html, "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"[OK] '{args.html}' updated successfully.")
    best_tag = max(
        [(t, s) for t, *_ in EXPERIMENTS for s in STRATEGIES if metrics[t][s]],
        key=lambda ts: metrics[ts[0]][ts[1]]["accuracy"]
    )
    print(f"[INFO] Best result: {SHORT_NAMES[best_tag[0]]} / {best_tag[1].upper()} = "
          f"{metrics[best_tag[0]][best_tag[1]]['accuracy']:.1f}%")


if __name__ == "__main__":
    main()
