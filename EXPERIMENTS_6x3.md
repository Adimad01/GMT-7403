# Unified 6-Experiment Design (Topological · Cardinal · Relative)

Each domain runs the **same 6 experiments × 3 strategies (CoT/ToT/GoT)** at a
**1024-token** budget, isolating *where* the OSM knowledge graph enters the model.

| Exp | Script | Adapter | KG @ train | KG @ input | KG @ inference |
|-----|--------|---------|:---:|:---:|:---:|
| 1 | `exp1_base.py`          | base        | — | — | — |
| 2 | `exp2_ft_nokg.py`       | no-KG LoRA  | — | — | — |
| 3 | `exp3_ft_osmkg.py`      | OSM-KG LoRA | OSM | — | — |
| 4 | `exp4_base_kg_input.py` | base        | — | OSM (static) | — |
| 5 | `exp5_ft_kg_input.py`   | no-KG LoRA  | — | OSM (static) | — |
| 6 | `exp6_base_kg_rag.py`   | base        | — | — | OSM (per-step RAG) |
| 7 | `exp7_base_graphrag.py` | base        | — | — | OSM (GraphRAG sub-graph) |

**KG mechanism = the `--kg-mode` flag** on every eval engine:
`none` (Exp 1/2/3) · `input` = evidence prepended once (Exp 4/5) ·
`rag` = per-step `NEXT_QUERY`→`RETRIEVED` loop during reasoning (Exp 6) ·
`graphrag` = k-hop sub-graph + connecting path, prepended once (Exp 7).

## Exp 7 — GraphRAG
Exp 6 retrieves one *place record* at a time and nothing connects the records, so
no multi-hop fact is reachable. Exp 7 retrieves a *sub-graph*:

- **Graph** — `graph_kg.py` derives nodes (geocoded places + administrative levels)
  and edges (`within`/`contains` from the Nominatim hierarchy, `near` from haversine)
  straight out of `results/osm_cache.json`. No LLM extraction; OSM is already structured.
- **Retrieval** — GraphRAG *local search*: both entities are given, so there is no
  embedding step. Pull each entity's containment chain and nearest neighbours, plus the
  shortest path between them. Global (community-summary) search is deliberately not
  implemented — every row is a two-entity classification, not corpus-wide sensemaking.
- **Evidence** = Exp 4's static OSM evidence **+** the sub-graph, so the Exp 4 → Exp 7
  delta isolates the graph structure itself.
- **Fallback** — entities missing from the graph keep Exp 4 evidence and stay in eval,
  so the row set matches the other six exactly.

Build the artifact locally (deterministic, network-free — it only reads the cache) and
commit it; the GPU server has no internet:
```bash
python build_osm_graph.py          # each */code dir → results/osm_graph.json
python exp7_base_graphrag.py       # zero-shot ; --shots 5 for few-shot
```

### ⚠ Known limitation, measured before running
On the Topological eval set only **54 / 105** geocodable rows have a connecting path, and
the path signature does **not** separate the labels (containment paths occur for `contains`
10, `within` 8, `equals` 8, `touches` 5, `disjoint` 4, `overlaps` 3, `crosses` 2).

**Geocode corruption (fixed 2026-08-28).** A legacy `countrycodes=us,mx` filter mis-resolved
global entities in **all three domains**, not just Cardinal: `Australia`, `Argentina`,
`Portugal`, `Chile` → Mexico. The filter is removed and all three caches re-warmed
(`warm_osm_cache.py --force`). Row coverage: Cardinal 95% → 99%, Relative 57% → 71%.
Residual: `State of Colorado` and `State of Tasmania` still mis-resolve because the
preferred-class heuristic in `osm_client.py` takes the first result whose class is in the
list rather than the highest-priority class, so a `natural/peak` can outrank a
`boundary/administrative`.

## Shared modules (duplicated into each `*/code/` dir)
- `osm_client.py` — Nominatim client + cache, geometry helpers (bearing, offset,
  haversine), `OSMEvidenceKG`, `NullKG`.
- `rag_loop.py` — generic bounded `RAGStrategy` (Exp 6), driven by a `DomainSpec`.
- `warm_osm_cache.py` — pre-warm `results/osm_cache.json` (run LOCALLY).

## ⚠️ The GPU server has no internet
OSM evidence at inference reads from `results/osm_cache.json`. **Warm it locally
first**, then commit + push the JSON:
```bash
# from each */code dir, with internet:
python warm_osm_cache.py --dataset ../dataset/topo_v2_eval.csv          # Topological
python warm_osm_cache.py --dataset ../dataset/cardinal_direction_relations.csv
python warm_osm_cache.py --dataset ../dataset/relative_direction_relations.csv
```

## Run order (per domain, on the A100)
```bash
# 1. Build training data  (OSM-KG builders run LOCALLY to embed real coordinates)
#    Topological: build_dataset_topological_v2.py ; osm_kg_balanced_train.jsonl exists
#    Cardinal:    python build_cardinal_train_data.py
#    Relative:    python build_relative_train_data.py

# 2. Train the 2 adapters per domain
#    Topological: train_runner_topo_v2.py        + train_runner_osm_kg.py
#    Cardinal:    train_runner_cardinal_nokg.py   + train_runner_cardinal_osm_kg.py
#    Relative:    train_runner_relative.py        + train_runner_relative_osm_kg.py

# 3. Run the experiments (each = 3 strategies)
for e in exp1_base exp2_ft_nokg exp3_ft_osmkg exp4_base_kg_input exp5_ft_kg_input exp6_base_kg_rag; do
    python $e.py            # or: python $e.py --strategy cot
done
```

## Adapters per domain
| Domain | no-KG (Exp 2/5) | OSM-KG (Exp 3) |
|--------|-----------------|----------------|
| Topological | `finetuned_gptoss_topo_v2` | `finetuned_gptoss_osm_kg` |
| Cardinal    | `finetuned_gptoss_cardinal` | `finetuned_gptoss_cardinal_osm_kg` |
| Relative    | `finetuned_gptoss_relative` | `finetuned_gptoss_relative_osm_kg` |

## Zero-shot vs few-shot
Every experiment runs in two prompting modes via `--shots`:
- **Zero-shot** (`--shots 0`, default) — the current pipeline.
- **Few-shot** (`--shots 5`) — prepends **5 demonstrations from the train split, one per
  ambiguity level (L1–L5), all sharing the target row's label**. Tagged `_fs5` in outputs.
  ⚠ **Label-conditioned by design:** the demos reveal the answer class, so few-shot numbers
  are a leakage-aware probe, *not* a clean baseline — compare them to zero-shot, not across labels.
```bash
python exp1_base.py              # zero-shot
python exp1_base.py --shots 5    # few-shot (uses the domain's train CSV for demos)
```

## OSM-failure filtering
Rows whose entities can't be geocoded (absent/`null` in `osm_cache.json`) are **dropped from
eval automatically**, uniformly across all 6 experiments, so the comparison stays fair. The
OSM-KG training builders likewise **skip ungeocodable rows**. Warm the cache first
(`warm_osm_cache.py`); pass `--keep-ungeocodable` to the engine to disable. The standalone
`drop_ungeocodable.py` can also post-filter already-produced checkpoints.

## Analysis
After eval, print the accuracy matrices per domain (zero-shot and few-shot shown separately):
```bash
python analyze_results.py        # run inside each */code dir
```

## Notes
- Relative OSM evidence is **informational only** (no coordinates/observer
  heading in the data); the LLM still infers left/right/front/behind from the
  corpus. Expect smaller KG gains than Topological/Cardinal.
- The repo was pruned to this pipeline only: the legacy experiments
  (Topological `exp01–06`/`exp_v2_*`, Wikidata + static-KG variants, Cardinal
  shore/compass task), their datasets, old result files, and stale
  analysis/visualization scripts were removed. They remain recoverable from git
  history if ever needed.

## Statistical protocol
Eval sets are small (Topological 105, Cardinal 40, Relative 25). The resolution floor —
the tightest 95% CI the set can produce — is **±9.4pp / ±14.8pp / ±18.2pp** respectively.
Any claimed effect smaller than that cannot be seen at this n, whatever the point estimate.

Generation uses `do_sample=True`, so every run is a stochastic draw. Each engine now takes
`--seed`, seeded **per prompt** so a row's output is independent of processing order and of
checkpoint-resume state. The seed is folded into the output filename, so seeds do not
overwrite one another.

```bash
for s in 1 2 3; do python exp1_base.py --seed $s; done
python ../../stats_analysis.py --by-level
```

`stats_analysis.py` reports:
- **Wilson 95% CIs** on every accuracy (not the normal approximation, which undercovers at
  small n and near 0/100%).
- **Exact McNemar** against the baseline arm. All experiments score the *same* eval rows, so
  comparisons are paired — far more powerful here than two-independent-proportions.
- **Holm–Bonferroni** correction across the comparison family.
- **Bootstrap CIs** on the paired accuracy delta, resampling rows to preserve pairing.

⚠ **Seeds are repeated measurements of the same rows, not extra rows.** They are collapsed
to one verdict per row before any test. Crossing baseline seeds with variant seeds would
replicate each row seeds² times and manufacture significance — verified against synthetic
data where only one arm carried a real effect.
