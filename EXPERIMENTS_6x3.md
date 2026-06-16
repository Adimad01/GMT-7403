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

**KG mechanism = the `--kg-mode` flag** on every eval engine:
`none` (Exp 1/2/3) · `input` = evidence prepended once (Exp 4/5) ·
`rag` = per-step `NEXT_QUERY`→`RETRIEVED` loop during reasoning (Exp 6).

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

## Analysis
After eval, print the 6×3 accuracy matrix per domain:
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
