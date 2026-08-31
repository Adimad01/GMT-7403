# Spatial Relation Reasoning — Prompting Strategy Evaluation

Evaluates how a large language model reasons about **spatial relations**
expressed in natural language, comparing five prompting strategies across three
relation families.

The design goal is comparability: every strategy is measured on **byte-identical
examples** with **byte-identical few-shot demonstrations**, so a difference
between two numbers is a difference between two strategies and nothing else.

---

## What is measured

**Task.** Given a natural-language description of how two places are arranged,
predict the relation that holds between them.

**Three relation families**, each with its own label vocabulary:

| relation | labels | eval rows | independent facts |
|---|---|---:|---:|
| `topological` | contains, within, touches, crosses, disjoint, overlaps, equals | 105 | 68 |
| `cardinal` | north_of, south_of, east_of, west_of, northeast_of, northwest_of, southeast_of, southwest_of | 40 | 39 |
| `relative` | left_of, right_of, in_front_of, behind, next_to | 25 | 20 |

**Five prompting strategies:**

| strategy | calls/row | what it does |
|---|---:|---|
| `zero_shot` | 1 | Direct question. No demonstrations, no reasoning scaffold. |
| `few_shot` | 1 | Five pinned demonstrations, then the question. |
| `cot` | 1 | One linear reasoning chain, then the answer. |
| `tot` | 4 | Three independent reasoning branches, then adjudication. |
| `got` | 4 | Three partial thoughts, explicitly cross-linked, then synthesis. |

`zero_shot` and `few_shot` differ in *context*; `cot`, `tot` and `got` differ in
*reasoning structure*. All five are run as peer arms.

---

## Why "independent facts" is a separate column

Some evaluation rows assert the **same** (subject, object, label) more than once
at different ambiguity levels. Topological has 105 rows but only 68 distinct
facts. Those rows are not independent observations, so every metric is reported
twice: over all rows, and clustered on `fact_id`. Use the clustered figure when
stating a result.

Each relation also has a **resolution floor** — the tightest confidence interval
its evaluation set can produce. Differences smaller than the floor cannot be
resolved regardless of the point estimates:

| relation | independent facts | resolution floor |
|---|---:|---|
| topological | 68 | ±11.7 pp |
| cardinal | 39 | ±15.0 pp |
| relative | 20 | ±20.1 pp |

---

## Repository layout

```
data/
  <relation>/
    corpus.csv            source corpus
    train.csv             training split (few-shot demo pool)
    eval.csv              evaluation split          (topological only)
    eval_indices.json     evaluation row indices    (cardinal, relative)
    eval_manifest.json    ** the pinned evaluation set, with sha256 **
    fewshot_manifest.json ** eval row -> exact demo rows, with sha256 **
    kg_train.jsonl        knowledge-graph training data (future experiments)
    osm/                  OpenStreetMap cache and derived graph (future)
  _source_collection/     original hand-authored corpora

src/spatial_eval/
  config.py               relations, labels, model settings, paths
  data.py                 manifest-verified dataset access
  model.py                backend abstraction  (hf | mock)
  parsing.py              completion -> label
  metrics.py              accuracy, Wilson intervals, per-label F1, clustering
  report.py               cross-strategy comparison
  runner.py               execution, resume, logging, error handling
  cli.py                  command line entry point
  strategies/
    base.py               Strategy interface + registry
    zero_shot.py few_shot.py cot.py tot.py got.py

results/<relation>/<strategy>/seed<N>/
  predictions.jsonl       one JSON object per evaluation row
  metrics.json            computed metrics
  run.json                config, manifest hashes, counts, timing
  traces.jsonl            every prompt and completion   (--save-traces)

tests/test_pipeline.py    offline suite, no GPU required
scripts/run_all.sh        full grid in one command
```

---

## Reproducibility

Everything that can change a number is recorded in `run.json`: model id,
temperature, `max_new_tokens`, dtype, seed, and the sha256 of both manifests.

**Seeding is per prompt**, derived from `(seed, prompt text)`. Seeding once at
startup would make a row's answer depend on how many rows ran before it, so a
resumed run would disagree with a clean one. Per-prompt seeding makes each row
reproducible on its own.

**Generation is stochastic** (`do_sample=True`, temperature 0.1). Run several
seeds and report the spread; a single seed leaves run-to-run variance unmeasured
and the report flags it.

---

## Setup

```bash
git clone https://github.com/Adimad01/GMT-7403.git && cd GMT-7403
python3 -m pip install --user -r requirements.txt
python3 -m pip install --user -e .
```

### Invoking the tool

Two equivalent forms. The module form needs no PATH setup and is used
throughout this README and in `scripts/run_all.sh`:

```bash
python3 -m spatial_eval.cli verify
```

The console script is shorter but `pip install --user` puts it in
`~/.local/bin`, which is often not on PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"        # add to ~/.bashrc to persist
spatial-eval verify
```

`torch` is deliberately **not** pinned — it ships with the cluster image
(2.5.1+cu121) and reinstalling it risks breaking the MIG-capable CUDA build.

### Environment notes for the GPU cluster

- `transformers` must be `>=4.55,<5`. Version 5.x replaced the MXFP4 loader; the
  dequantisation patch no longer applies and model loading fails on MIG with an
  NVML assert in the CUDA caching allocator. The code refuses to run on 5.x.
- `huggingface-hub` must be `>=0.34`. pip has been observed resolving it *down*
  to 0.16.4 while installing transformers 4.57, which breaks model download.
- TensorFlow is switched off (`USE_TF=0`). transformers imports TF through
  `image_transforms` when TF looks importable, and the cluster's TF has
  protobuf-incompatible generated code that takes the whole import chain down.
  Nothing here uses TF.

---

## Running

Verify the data before anything else. This checks both manifest hashes and
exits non-zero on a mismatch:

```bash
python3 -m spatial_eval.cli verify
```

Dry-run the whole pipeline without a GPU:

```bash
python3 -m spatial_eval.cli run --all --backend mock --limit 5
```

One cell:

```bash
python3 -m spatial_eval.cli run -r cardinal -s cot --seeds 1
```

Everything (3 relations × 5 strategies × 3 seeds = 45 cells):

```bash
python3 -m spatial_eval.cli run --all --seeds 1 2 3
```

Rerun only rows that errored, keeping every success:

```bash
python3 -m spatial_eval.cli run --all --seeds 1 2 3          # resume is the default
```

Metrics and comparison:

```bash
python3 -m spatial_eval.cli evaluate
spatial-eval report --metric accuracy_by_fact --per-label
spatial-eval report --csv results/comparison.csv --json results/comparison.json
```

---

## Adding a strategy

Create `src/spatial_eval/strategies/my_strategy.py`:

```python
from ..data import Example
from .base import Context, Strategy, register

@register
class MyStrategy(Strategy):
    name = "my_strategy"
    description = "One line, shown by `spatial-eval list`."

    def build_prompt(self, ex: Example, ctx: Context) -> str:
        return self.task_header(ctx.relation, ctx.labels) + self.question(ex) \
             + self.answer_instruction()
```

Import it in `strategies/__init__.py`. It is runnable immediately — the CLI,
runner and report all read the registry.

Override `run()` instead of `build_prompt()` for multi-call strategies; see
`tot.py`.

## Adding a model

Subclass `Backend` in `model.py`, implement `generate(prompt, seed)`, decorate
with `@register_backend("name")`, then pass `--backend name`.

---

## Tests

```bash
python3 tests/test_pipeline.py
```

No GPU and no model download: a mock backend stands in. The suite covers the
guarantees that would otherwise fail silently — identical examples across
strategies, pinned demonstrations, seed determinism, resume correctness,
per-row error isolation, and metric correctness against reference values.

---

## Known limitations

- **`relative` is underpowered.** 20 independent facts and a ±20.1 pp floor. Run
  it, report it, but do not draw a strategy conclusion from it alone.
- **Few-shot demonstrations are label-conditioned by design** — all five share
  the target row's gold label. That reveals the answer class, so few-shot is a
  leakage-aware probe against zero-shot, not a clean baseline, and it is not
  comparable across labels.
- **The OSM cache under `data/*/osm/` is not yet trustworthy.** Roughly 27 % of
  cardinal entity pairs have cached coordinates two or more compass sectors away
  from the gold label, because the geocoder's class heuristic prefers points of
  interest over administrative boundaries. It is unused by the five strategies
  here and must be repaired before any knowledge-graph experiment.
