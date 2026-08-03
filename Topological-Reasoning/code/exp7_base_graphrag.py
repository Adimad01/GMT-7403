"""
Experiment 7 — Base + OSM KG @ inference (GraphRAG local search)   [Topological]
================================================================================
KG queried as a GRAPH: the k-hop neighbourhood of each entity plus the
shortest path connecting them is retrieved from results/osm_graph.json and
prepended once. Unlike Exp 6 the retrieval is deterministic (no NEXT_QUERY
turns) and carries edges, so multi-hop facts are reachable.

Evidence = Exp 4's static OSM evidence + the retrieved sub-graph, so the
Exp 4 → Exp 7 delta isolates the graph structure itself.

  Model         : openai/gpt-oss-20b  (base, no adapter)
  KG @ training : —         KG @ input : —         KG @ inference : OSM graph
  Tokens        : 1024    Strategies : CoT, ToT, GoT
  Eval set      : topo_v2_eval.csv (105 rows · 15/predicate · 3/level)

Zero-shot (default) vs few-shot:
    python exp7_base_graphrag.py                 # zero-shot, all strategies
    python exp7_base_graphrag.py --strategy cot  # single strategy
    python exp7_base_graphrag.py --shots 5       # few-shot (5 same-label demos, one per level L1-5)

Note: rows whose entities fail OSM retrieval are dropped from eval automatically
(uniform across all experiments). Pass --keep-ungeocodable via the engine to disable.

Requires results/osm_graph.json — build it locally with build_osm_graph.py and commit.
"""
import os, sys, json, argparse

DATASET        = "../dataset/topo_v2_eval.csv"
INDICES_FILE   = "../dataset/topo_v2_eval_indices.json"
TRAIN_DATA     = "../dataset/topo_v2_train.csv"          # for --shots few-shot demo sampling
MODEL_ID       = "openai/gpt-oss-20b"
ADAPTER_PATH   = None
KG_MODE        = "graphrag"
MAX_NEW_TOKENS = 1024
MODEL_TAG      = "exp7_base_graphrag"
OUTPUT_DIR     = "results"
TEMPERATURE    = 0.1
STRATEGIES     = ["cot", "tot", "got"]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=STRATEGIES + ["all"], default="all")
    parser.add_argument("--shots", type=int, default=0,
                        help="0 = zero-shot (default); 5 = few-shot (same-label demos, one per level)")
    args = parser.parse_args()

    if ADAPTER_PATH and not os.path.exists(
            os.path.join(ADAPTER_PATH, "adapter_model.safetensors")):
        print(f"[ERROR] Adapter not found: {ADAPTER_PATH}")
        print("        Train it first (see train_runner_*.py).")
        sys.exit(1)

    print("=" * 70)
    print("  EXPERIMENT 7 — Base + OSM KG @ inference (GraphRAG) · Topological")
    print(f"  adapter={ADAPTER_PATH or 'none'}  kg-mode={KG_MODE}  shots={args.shots}  tokens={MAX_NEW_TOKENS}")
    print("=" * 70)

    argv = [
        "eval_engine_gpu.py",
        "--dataset",        DATASET,
        "--model-id",       MODEL_ID,
        "--filter-indices", INDICES_FILE,
        "--strategy",       args.strategy,
        "--output-dir",     OUTPUT_DIR,
        "--model-tag",      MODEL_TAG,
        "--temperature",    str(TEMPERATURE),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--kg-mode",        KG_MODE,
        "--shots",          str(args.shots),
    ]
    if args.shots > 0:
        argv += ["--train-data", TRAIN_DATA]
    if ADAPTER_PATH:
        argv += ["--adapter-path", ADAPTER_PATH]
    sys.argv = argv

    from eval_engine_gpu import main
    main()


if __name__ == "__main__":
    run()
