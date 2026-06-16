"""
Experiment 6 — Base + OSM KG @ inference (per-step RAG)   [Cardinal]
================================================================================
KG via per-step retrieval: base model, RAG loop during reasoning.

  Model         : openai/gpt-oss-20b  (base, no adapter)
  KG @ training : —         KG @ input : —         KG @ inference : OSM 
  Tokens        : 1024    Strategies : CoT, ToT, GoT
  Eval set      : cardinal_direction_relations.csv (32 balanced eval rows)

Run:
    python exp6_base_kg_rag.py                # all strategies
    python exp6_base_kg_rag.py --strategy cot
"""
import os, sys, json, argparse

DATASET        = "../dataset/cardinal_direction_relations.csv"
INDICES_FILE   = "../dataset/eval_32_balanced_indices.json"
MODEL_ID       = "openai/gpt-oss-20b"
ADAPTER_PATH   = None
KG_MODE        = "rag"
MAX_NEW_TOKENS = 1024
MODEL_TAG      = "exp6_base_kg_rag"
OUTPUT_DIR     = "results"
TEMPERATURE    = 0.1
STRATEGIES     = ["cot", "tot", "got"]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=STRATEGIES + ["all"], default="all")
    args = parser.parse_args()

    if ADAPTER_PATH and not os.path.exists(
            os.path.join(ADAPTER_PATH, "adapter_model.safetensors")):
        print(f"[ERROR] Adapter not found: {ADAPTER_PATH}")
        print("        Train it first (see train_runner_*.py).")
        sys.exit(1)

    print("=" * 70)
    print("  EXPERIMENT 6 — Base + OSM KG @ inference (per-step RAG) · Cardinal")
    print(f"  adapter={ADAPTER_PATH or 'none'}  kg-mode={KG_MODE}  tokens={MAX_NEW_TOKENS}")
    print("=" * 70)

    argv = [
        "eval_engine_cardinal.py",
        "--dataset",        DATASET,
        "--model-id",       MODEL_ID,
        "--filter-indices", INDICES_FILE,
        "--strategy",       args.strategy,
        "--output-dir",     OUTPUT_DIR,
        "--model-tag",      MODEL_TAG,
        "--temperature",    str(TEMPERATURE),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--kg-mode",        KG_MODE,
    ]
    if ADAPTER_PATH:
        argv += ["--adapter-path", ADAPTER_PATH]
    sys.argv = argv

    from eval_engine_cardinal import main
    main()


if __name__ == "__main__":
    run()
