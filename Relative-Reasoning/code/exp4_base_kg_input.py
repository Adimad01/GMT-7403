"""
Experiment 4 — Base + OSM KG @ input (static)   [Relative]
================================================================================
KG as static input: base model, OSM evidence prepended once.

  Model         : openai/gpt-oss-20b  (base, no adapter)
  KG @ training : —         KG @ input : OSM       KG @ inference : —   
  Tokens        : 1024    Strategies : CoT, ToT, GoT
  Eval set      : relative_direction_relations.csv (25 balanced eval rows: 5/label, one per level L1-5)

Zero-shot (default) vs few-shot:
    python exp4_base_kg_input.py                 # zero-shot, all strategies
    python exp4_base_kg_input.py --strategy cot  # single strategy
    python exp4_base_kg_input.py --shots 5       # few-shot (5 same-label demos, one per level L1-5)

Note: rows whose entities fail OSM retrieval are dropped from eval automatically
(uniform across all experiments). Pass --keep-ungeocodable via the engine to disable.
"""
import os, sys, json, argparse

DATASET        = "../dataset/relative_direction_relations.csv"
INDICES_FILE   = "../dataset/eval_25_balanced_indices.json"
TRAIN_DATA     = "../dataset/relative_balanced_train.csv"          # for --shots few-shot demo sampling
MODEL_ID       = "openai/gpt-oss-20b"
ADAPTER_PATH   = None
KG_MODE        = "input"
MAX_NEW_TOKENS = 1024
MODEL_TAG      = "exp4_base_kg_input"
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
    print("  EXPERIMENT 4 — Base + OSM KG @ input (static) · Relative")
    print(f"  adapter={ADAPTER_PATH or 'none'}  kg-mode={KG_MODE}  shots={args.shots}  tokens={MAX_NEW_TOKENS}")
    print("=" * 70)

    argv = [
        "eval_engine_relative.py",
        "--dataset",        DATASET,
        "--model-id",       MODEL_ID,
        "--filter-indices", INDICES_FILE,
        "--strategy",       args.strategy,
        "--output-dir",     OUTPUT_DIR,
        "--model-tag",      MODEL_TAG,
        "--temperature",    str(TEMPERATURE),
        "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--kg-mode",        KG_MODE,
        "--keep-ungeocodable",
        "--shots",          str(args.shots),
    ]
    if args.shots > 0:
        argv += ["--train-data", TRAIN_DATA]
    if ADAPTER_PATH:
        argv += ["--adapter-path", ADAPTER_PATH]
    sys.argv = argv

    from eval_engine_relative import main
    main()


if __name__ == "__main__":
    run()
