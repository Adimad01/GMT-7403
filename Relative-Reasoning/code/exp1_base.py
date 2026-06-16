"""
Experiment 1 — Base GPT-OSS-20B (no KG)   [Relative]
================================================================================
Pure baseline: base model, no adapter, no KG anywhere.

  Model         : openai/gpt-oss-20b  (base, no adapter)
  KG @ training : —         KG @ input : —         KG @ inference : —   
  Tokens        : 1024    Strategies : CoT, ToT, GoT
  Eval set      : relative_direction_relations.csv (20 balanced eval rows)

Run:
    python exp1_base.py                # all strategies
    python exp1_base.py --strategy cot
"""
import os, sys, json, argparse

DATASET        = "../dataset/relative_direction_relations.csv"
INDICES_FILE   = "../dataset/eval_20_balanced_indices.json"
MODEL_ID       = "openai/gpt-oss-20b"
ADAPTER_PATH   = None
KG_MODE        = "none"
MAX_NEW_TOKENS = 1024
MODEL_TAG      = "exp1_base"
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
    print("  EXPERIMENT 1 — Base GPT-OSS-20B (no KG) · Relative")
    print(f"  adapter={ADAPTER_PATH or 'none'}  kg-mode={KG_MODE}  tokens={MAX_NEW_TOKENS}")
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
    ]
    if ADAPTER_PATH:
        argv += ["--adapter-path", ADAPTER_PATH]
    sys.argv = argv

    from eval_engine_relative import main
    main()


if __name__ == "__main__":
    run()
