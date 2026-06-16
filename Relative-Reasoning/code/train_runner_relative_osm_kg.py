"""
train_runner_relative_osm_kg.py
================================================================================
OSM-KG relative-direction LoRA for the unified 6-experiment design (Exp 3
adapter).  Trains on relative_osm_kg_train.jsonl — each example embeds OSM
evidence so geographic facts are baked into the adapter weights.  Evaluated
WITHOUT KG at inference (kg-mode none).

  Dataset : ../dataset/relative_osm_kg_train.jsonl  (build_relative_train_data.py)
  Output  : finetuned_gptoss_relative_osm_kg/final_adapter

Run:
    python build_relative_train_data.py            # build data (locally, warms OSM)
    python train_runner_relative_osm_kg.py
"""
import sys

sys.argv = [
    "train_lora_adapter_relative.py",
    "--dataset",    "../dataset/relative_osm_kg_train.jsonl",
    "--model-id",   "openai/gpt-oss-20b",
    "--output-dir", "finetuned_gptoss_relative_osm_kg",
]
from train_lora_adapter_relative import main

main()
