import sys
sys.argv = [
    "train_lora_adapter_relative.py",
    "--dataset",    "../dataset/relative_train.jsonl",
    "--model-id",   "openai/gpt-oss-20b",
    "--output-dir", "finetuned_gptoss_relative",
]
from train_lora_adapter_relative import main
main()
