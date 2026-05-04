"""
finetune_qwen14b.py
--------------------
Fine-tune Qwen2.5-14B-Instruct on the 70% of city-pair data
that was NOT used during the 30% inference experiments.

Technique : QLoRA (4-bit base + LoRA adapters)
Library   : transformers + peft + trl (SFTTrainer)

Usage
-----
python finetune_qwen14b.py \
    --dataset       cities_with_coords.pkl \
    --subset-ids    outputs/subset_ids_cities_opt_30.json \
    --output-dir    finetuned_qwen14b \
    --epochs        3 \
    --batch-size    2 \
    --grad-accum    8
"""

import argparse
import json
import os
import pickle
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# Custom unpickler (numpy 2.x → 1.x compatibility)
# ---------------------------------------------------------------------------
class Numpy2to1Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_pickle(path):
    with open(path, "rb") as f:
        return Numpy2to1Unpickler(f).load()


# ---------------------------------------------------------------------------
# Prompt builders  (one per prompt type, same format as inference)
# ---------------------------------------------------------------------------
SYSTEM_MSG = "You are a helpful geography expert calculating distances between cities."


def build_prompt_p1(row):
    return (
        f"Question: What is the distance in kilometers between "
        f"{row['a_name']} and {row['b_name']}? "
        f"Answer exactly with the number only.\nAnswer: {row['distance']}"
    )


def build_prompt_p2(row):
    a_c = f"({row['a_lat']:.4f}, {row['a_lon']:.4f})"
    b_c = f"({row['b_lat']:.4f}, {row['b_lon']:.4f})"
    return (
        f"Question: What is the distance in kilometers between "
        f"{row['a_name']} {a_c} and {row['b_name']} {b_c}? "
        f"Answer exactly with the number only.\nAnswer: {row['distance']}"
    )


def build_prompt_p3(row):
    return (
        f"Question: What is the distance in kilometers between "
        f"{row['a_name']} and {row['b_name']}? "
        f"Explain your reasoning and how the distance is found.\n"
        f"Answer: The distance is approximately {row['distance']} km. "
        f"This is calculated using spatial geometry."
    )


def build_prompt_p4(row):
    a_c = f"({row['a_lat']:.4f}, {row['a_lon']:.4f})"
    b_c = f"({row['b_lat']:.4f}, {row['b_lon']:.4f})"
    return (
        f"Question: What is the distance in kilometers between "
        f"{row['a_name']} {a_c} and {row['b_name']} {b_c}? "
        f"Explain your reasoning and how the distance is found using these coordinates.\n"
        f"Answer: Using the Haversine formula on these coordinates, "
        f"the distance is {row['distance']} km."
    )


PROMPT_BUILDERS = {
    "p1": build_prompt_p1,
    "p2": build_prompt_p2,
    "p3": build_prompt_p3,
    "p4": build_prompt_p4,
}


def build_chat_text(tokenizer, user_content):
    """Wrap a user/assistant turn in the model's ChatML template."""
    messages = [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def prepare_dataset(df, tokenizer, has_coords, max_length=512):
    """
    Build one training example per row × prompt type.
    Uses all four prompt types if coordinates are available, else p1 + p3.
    """
    p_types = ["p1", "p2", "p3", "p4"] if has_coords else ["p1", "p3"]
    records = []

    for _, row in df.iterrows():
        for pt in p_types:
            user_content = PROMPT_BUILDERS[pt](row)
            full_text = build_chat_text(tokenizer, user_content)
            records.append({"text": full_text})

    hf_dataset = Dataset.from_list(records)
    print(f"--> Training examples built: {len(hf_dataset)} "
          f"({len(df)} rows × {len(p_types)} prompt types)")
    return hf_dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="cities_with_coords.pkl",
                        help="Pickle dataset (same one used for inference)")
    parser.add_argument("--subset-ids",  default="outputs/subset_ids_cities_opt_30.json",
                        help="JSON file listing the 30%% inference pairs to exclude. "
                             "70%% of rows NOT in this file are used for training.")
    parser.add_argument("--max-rows",    type=int, default=None,
                        help="Optional row cap. Default None = use the full dataset.")
    parser.add_argument("--output-dir",  default="finetuned_qwen14b",
                        help="Where to save LoRA adapter weights")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch-size",  type=int,   default=2,
                        help="Per-device train batch size")
    parser.add_argument("--grad-accum",  type=int,   default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--max-length",  type=int,   default=512,
                        help="Max token length per training example")
    parser.add_argument("--lora-r",      type=int,   default=16)
    parser.add_argument("--lora-alpha",  type=int,   default=32)
    parser.add_argument("--lora-dropout",type=float, default=0.05)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load full dataset
    # ------------------------------------------------------------------
    print(f"\n[1/6] Loading dataset: {args.dataset}")
    df = load_pickle(args.dataset)
    print(f"      Full dataset: {len(df)} rows")

    # Optional row cap (None by default = full dataset used for training)
    if args.max_rows is not None:
        df = df.head(args.max_rows)
        print(f"      After max-rows cap ({args.max_rows}): {len(df)} rows")
    else:
        print(f"      No row cap applied — using all {len(df)} rows.")

    # ------------------------------------------------------------------
    # 2. Exclude the 30% inference subset → keep the 70% training split
    # ------------------------------------------------------------------
    if args.subset_ids and os.path.exists(args.subset_ids):
        with open(args.subset_ids, "r", encoding="utf-8") as f:
            used_pairs = json.load(f)

        used_set = {(p["a_name"], p["b_name"]) for p in used_pairs}
        mask = df.apply(
            lambda r: (r["a_name"], r["b_name"]) not in used_set, axis=1
        )
        train_df = df[mask].copy()
        print(f"\n[2/6] Excluded {len(used_set)} inference pairs.")
        print(f"      Training split (70%%): {len(train_df)} rows")
    else:
        # Fallback: use rows not in the default 30% sample
        train_df = df.sample(frac=0.70, random_state=99).copy()
        print(f"\n[2/6] No subset-ids file found — sampled 70%% randomly: {len(train_df)} rows")

    has_coords = {"a_lat", "a_lon", "b_lat", "b_lon"}.issubset(train_df.columns)
    print(f"      Coordinates available: {has_coords}")

    # ------------------------------------------------------------------
    # 3. Load tokenizer
    # ------------------------------------------------------------------
    MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
    print(f"\n[3/6] Loading tokenizer from {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # right-pad for training

    # ------------------------------------------------------------------
    # 4. Build HuggingFace dataset
    # ------------------------------------------------------------------
    print("\n[4/6] Building training dataset ...")
    train_dataset = prepare_dataset(train_df, tokenizer, has_coords, args.max_length)

    # ------------------------------------------------------------------
    # 5. Load 4-bit quantised base model + LoRA
    # ------------------------------------------------------------------
    print(f"\n[5/6] Loading model {MODEL_ID} (4-bit QLoRA) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 6. Train with SFTTrainer
    # ------------------------------------------------------------------
    print(f"\n[6/6] Starting fine-tuning → {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,   # Qwen2.5 uses BFloat16 natively; fp16 causes grad scaler conflict
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        optim="paged_adamw_8bit",
        dataset_text_field="text",      # moved here from SFTTrainer
        max_length=args.max_length,      # renamed from max_seq_length in trl>=0.12
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    trainer.train()

    # Save final LoRA adapter
    adapter_path = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n✅ Fine-tuning complete. Adapter saved to: {adapter_path}")
    print("   To run inference, load the base model + adapter with:")
    print(f"   model = PeftModel.from_pretrained(base_model, '{adapter_path}')")


if __name__ == "__main__":
    main()