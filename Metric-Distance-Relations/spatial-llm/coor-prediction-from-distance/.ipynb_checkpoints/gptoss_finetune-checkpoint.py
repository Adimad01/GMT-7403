"""
finetune_gptoss.py
--------------------
Fine-tune openai/gpt-oss-20b on the remaining training data 
that was NOT used during the 100-instance inference experiments.

Modifications for GPT-OSS:
- Uses JSON data directly (no numpy pickle conflicts).
- Minimal LoRA Rank (r=8) and Target Modules (q_proj, v_proj only).
- Native model loading (bypassing BitsAndBytes because GPT-OSS is already MXFP4 quantized).
- Automatically resumes from the last saved checkpoint if one exists.
"""

import argparse
import json
import os
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Prompt builders (same format as inference)
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
    """Wrap a user/assistant turn in the model's native template."""
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
    parser.add_argument("--dataset",     default="cities_with_coords.json")
    parser.add_argument("--subset-ids",  default="100_target_ids.json",
                        help="The exact 100 test instances to EXCLUDE from training.")
    parser.add_argument("--max-rows",    type=int, default=None)
    parser.add_argument("--output-dir",  default="finetuned_gptoss",
                        help="Where to save LoRA adapter weights")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch-size",  type=int,   default=1, 
                        help="Per-device train batch size (lowered to 1 for 20B safety)")
    parser.add_argument("--grad-accum",  type=int,   default=16, 
                        help="Gradient accumulation steps")
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--max-length",  type=int,   default=512)
    parser.add_argument("--lora-r",      type=int,   default=8, 
                        help="LoRA Rank")
    parser.add_argument("--lora-alpha",  type=int,   default=16, 
                        help="LoRA Alpha")
    parser.add_argument("--lora-dropout",type=float, default=0.05)
    args = parser.parse_args()

    # 1. Load dataset
    print(f"\n[1/6] Loading dataset: {args.dataset}")
    df = pd.read_json(args.dataset)
    print(f"      Full dataset: {len(df)} rows")

    if args.max_rows is not None:
        df = df.head(args.max_rows)

    # 2. Exclude the 100 evaluation pairs → keep the rest for training
    if args.subset_ids and os.path.exists(args.subset_ids):
        with open(args.subset_ids, "r", encoding="utf-8") as f:
            used_pairs = json.load(f)

        used_set = {(p["a_name"], p["b_name"]) for p in used_pairs}
        
        # Filter OUT the rows that are in the used_set
        mask = df.apply(
            lambda r: (r["a_name"], r["b_name"]) not in used_set, axis=1
        )
        train_df = df[mask].copy()
        print(f"\n[2/6] Excluded {len(used_set)} test pairs.")
        print(f"      Training split available: {len(train_df)} rows")
    else:
        print(f"\n[2/6] ERROR: Test ID file '{args.subset_ids}' not found. Cannot safely filter dataset.")
        return

    has_coords = {"a_lat", "a_lon", "b_lat", "b_lon"}.issubset(train_df.columns)

    # 3. Load tokenizer
    MODEL_ID = "openai/gpt-oss-20b"
    print(f"\n[3/6] Loading tokenizer from {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. Build HuggingFace dataset
    print("\n[4/6] Building training dataset ...")
    train_dataset = prepare_dataset(train_df, tokenizer, has_coords, args.max_length)

    # 5. Load Native Base Model + Apply LoRA
    print(f"\n[5/6] Loading model {MODEL_ID} (Native loading) ...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # SMALL TARGET MODULES: Only q_proj and v_proj
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"], 
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Train
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
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        optim="paged_adamw_8bit",
        dataset_text_field="text",
        max_length=args.max_length,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    # CHECK FOR EXISTING CHECKPOINTS
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        # Look for folders that start with "checkpoint-"
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort them by the step number to find the most recent one
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            last_checkpoint = os.path.join(args.output_dir, checkpoints[-1])
            print(f"\n[!] Found existing checkpoint: {last_checkpoint}")
            print("Resuming training from this point...")

    # Start or Resume Training
    if last_checkpoint:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    # Save final LoRA adapter
    adapter_path = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n✅ Fine-tuning complete. Adapter saved to: {adapter_path}")

if __name__ == "__main__":
    main()