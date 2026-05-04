"""
gptoss_finetune.py
-----------------
Fine-tune GPT-OSS 20B on the 70% of worldcities100k data
using the ultra-strict "Zero-Yap" prompt format.

Technique : 16-bit MXFP4 Dequantization + LIGHTWEIGHT LoRA + Gradient Checkpointing
Library   : transformers + peft + trl (SFTTrainer / SFTConfig)
"""

import argparse
import json
import os
import shutil
import logging
import pandas as pd
from datetime import datetime
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Mxfp4Config,
)
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/finetune_gptoss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename),
    ]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & "Zero-Yap" Configuration
# ---------------------------------------------------------------------------
BASE_MODEL_ID   = "openai/gpt-oss-20b"
SUBSET_IDS_PATH = "outputs/subset_ids_gptoss_30pct.json"

# This MUST match your inference script for consistency
SYSTEM_MSG = (
    "You are a geospatial reasoning assistant. Reasoning: high. "
    "DIRECT OUTPUT ONLY. DO NOT output an 'analysis' block. DO NOT think out loud. "
    "Task: Provide coordinates exactly in this format: Latitude, Longitude. "
    "Output absolutely no other text."
)

TEMPLATES = {
    '0': "What are the geo-coordinates of {city}?",
    '1': "What are the latitude and longitude of {city}?",
}

def build_training_examples(df, tokenizer):
    records = []
    for _, row in df.iterrows():
        answer = f"{row.Latitude}, {row.Longitude}"
        for p_type, user_tmpl in TEMPLATES.items():
            user_content = user_tmpl.format(city=row.AccentCity)
            
            messages = [
                {"role": "system",    "content": SYSTEM_MSG},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": answer},
            ]
            
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            records.append({"text": full_text})

    log.info(f"Training examples built: {len(records):,} "
             f"({len(df):,} cities × {len(TEMPLATES)} templates)")
    return Dataset.from_list(records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      default="worldcities100k.pkl")
    parser.add_argument("--subset-ids",   default=SUBSET_IDS_PATH)
    parser.add_argument("--output-dir",   default="finetuned_gptoss20b_geocoords")
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--batch-size",   type=int,   default=1) 
    parser.add_argument("--grad-accum",   type=int,   default=16) 
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--max-length",   type=int,   default=256)
    parser.add_argument("--lora-r",       type=int,   default=8)
    parser.add_argument("--lora-alpha",   type=int,   default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    
    # New argument to control folder wiping (defaults to True)
    parser.add_argument("--overwrite",    type=bool,  default=True, 
                        help="Wipe existing output directory before starting")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("GPT-OSS 20B Fine-tuning — Zero-Yap Geospatial Mode (LIGHTWEIGHT LORA)")
    log.info("=" * 60)

    # 1. Load Data
    log.info(f"\n[1/6] Loading dataset: {args.dataset}")
    if args.dataset.endswith('.pkl'):
        cities = pd.read_pickle(args.dataset)
    else:
        cities = pd.read_csv(args.dataset)

    # 2. Split Data
    log.info(f"\n[2/6] Building 70% training split ...")
    if os.path.exists(args.subset_ids):
        with open(args.subset_ids, "r") as f:
            subset_ids = set(json.load(f))
        train_df = cities[~cities.index.isin(subset_ids)].copy()
        log.info(f"Successfully excluded {len(subset_ids)} cities found in {args.subset_ids}")
    else:
        log.warning("Subset ID file not found! Using random 70% split. Risk of data leakage.")
        train_df = cities.sample(frac=0.70, random_state=99).copy()

    # 3. Tokenizer
    log.info(f"\n[3/6] Loading tokenizer from {BASE_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. Prepare Dataset
    log.info(f"\n[4/6] Formatting prompts for training ...")
    train_dataset = build_training_examples(train_df, tokenizer)

    # 5. Load Model & Apply LoRA
    log.info(f"\n[5/6] Loading model {BASE_MODEL_ID} on GPU ...")
    mxfp4_config = Mxfp4Config(dequantize=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=mxfp4_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    model.gradient_checkpointing_enable()
    torch.cuda.empty_cache()

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

    # 6. Trainer Execution
    log.info(f"\n[6/6] Preparing output directory → {args.output_dir}")
    
    # --- FOLDER WIPE LOGIC ---
    if args.overwrite and os.path.exists(args.output_dir):
        log.warning(f"[!] Overwrite enabled. Deleting existing folder: {args.output_dir}")
        shutil.rmtree(args.output_dir)
        
    os.makedirs(args.output_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit",
        dataset_text_field="text",
        max_length=args.max_length,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
    )

    try:
        # If we just wiped the folder, resume will naturally be False
        resume = any(d.startswith("checkpoint-") for d in os.listdir(args.output_dir)) if os.path.exists(args.output_dir) else False
        trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:
        log.warning("\nTraining interrupted by user. Saving current adapter state...")
    
    # Final Save
    adapter_path = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    log.info(f"\n✅ Fine-tuning complete. Adapter saved to: {adapter_path}")

if __name__ == "__main__":
    main()