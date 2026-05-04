#!/usr/bin/env python3
"""
gptoss_finetune_cardinal.py
────────────────────────────
LoRA fine-tuning of GPT-OSS 20B on cardinal-direction questions.

Key design decisions:
  1. System prompt MUST match exactly what is used at inference time.
     Training with one prompt and evaluating with another causes the adapter
     to learn a response pattern the inference prompt never triggers.
  2. format_gptoss uses apply_chat_template with add_generation_prompt=False.
     The chat template serialises assistant content into the Harmony
     <|channel|>final channel automatically. The training target is the
     final-channel answer tokens only.
  3. Response template uses "<|channel|>final<|message|>" — the exact token
     sequence that immediately precedes the answer in the serialised text.
     This is more specific than "<|message|>" (which appears 3+ times per
     example — for system, user, and assistant turns) and guarantees we only
     compute loss on the answer tokens.
  4. LoRA rank=16 across q_proj + v_proj. With ~1000 training examples,
     rank 4 on q_proj alone is too weak to override the model's strong prior
     to start with <|channel|>analysis reasoning. Rank 16 on q+v projections
     provides enough capacity to learn the direct-answer pattern without
     catastrophic overfitting.
  5. Learning rate 2e-4 with warmup + cosine schedule for stable convergence.
"""

import os
import gc
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import argparse

# ==========================================
# CLI ARGUMENTS
# ==========================================
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--per-device-train-batch-size", type=int,   default=1)
parser.add_argument("--gradient-accumulation-steps",  type=int,   default=4)
parser.add_argument("--max-steps",                    type=int,   default=500)
parser.add_argument("--learning-rate",                type=float, default=2e-4)
parser.add_argument("--warmup-steps",                 type=int,   default=50)
parser.add_argument("--save-steps",                   type=int,   default=100)
parser.add_argument("--logging-steps",                type=int,   default=1)
parser.add_argument("--optim",                        type=str,   default="paged_adamw_8bit")
parser.add_argument("--fp16",              action="store_true")
parser.add_argument("--bf16",              action="store_true")
parser.add_argument("--weight-decay",      type=float, default=0.01)
parser.add_argument("--lr-scheduler-type", type=str,   default="cosine")
parser.add_argument("--gradient-clipping", type=float, default=1.0)
args, _ = parser.parse_known_args()

# ==========================================
# 0. DRIVER SAFETY & CLEANUP
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# ==========================================
# 1. PATHS
# ==========================================
MODEL_ID   = "openai/gpt-oss-20b"
OUTPUT_DIR = "Directional_Reasoning_GPT_OSS_20B_LoRA"

QUESTIONS_PATH = "data/questions.jsonl"
# FIX: lives in the current directory, not inside evaluation_results/
ANSWERS_PATH   = "evaluation_results/answers_finetune_subset.jsonl"

if not os.path.exists(QUESTIONS_PATH) or not os.path.exists(ANSWERS_PATH):
    print("ERROR: Could not find data files.")
    print(f"  Checked: {QUESTIONS_PATH}")
    print(f"  Checked: {ANSWERS_PATH}")
    import sys; sys.exit(1)

# ==========================================
# 2. LOAD & MERGE DATA
# ==========================================
print("Loading JSONL data...")
df_questions = pd.read_json(QUESTIONS_PATH, lines=True)
df_answers   = pd.read_json(ANSWERS_PATH,   lines=True)

try:
    df_merged = pd.merge(df_questions, df_answers, on="id")
    print(f"Successfully merged {len(df_merged)} training examples.")
except KeyError:
    print("ERROR: Both JSONL files must contain an 'id' column to merge them.")
    import sys; sys.exit(1)

dataset = Dataset.from_pandas(df_merged)

# ==========================================
# 3. LOAD TOKENIZER & MODEL
# ==========================================
print(f"Loading Tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading GPT-OSS-20B (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="eager",
)

# ==========================================
# 4. LoRA CONFIG
# ==========================================
model.gradient_checkpointing_enable()
for _, param in model.named_parameters():
    param.requires_grad = False

# FIX: rank 4 → 16, q_proj only → q_proj + v_proj.
# With ~1000 training examples, rank 4 on q_proj is too weak to override
# the model's strong prior to emit <|channel|>analysis reasoning before
# answering. Rank 16 on q+v provides enough capacity to learn the
# direct-answer pattern while the "Reasoning: low" system prompt hint
# and the training signal reinforce skipping the analysis channel.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,                          # standard 2x rank scaling
    target_modules=["q_proj", "v_proj"],    # query + value projections
    lora_dropout=0.05,                      # light regularisation
    bias="none",
    task_type="CAUSAL_LM",
)

# ==========================================
# 5. DATA FORMATTING
# ==========================================
SYSTEM_PROMPT = (
    "You are a helpful assistant. I will give you a question about directions. "
    "The answer is either north, south, east, west, north-east, north-west, "
    "south-east or south-west. Only reply with the answer, nothing else.\n"
    "Reasoning: low"
)

def format_gptoss(batch):
    """
    Build training text using apply_chat_template.

    The GPT-OSS Harmony chat template serialises the assistant turn as:
        <|start|>assistant<|channel|>final<|message|>{answer}<|return|>

    The "Reasoning: low" hint in the system prompt tells the model to skip
    the analysis channel. By training exclusively on final-channel answers,
    the adapter learns to output the direction directly.

    add_generation_prompt=False because we include the complete assistant
    turn — the template appends <|return|> so the model learns when to stop.
    """
    texts = []
    for q, a in zip(batch["question"], batch["absoluteAnswer"]):
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": q},
            {"role": "assistant", "content": a},  # plain answer, no markup
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}

print("Formatting dataset...")
dataset = dataset.map(format_gptoss, batched=True)

# ==========================================
# 6. CUSTOM COLLATOR
# ==========================================
class CompletionCollator:
    """
    Masks all prompt tokens with -100 so loss is computed only on the answer.

    response_template = "<|channel|>final<|message|>"

    After apply_chat_template the assistant turn is serialised as:
        <|start|>assistant<|channel|>final<|message|>{answer}<|return|>

    "<|channel|>final<|message|>" uniquely identifies the start of the answer.

    IMPORTANT: The previous template "<|message|>" was WRONG because it
    appears in EVERY turn (system, user, and assistant), so last-to-first
    search would always land on the assistant turn BUT on some tokenizers
    "<|message|>" is a single special token that may not match the
    multi-token encoding of the string. Using the full channel+final+message
    sequence is more reliable and semantically correct.

    We search LAST-TO-FIRST to guarantee we match the assistant turn's
    final channel, not any earlier occurrence.
    """

    def __init__(self, tokenizer, response_template: str = "<|channel|>final<|message|>"):
        """Initializes the collator with a tokenizer and encodes the response template."""
        self.tokenizer = tokenizer
        self.response_template_ids = self.tokenizer.encode(
            response_template, add_special_tokens=False
        )

    def __call__(self, features):
        """Pads a batch of features and masks all tokens before the response template."""
        clean_features = []
        for feature in features:
            item = {"input_ids": feature["input_ids"]}
            if "attention_mask" in feature:
                item["attention_mask"] = feature["attention_mask"]
            clean_features.append(item)

        batch       = self.tokenizer.pad(clean_features, padding=True, return_tensors="pt")
        labels      = batch["input_ids"].clone()
        window_size = len(self.response_template_ids)

        for i in range(len(labels)):
            seq_ids   = labels[i].tolist()
            match_idx = -1

            # Search last-to-first: guarantees we land on the assistant turn
            for j in range(len(seq_ids) - window_size, -1, -1):
                if seq_ids[j : j + window_size] == self.response_template_ids:
                    match_idx = j + window_size
                    break

            if match_idx != -1:
                labels[i, :match_idx] = -100
            else:
                # Template not found — mask entire sequence, do not train on it
                labels[i, :] = -100

            # Mask padding tokens
            if "attention_mask" in batch:
                labels[i][batch["attention_mask"][i] == 0] = -100

        batch["labels"] = labels
        return batch

# ==========================================
# 7. TRAINING
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=256,             # these examples are short (question + 1-word answer)
    dataset_text_field="text",
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    warmup_steps=args.warmup_steps,
    max_steps=args.max_steps,
    learning_rate=args.learning_rate,
    fp16=args.fp16,
    bf16=args.bf16,
    logging_steps=args.logging_steps,
    optim=args.optim,
    report_to="none",
    save_strategy="steps",
    save_steps=args.save_steps,
    gradient_checkpointing=True,
    weight_decay=args.weight_decay,
    lr_scheduler_type=args.lr_scheduler_type,
    max_grad_norm=args.gradient_clipping,
    packing=False,
    remove_unused_columns=False,
)

collator = CompletionCollator(tokenizer=tokenizer, response_template="<|channel|>final<|message|>")

print(f"\n{'='*65}")
print(f"  GPT-OSS 20B — LoRA Fine-tuning on Cardinal Directions")
print(f"  rank=16 | alpha=32 | modules=['q_proj', 'v_proj']")
print(f"  Training examples: {len(dataset)}")
print(f"{'='*65}\n")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    peft_config=peft_config,
    data_collator=collator,
    processing_class=tokenizer,
)

trainer.train()

# ==========================================
# 8. SAVE
# ==========================================
final_dir = os.path.join(OUTPUT_DIR, "final_adapter")
print(f"\nSaving final adapter to {final_dir} ...")
trainer.model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Done! Adapter saved to: {final_dir}")