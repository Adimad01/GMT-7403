#!/usr/bin/env python3
"""
gptoss_finetune.py
──────────────────
LoRA fine-tuning of GPT-OSS 20B on relative-direction spatial reasoning.

Key design decisions (aligned with the proven Directions Cardinaux approach):

  1. System prompt MUST match exactly what is used at inference time.
     Training with one prompt and evaluating with another causes the adapter
     to learn a response pattern the inference prompt never triggers.

  2. format_for_sft uses apply_chat_template with add_generation_prompt=False.
     The chat template serialises assistant content into the Harmony
     <|channel|>final channel automatically. The training target is the
     final-channel answer tokens only.

  3. Response template uses "<|channel|>final<|message|>" — the exact token
     sequence that immediately precedes the answer in the serialised text.
     Using just "<|message|>" is WRONG because it appears 3+ times per
     example (system, user, assistant turns) and causes incorrect loss
     masking, which was the #1 cause of the previous fine-tune degradation.

  4. LoRA rank=16 across q_proj + v_proj. With 630 training examples,
     rank 4 on q_proj alone is too weak to override the model's strong prior
     to start with <|channel|>analysis reasoning. Rank 16 on q+v provides
     enough capacity to learn the direct-answer pattern without catastrophic
     overfitting.

  5. Anti-overfitting measures for 630 examples:
     - max_steps=500 (hard cap, ~3.2 effective epochs at batch 4)
     - weight_decay=0.01, max_grad_norm=1.0
     - warmup_steps=50 for stable early training
     - lora_dropout=0.05 for regularisation
     - save every 100 steps to pick best checkpoint if needed
"""

import os
import gc
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig


# ═══════════════════════════════════════════════════════════════════════════ #
#  CONFIGURATION                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

MODEL_ID   = "openai/gpt-oss-20b"
TRAIN_FILE = "./SpatialEvalLLM/merged_global_70.jsonl"
OUTPUT_DIR = "./lora_output_improved_v2"

# ── LoRA hyperparameters ────────────────────────────────────────────────── #
# rank 16 on q+v provides enough capacity to learn direct-answer pattern;
# rank 4 on q_proj alone was too weak (caused degradation vs base model).
LORA_R              = 16
LORA_ALPHA          = 32                       # standard 2× rank scaling
LORA_TARGET_MODULES = ["q_proj", "v_proj"]     # query + value projections
LORA_DROPOUT        = 0.05                     # light regularisation

# ── Training hyperparameters ─────────────────────────────────────────────── #
# 630 training examples → effective batch 4 (1 × 4 accum) → ~158 steps/epoch.
# max_steps=500 ≈ 3.2 epochs — enough to converge without overfitting.
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUM_STEPS  = 4                      # effective batch = 4
MAX_STEPS             = 500                    # hard cap, ~3.2 epochs
LEARNING_RATE         = 2e-4                   # standard LoRA LR
WARMUP_STEPS          = 50                     # 10% warmup for stability
MAX_SEQ_LENGTH        = 256                    # questions + 1-word answer fit easily
SAVE_STEPS            = 100
LOGGING_STEPS         = 10
WEIGHT_DECAY          = 0.01                   # L2 regularisation
MAX_GRAD_NORM         = 1.0                    # gradient clipping

# ── System Prompt ────────────────────────────────────────────────────────── #
# MUST match the prompt used in gptoss_eval.py and gptoss_finetune_eval.py.
# "Reasoning: low" tells GPT-OSS to skip <|channel|>analysis and output
# the answer directly in <|channel|>final.
SYSTEM_PROMPT = (
    "You are a strict exact-match spatial reasoning AI. "
    "You track movements on a map or grid and identify the final destination. "
    "CRITICAL: You must respond with ONLY the exact name of the final object. "
    "Never provide steps, reasoning, indices, or punctuation. Just the object name.\n"
    "Reasoning: low"
)

# ═══════════════════════════════════════════════════════════════════════════ #
#  GPU MEMORY MANAGEMENT                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════ #
#  DATA LOADING                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

def load_spatial_data(filepath: str) -> List[Dict]:
    """Load JSONL training data with question/answer pairs."""
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if obj.get("question") and obj.get("answer"):
                    items.append({"question": obj["question"], "answer": obj["answer"]})
    print(f"  ✅  Loaded {len(items)} training examples from {Path(filepath).name}")
    return items


# ═══════════════════════════════════════════════════════════════════════════ #
#  CHAT FORMATTING                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

def format_for_sft(batch: Dict, tokenizer: AutoTokenizer) -> Dict:
    """
    Build training text using apply_chat_template.

    The GPT-OSS Harmony chat template serialises the assistant turn as:
        <|start|>assistant<|channel|>final<|message|>{answer}<|return|>

    add_generation_prompt=False because we include the complete assistant
    turn — the template appends <|return|> so the model learns when to stop.

    The "Reasoning: low" hint in the system prompt tells the model to skip
    the analysis channel. By training exclusively on final-channel answers,
    the adapter learns to output the object name directly.
    """
    texts = []
    for q, a in zip(batch["question"], batch["answer"]):
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"{q}\n\nFinal Object:"},
            {"role": "assistant", "content": a},   # plain answer, no markup
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


# ═══════════════════════════════════════════════════════════════════════════ #
#  CUSTOM COMPLETION COLLATOR                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

class CompletionCollator:
    """
    Masks all prompt tokens with -100 so loss is computed ONLY on the answer.

    response_template = "<|channel|>final<|message|>"

    After apply_chat_template the assistant turn is serialised as:
        <|start|>assistant<|channel|>final<|message|>{answer}<|return|>

    "<|channel|>final<|message|>" uniquely identifies the start of the answer.

    IMPORTANT: The previous approach using completion_only_loss=True in SFTConfig
    or response_template="<|message|>" was WRONG because "<|message|>" appears
    in EVERY turn (system, user, and assistant). This meant loss was computed
    on the wrong tokens or masked entirely, which is why the fine-tuned model
    performed WORSE than the base model.

    We search LAST-TO-FIRST to guarantee we match the assistant turn's
    final channel, not any earlier occurrence.
    """

    def __init__(self, tokenizer, response_template: str = "<|channel|>final<|message|>"):
        self.tokenizer = tokenizer
        self.response_template_ids = self.tokenizer.encode(
            response_template, add_special_tokens=False
        )

    def __call__(self, features):
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
                labels[i, :match_idx] = -100      # mask everything before the answer
            else:
                # Template not found — mask entire sequence, do not train on it
                labels[i, :] = -100

            # Mask padding tokens
            if "attention_mask" in batch:
                labels[i][batch["attention_mask"][i] == 0] = -100

        batch["labels"] = labels
        return batch


# ═══════════════════════════════════════════════════════════════════════════ #
#  MAIN                                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def main():
    print("=" * 65)
    print("  GPT-OSS 20B — LoRA Fine-tuning on SpatialEvalLLM (70%)")
    print(f"  rank={LORA_R} | alpha={LORA_ALPHA} | modules={LORA_TARGET_MODULES}")
    print("=" * 65)

    # ── 1. Load training data ────────────────────────────────────────────
    print("\n📂  Loading training data ...")
    raw_data   = load_spatial_data(TRAIN_FILE)
    hf_dataset = Dataset.from_list(raw_data)
    print(f"  Dataset size: {len(hf_dataset)} examples")
    n_train = len(hf_dataset)
    steps_per_epoch = n_train // (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUM_STEPS)
    total_epochs = MAX_STEPS / steps_per_epoch if steps_per_epoch > 0 else 0
    print(f"  Steps/epoch: {steps_per_epoch} | max_steps={MAX_STEPS} ≈ {total_epochs:.1f} epochs")

    # ── 2. Load tokenizer ────────────────────────────────────────────────
    print(f"\n🔧  Loading tokenizer for {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 3. Load base model ───────────────────────────────────────────────
    print("🔧  Loading base model (dequantizing MXFP4 → bfloat16) ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",           # required for gradient checkpointing
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    for _, param in base_model.named_parameters():
        param.requires_grad = False

    # ── 4. LoRA config ───────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 5. Format dataset ────────────────────────────────────────────────
    print("\n📝  Formatting dataset into chat-template text ...")
    formatted_dataset = hf_dataset.map(
        lambda batch: format_for_sft(batch, tokenizer),
        batched=True,
        desc="Formatting",
    )

    # ── 6. Build collator ────────────────────────────────────────────────
    collator = CompletionCollator(
        tokenizer=tokenizer,
        response_template="<|channel|>final<|message|>",
    )

    # ── 7. Training config ───────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        packing=False,
        run_name=f"gptoss_lora_spatial_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # ── 8. Train ─────────────────────────────────────────────────────────
    print("\n�  Starting training ...")
    trainer = SFTTrainer(
        model=base_model,
        processing_class=tokenizer,
        train_dataset=formatted_dataset,
        args=training_args,
        peft_config=peft_config,               # let SFTTrainer attach LoRA
        data_collator=collator,
    )

    trainer.train()

    # ── 9. Save ──────────────────────────────────────────────────────────
    final_dir = os.path.join(OUTPUT_DIR, "final")
    print(f"\n💾  Saving final LoRA adapter to {final_dir} ...")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("\n✅  Fine-tuning complete.")
    print(f"    Adapter saved at: {final_dir}")


if __name__ == "__main__":
    main()