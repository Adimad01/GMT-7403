import os
import gc
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional/late imports: handle mismatched environments (transformers vs peft/trl)
peft_import_err = None
trl_import_err = None
LoraConfig = None
get_peft_model = None
SFTTrainer = None
SFTConfig = None
try:
    from peft import LoraConfig, get_peft_model
except Exception as e:
    peft_import_err = e

try:
    from trl import SFTTrainer, SFTConfig
except Exception as e:
    trl_import_err = e

from huggingface_hub import login
import argparse

# === CLI / hyperparameter overrides ===
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--per-device-train-batch-size", type=int, default=1)
parser.add_argument("--gradient-accumulation-steps", type=int, default=4)       # ↑ was 1  — effective batch=4, stabilises gradients
parser.add_argument("--max-steps", type=int, default=500)                        # ↑ was 100 — model needs more steps to generalise
parser.add_argument("--learning-rate", type=float, default=5e-5)                 # ↓ was 2e-4 — lower LR prevents minority-class collapse
parser.add_argument("--warmup-steps", type=int, default=50)                      # ↑ was 10  — ~10% of max-steps, avoids early instability
parser.add_argument("--save-steps", type=int, default=100)                       # ↑ was 50  — scaled to new step count
parser.add_argument("--logging-steps", type=int, default=1)
parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
parser.add_argument("--fp16", action="store_true", help="Enable fp16 training")
parser.add_argument("--bf16", action="store_true", help="Enable bf16 training")
parser.add_argument("--weight-decay", type=float, default=0.01)                  # ↑ was 0.0 — light regularisation helps generalisation
parser.add_argument("--lr-scheduler-type", type=str, default="cosine",           # ↑ was linear — cosine decay is smoother for sparse classes
                    help="linear|cosine|cosine_with_restarts")
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
# 1. SETUP
# ==========================================
hf_token = os.environ.get("HF_API_TOKEN", "")
if hf_token:
    try:
        login(token=hf_token)
    except Exception as e:
        print("Warning: huggingface login failed with provided HF_API_TOKEN:", e)
        print("Continuing without explicit login (may fail for private models).")
else:
    print("No HF_API_TOKEN found in environment; continuing without explicit login.")

MODEL_ID   = "unsloth/gpt-oss-20b-bnb-4bit"
DATA_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "triplet_update_v3_70.csv"))
OUTPUT_DIR = "Topological_Reasoning_GPTOSS_Improved"

print(f"Loading {MODEL_ID}...")

# Pre-flight: ensure dataset exists before starting heavy model downloads
if not os.path.exists(DATA_PATH):
    print("ERROR: dataset file not found at:", DATA_PATH)
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    print("Dataset folder contents (if accessible):")
    try:
        for p in sorted(os.listdir(dataset_dir)):
            print(" -", p)
    except Exception:
        print("  (could not list dataset directory)")
    print("\nPlease set `DATA_PATH` to the correct CSV file location or place the file in the project's `dataset/` folder.")
    import sys
    sys.exit(1)

# ==========================================
# 2. LOAD TOKENIZER
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 3. LOAD MODEL (CPU -> GPU STRATEGY)
# ==========================================
print("Step 1: Loading model to System RAM (CPU)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)

print("Step 2: Moving model to GPU...")
model = model.to("cuda:0")

# ==========================================
# 4. ENABLE GRADIENT CHECKPOINTING
# ==========================================
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# ==========================================
# 5. LoRA CONFIG  (r/alpha raised for more capacity)
# ==========================================
if get_peft_model is None or LoraConfig is None:
    import sys
    print("ERROR: PEFT or its dependencies failed to import.")
    print("Import error details:", peft_import_err)
    print("Try: pip install --upgrade 'transformers<5.0.0,>=4.30.0' peft")
    sys.exit(1)

peft_config = LoraConfig(
    r=32,                    # ↑ was 16 — more representational capacity for 6-class task
    lora_alpha=32,           # keep equal to r
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,        # ↑ was 0.05 — slightly more regularisation
    bias="none",
    task_type="CAUSAL_LM",
)

# ==========================================
# 6. DATA PROCESSING  (with class-aware oversampling)
# ==========================================
def format_gptoss(batch):
    """Format rows into the GPT-OSS chat template."""
    texts = []
    sys_msg = "You are an expert in topological spatial reasoning."
    for sentence, subj, obj, relation in zip(
        batch["Sentence"],
        batch["place_name_subject"],
        batch["place_name_object"],
        batch["spatial_relation"],
    ):
        text = (
            f"<|start|>system\n{sys_msg}<|end|>\n"
            f"<|start|>user\nSentence: {sentence}\nSubject: {subj}\nObject: {obj}<|end|>\n"
            f"<|start|>assistant\n{str(relation)}<|end|>"
        )
        texts.append(text)
    return {"text": texts}


print("Loading data...")
df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------------
# Class-aware oversampling: upsample minority classes so every class
# appears at least `min_samples` times. This addresses the near-zero
# F1 scores on classes with very low support seen in evaluation.
# ------------------------------------------------------------------
LABEL_COL   = "spatial_relation"
MIN_SAMPLES = 50   # target floor — increase if you have more GPU memory

print("Class distribution before oversampling:")
print(df[LABEL_COL].value_counts().to_string())

class_counts = df[LABEL_COL].value_counts()
parts = [df]
for label, count in class_counts.items():
    if count < MIN_SAMPLES:
        shortage  = MIN_SAMPLES - count
        extra     = df[df[LABEL_COL] == label].sample(shortage, replace=True, random_state=42)
        parts.append(extra)

df_balanced = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nClass distribution after oversampling:")
print(df_balanced[LABEL_COL].value_counts().to_string())

dataset = Dataset.from_pandas(df_balanced)
dataset = dataset.map(format_gptoss, batched=True)

# ==========================================
# 7. TRAINING CONFIGURATION
# ==========================================
print("\nStarting Training...")

if SFTConfig is None or SFTTrainer is None:
    import sys
    print("ERROR: `trl` failed to import or is incompatible.")
    print("Import error details:", trl_import_err)
    print("Try: pip install --upgrade trl")
    sys.exit(1)

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=2048,
    dataset_text_field="text",
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    warmup_steps=args.warmup_steps,
    max_steps=args.max_steps,
    learning_rate=args.learning_rate,
    fp16=args.fp16,
    bf16=args.bf16 or True,
    logging_steps=args.logging_steps,
    optim=args.optim,
    report_to="none",
    save_strategy="steps",
    save_steps=args.save_steps,
    gradient_checkpointing=True,
    weight_decay=args.weight_decay,
    lr_scheduler_type=args.lr_scheduler_type,
    max_grad_norm=args.gradient_clipping,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final_adapter")
print("Done!")
print(f"\nAdapter saved to: {OUTPUT_DIR}/final_adapter")