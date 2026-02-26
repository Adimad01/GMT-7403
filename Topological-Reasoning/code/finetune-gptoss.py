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
    from trl import SFTTrainer, SFTConfig  # <--- NEW IMPORT
except Exception as e:
    trl_import_err = e
from huggingface_hub import login
import argparse

# === CLI / hyperparameter overrides ===
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--per-device-train-batch-size", type=int, default=1)
parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
parser.add_argument("--max-steps", type=int, default=100)
parser.add_argument("--learning-rate", type=float, default=2e-4)
parser.add_argument("--warmup-steps", type=int, default=10)
parser.add_argument("--save-steps", type=int, default=50)
parser.add_argument("--logging-steps", type=int, default=1)
parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
parser.add_argument("--fp16", action="store_true", help="Enable fp16 training")
parser.add_argument("--bf16", action="store_true", help="Enable bf16 training")
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--lr-scheduler-type", type=str, default="linear", help="linear|cosine|cosine_with_restarts")
parser.add_argument("--gradient-clipping", type=float, default=1.0)
args, _ = parser.parse_known_args()

# ==========================================
# 0. DRIVER SAFETY & CLEANUP
# ==========================================
# Force smaller memory chunks to prevent "NVML_SUCCESS" crash
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# ==========================================
# 1. SETUP
# ==========================================
# Prefer using an environment variable for the HF token so it's not hard-coded.
hf_token = ""
if hf_token:
    try:
        login(token=hf_token)
    except Exception as e:
        print("Warning: huggingface login failed with provided HF_API_TOKEN:", e)
        print("Continuing without explicit login (may fail for private models).")
else:
    print("No HF_API_TOKEN found in environment; continuing without explicit login.")

MODEL_ID = "unsloth/gpt-oss-20b-bnb-4bit"
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "triplet_update_v3_70.csv"))
OUTPUT_DIR = "Topological_Reasoning_GPTOSS_Standard"

print(f"Loading {MODEL_ID}...")

# Pre-flight: ensure dataset exists before starting heavy model downloads
if not os.path.exists(DATA_PATH):
    print("ERROR: dataset file not found at:", DATA_PATH)
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    print("Dataset folder contents (if accessible):")
    try:
        for p in sorted(os.listdir(dataset_dir)):
            print(" -", p)
    except Exception as _:
        print("  (could not list dataset directory)")
    print("\nPlease set `DATA_PATH` to the correct CSV file location or place the file in the project's `dataset/` folder.")
    import sys
    sys.exit(1)

# 2. LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. LOAD MODEL (CPU -> GPU STRATEGY)
# We load to CPU first to bypass the aggressive driver crash.
print("Step 1: Loading model to System RAM (CPU)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",         # Force CPU load
    trust_remote_code=True,
    low_cpu_mem_usage=True,   
    torch_dtype=torch.float16
)

print("Step 2: Moving model to GPU...")
model = model.to("cuda:0") # Manual move is safer for old drivers

# 4. CONFIGURATION
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# 5. LoRA CONFIG
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Ensure PEFT is available before trying to apply adapters
if get_peft_model is None or LoraConfig is None:
    import sys
    print("ERROR: PEFT or its dependencies failed to import.")
    print("Import error details:", peft_import_err)
    print("Common cause: incompatible `transformers` version (v5+) with PEFT expecting v4.x.")
    print("Try installing compatible versions:\n  pip install --upgrade 'transformers<5.0.0,>=4.30.0' peft")
    sys.exit(1)

# DO NOT wrap the model with PEFT here. `SFTTrainer` expects the base
# model plus a `peft_config` and will initialize PEFT internally.

# 6. DATA PROCESSING
def format_gpt_oss(batch):
    texts = []
    sys_msg = "You are an expert in topological spatial reasoning."
    for sentence, subj, obj, relation in zip(
        batch['Sentence'], 
        batch['place_name_subject'], 
        batch['place_name_object'], 
        batch['spatial_relation']
    ):
        text = (
            f"<|start|>system\n{sys_msg}<|end|>\n"
            f"<|start|>user\nSentence: {sentence}\nSubject: {subj}\nObject: {obj}<|end|>\n"
            f"<|start|>assistant\n{str(relation)}<|end|>"
        )
        texts.append(text)
    return { "text" : texts }

print("Loading data...")
df = pd.read_csv(DATA_PATH)
dataset = Dataset.from_pandas(df)
dataset = dataset.map(format_gpt_oss, batched=True)

# 7. TRAINING CONFIGURATION (THE FIX)
print("Starting Training...")

# Ensure TRL (SFT trainer) is available
if SFTConfig is None or SFTTrainer is None:
    import sys
    print("ERROR: `trl` failed to import or is incompatible.")
    print("Import error details:", trl_import_err)
    print("Try installing a compatible TRL version, for example:\n  pip install --upgrade trl")
    sys.exit(1)

# We use SFTConfig instead of TrainingArguments.
# This is required for TRL >= 0.28.0
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=2048,               # <--- renamed to match SFTConfig API
    dataset_text_field="text",         # <--- Moved INSIDE config
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
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,                   # <--- Pass the new config here
    peft_config=peft_config,
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final_adapter")
print("Done!")