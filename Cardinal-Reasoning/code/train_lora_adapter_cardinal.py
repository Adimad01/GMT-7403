"""
train_lora_adapter_cardinal.py
================================================================================
Fine-tunes GPT-OSS-20B with LoRA on cardinal direction instruction datasets.

Dataset-agnostic: point --dataset at either:
  ../dataset/cardinal_train.jsonl      → plain (Config 2)
  ../dataset/cardinal_kg_train.jsonl   → KG-enriched (Config 3)

JSONL records must have:  {"text": "<full instruction+answer>", "label": "<direction>"}

Usage:
  python train_lora_adapter_cardinal.py \\
      --dataset   ../dataset/cardinal_train.jsonl \\
      --run-name  cardinal \\
      --output-dir finetuned_gptoss_cardinal
"""

import argparse
import inspect
import os
import json
import sys

# ---------------------------------------------------------------------------
# Torchvision stub — must run before peft/transformers are imported.
# ---------------------------------------------------------------------------
import importlib.machinery
import importlib.abc

class _TvStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torchvision" or fullname.startswith("torchvision."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.__path__ = []
        module.__file__ = "<torchvision-stub>"
        class _Stub:
            def __init__(self, n=""): self._n = n
            def __getattr__(self, n): return _Stub(n)
            def __call__(self, *a, **k): return _Stub()
            def __iter__(self): return iter([])
        def _catchall(name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _Stub(name)
        module.__getattr__ = _catchall
        if module.__name__ == "torchvision.io":
            class _ImageReadMode:
                RGB = 0; GRAY = 1; RGB_ALPHA = 2; GRAY_ALPHA = 3; UNCHANGED = 4
            module.ImageReadMode = _ImageReadMode
            module.decode_image = None
        elif module.__name__ == "torchvision.transforms":
            class _InterpolationMode:
                NEAREST = "nearest"; NEAREST_EXACT = "nearest-exact"
                BILINEAR = "bilinear"; BICUBIC = "bicubic"
                BOX = "box"; HAMMING = "hamming"; LANCZOS = "lanczos"
            module.InterpolationMode = _InterpolationMode

if not any(isinstance(f, _TvStubFinder) for f in sys.meta_path):
    for _k in [k for k in list(sys.modules) if k == "torchvision" or k.startswith("torchvision.")]:
        del sys.modules[_k]
    sys.meta_path.insert(0, _TvStubFinder())
# ---------------------------------------------------------------------------

import torch

if not hasattr(torch, "accelerator"):
    class _DummyAccelerator:
        @staticmethod
        def current_accelerator():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.accelerator = _DummyAccelerator()

if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: torch.nn.Module) -> None:
        if target == "":
            raise ValueError("target cannot be empty")
        atoms = target.split(".")
        name = atoms.pop(-1)
        mod = self
        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(f"'{type(mod).__name__}' has no attribute '{item}'")
            mod = getattr(mod, item)
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError(f"'{item}' is not an nn.Module")
        setattr(mod, name, module)
    torch.nn.Module.set_submodule = _set_submodule

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e5m2", None)

import importlib.machinery
import importlib.abc

class _TorchaudioStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torchaudio" or fullname.startswith("torchaudio."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec): return None
    def exec_module(self, module):
        module.__path__ = []; module.__file__ = "<torchaudio-stub>"
        class _S:
            def __getattr__(self, n): return _S()
            def __call__(self, *a, **k): return _S()
            def __iter__(self): return iter([])
        module.__getattr__ = lambda n: _S() if not (n.startswith("__") and n.endswith("__")) else (_ for _ in ()).throw(AttributeError(n))
for _k in [k for k in list(sys.modules) if k == "torchaudio" or k.startswith("torchaudio.")]:
    del sys.modules[_k]
sys.meta_path.insert(0, _TorchaudioStubFinder())

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from trl import SFTTrainer, SFTConfig

MAX_SEQ_LENGTH = 512   # cardinal prompts are short (plain) to 768 (KG-enriched)

_SFTCONFIG_PARAMS  = set(inspect.signature(SFTConfig.__init__).parameters)
_SFTTRAINER_PARAMS = set(inspect.signature(SFTTrainer.__init__).parameters)

print(f"[INFO] TRL version: {trl.__version__}")
if "max_seq_length" in _SFTCONFIG_PARAMS:
    _SEQ_LEN_IN = "SFTConfig"
elif "max_seq_length" in _SFTTRAINER_PARAMS:
    _SEQ_LEN_IN = "SFTTrainer"
else:
    _SEQ_LEN_IN = "neither"

_CUDA_OK = torch.cuda.is_available()
_BF16    = _CUDA_OK and torch.cuda.is_bf16_supported()
_FP16    = _CUDA_OK and not _BF16
print(f"[INFO] CUDA: {_CUDA_OK}  |  bf16: {_BF16}  |  fp16: {_FP16}")

# ===========================================================================
# PATCH: same as eval_engine_gpu.py — suppresses AutoHfQuantizer None-qcfg
# crash on some transformers versions.
# NOTE: The real fix for MXFP4 MoE loading is installing Triton (>=3.4.0),
# which makes the model load natively in MXFP4 without dequantization,
# avoiding the CUDA CachingAllocator NVML assertion entirely.
# ===========================================================================
try:
    from transformers.quantizers.auto import AutoHfQuantizer as _AHQ
    _orig_sqm = getattr(_AHQ, "supports_quant_method", None)
    if _orig_sqm is not None:
        @staticmethod
        def _safe_sqm(qcfg):
            if qcfg is None:
                return False
            return _orig_sqm(qcfg)
        _AHQ.supports_quant_method = _safe_sqm
except Exception:
    pass
# ===========================================================================


_VALID_CARDINAL = (
    "north_of, south_of, east_of, west_of, "
    "northeast_of, northwest_of, southeast_of, southwest_of"
)


def load_csv_records(path: str) -> list[dict]:
    import csv
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("relation_label"):
                records.append(row)
    return records


def build_cardinal_training_prompt(row: dict) -> str:
    src    = row["source_entity"]
    tgt    = row["target_entity"]
    corpus = row["corpus"]
    label  = row["relation_label"]
    return (
        f"You are an expert in spatial geography and cardinal directions.\n\n"
        f"Given the following description, determine the cardinal direction of "
        f"'{src}' relative to '{tgt}'.\n\n"
        f"Corpus: \"{corpus}\"\n\n"
        f"Possible directions: {_VALID_CARDINAL}\n\n"
        f"Answer: [{label}]"
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-OSS on cardinal direction dataset")
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--run-name",   default="cardinal")
    parser.add_argument("--model-id",   default="openai/gpt-oss-20b")
    parser.add_argument("--output-dir", default="finetuned_gptoss_cardinal")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int,   default=1)
    parser.add_argument("--grad-accum", type=int,   default=16)
    args = parser.parse_args()

    print(f"[1/5] Loading dataset: {args.dataset}")
    raw_records = load_csv_records(args.dataset)
    print(f"      -> {len(raw_records)} instruction examples loaded")

    from collections import Counter
    label_counts = Counter(r.get("relation_label", "unknown") for r in raw_records)
    print("      Label distribution:", dict(sorted(label_counts.items())))

    print(f"[2/5] Loading tokenizer from {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    records_with_eos = [
        {"text": build_cardinal_training_prompt(row) + tokenizer.eos_token}
        for row in raw_records
    ]
    train_dataset = Dataset.from_list(records_with_eos)
    print(f"      -> {len(train_dataset)} training examples ready.")

    dtype = torch.bfloat16 if _BF16 else torch.float16
    print(f"[3/5] Loading {args.model_id} in {dtype} (no Mxfp4 dequantize — base weights frozen for LoRA) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        dtype=dtype,
    )

    print("[4/5] Attaching LoRA adapters ...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[5/5] Starting cardinal fine-tuning ...")
    sftconfig_kwargs  = {}
    sfttrainer_kwargs = {}
    if _SEQ_LEN_IN == "SFTConfig":
        sftconfig_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    elif _SEQ_LEN_IN == "SFTTrainer":
        sfttrainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=_BF16,
        fp16=_FP16,
        use_cpu=not _CUDA_OK,
        optim="paged_adamw_8bit" if _CUDA_OK else "adamw_torch",
        dataset_text_field="text",
        report_to="none",
        run_name=args.run_name,
        **sftconfig_kwargs,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        processing_class=tokenizer,
        **sfttrainer_kwargs,
    )

    trainer.train()

    save_path = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n[DONE] Fine-tuning complete. Adapter saved → {save_path}")
    print(f"       Run: {args.run_name}  |  Epochs: {args.epochs}  |  LR: {args.lr}")


if __name__ == "__main__":
    main()
