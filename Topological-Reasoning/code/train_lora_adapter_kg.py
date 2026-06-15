"""
train_lora_adapter_kg.py
================================================================================
Phase 3 — KGs Instruction-Tuning Fine-Tuning

Fine-tunes GPT-OSS-20B with LoRA on a KG-enriched instruction dataset.

This script is intentionally dataset-agnostic: point --dataset at either
  ../dataset/osm_kg_train.jsonl      → OSM-KG run  (Run A)
  ../dataset/wikidata_kg_train.jsonl → Wikidata-KG run (Run B)

The JSONL records are produced by the two dataset-builder scripts and already
contain the full instruction text (KG evidence in INPUT, label in OUTPUT).
The fine-tuning objective is identical to the data-only baseline (train_lora_adapter.py):
SFT with LoRA (r=8, lora_alpha=16) on the full text — same hyperparameters for
a fair comparison.

Usage:
  # Run A — OSM-KG
  python train_lora_adapter_kg.py \
      --dataset   ../dataset/osm_kg_train.jsonl \
      --run-name  osm_kg \
      --output-dir finetuned_gptoss_osm_kg

  # Run B — Wikidata-KG
  python train_lora_adapter_kg.py \
      --dataset   ../dataset/wikidata_kg_train.jsonl \
      --run-name  wikidata_kg \
      --output-dir finetuned_gptoss_wikidata_kg
"""

import argparse
import inspect
import os
import json
import sys

# ---------------------------------------------------------------------------
# Torchvision stub — must run before peft/transformers are imported.
# Prevents AttributeError: NEAREST_EXACT in transformers.image_utils caused
# by a broken torchvision native extension on this server.
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

# Torchaudio stub — libcudart.so.11.0 missing on this server causes torchaudio
# to crash on import, which is triggered by transformers.loss.loss_rnnt → torchaudio.
class _TaStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
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

if not any(isinstance(f, _TaStubFinder) for f in sys.meta_path):
    for _k in [k for k in list(sys.modules) if k == "torchaudio" or k.startswith("torchaudio.")]:
        del sys.modules[_k]
    sys.meta_path.insert(0, _TaStubFinder())
# ---------------------------------------------------------------------------

import torch

# ===========================================================================
# DEPENDENCY MONKEY PATCHES (same as train_lora_adapter.py)
# ===========================================================================
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

# Patch: transformers >=5.9.0 on torch <2.6 — float8_e8m0fnu added in torch 2.6
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e5m2", None)

# Patch: mxfp4 MoE dequantization → CPU to avoid NVML_SUCCESS assert on MIG A100
try:
    import transformers.integrations.mxfp4 as _mxfp4_m
    _orig_moe_cvt = _mxfp4_m._convert_moe_packed_tensors
    def _cpu_moe_cvt(blocks, scales, dtype=torch.bfloat16, rows_per_chunk=None):
        target_device = blocks.device
        b = blocks.cpu() if blocks.device.type != "cpu" else blocks
        s = scales.cpu() if scales.device.type != "cpu" else scales
        result = _orig_moe_cvt(b, s, dtype=dtype, rows_per_chunk=rows_per_chunk)
        return result.to(target_device)
    _mxfp4_m._convert_moe_packed_tensors = _cpu_moe_cvt
    print("[PATCH] mxfp4 MoE dequantization → CPU  (MIG A100 NVML fix)")
except Exception as _e:
    print(f"[WARN] mxfp4 MoE patch skipped: {_e}")

# Patch: caching_allocator_warmup pre-allocates ~40 GB; skip on MIG to avoid OOM
try:
    import transformers.modeling_utils as _tmu
    _tmu.caching_allocator_warmup = lambda *_a, **_k: None
    print("[PATCH] caching_allocator_warmup → no-op  (MIG A100 OOM fix)")
except Exception as _e:
    print(f"[WARN] warmup patch skipped: {_e}")
# ===========================================================================

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
import trl
from trl import SFTTrainer, SFTConfig

MAX_SEQ_LENGTH = 768   # KG prompts are longer than the data-only baseline

# Detect where max_seq_length lives in this TRL version (moves between releases)
_SFTCONFIG_PARAMS  = set(inspect.signature(SFTConfig.__init__).parameters)
_SFTTRAINER_PARAMS = set(inspect.signature(SFTTrainer.__init__).parameters)

print(f"[INFO] TRL version: {trl.__version__}")
if "max_seq_length" in _SFTCONFIG_PARAMS:
    _SEQ_LEN_IN = "SFTConfig"
    print("[INFO] max_seq_length -> SFTConfig")
elif "max_seq_length" in _SFTTRAINER_PARAMS:
    _SEQ_LEN_IN = "SFTTrainer"
    print("[INFO] max_seq_length -> SFTTrainer")
else:
    _SEQ_LEN_IN = "neither"
    print("[INFO] TRL 1.x detected — max_seq_length set via tokenizer.model_max_length")

# Determine bf16/fp16 based on actual GPU availability
_CUDA_OK = torch.cuda.is_available()
_BF16    = _CUDA_OK and torch.cuda.is_bf16_supported()
_FP16    = _CUDA_OK and not _BF16
print(f"[INFO] CUDA: {_CUDA_OK}  |  bf16: {_BF16}  |  fp16: {_FP16}")


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-OSS on a KG instruction dataset")
    parser.add_argument("--dataset",    required=True,
                        help="Path to the KG instruction JSONL (osm_kg_train.jsonl or wikidata_kg_train.jsonl)")
    parser.add_argument("--run-name",   default="kg",
                        help="Short identifier for this run (osm_kg / wikidata_kg) — used in logs")
    parser.add_argument("--model-id",   default="openai/gpt-oss-20b")
    parser.add_argument("--output-dir", default="finetuned_gptoss_kg")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int,   default=1)
    parser.add_argument("--grad-accum", type=int,   default=16)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load JSONL dataset
    # ------------------------------------------------------------------
    print(f"[1/5] Loading KG instruction dataset: {args.dataset}")
    raw_records = load_jsonl(args.dataset)
    print(f"      -> {len(raw_records)} instruction examples loaded (source: {args.run_name})")

    label_counts: dict[str, int] = {}
    for rec in raw_records:
        lbl = rec.get("label", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print("      Label distribution:", label_counts)

    # ------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------
    print(f"[2/5] Loading tokenizer from {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # TRL 1.x uses tokenizer.model_max_length for truncation instead of max_seq_length
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    # Append the EOS token to each training example (the JSONL text does not
    # include it so the fine-tuning objective is clear across all tokenizer variants)
    records_with_eos = [
        {"text": rec["text"] + tokenizer.eos_token}
        for rec in raw_records
    ]
    train_dataset = Dataset.from_list(records_with_eos)
    print(f"      -> {len(train_dataset)} training examples ready.")

    # ------------------------------------------------------------------
    # 3. Load model — dequantize MXFP4 to bf16 for training
    # ------------------------------------------------------------------
    print(f"[3/5] Loading {args.model_id} (dequantizing MXFP4 → bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),
        use_cache=False,
    )
    print(f"      -> Model dtype: {next(model.parameters()).dtype}")

    # ------------------------------------------------------------------
    # 4. Attach LoRA adapters
    # Identical configuration to train_lora_adapter.py for a fair comparison.
    # prepare_model_for_kbit_training() is intentionally skipped — this is
    # a plain bf16 model, not a bnb quantized one (would crash the allocator).
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("[5/5] Starting KG instruction fine-tuning ...")

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

    # ------------------------------------------------------------------
    # Save adapter
    # ------------------------------------------------------------------
    save_path = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n[DONE] KG fine-tuning complete. Adapter saved → {save_path}")
    print(f"       Run:    {args.run_name}")
    print(f"       Epochs: {args.epochs}  |  LR: {args.lr}  |  max_seq_length: {MAX_SEQ_LENGTH}")


if __name__ == "__main__":
    main()
