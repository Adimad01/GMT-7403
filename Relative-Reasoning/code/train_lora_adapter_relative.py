import argparse
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:native")
import json
import torch

# ===========================================================================
# DEPENDENCY MONKEY PATCHES
# ===========================================================================

# Patch 1: Missing torch.accelerator (PyTorch < 2.4/2.5)
if not hasattr(torch, "accelerator"):
    class DummyAccelerator:
        @staticmethod
        def current_accelerator():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.accelerator = DummyAccelerator()

# Patch 2: Missing torch.nn.Module.set_submodule (PyTorch < 2.5)
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

# Patch 3: transformers >=5.9.0 on torch <2.6 -- float8_e8m0fnu added in torch 2.6
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e5m2", None)

# Patch 4: transformers >=5.9.0 bug -- supports_quant_method crashes when
# the model's config.quantization_config is None instead of a dict.
try:
    from transformers.quantizers.auto import AutoHfQuantizer as _AHQ
    _orig_sqm = getattr(_AHQ, "supports_quant_method", None)
    if _orig_sqm is not None:
        @staticmethod  # type: ignore[misc]
        def _safe_sqm(qcfg):
            if qcfg is None:
                return False
            return _orig_sqm(qcfg)
        _AHQ.supports_quant_method = _safe_sqm
except Exception:
    pass

# Patch 5: kernels.LayerRepository -- newer versions require revision= or version=
try:
    from kernels.layer.layer import LayerRepository as _LayerRepo
    _orig_lr_init = _LayerRepo.__init__
    def _patched_lr_init(self, *args, revision=None, version=None, **kwargs):
        if revision is None and version is None:
            version = "0"
        _orig_lr_init(self, *args, revision=revision, version=version, **kwargs)
    _LayerRepo.__init__ = _patched_lr_init
    print("[PATCH] kernels.LayerRepository patched OK")
except Exception as _kp_err:
    print(f"[WARN] kernels patch skipped: {_kp_err}")
# ===========================================================================

import sys
import inspect
import importlib.machinery
import importlib.abc

class _TorchvisionStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torchvision" or fullname.startswith("torchvision."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec): return None
    def exec_module(self, module):
        module.__path__ = []; module.__file__ = "<torchvision-stub>"
        class _S:
            def __init__(self, n=""): self._n = n
            def __getattr__(self, n): return _S(n)
            def __call__(self, *a, **k): return _S()
            def __iter__(self): return iter([])
        def _catchall(name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _S(name)
        module.__getattr__ = _catchall
        if module.__name__ == "torchvision.transforms":
            class _InterpolationMode:
                NEAREST = "nearest"; NEAREST_EXACT = "nearest-exact"
                BILINEAR = "bilinear"; BICUBIC = "bicubic"
                BOX = "box"; HAMMING = "hamming"; LANCZOS = "lanczos"
            module.InterpolationMode = _InterpolationMode
        elif module.__name__ == "torchvision.io":
            class _ImageReadMode:
                RGB = 0; GRAY = 1; RGB_ALPHA = 2; GRAY_ALPHA = 3; UNCHANGED = 4
            module.ImageReadMode = _ImageReadMode
            module.decode_image = None

for _k in [k for k in list(sys.modules) if k == "torchvision" or k.startswith("torchvision.")]:
    del sys.modules[_k]
sys.meta_path.insert(0, _TorchvisionStubFinder())

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
# NOTE: prepare_model_for_kbit_training is intentionally NOT imported.
# It is a bitsandbytes helper for 4-bit/8-bit bnb models only.
# Our model is dequantized to plain bf16, so calling it causes a
# CUDA allocator crash when it tries to upcast params to float32.
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config

# Monkey-patch mxfp4 MoE dequantization to run on CPU — prevents
# NVML_SUCCESS assert at CUDACachingAllocator.cpp:995 on MIG A100.
# sub[:, 0::2] = lut[idx_lo] (mxfp4.py:293) triggers an NVML memory query
# via the CUDA indexed-assignment temp-buffer allocator; running on CPU avoids it.
try:
    import transformers.integrations.mxfp4 as _mxfp4_m
    _orig_moe_cvt = _mxfp4_m._convert_moe_packed_tensors
    def _cpu_moe_cvt(blocks, scales, dtype=torch.bfloat16, rows_per_chunk=None):
        target_device = blocks.device
        b = blocks.cpu() if blocks.device.type != "cpu" else blocks
        s = scales.cpu() if scales.device.type != "cpu" else scales
        result = _orig_moe_cvt(b, s, dtype=dtype, rows_per_chunk=rows_per_chunk)
        return result.to(target_device)  # move back to GPU after CPU dequantization
    _mxfp4_m._convert_moe_packed_tensors = _cpu_moe_cvt
    print("[PATCH] mxfp4 MoE dequantization → CPU  (MIG A100 NVML fix)")
except Exception as _mxfp4_e:
    print(f"[WARN] mxfp4 MoE patch skipped: {_mxfp4_e}")

# Disable caching_allocator_warmup — it pre-allocates a tensor the size of the
# whole model (~40 GB) as a CUDA allocator warmup.  On MIG the model skeleton +
# warmup tensor together exceed available VRAM even though the model fits.
# The warmup is a performance optimisation only; skipping it has no effect on accuracy.
try:
    import transformers.modeling_utils as _tmu
    _tmu.caching_allocator_warmup = lambda *_a, **_k: None
    print("[PATCH] caching_allocator_warmup → no-op  (MIG A100 OOM fix)")
except Exception as _wpe:
    print(f"[WARN] warmup patch skipped: {_wpe}")

import trl
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Detect where max_seq_length lives in this TRL version (moves between releases)
# ---------------------------------------------------------------------------
_SFTCONFIG_PARAMS  = set(inspect.signature(SFTConfig.__init__).parameters)
_SFTTRAINER_PARAMS = set(inspect.signature(SFTTrainer.__init__).parameters)

MAX_SEQ_LENGTH = 512

print(f"[INFO] TRL version: {trl.__version__}")
if "max_seq_length" in _SFTCONFIG_PARAMS:
    _SEQ_LEN_IN = "SFTConfig"
    print("[INFO] max_seq_length -> SFTConfig")
elif "max_seq_length" in _SFTTRAINER_PARAMS:
    _SEQ_LEN_IN = "SFTTrainer"
    print("[INFO] max_seq_length -> SFTTrainer")
else:
    _SEQ_LEN_IN = "neither"
    print("[WARN] max_seq_length not found in either class -- omitting (TRL uses its own default).")


# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------
_VALID_RELATIVE = "behind, in_front_of, left_of, next_to, right_of"


def build_training_prompt(row, tokenizer):
    """Build a relative direction instruction-following training example."""
    src    = str(row.get("source_entity", "")).strip()
    tgt    = str(row.get("target_entity", "")).strip()
    corpus = str(row.get("corpus",        "")).strip()
    label  = str(row.get("relation_label","")).strip()

    text = (
        f"You are an expert in spatial and relative directions.\n\n"
        f"Given the following description, determine the relative direction of "
        f"'{src}' relative to '{tgt}'.\n\n"
        f"Corpus: \"{corpus}\"\n\n"
        f"Possible directions: {_VALID_RELATIVE}\n\n"
        f"Answer: [{label}]"
        + tokenizer.eos_token
    )
    return text


# ---------------------------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------------------------
def main():
    """Load relative navigation data, attach LoRA adapters, and fine-tune GPT-OSS."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="../dataset/relative_train.jsonl")
    parser.add_argument("--model-id",   default="openai/gpt-oss-20b")
    parser.add_argument("--output-dir", default="finetuned_gptoss_relative")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load Data & Tokenizer
    # ------------------------------------------------------------------
    print("[1/5] Loading dataset and tokenizer...")
    import csv as _csv
    rows = []
    with open(args.dataset, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            if row.get("relation_label"):
                rows.append(row)
    print(f"      -> {len(rows)} rows loaded from {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 2. Build Dataset
    # ------------------------------------------------------------------
    print("[2/5] Building training dataset...")
    records = [
        {"text": build_training_prompt(row, tokenizer)}
        for row in rows
    ]
    train_dataset = Dataset.from_list(records)
    print(f"      -> {len(records)} training examples loaded.")

    # ------------------------------------------------------------------
    # 3. Load Model -- dequantize MXFP4 to bf16 so the model is trainable.
    #
    # gpt-oss-20b ships as a native MXFP4 checkpoint (inference-only).
    # Mxfp4Config(dequantize=True) decompresses weights to bf16 on load.
    # ------------------------------------------------------------------
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated(0)) / 1e9
        print(f"[GPU] CUDA  {torch.cuda.get_device_name(0)}  free~{free_gb:.0f} GB")
    else:
        print("[GPU] CUDA not available -- training will be very slow on CPU!")
    dtype = torch.bfloat16 if (cuda_ok and torch.cuda.is_bf16_supported()) else torch.float16

    # Always dequantize MXFP4 -> bf16 on load.  save_pretrained does NOT preserve
    # MoE expert key names correctly for gpt-oss-20b (gate_up_proj saved at root
    # level instead of mlp.experts.gate_up_proj), so a pre-saved cache silently
    # reinitialises expert layers -> garbage outputs.
    if cuda_ok:
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated(0)) / 1e9
        print(f"[3/5] Loading {args.model_id} with MXFP4->bf16 dequantize "
              f"(free GPU~{free_gb:.0f} GB)")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=Mxfp4Config(dequantize=True),
        use_cache=False,
    )
    print(f"      -> Model dtype: {next(model.parameters()).dtype}")

    # ------------------------------------------------------------------
    # 4. Attach LoRA adapters directly on the bf16 model.
    #
    # prepare_model_for_kbit_training() is intentionally skipped here.
    # That helper is only for bitsandbytes 4-bit/8-bit quantized models.
    # ------------------------------------------------------------------
    print("[4/5] Attaching LoRA adapters...")

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
    print("[5/5] Starting training...")

    sftconfig_kwargs  = {}
    sfttrainer_kwargs = {}
    if _SEQ_LEN_IN == "SFTConfig":
        sftconfig_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
    elif _SEQ_LEN_IN == "SFTTrainer":
        sfttrainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        bf16=cuda_ok and torch.cuda.is_bf16_supported(),
        fp16=cuda_ok and not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        dataset_text_field="text",
        report_to="none",
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
    print(f"Training complete. Adapter saved to: {save_path}")


if __name__ == "__main__":
    main()
