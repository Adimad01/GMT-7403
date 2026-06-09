"""
save_bf16_model.py
================================================================================
One-time utility: load GPT-OSS-20B from its native MXFP4 checkpoint, dequantize
to bf16, then save the plain-bf16 weights to disk.

Why: dequantizing MXFP4 → bf16 on the fly requires ~55-60 GB transient peak
(original MXFP4 weights + converted bf16 weights both resident).  Loading the
pre-saved bf16 checkpoint directly costs only ~40 GB steady-state — no spike —
so training and inference both fit on a half-occupied MIG slice.

Run once (needs a free GPU with ≥60 GB):
    python save_bf16_model.py
    python save_bf16_model.py --output-dir /home/jovyan/models/gpt-oss-20b-bf16

After saving, update MODEL_BF16_DIR in this script and all other scripts will
load from the saved checkpoint automatically.
"""

import os
import sys
import argparse
import torch

# ── Torchvision stub (same as other scripts) ────────────────────────────────
import importlib.machinery
import importlib.abc

class _TvStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torchvision" or fullname.startswith("torchvision."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec): return None
    def exec_module(self, module):
        module.__path__ = []
        module.__file__ = "<torchvision-stub>"
        class _Stub:
            def __init__(self, n=""): self._n = n
            def __getattr__(self, n): return _Stub(n)
            def __call__(self, *a, **k): return _Stub()
            def __iter__(self): return iter([])
        def _catchall(name):
            if name.startswith("__") and name.endswith("__"): raise AttributeError(name)
            return _Stub(name)
        module.__getattr__ = _catchall

for _k in [k for k in list(sys.modules) if k == "torchvision" or k.startswith("torchvision.")]:
    del sys.modules[_k]
sys.meta_path.insert(0, _TvStubFinder())
# ────────────────────────────────────────────────────────────────────────────

# ── Torch patches ────────────────────────────────────────────────────────────
if not hasattr(torch, "accelerator"):
    class _DA:
        @staticmethod
        def current_accelerator():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.accelerator = _DA()

if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submod(self, target, module):
        if not target: raise ValueError("target cannot be empty")
        atoms = target.split(".")
        name = atoms.pop(-1)
        mod = self
        for item in atoms:
            mod = getattr(mod, item)
        setattr(mod, name, module)
    torch.nn.Module.set_submodule = _set_submod

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e5m2", None)

try:
    from kernels.layer.layer import LayerRepository as _LR
    _orig = _LR.__init__
    def _patched(self, *a, revision=None, version=None, **kw):
        if revision is None and version is None: version = "0"
        _orig(self, *a, revision=revision, version=version, **kw)
    _LR.__init__ = _patched
    print("[PATCH] kernels OK")
except Exception as e:
    print(f"[WARN] kernels patch skipped: {e}")
# ────────────────────────────────────────────────────────────────────────────

from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config


DEFAULT_MODEL_ID  = "openai/gpt-oss-20b"
DEFAULT_OUTPUT    = "/home/jovyan/models/gpt-oss-20b-bf16"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id",   default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        free    = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated(0)) / 1e9
        print(f"[GPU] {torch.cuda.get_device_name(0)}  total={gpu_mem:.0f} GB  free≈{free:.0f} GB")
        if free < 55:
            print(f"[WARN] Only {free:.0f} GB free — dequantize needs ~55-60 GB peak. "
                  f"Risk of OOM. Continue? (Ctrl-C to abort)")
    else:
        print("[GPU] No CUDA — will save on CPU (very slow, weights on RAM)")

    out = args.output_dir
    if os.path.isfile(os.path.join(out, "config.json")):
        print(f"[DONE] bf16 checkpoint already exists at {out} — nothing to do.")
        print("       Delete the directory to regenerate.")
        return

    print(f"\n[1/3] Loading tokenizer from {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    print(f"[2/3] Loading model (MXFP4 → dequantize to bf16) ...")
    dtype = torch.bfloat16 if (cuda_ok and torch.cuda.is_bf16_supported()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        dtype=dtype,
        quantization_config=Mxfp4Config(dequantize=True),
    )
    print(f"       Model dtype : {next(model.parameters()).dtype}")
    if cuda_ok:
        used = torch.cuda.memory_allocated(0) / 1e9
        print(f"       GPU used    : {used:.1f} GB")

    print(f"[3/3] Saving bf16 weights to {out} ...")
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    size_gb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(out) for f in files
    ) / 1e9
    print(f"\n✅  Saved {size_gb:.1f} GB → {out}")
    print("   Future loads: AutoModelForCausalLM.from_pretrained(out, device_map='auto', dtype=torch.bfloat16)")
    print("   No quantization_config needed — plain bf16, ~40 GB peak, no conversion spike.")


if __name__ == "__main__":
    main()
