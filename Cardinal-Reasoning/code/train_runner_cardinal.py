"""
train_runner_cardinal.py
================================================================================
Fine-tunes GPT-OSS-20B with LoRA on cardinal_train.jsonl (5680 rows,
710 per direction × 8 directions).  No KG evidence in training prompts.

Used by: Config 2 (Cardinal-LoRA) and Config 5 (extended budget).
Output  : finetuned_gptoss_cardinal/final_adapter

Run:
    python train_runner_cardinal.py
"""

import os
import sys

# ---------------------------------------------------------------------------
# Torchvision stub
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

for _k in [k for k in list(sys.modules) if k == "torchvision" or k.startswith("torchvision.")]:
    del sys.modules[_k]
sys.meta_path.insert(0, _TvStubFinder())
# ---------------------------------------------------------------------------

DATASET    = "../dataset/cardinal_train.jsonl"
OUTPUT_DIR = "finetuned_gptoss_cardinal"
MODEL_ID   = "openai/gpt-oss-20b"


def check_done() -> bool:
    final = os.path.join(OUTPUT_DIR, "final_adapter", "adapter_model.safetensors")
    if os.path.exists(final):
        print(f"[DONE] ✅  Adapter already exists: {final}")
        print("         Delete the directory to re-run fine-tuning.")
        return True
    return False


def preflight():
    if not os.path.exists(DATASET):
        print(f"[ERROR] Training dataset not found: {DATASET}")
        print("        Build it first with: python build_cardinal_training_data.py")
        sys.exit(1)
    n = sum(1 for _ in open(DATASET))
    print(f"[OK] Dataset: {DATASET}  ({n} lines)")


def run():
    print("\n" + "=" * 70)
    print("  FINE-TUNING — Cardinal-LoRA  (plain, no KG)")
    print("=" * 70)
    print(f"  Model      : {MODEL_ID}")
    print(f"  Dataset    : {DATASET}")
    print(f"  Output dir : {OUTPUT_DIR}/final_adapter")
    print("=" * 70 + "\n")

    if check_done():
        sys.exit(0)

    preflight()

    sys.argv = [
        "train_lora_adapter_cardinal.py",
        "--dataset",    DATASET,
        "--run-name",   "cardinal",
        "--model-id",   MODEL_ID,
        "--output-dir", OUTPUT_DIR,
    ]

    from train_lora_adapter_cardinal import main
    main()


if __name__ == "__main__":
    run()
