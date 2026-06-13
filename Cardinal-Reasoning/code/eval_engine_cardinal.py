"""
eval_engine_cardinal.py
================================================================================
GPU-based cardinal direction inference (CoT / ToT / GoT).
Supports an optional PEFT adapter for the fine-tuned model.

Dataset:  ../dataset/cardinal_eval_80.csv   (80 rows, 10/direction × 8 directions)
Fields:   id, question, direction

Usage:
    python eval_engine_cardinal.py \\
        --dataset  ../dataset/cardinal_eval_80.csv \\
        --model-id openai/gpt-oss-20b \\
        --strategy all \\
        --output-dir results \\
        --model-tag exp1_card_base_gpu

    # with adapter
    python eval_engine_cardinal.py \\
        --dataset  ../dataset/cardinal_eval_80.csv \\
        --model-id openai/gpt-oss-20b \\
        --adapter-path finetuned_gptoss_cardinal/final_adapter \\
        --strategy all \\
        --output-dir results \\
        --model-tag exp2_card_topo_gpu
"""

import os
import sys
import json
import argparse
import types
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime

# ===========================================================================
# TORCHVISION STUB
# ===========================================================================
import importlib.machinery
import importlib.abc

class _TorchvisionStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
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
            module.decode_image  = None
        elif module.__name__ == "torchvision.transforms":
            class _InterpolationMode:
                NEAREST = "nearest"; NEAREST_EXACT = "nearest-exact"
                BILINEAR = "bilinear"; BICUBIC = "bicubic"
                BOX = "box"; HAMMING = "hamming"; LANCZOS = "lanczos"
            module.InterpolationMode = _InterpolationMode

for _k in [k for k in list(sys.modules) if k == "torchvision" or k.startswith("torchvision.")]:
    del sys.modules[_k]
sys.meta_path.insert(0, _TorchvisionStubFinder())
# ===========================================================================

# ===========================================================================
# TORCHAUDIO STUB — transformers 5.x imports torchaudio in loss_rnnt.py but
# torchaudio 0.13 was compiled for CUDA 11 while torch 2.5 uses CUDA 12.
# ===========================================================================
class _TorchaudioStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torchaudio" or fullname.startswith("torchaudio."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.__path__ = []
        module.__file__ = "<torchaudio-stub>"
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

for _k in [k for k in list(sys.modules) if k == "torchaudio" or k.startswith("torchaudio.")]:
    del sys.modules[_k]
sys.meta_path.insert(0, _TorchaudioStubFinder())
# ===========================================================================

# ===========================================================================
# DEPENDENCY MONKEY PATCHES
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

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e5m2", None)

# Patch: kernels.LayerRepository requires revision= or version=
try:
    from kernels.layer.layer import LayerRepository as _LayerRepo
    _orig_lr_init = _LayerRepo.__init__
    def _patched_lr_init(self, *args, revision=None, version=None, **kwargs):
        if revision is None and version is None:
            version = "0"
        _orig_lr_init(self, *args, revision=revision, version=version, **kwargs)
    _LayerRepo.__init__ = _patched_lr_init
except Exception as _kp_err:
    pass
# ===========================================================================

from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))

from strategies_cardinal import (
    get_strategy,
    VALID_DIRECTIONS,
    STRATEGY_MAP,
    CardinalKnowledgeGraph,
)

EXPERIMENT_SUFFIX = "cardinal_dir_32_sample"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _save_json_atomic(path: str, data):
    import tempfile
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_checkpoint(ckpt_path: str) -> dict:
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"processed_indices": [], "results": []}


# ---------------------------------------------------------------------------
# EVALUATION LOOP
# ---------------------------------------------------------------------------
def evaluate_strategy(strategy, df: pd.DataFrame, output_dir: str,
                      model_tag: str, adapter_tag: str = "none"):

    strategy_name = strategy.name.lower()

    log_path  = os.path.join(output_dir, f"voletc_{model_tag}_{strategy_name}_{EXPERIMENT_SUFFIX}.txt")
    ckpt_path = os.path.join(output_dir, f"voletc_{model_tag}_{strategy_name}_{EXPERIMENT_SUFFIX}_ckpt.json")

    ckpt = _load_checkpoint(ckpt_path)
    processed_indices = set(ckpt.get("processed_indices", []))
    results = list(ckpt.get("results", []))

    if processed_indices:
        print(f"♻️  Resuming — {len(processed_indices)} rows already done.")

    log_f = open(log_path, "a", encoding="utf-8")

    if not processed_indices:
        log_f.write(
            f"{'=' * 90}\n"
            f"  CARDINAL — {strategy_name.upper()} — {model_tag.upper()} [{EXPERIMENT_SUFFIX}]\n"
            f"  Inference: GPU  |  Adapter: {adapter_tag}\n"
            f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'=' * 90}\n\n"
        )
        log_f.flush()

    desc = f"[{model_tag}/{strategy_name}]"
    total_expected = len(df)

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            real_idx = row.name

            if real_idx in processed_indices:
                continue

            entity = {
                "source_entity": str(row.get("source_entity", "")).strip(),
                "target_entity": str(row.get("target_entity", "")).strip(),
                "corpus":        str(row.get("corpus", "")).strip(),
                "relation_label": str(row.get("relation_label", "")).strip().lower(),
            }

            expected = entity["relation_label"]

            def row_logger(msg: str):
                log_f.write(msg + "\n")
                log_f.flush()

            row_logger(f"\n{'=' * 90}")
            row_logger(f"ROW {real_idx} | {entity['source_entity']} ? {entity['target_entity']}")
            row_logger(f"Expected: {expected}")
            row_logger(f"{'=' * 90}")

            predicted = "invalid"
            for attempt in range(3):
                try:
                    predicted, _ = strategy.reason(entity, log_fn=row_logger)
                except Exception as exc:
                    row_logger(f"ERROR attempt {attempt + 1}: {exc}")
                    predicted = "invalid"
                if predicted in VALID_DIRECTIONS:
                    break
                if attempt < 2:
                    row_logger(f"  ↻ invalid — retrying (attempt {attempt + 2}/3)...")

            is_match = (expected == predicted)
            correct_so_far = sum(1 for r in results if r.get("match")) + (1 if is_match else 0)
            total_so_far   = len(results) + 1
            running_acc    = correct_so_far / total_so_far * 100

            log_f.write(
                f"\nRESULT | Expected={expected} | Predicted={predicted} | "
                f"{'CORRECT' if is_match else 'WRONG'} | Acc={running_acc:.2f}%\n"
            )
            tqdm.write(f"{real_idx} {expected} → {predicted} | acc={running_acc:.1f}%")

            results.append({"index": real_idx, "expected": expected,
                            "predicted": predicted, "match": is_match})
            processed_indices.add(real_idx)

            if len(results) % 10 == 0:
                _save_json_atomic(ckpt_path, {
                    "processed_indices": sorted(processed_indices),
                    "results": results,
                })

    finally:
        _save_json_atomic(ckpt_path, {"processed_indices": sorted(processed_indices), "results": results})
        if results:
            rdf = pd.DataFrame(results)
            acc = rdf["match"].mean() * 100
            log_f.write(f"\nFINAL ACCURACY: {acc:.2f}% ({rdf['match'].sum()}/{len(rdf)})\n")
        log_f.close()

    if results:
        rdf = pd.DataFrame(results)
        acc = rdf["match"].mean() * 100
        print(f"\n✅ {desc} Finished — Accuracy: {acc:.2f}%")
        print(f"   Log : {log_path}")
        print(f"   CKPT: {ckpt_path}")

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GPU cardinal direction inference (CoT / ToT / GoT)")
    parser.add_argument("--dataset",        required=True,
                        help="Path to cardinal eval CSV")
    parser.add_argument("--filter-indices", default=None,
                        help="JSON file with row indices to evaluate (for balanced eval split)")
    parser.add_argument("--model-id",       default="openai/gpt-oss-20b")
    parser.add_argument("--adapter-path",   default=None)
    parser.add_argument("--use-kg",         action="store_true",
                        help="Pass rule-based CardinalKnowledgeGraph to strategy")
    parser.add_argument("--strategy",       required=True,
                        choices=list(STRATEGY_MAP.keys()) + ["all"])
    parser.add_argument("--output-dir",     default="./results")
    parser.add_argument("--temperature",    type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int,   default=512)
    parser.add_argument("--model-tag",      default="exp1_card_base_gpu")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(args.dataset)
    if args.filter_indices:
        with open(args.filter_indices) as f:
            keep = set(json.load(f))
        df = df[df.index.isin(keep)]
        print(f"[DATA] Filtered to {len(df)} eval rows from {args.filter_indices}")
    print(f"[DATA] {len(df)} eval rows ready  ({args.dataset})")

    dist = df["relation_label"].value_counts().to_dict()
    print("[DATA] Label distribution:", dist)

    # ------------------------------------------------------------------
    # 2. Load model on GPU
    # ------------------------------------------------------------------
    tok_path = args.model_id
    if args.adapter_path and os.path.isfile(os.path.join(args.adapter_path, "tokenizer_config.json")):
        tok_path = args.adapter_path

    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] CUDA available ✅  {gpu_name}  ({gpu_mem:.1f} GB)")
    else:
        print("[GPU] ⚠️  CUDA not available — will run on CPU")

    dtype = torch.bfloat16 if (cuda_ok and torch.cuda.is_bf16_supported()) else torch.float16

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

    # Always dequantize MXFP4 → bf16.  save_pretrained breaks MoE expert key names
    # for gpt-oss-20b so a cached bf16 model produces all-None predictions.
    if cuda_ok:
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated(0)) / 1e9
        print(f"[MODEL] Loading {args.model_id} with MXFP4→bf16 dequantize "
              f"(free GPU≈{free_gb:.0f} GB)")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            trust_remote_code=True,
            dtype=dtype,
            quantization_config=Mxfp4Config(dequantize=True),
        )
    except RuntimeError as _mig_err:
        # transformers wraps the raw NVML assert in its own RuntimeError, so walk
        # the full __cause__/__context__ chain to find the real error.
        _exc, _nvml = _mig_err, False
        _seen: set = set()
        while _exc is not None and id(_exc) not in _seen:
            _seen.add(id(_exc))
            if "NVML_SUCCESS" in str(_exc) or "CUDACachingAllocator" in str(_exc):
                _nvml = True
                break
            _exc = _exc.__cause__ if _exc.__cause__ is not None else _exc.__context__
        if not _nvml:
            raise
        print("[MODEL] NVML assert on MIG partition — retrying: dequantize on CPU then move to CUDA")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="cpu",
            trust_remote_code=True,
            dtype=dtype,
            quantization_config=Mxfp4Config(dequantize=True),
        )
        model = model.cuda()

    adapter_tag = "base"
    if args.adapter_path:
        adapter_abs = os.path.abspath(args.adapter_path)
        print(f"[MODEL] Applying LoRA adapter: {adapter_abs}")
        import peft.utils.save_and_load as _peft_sl

        _orig_fe = _peft_sl.file_exists
        def _local_file_exists(repo_id, *a, **kw):
            if os.path.isabs(str(repo_id)):
                return False
            return _orig_fe(repo_id, *a, **kw)
        _peft_sl.file_exists = _local_file_exists

        _orig_hhd = getattr(_peft_sl, "hf_hub_download", None)
        def _local_hf_hub_download(repo_id, filename, **kw):
            if os.path.isabs(str(repo_id)):
                candidates = [repo_id, os.path.dirname(repo_id)]
                stem, ext = os.path.splitext(filename)
                alt_exts = [".safetensors", ".bin"] if ext == ".bin" else [ext, ".safetensors", ".bin"]
                for d in candidates:
                    for e in alt_exts:
                        p = os.path.join(d, stem + e)
                        if os.path.exists(p):
                            return p
                raise FileNotFoundError(f"Adapter weight not found for {filename}")
            return _orig_hhd(repo_id, filename, **kw)
        if _orig_hhd is not None:
            _peft_sl.hf_hub_download = _local_hf_hub_download

        try:
            model = PeftModel.from_pretrained(model, adapter_abs, local_files_only=True)
        finally:
            _peft_sl.file_exists = _orig_fe
            if _orig_hhd is not None:
                _peft_sl.hf_hub_download = _orig_hhd
        adapter_tag = os.path.basename(args.adapter_path.rstrip("/"))

    model.eval()
    print(f"[MODEL] Ready. Adapter: {adapter_tag}  |  Device: {next(model.parameters()).device}")

    # ------------------------------------------------------------------
    # 3. Build GPU inference callable
    # ------------------------------------------------------------------
    _temperature    = args.temperature
    _max_new_tokens = args.max_new_tokens
    _model_max_len  = getattr(model.config, "max_position_embeddings", None) \
                   or getattr(tokenizer, "model_max_length", 2048)
    _max_input_len  = max(64, _model_max_len - _max_new_tokens - 32)

    import time as _time
    _t0_first = [None]

    def gpu_inference_fn(prompt: str) -> str:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=_max_input_len,
        ).to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        t0 = _time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=_max_new_tokens,
                do_sample=True,
                temperature=_temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = _time.time() - t0
        if _t0_first[0] is None:
            _t0_first[0] = elapsed
            print(f"[TIMING] First sample: {elapsed:.1f}s  ({input_len} input tokens)")
        return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # 4. Optionally build KG
    # ------------------------------------------------------------------
    kg = CardinalKnowledgeGraph() if args.use_kg else None

    # ------------------------------------------------------------------
    # 5. Run selected strategies
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    strategies = list(STRATEGY_MAP.keys()) if args.strategy == "all" else [args.strategy]

    for strat in strategies:
        print(f"\n🚀 Running {strat.upper()} (adapter={adapter_tag}, kg={'yes' if kg else 'no'})")
        strategy_obj = get_strategy(strat, kg=kg, model_fn=gpu_inference_fn,
                                    max_new_tokens=_max_new_tokens,
                                    temperature=_temperature)
        evaluate_strategy(strategy_obj, df, args.output_dir, args.model_tag, adapter_tag)


if __name__ == "__main__":
    main()
