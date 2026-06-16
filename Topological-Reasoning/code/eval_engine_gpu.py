"""
eval_engine_gpu.py
================================================================================
GPU-based enriched LLM inference (CoT / ToT / GoT) grounded on OSM KG evidence.
Supports an optional PEFT adapter for the fine-tuned model.

Experiment 5 — GPTOSS Fine-tuné + Inférence LLM enrichie par KG:
  python eval_engine_gpu.py \\
      --dataset  ../dataset/triplet_update_v3_30.csv \\
      --model-id openai/gpt-oss-20b \\
      --adapter-path finetuned_gptoss_topological/final_adapter \\
      --filter-indices ../dataset/eval_96_balanced_indices.json \\
      --strategy all \\
      --output-dir results \\
      --model-tag dynamic_osm_finetuned_gpu

Base-model GPU variant (no adapter):
  python eval_engine_gpu.py \\
      --dataset  ../dataset/triplet_update_v3_30.csv \\
      --model-id openai/gpt-oss-20b \\
      --filter-indices ../dataset/eval_96_balanced_indices.json \\
      --strategy all \\
      --output-dir results \\
      --model-tag dynamic_osm_base_gpu
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:native")
import sys
import json
import argparse
import types
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime

# ===========================================================================
# TORCHVISION STUB — custom MetaPathFinder
# On this server torch and torchvision versions mismatch: importing torchvision
# triggers ImportError(_cast_Long) deep inside its ops module.
# transformers.__init__ calls importlib.util.find_spec("torchvision") which
# raises ValueError if __spec__ is None (plain ModuleType stubs won't work).
# Solution: register a MetaPathFinder at priority 0 that intercepts ALL
# torchvision imports and returns minimal empty stubs with proper __spec__.
# ===========================================================================
import importlib.machinery
import importlib.abc

class _TorchvisionStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torchvision" or fullname.startswith("torchvision."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return None  # default module creation — __spec__ set by import machinery

    def exec_module(self, module):
        module.__path__ = []
        # Must be a real string so inspect.getsourcefile / os.fspath never
        # receive a _Stub object (bitsandbytes iterates sys.modules calling
        # inspect on each one during @torch.library.register).
        module.__file__ = "<torchvision-stub>"

        # Self-referencing catch-all for non-dunder attribute access.
        # Dunders raise AttributeError so Python's default machinery handles
        # them (avoids _Stub objects leaking into inspect / os.fspath paths).
        class _Stub:
            def __init__(self, n=""):  self._n = n
            def __getattr__(self, n):  return _Stub(n)
            def __call__(self, *a, **k): return _Stub()
            def __iter__(self):        return iter([])

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
                NEAREST       = "nearest"
                NEAREST_EXACT = "nearest-exact"
                BILINEAR      = "bilinear"
                BICUBIC       = "bicubic"
                BOX           = "box"
                HAMMING       = "hamming"
                LANCZOS       = "lanczos"
            module.InterpolationMode = _InterpolationMode

# Remove any partial / broken cached state, then install our finder first
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
# PATCH: kernels.LayerRepository — newer kernels versions require revision= or
# version= in the constructor; transformers/hub_kernels.py calls it without
# those args at module level.  Patching the __init__ METHOD on the class
# (rather than replacing the class) ensures the fix applies even when
# hub_kernels.py already holds a direct reference to the class object.
# ===========================================================================
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

# ===========================================================================
# DEPENDENCY MONKEY PATCHES (same as eval_kg_instruction_finetuned.py)
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
# ===========================================================================

from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config

# Monkey-patch mxfp4 MoE dequantization to run on CPU — prevents
# NVML_SUCCESS assert at CUDACachingAllocator.cpp:995 on MIG A100.
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

try:
    import transformers.modeling_utils as _tmu
    _tmu.caching_allocator_warmup = lambda *_a, **_k: None
    print("[PATCH] caching_allocator_warmup → no-op  (MIG A100 OOM fix)")
except Exception as _wpe:
    print(f"[WARN] warmup patch skipped: {_wpe}")

from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))

from strategies_osm import (
    get_strategy,
    VALID_PREDICATES,
    VALID_LIST,
    STRATEGY_MAP,
    GeographicKnowledgeGraph,
    extract_predicate,
)
from osm_client import NullKG
from rag_loop import RAGStrategy, DomainSpec

EXPERIMENT_SUFFIX = "neighborhood_details_spatial_relation_16_sample"


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
            f"  VOLET C — {strategy_name.upper()} — {model_tag.upper()} [{EXPERIMENT_SUFFIX}]\n"
            f"  Inference: GPU  |  Adapter: {adapter_tag}\n"
            f"  KG: Dynamic OSM API\n"
            f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'=' * 90}\n\n"
        )
        log_f.flush()

    desc = f"[{model_tag}/{strategy_name}]"

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            real_idx = row.name

            if real_idx in processed_indices:
                continue

            entity = {
                "place_name_subject":    str(row.get("source_entity",        row.get("place_name_subject", ""))).strip(),
                "place_name_object":     str(row.get("target_entity",         row.get("place_name_object",  ""))).strip(),
                "placetype_subject":     str(row.get("placetype_subject",     "place")).strip(),
                "placetype_object":      str(row.get("placetype_object",      "place")).strip(),
                "geometry_type_subject": str(row.get("source_geometry",       row.get("geometry_type_subject", "unknown"))).strip(),
                "geometry_type_object":  str(row.get("target_geometry",       row.get("geometry_type_object",  "unknown"))).strip(),
                "relation_predicate":    str(row.get("corpus",                row.get("Sentence", row.get("relation_predicate", "")))).strip(),
                "vernacular_relation":   str(row.get("explanation",           row.get("vernacular_relation", ""))).strip(),
                "sentence":              str(row.get("corpus",                row.get("Sentence", ""))).strip(),
            }

            expected = str(row.get("relation_label", row.get("spatial_relation", ""))).lower().strip()

            def row_logger(msg: str):
                log_f.write(msg + "\n")
                log_f.flush()

            row_logger(f"\n{'=' * 90}")
            row_logger(f"ROW {real_idx} | {entity['place_name_subject']} → {entity['place_name_object']}")
            row_logger(f"Expected: {expected}")
            row_logger(f"{'=' * 90}")

            predicted = "invalid"
            for attempt in range(3):
                try:
                    predicted, _ = strategy.reason(entity, log_fn=row_logger)
                except Exception as exc:
                    row_logger(f"ERROR attempt {attempt + 1}: {exc}")
                    predicted = "invalid"
                if predicted in VALID_PREDICATES:
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

            if len(results) % 16 == 0:
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
    parser = argparse.ArgumentParser(description="GPU-based enriched OSM inference (CoT / ToT / GoT)")
    parser.add_argument("--dataset",         required=True,
                        help="Path to the 30%% test CSV (triplet_update_v3_30.csv)")
    parser.add_argument("--model-id",        default="openai/gpt-oss-20b")
    parser.add_argument("--adapter-path",    default=None,
                        help="LoRA adapter directory (omit for base model without adapter)")
    parser.add_argument("--filter-indices",  default=None,
                        help="JSON file of row indices to evaluate (eval_96_balanced_indices.json). "
                             "If omitted, falls back to stratified sampling (random_state=42).")
    parser.add_argument("--strategy",        required=True,
                        choices=list(STRATEGY_MAP.keys()) + ["all"])
    parser.add_argument("--output-dir",      default="./results")
    parser.add_argument("--temperature",     type=float, default=0.1)
    parser.add_argument("--max-new-tokens",  type=int,   default=1024)
    parser.add_argument("--model-tag",       default="dynamic_osm_finetuned_gpu")
    parser.add_argument("--kg-mode",         default="none",
                        choices=["none", "input", "rag"],
                        help="none = no KG evidence (Exp 1/2/3); "
                             "input = OSM evidence prepended once (Exp 4/5); "
                             "rag = per-step OSM retrieval during reasoning (Exp 6)")
    parser.add_argument("--no-kg",           action="store_true",
                        help="(deprecated) alias for --kg-mode none")
    args = parser.parse_args()
    if args.no_kg:
        args.kg_mode = "none"

    # ------------------------------------------------------------------
    # 1. Load & filter dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(args.dataset)

    if args.filter_indices:
        with open(args.filter_indices, "r") as fh:
            keep = set(json.load(fh))
        df = df.loc[df.index.isin(keep)]
        print(f"[DATA] Filtered to {len(df)} rows using {args.filter_indices}")
    else:
        label_col = "relation_label" if "relation_label" in df.columns else "spatial_relation"
        df["_sr_clean"] = df[label_col].astype(str).str.lower().str.strip()
        sampled = [
            grp.sample(n=min(16, len(grp)), random_state=42)
            for grp in (df[df["_sr_clean"] == p] for p in VALID_PREDICATES)
            if len(grp) > 0
        ]
        df = pd.concat(sampled).drop(columns=["_sr_clean"])
        print(f"[DATA] Stratified sample: {len(df)} rows (16 per predicate)")

    print(f"[DATA] {len(df)} test rows ready.")

    # ------------------------------------------------------------------
    # 2. Load model on GPU
    # ------------------------------------------------------------------
    tok_path = args.model_id
    if args.adapter_path and os.path.isfile(os.path.join(args.adapter_path, "tokenizer_config.json")):
        tok_path = args.adapter_path

    print(f"[MODEL] Tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- GPU diagnostics ---------------------------------------------------
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] CUDA available ✅  {gpu_name}  ({gpu_mem:.1f} GB)")
    else:
        print("[GPU] ⚠️  CUDA not available — model will load on CPU (very slow!)")
        print("[GPU] Check: nvidia-smi  and  python -c 'import torch; print(torch.version.cuda)'")

    dtype = torch.bfloat16 if (cuda_ok and torch.cuda.is_bf16_supported()) else torch.float16

    # Patch transformers ≥5.9.0 bug: supports_quant_method crashes when model's
    # quantization_config is None instead of a dict.
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

    # Always dequantize MXFP4 → bf16 on load.  save_pretrained does NOT preserve
    # MoE expert key names correctly for gpt-oss-20b (gate_up_proj saved at root
    # level instead of mlp.experts.gate_up_proj), so a pre-saved bf16 cache would
    # silently reinitialise the expert layers and produce all-None predictions.
    # Dequantize needs ~60 GB peak; with ≥80 GB free this is always safe.
    if cuda_ok:
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated(0)) / 1e9
        print(f"[MODEL] Loading {args.model_id} with MXFP4→bf16 dequantize "
              f"(free GPU≈{free_gb:.0f} GB)")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=Mxfp4Config(dequantize=True),
    )

    adapter_tag = "base"
    if args.adapter_path:
        adapter_abs = os.path.abspath(args.adapter_path)
        print(f"[MODEL] Applying LoRA adapter: {adapter_abs}")
        import peft.utils.save_and_load as _peft_sl

        # Patch 1 — file_exists (older PEFT versions)
        _orig_fe = _peft_sl.file_exists
        def _local_file_exists(repo_id, *a, **kw):
            if os.path.isabs(str(repo_id)):
                return False
            return _orig_fe(repo_id, *a, **kw)
        _peft_sl.file_exists = _local_file_exists

        # Patch 2 — hf_hub_download (newer PEFT versions go here directly).
        # PEFT may request "adapter_model.bin" first; fall back to .safetensors.
        _orig_hhd = getattr(_peft_sl, "hf_hub_download", None)
        def _local_hf_hub_download(repo_id, filename, **kw):
            if os.path.isabs(str(repo_id)):
                # Candidate directories to search (given dir + parent)
                candidates = [repo_id, os.path.dirname(repo_id)]
                stem, ext = os.path.splitext(filename)
                alt_exts = [".safetensors", ".bin"] if ext == ".bin" else [ext, ".safetensors", ".bin"]
                for d in candidates:
                    for e in alt_exts:
                        p = os.path.join(d, stem + e)
                        if os.path.exists(p):
                            print(f"[PEFT-PATCH] resolved {filename} → {p}")
                            return p
                # Emit a diagnostic listing so the server log shows what IS there
                for d in candidates:
                    if os.path.isdir(d):
                        files = os.listdir(d)
                        print(f"[PEFT-PATCH] dir {d!r} contents: {files}")
                raise FileNotFoundError(
                    f"Adapter weight not found: tried {[os.path.join(c, filename) for c in candidates]}"
                )
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
    _model_device = next(model.parameters()).device
    print(f"[MODEL] Ready. Adapter: {adapter_tag}  |  Device: {_model_device}")

    # ------------------------------------------------------------------
    # 3. Build GPU inference callable
    # ------------------------------------------------------------------
    _temperature     = args.temperature
    _max_new_tokens  = args.max_new_tokens

    # Determine safe max input length (model config takes precedence over tokenizer)
    _model_max_len = getattr(model.config, "max_position_embeddings", None) \
                  or getattr(tokenizer, "model_max_length", 2048)
    # Leave room for the generated tokens
    _max_input_len = max(64, _model_max_len - _max_new_tokens - 32)
    print(f"[MODEL] max_position_embeddings={_model_max_len}  →  truncating inputs to {_max_input_len} tokens")

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
            print(f"[TIMING] First sample: {elapsed:.1f}s  ({input_len} input tokens → {_max_new_tokens} max new)")
        return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # 4. Build the KG for the selected inference mode
    # ------------------------------------------------------------------
    if args.kg_mode == "none":
        kg = NullKG()
        print("[KG] kg-mode=none — no OSM evidence at inference (Exp 1/2/3)")
    else:
        kg = GeographicKnowledgeGraph("results/osm_cache.json")
        print(f"[KG] kg-mode={args.kg_mode} — OSM evidence active "
              f"({'static input' if args.kg_mode == 'input' else 'per-step RAG'})")

    topo_spec = DomainSpec(
        task_noun="DE-9IM topological predicate",
        valid_list=VALID_LIST,
        extract_fn=extract_predicate,
        parse_entity=lambda e: (
            e["place_name_subject"], e["place_name_object"],
            e.get("sentence") or e.get("vernacular_relation", ""),
        ),
    )

    # ------------------------------------------------------------------
    # 5. Run selected strategies
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    strategies = list(STRATEGY_MAP.keys()) if args.strategy == "all" else [args.strategy]

    for strat in strategies:
        print(f"\n🚀 Running {strat.upper()} with GPU inference "
              f"(adapter={adapter_tag}, kg-mode={args.kg_mode})")
        if args.kg_mode == "rag":
            strategy_obj = RAGStrategy(strat, kg, gpu_inference_fn, topo_spec)
        else:
            strategy_obj = get_strategy(strat, kg, model_fn=gpu_inference_fn)
        evaluate_strategy(strategy_obj, df, args.output_dir, args.model_tag, adapter_tag)


if __name__ == "__main__":
    main()
