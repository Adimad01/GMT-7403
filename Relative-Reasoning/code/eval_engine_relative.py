"""
eval_engine_relative.py
================================================================================
GPU-based relative direction inference (CoT / ToT / GoT).
Supports an optional PEFT adapter for the fine-tuned model.

Dataset:  ../dataset/relative_direction_relations.csv
Fields:   source_entity, target_entity, corpus, relation_label
Labels:   behind, in_front_of, left_of, next_to, right_of

Evaluation: exact match (predicted label == relation_label)

Usage:
    python eval_engine_relative.py \\
        --dataset  ../dataset/relative_direction_relations.csv \\
        --filter-indices ../dataset/eval_25_balanced_indices.json \\
        --model-id openai/gpt-oss-20b \\
        --strategy all \\
        --output-dir results \\
        --model-tag exp1_rel_base_gpu

    # with adapter
    python eval_engine_relative.py \\
        --dataset  ../dataset/relative_direction_relations.csv \\
        --filter-indices ../dataset/eval_25_balanced_indices.json \\
        --model-id openai/gpt-oss-20b \\
        --adapter-path finetuned_gptoss_relative/final_adapter \\
        --strategy all \\
        --output-dir results \\
        --model-tag exp2_rel_lora_gpu
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:native")

# transformers pulls TensorFlow in through image_transforms whenever TF looks
# importable. The cluster's TF is old enough that its generated protobuf code is
# rejected by the installed protobuf ("Descriptors cannot not be created
# directly"), and that kills the whole import chain
# (peft -> transformers.models.bloom -> ... -> import tensorflow).
# These engines are torch-only, so switch TF off rather than repair it. Must be
# set before transformers is imported anywhere in the process.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Belt and braces: if something still drags TF in, the pure-Python protobuf
# implementation accepts the older generated descriptors.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
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

# --- transformers version guard ---------------------------------------------
# The MXFP4 patches below target the transformers 4.5x loader. 5.x replaced it
# with core_model_loading / Mxfp4Dequantize: the patches still get called, but
# the surrounding allocation path differs and the run dies ~4 minutes in with an
# opaque "NVML_SUCCESS == r INTERNAL ASSERT FAILED" from the CUDA caching
# allocator, plus a randomly varying set of MISSING expert layers. Fail now,
# with the fix, instead of after the weight download.
import transformers as _tf
try:
    _tf_major = int(str(_tf.__version__).split(".")[0])
except Exception:
    _tf_major = 0
if _tf_major >= 5:
    sys.stderr.write(
        f"\n[FATAL] transformers {_tf.__version__} is not supported by this engine.\n"
        "        These engines patch the 4.5x MXFP4 loader; under 5.x that path no\n"
        "        longer exists and model loading fails on MIG with an NVML assert.\n"
        "\n        Fix:  pip install 'transformers>=4.55,<5'\n"
        "        Then: python3 diagnose_gpu.py\n"
        "        See:  requirements.txt\n\n")
    sys.exit(2)

from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config

# Monkey-patch mxfp4 MoE dequantization to run on CPU — prevents
# NVML_SUCCESS assert at CUDACachingAllocator.cpp:995 on MIG A100.
try:
    import transformers.integrations.mxfp4 as _mxfp4_m

    def _make_cpu_moe_cvt(_orig):
        # Signature-agnostic: forward *args/**kwargs untouched so this keeps
        # working when the upstream signature changes between releases.
        def _cpu_moe_cvt(blocks, scales, *args, **kwargs):
            target_device = blocks.device
            b = blocks.cpu() if blocks.device.type != "cpu" else blocks
            s = scales.cpu() if scales.device.type != "cpu" else scales
            result = _orig(b, s, *args, **kwargs)
            return result.to(target_device)  # back to GPU after CPU dequantization
        return _cpu_moe_cvt

    # The function was renamed across transformers releases (4.55 exposes
    # _convert_moe_packed_tensors, later ones convert_moe_packed_tensors, and
    # the public wrapper calls the private one). Patch every name that exists
    # rather than pinning to one, and say clearly when none matched -- the
    # previous version silently skipped the patch and the run died minutes
    # later with an opaque allocator assert.
    _patched = []
    for _nm in ("_convert_moe_packed_tensors", "convert_moe_packed_tensors"):
        _orig_fn = getattr(_mxfp4_m, _nm, None)
        if callable(_orig_fn):
            setattr(_mxfp4_m, _nm, _make_cpu_moe_cvt(_orig_fn))
            _patched.append(_nm)
    if _patched:
        print(f"[PATCH] mxfp4 MoE dequantization → CPU  ({', '.join(_patched)})")
    else:
        print("[WARN] mxfp4 MoE patch found NO conversion function to patch — "
              "model loading may fail on MIG. Check the transformers version "
              "against requirements.txt.")
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

from strategies_relative import normalize, VALID_DIRECTIONS, VALID_LIST, extract_direction
from strategies_osm_relative import get_strategy, STRATEGY_MAP
from osm_client import OSMEvidenceKG, NullKG, load_cache, is_geocodable
from graph_kg import GraphKG
from rag_loop import RAGStrategy, DomainSpec
from fewshot import FewShotSelector

EXPERIMENT_SUFFIX = "relative_dir_25_sample"


def _seed_for_prompt(run_seed: int, prompt: str) -> None:
    """Seed every RNG from (run_seed, prompt).

    Generation uses do_sample=True, so without this each run is an
    unreproducible stochastic draw. Deriving the seed from the prompt (rather
    than seeding once at startup) makes a row's output independent of the order
    rows are processed in and of any checkpoint-resume state, so re-running a
    partially-complete job reproduces the same predictions.
    """
    import zlib
    s = (run_seed * 1000003 + zlib.crc32(prompt.encode("utf-8"))) % (2 ** 31 - 1)
    try:
        from transformers import set_seed as _hf_set_seed
        _hf_set_seed(s)
    except Exception:
        import random
        random.seed(s)
        try:
            import torch
            torch.manual_seed(s)
        except Exception:
            pass




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
                      model_tag: str, adapter_tag: str = "none",
                      fewshot=None, prefix_holder=None):

    strategy_name = strategy.name.lower()

    log_path  = os.path.join(output_dir, f"voletc_{model_tag}_{strategy_name}_{EXPERIMENT_SUFFIX}.txt")
    ckpt_path = os.path.join(output_dir, f"voletc_{model_tag}_{strategy_name}_{EXPERIMENT_SUFFIX}_ckpt.json")

    ckpt = _load_checkpoint(ckpt_path)
    processed_indices = set(ckpt.get("processed_indices", []))
    results = list(ckpt.get("results", []))

    if processed_indices:
        print(f"Resuming -- {len(processed_indices)} rows already done.")

    log_f = open(log_path, "a", encoding="utf-8")

    if not processed_indices:
        log_f.write(
            f"{'=' * 90}\n"
            f"  RELATIVE DIR -- {strategy_name.upper()} -- {model_tag.upper()} [{EXPERIMENT_SUFFIX}]\n"
            f"  Inference: GPU  |  Adapter: {adapter_tag}\n"
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
                "source_entity":  str(row.get("source_entity",  "")).strip(),
                "target_entity":  str(row.get("target_entity",  "")).strip(),
                "corpus":         str(row.get("corpus",         "")).strip(),
                "relation_label": str(row.get("relation_label", "")).strip().lower(),
            }

            expected = entity["relation_label"]

            if prefix_holder is not None:
                prefix_holder["v"] = fewshot.build_block(expected) if fewshot else ""

            def row_logger(msg: str):
                log_f.write(msg + "\n")
                log_f.flush()

            row_logger(f"\n{'=' * 90}")
            row_logger(f"ROW {real_idx} | {entity['source_entity']} ? {entity['target_entity']}")
            row_logger(f"Expected: {expected}")
            row_logger(f"{'=' * 90}")

            predicted = None
            for attempt in range(3):
                try:
                    predicted, _ = strategy.reason(entity, log_fn=row_logger)
                except Exception as exc:
                    import traceback as _tb
                    tb_str = _tb.format_exc()
                    row_logger(f"ERROR attempt {attempt + 1}: {exc}\n{tb_str}")
                    tqdm.write(f"  [ERR] row {real_idx} attempt {attempt+1}: {type(exc).__name__}: {exc}")
                    predicted = None
                if predicted in VALID_DIRECTIONS:
                    break
                if attempt < 2:
                    row_logger(f"  retrying (attempt {attempt + 2}/3)...")

            is_match = (predicted == expected)
            correct_so_far = sum(1 for r in results if r.get("match")) + (1 if is_match else 0)
            total_so_far   = len(results) + 1
            running_acc    = correct_so_far / total_so_far * 100

            log_f.write(
                f"\nRESULT | Expected={expected} | Predicted={predicted} | "
                f"{'CORRECT' if is_match else 'WRONG'} | Acc={running_acc:.2f}%\n"
            )
            tqdm.write(f"{real_idx} {expected[:40]} -> {predicted} | acc={running_acc:.1f}%")

            results.append({"index": real_idx, "expected": expected,
                            "predicted": predicted, "match": is_match})
            processed_indices.add(real_idx)

            # Checkpoint after every row so an interruption never loses
            # completed work (each row is expensive; the file is tiny).
            _save_json_atomic(ckpt_path, {
                "processed_indices": sorted(processed_indices),
                "results": results,
            })

    finally:
        _save_json_atomic(ckpt_path, {"processed_indices": sorted(processed_indices), "results": results})
        if results:
            import pandas as pd
            rdf = pd.DataFrame(results)
            acc = rdf["match"].mean() * 100
            log_f.write(f"\nFINAL ACCURACY: {acc:.2f}% ({rdf['match'].sum()}/{len(rdf)})\n")
        log_f.close()

    if results:
        import pandas as pd
        rdf = pd.DataFrame(results)
        acc = rdf["match"].mean() * 100
        print(f"\n{desc} Finished -- Accuracy: {acc:.2f}%")
        print(f"   Log : {log_path}")
        print(f"   CKPT: {ckpt_path}")

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GPU relative direction inference (CoT / ToT / GoT)")
    parser.add_argument("--dataset",        required=True,
                        help="Path to relative_direction_relations.csv")
    parser.add_argument("--filter-indices", default=None,
                        help="JSON file with row indices to evaluate (balanced split)")
    parser.add_argument("--model-id",       default="openai/gpt-oss-20b")
    parser.add_argument("--adapter-path",   default=None)
    parser.add_argument("--kg-mode",        default="none",
                        choices=["none", "input", "rag", "graphrag"],
                        help="none = no KG (Exp 1/2/3); input = OSM evidence prepended "
                             "once (Exp 4/5); rag = per-step OSM retrieval (Exp 6); "
                             "graphrag = k-hop sub-graph + connecting path (Exp 7)")
    parser.add_argument("--strategy",       required=True,
                        choices=list(STRATEGY_MAP.keys()) + ["all"])
    parser.add_argument("--output-dir",     default="./results")
    parser.add_argument("--temperature",    type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed. Sampling is stochastic (do_sample=True), so "
                             "vary this across runs to measure run-to-run variance.")
    parser.add_argument("--max-new-tokens", type=int,   default=1024)
    parser.add_argument("--model-tag",      default="exp1_rel_base_gpu")
    parser.add_argument("--shots",          type=int, default=0,
                        help="0 = zero-shot; 5 = few-shot (5 same-label demos, one per level)")
    parser.add_argument("--train-data",     default=None,
                        help="Train CSV for few-shot demo sampling (required when --shots > 0)")
    parser.add_argument("--keep-ungeocodable", action="store_true",
                        help="Keep rows whose entities failed OSM retrieval (default: drop)")
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

    if not args.keep_ungeocodable:
        cache = load_cache("results/osm_cache.json")
        if cache:
            before = len(df)
            df = df[df.apply(lambda r: is_geocodable(cache, r.get("source_entity"),
                                                     r.get("target_entity")), axis=1)]
            print(f"[OSM-FILTER] dropped {before - len(df)} ungeocodable rows; {len(df)} remain")
            if len(df) == 0:
                print("[ERROR] all rows dropped — warm the cache first (warm_osm_cache.py)")
                sys.exit(1)
        else:
            print("[OSM-FILTER] no osm_cache.json — skipping geocodability filter")

    dist = df["relation_label"].value_counts().to_dict()
    print("[DATA] Label distribution:", dist)

    # ------------------------------------------------------------------
    # 1b. Early-resume: if every selected strategy's checkpoint already
    #     covers all eval rows, skip the expensive model load entirely.
    # ------------------------------------------------------------------
    _strats  = list(STRATEGY_MAP.keys()) if args.strategy == "all" else [args.strategy]
    _eff_tag = f"{args.model_tag}_fs{args.shots}" if args.shots > 0 else args.model_tag
    # Seed goes in the tag so multiple seeds do not overwrite each other.
    if args.seed:
        _eff_tag = f"{_eff_tag}_s{args.seed}"
    _expected = set(df.index.tolist())

    def _strategy_done(sname: str) -> bool:
        cp = os.path.join(args.output_dir,
                          f"voletc_{_eff_tag}_{sname}_{EXPERIMENT_SUFFIX}_ckpt.json")
        done = set(_load_checkpoint(cp).get("processed_indices", []))
        return _expected.issubset(done)

    if _expected and all(_strategy_done(s) for s in _strats):
        print(f"[RESUME] All strategies ({', '.join(_strats)}) already complete for "
              f"'{_eff_tag}' ({len(_expected)} rows) — skipping model load and this "
              f"experiment.")
        return

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
        print(f"[GPU] CUDA available  {gpu_name}  ({gpu_mem:.1f} GB)")
    else:
        print("[GPU] CUDA not available -- will run on CPU")

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

    # Always dequantize MXFP4 -> bf16.  save_pretrained breaks MoE expert key names
    # for gpt-oss-20b so a cached bf16 model produces all-None predictions.
    if cuda_ok:
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated(0)) / 1e9
        print(f"[MODEL] Loading {args.model_id} with MXFP4->bf16 dequantize "
              f"(free GPU~{free_gb:.0f} GB)")
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
    prefix_holder = {"v": ""}   # few-shot demo prefix, set per row

    def gpu_inference_fn(prompt: str) -> str:

        _seed_for_prompt(args.seed, prompt)
        prompt = prefix_holder["v"] + prompt
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
        decoded = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        if _t0_first[0] is None:
            _t0_first[0] = elapsed
            print(f"[TIMING] First sample: {elapsed:.1f}s  ({input_len} input tokens)")
            print(f"[DEBUG]  First response: {decoded[:300]!r}")
        return decoded

    # ------------------------------------------------------------------
    # 4. Build the KG for the selected inference mode
    # ------------------------------------------------------------------
    if args.kg_mode == "none":
        kg = NullKG()
        print("[KG] kg-mode=none — no OSM evidence (Exp 1/2/3)")
    else:
        kg = OSMEvidenceKG("results/osm_cache.json")
        if args.kg_mode == "graphrag":
            kg = GraphKG("results/osm_graph.json", kg)
        _mode_desc = {"input": "static input",
                      "rag": "per-step RAG",
                      "graphrag": "GraphRAG sub-graph"}[args.kg_mode]
        print(f"[KG] kg-mode={args.kg_mode} — OSM evidence active ({_mode_desc})")

    rel_spec = DomainSpec(
        task_noun="relative direction",
        valid_list=VALID_LIST,
        extract_fn=extract_direction,
        parse_entity=lambda e: (e["source_entity"], e["target_entity"], e["corpus"]),
    )

    # ------------------------------------------------------------------
    # 5. Few-shot demo selector (optional, label-conditioned)
    # ------------------------------------------------------------------
    fewshot = None
    if args.shots > 0:
        if not args.train_data or not os.path.exists(args.train_data):
            print(f"[ERROR] --shots {args.shots} requires --train-data <train csv>")
            sys.exit(1)
        fewshot = FewShotSelector(args.train_data)
        args.model_tag = f"{args.model_tag}_fs{args.shots}"
        print(f"[FEWSHOT] {args.shots}-shot label-conditioned demos from {args.train_data} "
              f"(tag → {args.model_tag})")

    # The seed MUST reach the output tag, because the tag is the checkpoint
    # filename and the checkpoint is resumed by row index. Without it, seed 2
    # opens seed 1's checkpoint, sees every row already processed, skips the
    # whole run and reports seed 1's predictions as its own -- producing
    # perfectly identical "seeds" and an apparent zero run-to-run variance.
    # That would silently invalidate every confidence interval built on them.
    if args.seed:
        args.model_tag = f"{args.model_tag}_s{args.seed}"
        print(f"[SEED] {args.seed}  (tag → {args.model_tag})")

    # ------------------------------------------------------------------
    # 6. Run selected strategies
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    strategies = list(STRATEGY_MAP.keys()) if args.strategy == "all" else [args.strategy]

    for strat in strategies:
        print(f"\nRunning {strat.upper()} (adapter={adapter_tag}, kg-mode={args.kg_mode}, shots={args.shots})")
        if args.kg_mode == "rag":
            strategy_obj = RAGStrategy(strat, kg, gpu_inference_fn, rel_spec)
        else:
            strategy_obj = get_strategy(strat, kg=kg, model_fn=gpu_inference_fn,
                                        max_new_tokens=_max_new_tokens,
                                        temperature=_temperature)
        evaluate_strategy(strategy_obj, df, args.output_dir, args.model_tag, adapter_tag,
                          fewshot=fewshot, prefix_holder=prefix_holder)


if __name__ == "__main__":
    main()
