"""
eval_kg_instruction_finetuned.py
================================================================================
Phase 4 — Unified Evaluation for All Four Model Variants

Evaluates any of the four models on the 30% test split (triplet_update_v3_30.csv):

  Variant A: Base GPT-OSS (no adapter, no KG evidence)     → --kg-source none
  Variant B: Data-only fine-tuned (existing adapter)        → --kg-source none  + --adapter-path finetuned_gptoss_topological/final_adapter
  Variant C: OSM-KG fine-tuned                             → --kg-source osm   + --adapter-path finetuned_gptoss_osm_kg/final_adapter
  Variant D: Wikidata-KG fine-tuned                        → --kg-source wikidata + --adapter-path finetuned_gptoss_wikidata_kg/final_adapter

Key design principle:
  Models trained WITH KG evidence are evaluated WITH the same KG evidence in
  the prompt (--kg-source osm / wikidata).  The base and data-only models are
  evaluated WITHOUT KG evidence (--kg-source none) to match their training setup.
  This ensures every variant is tested under the conditions it was trained on.

Output: a CSV log with columns [index, expected, predicted, match, description]
  placed in --output-dir with a filename that encodes the model variant and KG source.
  The filename convention matches the existing logs so compare_kg_finetune_results.py
  can auto-discover all files.

Usage examples:
  # Variant A — base model, no KG evidence
  python eval_kg_instruction_finetuned.py \
      --dataset    ../dataset/triplet_update_v3_30.csv \
      --model-id   openai/gpt-oss-20b \
      --kg-source  none \
      --model-tag  gptoss_base_no_kg \
      --output-dir results

  # Variant C — OSM-KG fine-tuned
  python eval_kg_instruction_finetuned.py \
      --dataset      ../dataset/triplet_update_v3_30.csv \
      --model-id     openai/gpt-oss-20b \
      --adapter-path finetuned_gptoss_osm_kg/final_adapter \
      --kg-source    osm \
      --osm-cache    results/osm_cache.json \
      --model-tag    gptoss_osm_kg_finetuned \
      --output-dir   results

  # Variant D — Wikidata-KG fine-tuned
  python eval_kg_instruction_finetuned.py \
      --dataset        ../dataset/triplet_update_v3_30.csv \
      --model-id       openai/gpt-oss-20b \
      --adapter-path   finetuned_gptoss_wikidata_kg/final_adapter \
      --kg-source      wikidata \
      --wikidata-cache results/wikidata_cache.json \
      --model-tag      gptoss_wikidata_kg_finetuned \
      --output-dir     results
"""

import argparse
import os
import re
import sys

import pandas as pd
import torch
from tqdm import tqdm

# ===========================================================================
# DEPENDENCY MONKEY PATCH (identical to all other scripts in this project)
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
# ===========================================================================

from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))

VALID_PREDICATES = {"disjoint", "touches", "crosses", "within", "contains", "overlaps"}

# ---------------------------------------------------------------------------
# Prompt builders — must mirror the training templates exactly
# ---------------------------------------------------------------------------

_SYSTEM = "You are an expert in geospatial topological reasoning using the DE-9IM model."

_VALID_PREDS = "contains, within, touches, crosses, disjoint, overlaps"

_RULES_NO_KG = """\
1. The relation is DIRECTED from entity A to entity B.
2. Consider geometry types (Point, LineString, Polygon, MultiPolygon).
3. Direction: "contains" means A encloses B; "within" means A is inside B.
4. You must pick EXACTLY ONE predicate from the list above."""

_RULES_OSM = """\
1. The relation is DIRECTED from entity A to entity B.
2. Use the KG evidence (coordinates, bounding boxes, administrative hierarchy) \
to reason geometrically.
3. Bounding box containment ≠ polygon containment — treat bbox as a clue, not proof.
4. Direction: "contains" means A encloses B; "within" means A is inside B.
5. You must pick EXACTLY ONE predicate from the list above."""

_RULES_WIKIDATA = """\
1. The relation is DIRECTED from entity A to entity B.
2. Use the Wikidata KG evidence (P131 located-in chain, P31 instance-of, coordinates).
3. Wikidata provides centroid coordinates only, NOT polygon geometry.
4. If A appears in B's P131 chain → within.  If B appears in A's P131 chain → contains.
5. Direction: "contains" means A encloses B; "within" means A is inside B.
6. You must pick EXACTLY ONE predicate from the list above."""


def build_prompt_no_kg(row: dict) -> str:
    subj      = row.get("place_name_subject", "Entity A")
    obj       = row.get("place_name_object",  "Entity B")
    vernacular = row.get("vernacular_relation") or row.get("relation_predicate", "")
    subj_type = row.get("placetype_subject", "place")
    obj_type  = row.get("placetype_object",  "place")
    geom_A    = row.get("geometry_type_subject", "unknown")
    geom_B    = row.get("geometry_type_object",  "unknown")

    return (
        f"### SYSTEM PROMPT ###\n{_SYSTEM}\n\n"
        f"### TASK ###\n"
        f"Given a vernacular spatial relation between two geographic entities A and B, "
        f"determine the correct DE-9IM topological predicate.\n\n"
        f"### VALID PREDICATES ###\n{_VALID_PREDS}\n\n"
        f"### RULES ###\n{_RULES_NO_KG}\n\n"
        f"### OUTPUT FORMAT ###\n"
        f"Reasoning: <brief justification>\n"
        f"Answer: A [<PREDICATE>] B\n\n"
        f"### INPUT ###\n"
        f"Vernacular relation: \"{subj} {vernacular} {obj}\"\n"
        f"Entity A: {subj_type} (geometry: {geom_A})\n"
        f"Entity B: {obj_type} (geometry: {geom_B})\n\n"
        f"### OUTPUT ###\n"
        f"Reasoning:"
    )


def build_prompt_with_kg(row: dict, kg_evidence: str, source: str) -> str:
    subj       = row.get("place_name_subject", "Entity A")
    obj        = row.get("place_name_object",  "Entity B")
    vernacular = row.get("vernacular_relation") or row.get("relation_predicate", "")
    subj_type  = row.get("placetype_subject", "place")
    obj_type   = row.get("placetype_object",  "place")
    geom_A     = row.get("geometry_type_subject", "unknown")
    geom_B     = row.get("geometry_type_object",  "unknown")

    if source == "osm":
        evidence_header = "### KG EVIDENCE (OpenStreetMap) ###"
        task_line = (
            "Given a vernacular spatial relation between two geographic entities A and B, "
            "use the OpenStreetMap knowledge graph evidence provided to determine the correct "
            "DE-9IM topological predicate."
        )
        rules = _RULES_OSM
    else:  # wikidata
        evidence_header = "### KG EVIDENCE (Wikidata) ###"
        task_line = (
            "Given a vernacular spatial relation between two geographic entities A and B, "
            "use the Wikidata knowledge graph evidence provided (semantic hierarchy, ontological "
            "type, and coordinates) to determine the correct DE-9IM topological predicate."
        )
        rules = _RULES_WIKIDATA

    return (
        f"### SYSTEM PROMPT ###\n{_SYSTEM}\n\n"
        f"### TASK ###\n{task_line}\n\n"
        f"### VALID PREDICATES ###\n{_VALID_PREDS}\n\n"
        f"### RULES ###\n{rules}\n\n"
        f"{evidence_header}\n{kg_evidence}\n\n"
        f"### INPUT ###\n"
        f"Vernacular relation: \"{subj} {vernacular} {obj}\"\n"
        f"Entity A: {subj_type} (geometry: {geom_A})\n"
        f"Entity B: {obj_type} (geometry: {geom_B})\n\n"
        f"### OUTPUT ###\n"
        f"Reasoning:"
    )


# ---------------------------------------------------------------------------
# Predicate extraction (robust multi-pattern)
# ---------------------------------------------------------------------------
def extract_predicate(text: str) -> str:
    if not text:
        return "invalid"

    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text_clean = re.sub(r"[*_`]", "", text_clean)
    text_lower = text_clean.lower()

    patterns = [
        r"answer\s*[:\s]+a\s*\[(\w+)\]",
        r"answer\s*[:=]\s*\[?(\w+)\]?",
        r"predicate\s*[:=]\s*\[?(\w+)\]?",
        r"the\s+(?:relation|predicate|answer)\s+is\s+[:\[]?(\w+)",
        r"final\s+(?:answer|predicate)\s*[:=]\s*\[?(\w+)\]?",
    ]
    for pat in patterns:
        for m in reversed(list(re.finditer(pat, text_lower))):
            pred = m.group(1).strip()
            if pred in VALID_PREDICATES:
                return pred

    # fallback: last occurrence of any valid predicate
    found = [(text_lower.rfind(p), p) for p in VALID_PREDICATES if p in text_lower]
    if found:
        return max(found, key=lambda x: x[0])[1]

    return "invalid"


# ---------------------------------------------------------------------------
# Native GPU inference
# ---------------------------------------------------------------------------
def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 150,
                  temperature: float = 0.1) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-OSS variants on topological reasoning")
    parser.add_argument("--dataset",        required=True,
                        help="Path to the 30%% test CSV (triplet_update_v3_30.csv)")
    parser.add_argument("--model-id",       default="openai/gpt-oss-20b")
    parser.add_argument("--adapter-path",   default=None,
                        help="Path to LoRA adapter directory (omit for base model)")
    parser.add_argument("--kg-source",      choices=["none", "osm", "wikidata"], default="none",
                        help="KG to fetch evidence from at inference time")
    parser.add_argument("--osm-cache",      default="results/osm_cache.json",
                        help="OSM Nominatim cache file (used when --kg-source osm)")
    parser.add_argument("--wikidata-cache", default="results/wikidata_cache.json",
                        help="Wikidata cache file (used when --kg-source wikidata)")
    parser.add_argument("--model-tag",      default="gptoss",
                        help="Short identifier embedded in the output filename")
    parser.add_argument("--output-dir",     default="results")
    parser.add_argument("--temperature",    type=float, default=0.1)
    parser.add_argument("--max-new-tokens", type=int,   default=150)
    parser.add_argument("--max-rows",        type=int,   default=None,
                        help="Limit rows (for quick smoke-testing)")
    parser.add_argument("--filter-indices",  default=None,
                        help="Path to a JSON file containing a list of row indices to evaluate "
                             "(e.g. the 96 balanced indices used in the prompting experiments). "
                             "Rows NOT in the list are skipped.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Optionally load the KG for evidence fetching at inference time
    # ------------------------------------------------------------------
    kg = None
    if args.kg_source == "osm":
        from strategies_osm import GeographicKnowledgeGraph as OsmKG
        kg = OsmKG(kg_path=args.osm_cache)
        print(f"[KG] OSM evidence fetcher loaded (cache: {args.osm_cache})")
    elif args.kg_source == "wikidata":
        from strategies_wikidata import GeographicKnowledgeGraph as WdKG
        kg = WdKG(kg_path=args.wikidata_cache)
        print(f"[KG] Wikidata evidence fetcher loaded (cache: {args.wikidata_cache})")

    # ------------------------------------------------------------------
    # 2. Load test dataset
    # ------------------------------------------------------------------
    import json as _json
    df = pd.read_csv(args.dataset)
    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"[DATA] Truncated to {args.max_rows} rows for testing.")
    if args.filter_indices:
        with open(args.filter_indices, "r") as _f:
            _keep = set(_json.load(_f))
        df = df.loc[df.index.isin(_keep)]
        print(f"[DATA] Filtered to {len(df)} rows using index list from {args.filter_indices}")
    print(f"[DATA] {len(df)} test rows loaded from {args.dataset}")

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    tok_path = args.model_id
    if args.adapter_path and os.path.isfile(os.path.join(args.adapter_path, "tokenizer_config.json")):
        tok_path = args.adapter_path
    print(f"[MODEL] Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[MODEL] Loading base model on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="auto",
        trust_remote_code=True,
        dtype=dtype,
    )

    if args.adapter_path:
        adapter_abs = os.path.abspath(args.adapter_path)
        print(f"[MODEL] Applying LoRA adapter from: {adapter_abs}")
        # Older PEFT + newer huggingface_hub rejects absolute paths as invalid repo IDs.
        # Patch validate_repo_id to pass through absolute local paths without validation.
        try:
            import huggingface_hub.utils._validators as _hfv
            _orig_validate = _hfv.validate_repo_id
            def _allow_local_abs(repo_id, *a, **kw):
                if os.path.isabs(repo_id):
                    return
                return _orig_validate(repo_id, *a, **kw)
            _hfv.validate_repo_id = _allow_local_abs
            model = PeftModel.from_pretrained(model, adapter_abs, local_files_only=True)
        finally:
            _hfv.validate_repo_id = _orig_validate

    model.eval()
    print(f"[MODEL] Ready. KG source: {args.kg_source}")

    # ------------------------------------------------------------------
    # 4. Determine output log file and load checkpoint
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"kg_eval_{args.model_tag}_{args.kg_source}.csv")

    processed_indices: set[int] = set()
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            next(f, None)  # skip header
            for line in f:
                parts = line.strip().split(",")
                if parts:
                    try:
                        processed_indices.add(int(parts[0]))
                    except ValueError:
                        pass
        print(f"[RESUME] {len(processed_indices)} rows already evaluated.")

    if not os.path.exists(log_file):
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("index,expected,predicted,match,description\n")

    # ------------------------------------------------------------------
    # 5. Evaluation loop
    # ------------------------------------------------------------------
    correct = total = 0

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval [{args.model_tag}]"):
        if index in processed_indices:
            # Count already-evaluated rows toward accuracy tracking
            continue

        expected = str(row.get("spatial_relation", "")).lower().strip()
        description = str(row.get("Sentence", "")).replace(",", " ")

        # Build prompt
        if args.kg_source == "none" or kg is None:
            prompt = build_prompt_no_kg(row.to_dict())
        else:
            try:
                kg_evidence = kg.gather_evidence(
                    str(row.get("place_name_subject", "")),
                    str(row.get("place_name_object",  "")),
                    sentence=str(row.get("Sentence", "")),
                    entity=row.to_dict(),
                )
            except Exception as exc:
                print(f"\n[WARN] KG fetch failed at row {index}: {exc}")
                kg_evidence = "No KG evidence available."
            prompt = build_prompt_with_kg(row.to_dict(), kg_evidence, args.kg_source)

        # Generate
        raw_output = run_inference(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        predicted = extract_predicate(raw_output)
        is_match  = (expected == predicted) and (predicted in VALID_PREDICATES)

        if is_match:
            correct += 1
        total += 1

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{index},{expected},{predicted},{is_match},{description}\n")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    # Re-read the full log for final accuracy (includes rows from prior runs)
    results_df = pd.read_csv(log_file)
    overall_acc = results_df["match"].astype(str).str.lower().eq("true").mean()

    print(f"\n{'='*60}")
    print(f"Model tag : {args.model_tag}")
    print(f"KG source : {args.kg_source}")
    print(f"Adapter   : {args.adapter_path or 'none (base model)'}")
    print(f"Overall accuracy: {overall_acc:.4f}  ({results_df['match'].astype(str).str.lower().eq('true').sum()}/{len(results_df)})")
    print(f"Log saved → {log_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
