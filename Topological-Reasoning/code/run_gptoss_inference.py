"""Run generation using the finetuned GPT-OSS adapter.

Usage example:
  python run_gptoss_inference.py \
    --base-model unsloth/gpt-oss-20b-bnb-4bit \
    --adapter-dir ../Topological_Reasoning_GPTOSS_Standard/final_adapter \
    --data ../dataset/triplet_update_v3_70.csv \
    --output ../outputs/gptoss_preds.jsonl

The script loads the base model, applies the PEFT adapter, and generates answers
for the dataset in batches. It includes guards for missing packages and prints
helpful error messages when compatibility issues arise.
"""

import os
import json
import argparse
import math
import pandas as pd
import torch
from tqdm.auto import tqdm
import shutil
import re

# Optional imports with friendly errors
peft_err = None
peft_PeftModel = None
try:
    from peft import PeftModel
    peft_PeftModel = PeftModel
except Exception as e:
    peft_err = e

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as e:
    raise RuntimeError("transformers import failed: " + str(e))


def build_prompt(sentence, subj, obj, relation=None):
    sys_msg = "You are an expert in topological spatial reasoning."
    prompt = (
        f"<|start|>system\n{sys_msg}<|end|>\n"
        f"<|start|>user\nSentence: {sentence}\nSubject: {subj}\nObject: {obj}<|end|>\n"
        f"<|start|>assistant\n"
    )
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="Base model id (HuggingFace)")
    parser.add_argument("--adapter-dir", required=True, help="Path to the finetuned adapter directory")
    parser.add_argument("--data", required=True, help="CSV file with columns: Sentence, place_name_subject, place_name_object")
    parser.add_argument("--output", required=False, default=None, help="JSONL output file for predictions (defaults to ./results/gptoss_preds.jsonl)")
    parser.add_argument("--system-prompt", required=False, default=None, help="Optional system prompt to prepend to each generated example")
    parser.add_argument("--system-prompt-file", required=False, default=None, help="Path to a file containing the system prompt")
    parser.add_argument("--user-prompt-template", required=False, default=None, help="Optional user prompt template using row fields, e.g. 'given the sentence: A {relation_predicate} B and A is {place_type_subject}, B is {place_type_object}.')")
    parser.add_argument("--user-prompt-file", required=False, default=None, help="Path to a file containing the user prompt template")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mode", choices=["zero","few"], default="zero", help="zero-shot or few-shot prompting mode")
    parser.add_argument("--n-shots", type=int, default=4, help="number of few-shot exemplars to prepend when --mode=few")
    parser.add_argument("--shot-seed", type=int, default=42, help="random seed for selecting few-shot exemplars")
    parser.add_argument("--shot-file", default=None, help="optional CSV file to pull few-shot exemplars from (defaults to --data)")
    parser.add_argument("--device", default=None, help="torch device (auto-detect if omitted)")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="save output checkpoint every N batches (0 = disabled)")
    parser.add_argument("--checkpoint-dir", default=None, help="directory to write checkpoints (defaults to <output>.checkpoints)")
    parser.add_argument("--llm-offload", action="store_true", help="enable llm int8 fp32 cpu offload for large quantized models")
    parser.add_argument("--evaluate", action="store_true", help="check predictions against gold 'spatial_relation' and include 'is_correct' in output; prints running accuracy")
    args = parser.parse_args()

    if peft_PeftModel is None:
        raise RuntimeError(
            "PEFT is required to load the finetuned adapter but failed to import: "
            + str(peft_err)
            + "\nTry: pip install peft and use transformers<5.0.0 if needed."
        )

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model (CPU or auto device map to reduce memory spikes)
    # For large quantized models you can enable fp32 CPU offload via --llm-offload
    print("Loading base model (may be slow)...")
    load_kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device.startswith("cuda"):
        load_kwargs.update({
            "device_map": "auto",
            "torch_dtype": torch.float16,
        })
        if args.llm_offload:
            load_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
    else:
        load_kwargs.update({"device_map": {"": "cpu"}})

    try:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    except TypeError as e:
        # Some custom model classes don't accept newer kwargs like
        # llm_int8_enable_fp32_cpu_offload; fall back to CPU load then move to GPU.
        msg = str(e)
        if "llm_int8_enable_fp32_cpu_offload" in msg or "unexpected keyword" in msg:
            print("Falling back to CPU load then moving model to CUDA to accommodate model class limitations.")
            load_kwargs.pop("llm_int8_enable_fp32_cpu_offload", None)
            load_kwargs["device_map"] = {"": "cpu"}
            load_kwargs.pop("torch_dtype", None)
            model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
            if device.startswith("cuda"):
                print("Moving model to GPU (this may take some time and memory)...")
                model = model.to("cuda:0")
        else:
            raise

    print("Applying PEFT adapter from:", args.adapter_dir)
    peft_map = "auto" if device.startswith("cuda") else {"": "cpu"}
    model = PeftModel.from_pretrained(model, args.adapter_dir, device_map=peft_map)
    model.eval()

    # Read data
    df = pd.read_csv(args.data)
    required_cols = ["Sentence", "place_name_subject", "place_name_object"]
    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column in CSV: {c}")

    # If system/user prompt files are provided, load them
    if args.system_prompt is None and args.system_prompt_file:
        try:
            with open(args.system_prompt_file, "r", encoding="utf-8") as f:
                args.system_prompt = f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read system prompt file: {e}")
    if args.user_prompt_template is None and args.user_prompt_file:
        try:
            with open(args.user_prompt_file, "r", encoding="utf-8") as f:
                args.user_prompt_template = f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read user prompt file: {e}")

    # Default system prompt (use the user's provided instruction if not supplied)
    if not args.system_prompt:
        args.system_prompt = (
            "The given seven PREDICATES: contains, is within, touches, crosses, disjoint, overlaps, equals.\n"
            "Given a sentence that include a vernacular spatial relation term: please give the corresponding PREDICATES.\n"
            "- Indicate spatial relation from A to B only.\n"
            "- The vernacular spatial relation term should consider its typical meanings of the terms and the general geographical relationship it represent.\n"
            "- MAKE SURE the [PREDICATE] satisfies the dimensions defined by DE-9IM.\n"
            "- MAKE SURE that the output includes all plausible predicates.\n"
            "OUTPUT FORMAT:\n    Analysis: Provide a detailed examination of the spatial relation.\n    Answer: A [PREDICATE1/PREDICATE2...] B."
        )

    # If output not provided, default to ./results/gptoss_preds.jsonl within code dir
    if args.output is None:
        default_results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(default_results_dir, exist_ok=True)
        args.output = os.path.join(default_results_dir, "gptoss_preds.jsonl")

    # Helper to format a user prompt from a row and optional template
    def format_user_prompt_from_row(row):
        # Build a mapping of common keys to support different column names
        mapping = {
            "relation_predicate": row.get("relation_predicate") if "relation_predicate" in row.index else row.get("vernacular_relation", ""),
            "place_type_subject": row.get("placetype_subject") if "placetype_subject" in row.index else row.get("place_type_subject", ""),
            "place_type_object": row.get("placetype_object") if "placetype_object" in row.index else row.get("place_type_object", ""),
            "place_name_subject": row.get("place_name_subject") if "place_name_subject" in row.index else row.get("place_name_subject", ""),
            "place_name_object": row.get("place_name_object") if "place_name_object" in row.index else row.get("place_name_object", ""),
            "Sentence": row.get("Sentence") if "Sentence" in row.index else row.get("sentence", ""),
            "sentence": row.get("Sentence") if "Sentence" in row.index else row.get("sentence", ""),
        }

        if args.user_prompt_template:
            try:
                return args.user_prompt_template.format(**mapping)
            except Exception:
                # fallback to a simple constructed prompt
                return f"given the sentence: A {mapping.get('relation_predicate','')} B and A is {mapping.get('place_type_subject','')}, B is {mapping.get('place_type_object','')}."
        else:
            # default user prompt
            return f"Sentence: {mapping.get('Sentence','')}\nSubject: {mapping.get('place_name_subject','')}\nObject: {mapping.get('place_name_object','')}"

    # Prepare few-shot source if requested (we will sample per-row to avoid leakage)
    df_shots = None
    if args.mode == "few":
        src = args.shot_file or args.data
        df_shots = pd.read_csv(src)
        if "spatial_relation" not in df_shots.columns:
            raise RuntimeError("Few-shot source must contain 'spatial_relation' column for exemplar answers")

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    # checkpoint dir
    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
    else:
        ckpt_dir = args.output + ".checkpoints"
    if args.checkpoint_interval and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    # Run batched generation
    bs = max(1, args.batch_size)
    total = len(df)
    steps = math.ceil(total / bs)

    # canonicalization map for vernacular synonyms -> canonical predicate
    syn_map = {
        "contains": ["contains", "is home to", "contains/has", "has", "includes", "include", "home to"],
        "within": ["is within", "within", "inside", "in"],
        "touches": ["touches", "adjacent", "borders", "adjacent to", "bordered"],
        "crosses": ["crosses", "straddles", "spans", "crossed"],
        "disjoint": ["disjoint", "between", "separate", "separated"],
        "overlaps": ["overlaps", "overlap", "extends into", "partly"]
    }

    def canonicalize(gen_text):
        if gen_text is None:
            return None
        # prefer bracketed predicates and allow multiple mapped predicates
        m = re.search(r"\[([^\]]+)\]", gen_text)
        mapped = []
        if m:
            preds = [p.strip().lower() for p in m.group(1).split("/") if p.strip()]
            for p in preds:
                matched = None
                for canon, syns in syn_map.items():
                    if p == canon or p in syns:
                        matched = canon
                        break
                if matched:
                    mapped.append(matched)
                else:
                    mapped.append(p)
            # return joined canonical preds if multiple
            return "/".join(mapped) if mapped else None

        # fallback: search for synonyms in text (first match wins)
        txt = str(gen_text).lower()
        for canon, syns in syn_map.items():
            for s in syns:
                if s in txt:
                    return canon
        # fallback: try to find any canonical key in text
        for canon in syn_map.keys():
            if canon in txt:
                return canon
        return None

    with open(args.output, "w", encoding="utf-8") as out_f:
        pbar = tqdm(total=steps, desc="Batches")
        # running evaluation counters (if --evaluate)
        processed_scored = 0
        processed_correct = 0
        for i in range(steps):
            batch = df.iloc[i * bs : (i + 1) * bs]
            base_prompts = []
            for _, r in batch.iterrows():
                if args.user_prompt_template or args.system_prompt:
                    user_text = format_user_prompt_from_row(r)
                    if args.system_prompt:
                        full = args.system_prompt + "\n\n" + user_text + "\n\n"
                    else:
                        full = user_text + "\n\n"
                else:
                    full = build_prompt(r.Sentence, r.place_name_subject, r.place_name_object)
                base_prompts.append(full)
            prompts = []
            prompts = []
            for idx_row, bp in enumerate(base_prompts):
                if args.mode == "few":
                    # sample exemplars excluding the test row to avoid leakage
                    try:
                        row = batch.iloc[idx_row]
                        candidates = df_shots[~(
                            (df_shots.get("Sentence") == row.get("Sentence")) &
                            (df_shots.get("place_name_subject") == row.get("place_name_subject")) &
                            (df_shots.get("place_name_object") == row.get("place_name_object"))
                        )]
                    except Exception:
                        candidates = df_shots

                    n_samples = min(args.n_shots, len(candidates))
                    if n_samples <= 0:
                        few_shot_section = ""
                    else:
                        sampled = candidates.sample(n=n_samples, random_state=args.shot_seed)
                        few_shot_section = "\n\n--- EXAMPLES ---\n"
                        for _, ex in sampled.iterrows():
                            ex_text = ex.get("Sentence", "")
                            ex_subj_type = ex.get("placetype_subject", ex.get("place_type_subject", "place"))
                            ex_obj_type = ex.get("placetype_object", ex.get("place_type_object", "place"))
                            ex_truth = ex.get("spatial_relation", "disjoint")
                            few_shot_section += f"\nUser query: given the sentence: A {ex_text} B and A is {ex_subj_type}, B is {ex_obj_type}.\n"
                            few_shot_section += f"Analysis: Based on the spatial context of '{ex_text}' between a {ex_subj_type} and a {ex_obj_type}, the standard topological relation matches.\n"
                            few_shot_section += f"Answer: A [{ex_truth}] B\n"
                        few_shot_section += "--- END EXAMPLES ---\n\n"

                    if args.system_prompt:
                        full = args.system_prompt + "\n\n" + few_shot_section + bp
                    else:
                        full = few_shot_section + bp
                else:
                    full = bp
                prompts.append(full)

            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            input_ids = enc.input_ids.to(model.device)
            attention_mask = enc.attention_mask.to(model.device)

            # deterministic vs sampled generation
            do_sample = bool(args.temperature and args.temperature > 0.0)
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=do_sample,
            )
            if do_sample:
                gen_kwargs.update({"temperature": args.temperature, "top_p": 0.95})

            with torch.no_grad():
                outputs = model.generate(**gen_kwargs)

            # Strip prompt from generated text
            for j, out_ids in enumerate(outputs):
                full_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                prompt_text = prompts[j]
                if full_text.startswith(prompt_text):
                    gen_text = full_text[len(prompt_text) :].strip()
                else:
                    # fallback: return full_text
                    gen_text = full_text

                orig_row = batch.iloc[j].to_dict()
                # canonicalize prediction
                pred_canon = canonicalize(gen_text)

                # prepare output row
                out_row = dict(orig_row)
                out_row.update({"generated": gen_text, "predicate": pred_canon})

                # optional evaluation against gold
                if args.evaluate:
                    gold_raw = orig_row.get("spatial_relation") or orig_row.get("relation_predicate") or orig_row.get("vernacular_relation")
                    gold_canon = canonicalize(gold_raw) if gold_raw is not None else None
                    is_correct = None
                    if gold_canon is not None and pred_canon is not None:
                        # handle multiple predicted predicates joined by '/'
                        pred_set = [p.strip() for p in str(pred_canon).split("/") if p.strip()]
                        is_correct = gold_canon in pred_set
                        processed_scored += 1
                        if is_correct:
                            processed_correct += 1
                    out_row.update({"is_correct": is_correct})

                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            out_f.flush()
            try:
                os.fsync(out_f.fileno())
            except Exception:
                pass

            # print running accuracy to log (do not save)
            if args.evaluate:
                if processed_scored > 0:
                    running_acc = processed_correct / processed_scored
                    print(f"Running accuracy (scored {processed_scored}): {running_acc:.4f}")
                else:
                    print("Running accuracy: no gold+pred pairs scored yet")

            # checkpoint copy
            if args.checkpoint_interval and (i + 1) % args.checkpoint_interval == 0:
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_batch_{i+1}.jsonl")
                try:
                    shutil.copyfile(args.output, ckpt_path)
                except Exception as e:
                    print("Warning: failed to write checkpoint:", e)

            pbar.update(1)
        pbar.close()

    print("Done. Predictions saved to", args.output)


if __name__ == "__main__":
    main()
