import argparse
import json
import pandas as pd
import os
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
MAX_NEW_TOKENS = 400
TEMPERATURE = 0.1
# ──────────────────────────────────────────────────────────────────────────────

def load_model(base_model_path, adapter_path=None):
    # Load tokenizer from adapter if available, otherwise base model
    tok_path = adapter_path if adapter_path else base_model_path
    print(f"Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {base_model_path} on {DEVICE} with dtype {DTYPE} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )

    # Apply the LoRA adapter if provided
    if adapter_path:
        print(f"Applying LoRA adapter from: {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    print("Model loaded successfully.\n")
    return tokenizer, model


def query_model(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def get_few_shot_prefix(cities, current_idx, p_type):
    samples = cities.drop(current_idx).sample(3)
    prefix = "Here are some examples:\n\n"

    for _, row in samples.iterrows():
        if p_type == 'p1':
            prefix += f"Question: Distance between {row['a_name']} and {row['b_name']}? Answer: {row['distance']}\n"
        elif p_type == 'p2':
            a_coords = f"({row['a_lat']:.4f}, {row['a_lon']:.4f})"
            b_coords = f"({row['b_lat']:.4f}, {row['b_lon']:.4f})"
            prefix += f"Question: Distance between {row['a_name']} {a_coords} and {row['b_name']} {b_coords}? Answer: {row['distance']}\n"
        elif p_type == 'p3':
            prefix += f"Question: Distance between {row['a_name']} and {row['b_name']}? Answer: The distance is approximately {row['distance']} km. This is calculated using spatial geometry.\n"
        elif p_type == 'p4':
            a_coords = f"({row['a_lat']:.4f}, {row['a_lon']:.4f})"
            b_coords = f"({row['b_lat']:.4f}, {row['b_lon']:.4f})"
            prefix += f"Question: Distance between {row['a_name']} {a_coords} and {row['b_name']} {b_coords}? Answer: Using the Haversine formula on these coordinates, the distance is {row['distance']} km.\n"

    return prefix + "\nNow fulfill the following:\n"


def gen_dis(cities, p_type, shots, output_file, tokenizer, model):
    results = []
    processed_keys = set()

    # --- CHECKPOINT LOADING ---
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                results = data
                processed_keys = {(r['a_name'], r['b_name']) for r in results}
                print(f"--> Resuming {p_type}_{shots}: Found {len(processed_keys)} existing entries.")
        except Exception as e:
            print(f"--> Warning: Could not load existing file {output_file}. Starting fresh. Error: {e}")

    for i, each in tqdm(cities.iterrows(), total=cities.shape[0], desc=f"Processing {p_type} ({shots})"):
        if (each["a_name"], each["b_name"]) in processed_keys:
            continue

        if p_type in ('p2', 'p4'):
            a_c = f"({each['a_lat']:.4f}, {each['a_lon']:.4f})"
            b_c = f"({each['b_lat']:.4f}, {each['b_lon']:.4f})"

        # 1. Base Prompt Construction
        if p_type == 'p1':
            core_prompt = f"Question: What is the distance in kilometers between {each['a_name']} and {each['b_name']}? Answer exactly with the number only.\nAnswer:"
        elif p_type == 'p2':
            core_prompt = f"Question: What is the distance in kilometers between {each['a_name']} {a_c} and {each['b_name']} {b_c}? Answer exactly with the number only.\nAnswer:"
        elif p_type == 'p3':
            core_prompt = f"Question: What is the distance in kilometers between {each['a_name']} and {each['b_name']}? Explain your reasoning and how the distance is found.\nAnswer:"
        elif p_type == 'p4':
            core_prompt = f"Question: What is the distance in kilometers between {each['a_name']} {a_c} and {each['b_name']} {b_c}? Explain your reasoning and how the distance is found using these coordinates.\nAnswer:"

        # 2. Few-Shot Logic
        final_prompt = core_prompt
        if shots == '3-shot':
            prefix = get_few_shot_prefix(cities, i, p_type)
            final_prompt = prefix + core_prompt

        generated_text = query_model(final_prompt, tokenizer, model)

        # 3. Store and Atomic Save
        res = each.to_dict()
        res.update({'output': generated_text, 'prompt': final_prompt, 'p_type': p_type, 'shots': shots})
        results.append(res)

        temp_file = output_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(results, f, indent=4)
        shutil.move(temp_file, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_type', choices=['p1', 'p2', 'p3', 'p4'], required=True)
    parser.add_argument('--shots', choices=['zero-shot', '3-shot'], required=True)
    parser.add_argument('--dataset', default='cities_with_coords.json', help='Dataset to use')
    parser.add_argument('--model-id', default="openai/gpt-oss-20b", help='HuggingFace ID of the base model')
    parser.add_argument('--adapter-path', default=None, help='Path to the trained LoRA adapter')
    parser.add_argument('--subset-percent', type=float, default=1.0, help='Fraction of rows to sample')
    parser.add_argument('--input-ids', default=None, help='Path to JSON file of specific a_name/b_name pairs')
    parser.add_argument('--subset-ids-output', default=None, help='Path to save sampled IDs')
    parser.add_argument('--output-tag', default=None, help='Suffix tag to append to output filename')
    parser.add_argument('--max-rows', type=int, default=1000, help='Cap rows for bounded runs')
    args = parser.parse_args()

    # Load model once
    tokenizer, model = load_model(args.model_id, args.adapter_path)

    # Read data
    cities_df = pd.read_json(args.dataset)

    # Filtering Logic
    if args.input_ids and os.path.exists(args.input_ids):
        print(f"Loading exact target IDs from {args.input_ids}...")
        with open(args.input_ids, 'r') as f:
            target_ids = json.load(f)
        
        # Faster inner merge instead of .apply() lambda
        target_df = pd.DataFrame(target_ids)[['a_name', 'b_name']]
        cities_subset = pd.merge(cities_df, target_df, on=['a_name', 'b_name'], how='inner')
        
    else:
        # Proper order: Sample first, then truncate
        if args.subset_percent and 0.0 < args.subset_percent < 1.0:
            cities_df = cities_df.sample(frac=args.subset_percent, random_state=42).copy()
        if args.max_rows:
            cities_df = cities_df.head(args.max_rows)
        cities_subset = cities_df

    if args.subset_ids_output:
        try:
            ids = [{"a_name": r["a_name"], "b_name": r["b_name"]} for _, r in cities_subset.iterrows()]
        except KeyError:
            ids = [{"index": int(idx)} for idx in cities_subset.index]
        with open(args.subset_ids_output, 'w', encoding='utf-8') as f:
            json.dump(ids, f, indent=2)

    print(f"Experiment rows: {len(cities_subset)} selected")

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # Add a finetuned tag to the output file so you don't overwrite your base model results!
    suffix = f'_{args.output_tag}' if args.output_tag else ''
    adapter_tag = "_finetuned" if args.adapter_path else ""
    out_path = f'outputs/gen_dis_gptoss{adapter_tag}_{args.p_type}_{args.shots}{suffix}.json'

    gen_dis(cities_subset, args.p_type, args.shots, out_path, tokenizer, model)