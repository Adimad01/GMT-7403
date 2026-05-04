import argparse
import json
import pandas as pd
import os
import shutil
import torch
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# --- CUSTOM UNPICKLER FOR NUMPY 2.0 -> 1.x ---
class Numpy2to1Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_pickle(path):
    """Load a pickle file using the numpy-compat unpickler."""
    with open(path, 'rb') as f:
        return Numpy2to1Unpickler(f).load()
# ---------------------------------------------


# Configuration
BASE_MODEL_ID  = "Qwen/Qwen2.5-14B-Instruct"
ADAPTER_PATH   = "finetuned_qwen14b/final_adapter"   # LoRA adapter saved by qwan_finetune.py


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


def gen_dis(cities, p_type, shots, output_file):
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

    # --- LOAD BASE MODEL + LORA ADAPTER ---
    print(f"\nLoading tokenizer from adapter: {ADAPTER_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base model {BASE_MODEL_ID} (4-bit) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    print(f"Attaching LoRA adapter from: {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("✅ Fine-tuned model loaded successfully!\n")

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

        # 3. Model Generation
        messages = [
            {"role": "system", "content": "You are a helpful geography expert calculating distances between cities."},
            {"role": "user",   "content": final_prompt}
        ]

        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=400,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids  = outputs[0][input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # 4. Store and Atomic Save
        res = each.to_dict()
        res.update({'output': generated_text, 'prompt': final_prompt, 'p_type': p_type, 'shots': shots})
        results.append(res)

        temp_file = output_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(results, f, indent=4)
        shutil.move(temp_file, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_type',            choices=['p1', 'p2', 'p3', 'p4'], required=True)
    parser.add_argument('--shots',             choices=['zero-shot', '3-shot'],   required=True)
    parser.add_argument('--dataset',           default='cities_with_coords.pkl')
    parser.add_argument('--adapter-path',      default=ADAPTER_PATH,
                        help='Path to the fine-tuned LoRA adapter directory')
    parser.add_argument('--subset-percent',    type=float, default=1.0)
    parser.add_argument('--subset-ids-output', default=None)
    parser.add_argument('--output-tag',        default=None)
    parser.add_argument('--max-rows',          type=int, default=1000)
    args = parser.parse_args()

    # Override adapter path if provided via CLI
    ADAPTER_PATH = args.adapter_path

    print(f"--> Loading dataset: {args.dataset}")
    cities_df = load_pickle(args.dataset)
    print(f"--> Dataset loaded: {len(cities_df)} rows")

    if args.max_rows:
        cities_df = cities_df.head(args.max_rows)

    if args.subset_percent and 0.0 < args.subset_percent < 1.0:
        cities_df = cities_df.sample(frac=args.subset_percent, random_state=42).copy()

    cities_subset = cities_df

    if args.subset_ids_output:
        try:
            ids = [{"a_name": r["a_name"], "b_name": r["b_name"]} for _, r in cities_subset.iterrows()]
        except KeyError:
            ids = [{"index": int(idx)} for idx in cities_subset.index]
        with open(args.subset_ids_output, 'w', encoding='utf-8') as f:
            json.dump(ids, f, indent=2)

    print(f"Experiment rows: {len(cities_subset)} selected (dataset={args.dataset}, subset={args.subset_percent})")

    os.makedirs('outputs', exist_ok=True)

    suffix   = f'_{args.output_tag}' if args.output_tag else ''
    out_path = f'outputs/gen_dis_qwen14b_finetuned_{args.p_type}_{args.shots}{suffix}.json'

    gen_dis(cities_subset, args.p_type, args.shots, out_path)