#!/usr/bin/env python3
import json
import os
import random
import uuid
from datetime import datetime
from typing import List, Dict
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= CONFIGURATION =================
BASE_MODEL  = "openai/gpt-oss-20b"
ADAPTER_DIR = "Directional_Reasoning_GPT_OSS_20B_LoRA/final_adapter"

INPUT_QUESTIONS = "./data/questions.jsonl"
INPUT_ANSWERS   = "./data/answers.jsonl"
OUTPUT_FILE     = "evaluation_results/experiment_results_gptoss20b_finetuned.jsonl"
TEMPERATURES    = [0.25, 0.5, 1.0]
FEW_SHOT_K      = 3
# FIX: file lives in current dir, not inside evaluation_results/
ANSWERS_EVAL_SUBSET = "evaluation_results/answers_eval_subset.jsonl"
USE_EVAL_SUBSET = True

SYSTEM_PROMPT = (
    "You are a helpful assistant. I will give you a question about directions. "
    "The answer is either north, south, east, west, north-east, north-west, "
    "south-east or south-west. Only reply with the answer, nothing else.\n"
    "Reasoning: low"
)
# =================================================


def load_jsonl(path: str) -> List[Dict]:
    """Reads a JSONL file and returns a list of parsed record dicts."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out


def load_dataset(q_path: str, a_path: str) -> List[Dict]:
    """Merges question and answer JSONL files into a unified list of dicts keyed by id."""
    questions = load_jsonl(q_path)
    answers   = {str(item["id"]): item["absoluteAnswer"] for item in load_jsonl(a_path)}
    merged    = []
    for q in questions:
        qid = str(q.get("id"))
        merged.append({"id": qid, "question": q.get("question"), "answer": answers.get(qid, "")})
    return merged


def load_eval_ids_from_answers(eval_subset_path: str) -> List[str]:
    """Returns a list of question IDs from the evaluation subset JSONL file."""
    if not os.path.exists(eval_subset_path):
        return []
    try:
        ids = []
        with open(eval_subset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                if "id" in obj:
                    ids.append(str(obj["id"]))
        return ids
    except Exception:
        return []


def get_completed_tasks(filename: str):
    """Returns a set of (temperature, question_id) pairs already written to the output file."""
    completed = set()
    if not os.path.exists(filename):
        return completed
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    temp = float(data["request"]["temperature"])
                    q_id = str(data["request"]["id"])
                    completed.add((temp, q_id))
                except Exception:
                    continue
    except Exception:
        pass
    return completed


def save_result(result_obj, filename):
    """Appends a result object as a JSON line to the output file."""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_obj) + "\n")


def sample_few_shot(data: List[Dict], current_id: str, k: int) -> List[Dict]:
    """Randomly samples k few-shot examples from data, excluding the current question."""
    pool = [item for item in data if item["id"] != current_id and item.get("answer")]
    if not pool:
        return []
    return random.sample(pool, min(k, len(pool)))


# FIX: replaced build_harmony_prompt with build_messages + apply_chat_template.
# GPT-OSS forbids <|start|>/<|channel|> tags in the `content` field of input
# messages — those tokens are reserved for the model's own generated output.
# apply_chat_template injects all Harmony markup automatically and safely.
def build_messages(question_text: str, shots: List[Dict]) -> List[Dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for shot in shots:
        messages.append({"role": "user",      "content": shot["question"]})
        messages.append({"role": "assistant",  "content": shot["answer"]})  # plain answer only

    messages.append({"role": "user", "content": question_text})
    return messages


def run_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file",    default=None)
    parser.add_argument("--few-shot-k",     type=int, default=FEW_SHOT_K)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()

    if args.output_file is None:
        directory = os.path.dirname(OUTPUT_FILE)
        base_name = os.path.basename(OUTPUT_FILE).replace(".jsonl", "")
        args.output_file = os.path.join(directory, f"{base_name}_{args.few_shot_k}shot.jsonl")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    print(f"Results will be saved to: {args.output_file}")

    # 1. Load data
    data = load_dataset(INPUT_QUESTIONS, INPUT_ANSWERS)
    if not data:
        print("No data found.")
        return

    eval_ids = load_eval_ids_from_answers(ANSWERS_EVAL_SUBSET) if USE_EVAL_SUBSET else []
    if USE_EVAL_SUBSET and eval_ids:
        before = len(data)
        data   = [d for d in data if d["id"] in eval_ids]
        print(f"Using eval subset: {len(data)} items (from {before})")

    completed = get_completed_tasks(args.output_file)
    print(f"Loaded {len(data)} items. Completed entries: {len(completed)}")

    # 2. Load base model + LoRA adapter
    print(f"\nLoading tokenizer and Base Model ({BASE_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    print(f"Loading Fine-Tuned Adapter from: {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    # 3. Inference loop
    for temp in TEMPERATURES:
        print("\n" + "=" * 60)
        print(f"Starting temperature {temp}")
        print("=" * 60)

        if all((float(temp), item["id"]) in completed for item in data):
            print("All questions already processed for this temperature. Skipping.")
            continue

        for item in tqdm(data, desc=f"Temp {temp}"):
            qid = item["id"]
            if (float(temp), qid) in completed:
                continue

            question_text = item.get("question", "")

            shots    = sample_few_shot(data, qid, args.few_shot_k) if args.few_shot_k > 0 else []
            messages = build_messages(question_text, shots)

            # FIX: apply_chat_template instead of tokenizer(raw_string)
            # add_generation_prompt=True appends the opening assistant turn automatically
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            input_len = inputs["input_ids"].shape[1]

            do_sample  = temp > 0.0
            gen_kwargs = {
                **inputs,
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id":   tokenizer.pad_token_id,
                "eos_token_id":   tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs.update({"do_sample": True, "temperature": temp, "top_p": 0.95})
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                outputs = model.generate(**gen_kwargs)

            generated_ids = outputs[0][input_len:]
            answer_text   = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            output_tokens = len(generated_ids)

            # 4. Build log entry
            msg_uuid  = f"msg_{uuid.uuid4().hex[:24]}"
            timestamp = datetime.now().isoformat()

            log_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for s in shots:
                log_messages.append({"role": "user",      "content": s["question"]})
                log_messages.append({"role": "assistant",  "content": s["answer"]})
            log_messages.append({"role": "user", "content": question_text})

            log_entry = {
                "request": {
                    "id":           qid,
                    "datetime":     timestamp,
                    "messages":     log_messages,
                    "temperature":  temp,
                    "max_tokens":   args.max_new_tokens,
                    "few_shot_ids": [s["id"] for s in shots],
                },
                "response": {
                    "id":            msg_uuid,
                    "content":       [{"text": answer_text, "type": "text"}],
                    "model":         "gpt-oss-20b-finetuned",
                    "role":          "assistant",
                    "stop_reason":   "end_turn",
                    "stop_sequence": None,
                    "type":          "message",
                    "usage": {
                        "input_tokens":  input_len,
                        "output_tokens": output_tokens,
                    },
                },
            }

            save_result(log_entry, args.output_file)
            completed.add((float(temp), qid))

    print(f"\nDone! Results saved to {args.output_file}")


if __name__ == "__main__":
    run_experiments()