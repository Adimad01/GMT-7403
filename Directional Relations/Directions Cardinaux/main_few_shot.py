import json
import os
import random
import traceback
import uuid
import time
from datetime import datetime
from typing import List, Dict, Tuple
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# ================= CONFIGURATION =================
MODEL = "gpt-oss"
BASE_URL = "http://ollama.apps.crdig.ulaval.ca"
INPUT_QUESTIONS = "./data/questions.jsonl"
INPUT_ANSWERS = "./data/answers.jsonl"
OUTPUT_FILE = "experiment_results_gptoss_fewshot.jsonl"
TEMPERATURES = [0.25, 0.5, 1]
FEW_SHOT_K = 3  
ANSWERS_EVAL_SUBSET = "./evaluation_results/answers_eval_subset.jsonl"  # produced by evaluate_model.py
USE_EVAL_SUBSET = True  # when True, restrict runs to eval IDs from answers_eval_subset

SYSTEM_PROMPT = (
    "You are a helpful assistant. I will give you a question about directions. "
    "The answer is either north, south, east, west, north-east, north-west, "
    "south-east or south-west. Only reply with the answer, nothing else."
)
# =================================================


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_dataset(q_path: str, a_path: str) -> List[Dict]:
    questions = load_jsonl(q_path)
    answers = {str(item["id"]): item["absoluteAnswer"] for item in load_jsonl(a_path)}
    merged = []
    for q in questions:
        qid = str(q.get("id"))
        merged.append({"id": qid, "question": q.get("question"), "answer": answers.get(qid, "")})
    return merged


def load_eval_ids_from_answers(eval_subset_path: str) -> List[str]:
    """Read IDs from answers_eval_subset.jsonl."""
    if not os.path.exists(eval_subset_path):
        return []
    try:
        ids = []
        with open(eval_subset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "id" in obj:
                    ids.append(str(obj["id"]))
        return ids
    except Exception:
        return []


def get_completed_tasks(filename: str):
    completed = set()
    if not os.path.exists(filename):
        return completed
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
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
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_obj) + "\n")


def _invoke_with_timeout(llm, messages, timeout_sec: float):
    """Invoke the LLM with a timeout to avoid hanging requests."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(llm.invoke, messages)
        try:
            return fut.result(timeout=timeout_sec)
        except FuturesTimeout:
            return None


def sample_few_shot(data: List[Dict], current_id: str, k: int) -> List[Tuple[str, str]]:
    pool = [item for item in data if item["id"] != current_id and item.get("answer")]
    if not pool:
        return []
    k = min(k, len(pool))
    return random.sample(pool, k)


def build_messages(question_text: str, shots: List[Tuple[str, str]]) -> List:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if shots:
        messages.append(HumanMessage(content="Here are a few shots (question then answer):"))
    for shot in shots:
        messages.append(HumanMessage(content=shot["question"]))
        messages.append(AIMessage(content=shot["answer"]))
    messages.append(HumanMessage(content=question_text))
    return messages


def run_experiments(
    input_questions: str = INPUT_QUESTIONS,
    input_answers: str = INPUT_ANSWERS,
    output_file: str = OUTPUT_FILE,
    temperatures: List[float] = TEMPERATURES,
    few_shot_k: int = FEW_SHOT_K,
    eval_subset_path: str = ANSWERS_EVAL_SUBSET,
    use_eval_subset: bool = USE_EVAL_SUBSET,
    request_timeout: float = 60.0,
    model: str = MODEL,
    base_url: str = BASE_URL,
    max_questions: int = 0,
    retries: int = 2,
    per_question_retries: int = 1,
    sleep_on_timeout: float = 5.0,
):
    data = load_dataset(input_questions, input_answers)
    if not data:
        print("No data found.")
        return

    # Restrict to eval subset if configured
    eval_ids = load_eval_ids_from_answers(eval_subset_path) if use_eval_subset else []
    if use_eval_subset and eval_ids:
        before = len(data)
        data = [d for d in data if d["id"] in eval_ids]
        print(f"Using eval subset: {len(data)} items (from {before}) based on {eval_subset_path}")
    elif use_eval_subset:
        print(f"Eval subset requested but {eval_subset_path} not found or empty. Using full dataset.")

    # Optionally limit number of questions for quick tests
    if max_questions and max_questions > 0:
        data = data[:max_questions]

    completed = get_completed_tasks(output_file)
    print(f"Loaded {len(data)} items. Completed entries: {len(completed)}")
    print(f"Temperatures: {temperatures}, few-shot k={few_shot_k}")
    print(f"Model: {model} @ {base_url}")

    for temp in temperatures:
        print("\n" + "=" * 60)
        print(f"Starting temperature {temp}")
        print("=" * 60)

        # check if all done
        if all((float(temp), item["id"]) in completed for item in data):
            print("All questions already processed for this temperature. Skipping.")
            continue

        try:
            llm = ChatOllama(model=model, temperature=temp, base_url=base_url)
            print("   Testing connection...")
            ok = False
            for attempt in range(1, max(1, retries) + 1):
                test_resp = _invoke_with_timeout(
                    llm,
                    [HumanMessage(content="Reply with exactly: TEST OK")],
                    timeout_sec=min(15.0, request_timeout),
                )
                if test_resp is not None:
                    ok = True
                    break
                print(f"   Attempt {attempt} failed; sleeping {sleep_on_timeout}s then retry...")
                time.sleep(sleep_on_timeout)
            if not ok:
                print("   ❌ Connection timed out or failed. Skipping this temperature.")
                continue
            print("   ✅ Connection OK.")
        except Exception as e:
            print(f"   ❌ Failed to init LLM at temp {temp}: {e}")
            continue

        for item in data:
            qid = item["id"]
            if (float(temp), qid) in completed:
                continue

            question_text = item.get("question", "")
            # sample few-shots excluding this instance
            shots = sample_few_shot(data, qid, few_shot_k)
            messages = build_messages(question_text, shots)

            msg_uuid = f"msg_{uuid.uuid4().hex[:24]}"
            timestamp = datetime.now().isoformat()

            try:
                response = None
                for attempt in range(1, max(1, per_question_retries) + 2):
                    response = _invoke_with_timeout(llm, messages, timeout_sec=request_timeout)
                    if response is not None:
                        break
                    print(f"   QID {qid}: attempt {attempt} timed out; sleeping {sleep_on_timeout}s then retry...")
                    time.sleep(sleep_on_timeout)
                if response is None:
                    print(f"   [Timeout -> Skipped QID {qid}]")
                    continue
                answer_text = response.content.strip()

                log_entry = {
                    "request": {
                        "id": qid,
                        "datetime": timestamp,
                        "messages": [
                            {"role": m.type if hasattr(m, "type") else m.__class__.__name__.replace("Message", "").lower(), "content": m.content}
                            for m in messages
                        ],
                        "temperature": temp,
                        "max_tokens": 1024,
                        "few_shot_ids": [s["id"] for s in shots],
                    },
                    "response": {
                        "id": msg_uuid,
                        "content": [{"text": answer_text, "type": "text"}],
                        "model": MODEL,
                        "role": "assistant",
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "type": "message",
                        "usage": {
                            "input_tokens": response.response_metadata.get("prompt_eval_count", 0),
                            "output_tokens": response.response_metadata.get("eval_count", 0),
                        },
                    },
                }

                save_result(log_entry, output_file)
                completed.add((float(temp), qid))
                print(f"Done QID {qid} @ temp {temp} with {len(shots)} shots")
            except Exception as e:
                print(f"   ❌ Error on QID {qid}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run few-shot experiments for directional reasoning.")
    parser.add_argument("--input-questions", default=INPUT_QUESTIONS, help="Path to questions JSONL file.")
    parser.add_argument("--input-answers", default=INPUT_ANSWERS, help="Path to answers JSONL file.")
    parser.add_argument("--output-file", default=OUTPUT_FILE, help="Path to output results JSONL file.")
    parser.add_argument("--temperatures", nargs="*", type=float, default=TEMPERATURES, help="List of temperatures to run.")
    parser.add_argument("--few-shot-k", type=int, default=FEW_SHOT_K, help="Number of few-shot exemplars.")
    parser.add_argument("--eval-subset-file", default=ANSWERS_EVAL_SUBSET, help="Path to answers_eval_subset JSONL for ID filtering.")
    parser.add_argument("--use-eval-subset", action="store_true", default=USE_EVAL_SUBSET, help="Restrict runs to eval subset IDs.")
    parser.add_argument("--no-eval-subset", dest="use_eval_subset", action="store_false", help="Disable eval subset filtering.")
    parser.add_argument("--request-timeout", type=float, default=60.0, help="Per-request timeout in seconds.")
    parser.add_argument("--model", default=MODEL, help="Model name to use in Ollama.")
    parser.add_argument("--base-url", default=BASE_URL, help="Ollama base URL.")
    parser.add_argument("--max-questions", type=int, default=0, help="Limit number of questions after filtering (0 = no limit).")
    parser.add_argument("--retries", type=int, default=2, help="Connection test retry attempts per temperature.")
    parser.add_argument("--per-question-retries", type=int, default=1, help="Additional retry attempts per question on timeout.")
    parser.add_argument("--sleep-on-timeout", type=float, default=5.0, help="Seconds to sleep between retries on timeouts.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(
        input_questions=args.input_questions,
        input_answers=args.input_answers,
        output_file=args.output_file,
        temperatures=args.temperatures,
        few_shot_k=args.few_shot_k,
        eval_subset_path=args.eval_subset_file,
        use_eval_subset=args.use_eval_subset,
        request_timeout=args.request_timeout,
        model=args.model,
        base_url=args.base_url,
        max_questions=args.max_questions,
        retries=args.retries,
        per_question_retries=args.per_question_retries,
        sleep_on_timeout=args.sleep_on_timeout,
    )