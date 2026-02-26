import json
import traceback
import uuid
import os
import argparse
from typing import Optional, Set, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ================= CONFIGURATION =================
# The model name as it appears in your Ollama instance
MODEL = "gpt-oss" 
BASE_URL = "http://ollama.apps.crdig.ulaval.ca"
DEFAULT_INPUT_FILE = ".//data//questions.jsonl"
DEFAULT_OUTPUT_FILE = "experiment_results_gptoss.jsonl"
DEFAULT_IDS_FILE = ".//evaluation_results//answers_eval_subset.jsonl"
DEFAULT_TEMPERATURES = [0.25, 0.5, 1]

# The system prompt strictly enforcing the answer format
SYSTEM_PROMPT = (
    "You are a helpful assistant. I will give you a question about directions. "
    "The answer is either north, south, east, west. north-east, north-west, "
    "south-east or south-west. Please only reply with the answer. No yapping."
)
# =================================================

def load_questions(filename: str, max_questions: int = 1400, ids_filter: Optional[Set[str]] = None) -> List[dict]:
    """Reads questions from the jsonl file (limited to max_questions), optionally filtering by IDs."""
    q_list = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    obj = json.loads(line)
                    # If filtering by IDs, include only matching IDs
                    if ids_filter is not None:
                        qid = str(obj.get('id'))
                        if qid not in ids_filter:
                            continue
                    q_list.append(obj)
                    if len(q_list) >= max_questions:
                        break
        src_info = f"{filename} (max: {max_questions})"
        if ids_filter is not None:
            src_info += f", filtered by {len(ids_filter)} IDs"
        print(f"📂 Loaded {len(q_list)} questions from {src_info}")
        return q_list
    except FileNotFoundError:
        print(f"❌ Error: Could not find {filename}. Please ensure the file exists.")
        return []

def load_ids_filter(filename: Optional[str]) -> Optional[Set[str]]:
    """Loads a set of IDs from a JSONL file with objects containing an 'id' field."""
    if not filename:
        return None
    if not os.path.exists(filename):
        print(f"⚠️  IDs filter file not found: {filename}. Proceeding without filter.")
        return None
    ids: Set[str] = set()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    q_id = str(obj.get('id'))
                    if q_id:
                        ids.add(q_id)
                except json.JSONDecodeError:
                    continue
        print(f"🔖 Loaded {len(ids)} IDs from {filename}")
        return ids if ids else None
    except Exception as e:
        print(f"⚠️  Failed to load IDs filter from {filename}: {e}")
        return None

def get_completed_tasks(filename):
    """
    Scans the output file to find which (temperature, question_id) pairs 
    have already been processed.
    """
    completed = set()
    if not os.path.exists(filename):
        return completed
    
    print("🔎 Scanning existing results to resume progress...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    # We track progress based on Temperature and Request ID
                    # Ensure we handle float/string conversions consistently
                    temp = float(data['request']['temperature'])
                    q_id = str(data['request']['id'])
                    completed.add((temp, q_id))
                except (KeyError, ValueError, json.JSONDecodeError):
                    # Ignore corrupted lines or incomplete writes
                    continue
    except Exception as e:
        print(f"⚠️  Warning: Error reading existing file ({e}). Starting fresh.")
        
    print(f"   Found {len(completed)} completed tasks.")
    return completed

def save_result(result_obj, filename):
    """Appends a single result to the output file."""
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result_obj) + "\n")

def _invoke_with_timeout(llm, messages, timeout_sec: float):
    """Invoke the LLM with a timeout to avoid hanging requests."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(llm.invoke, messages)
        try:
            return fut.result(timeout=timeout_sec)
        except FuturesTimeout:
            return None

def run_experiments(input_file: str, output_file: str, temperatures: List[float], ids_file: Optional[str] = None, max_questions: int = 1400, request_timeout: float = 60.0):
    ids_filter = load_ids_filter(ids_file)
    questions = load_questions(input_file, max_questions=max_questions, ids_filter=ids_filter)
    if not questions:
        return

    # Load the set of already completed tasks
    completed_tasks = get_completed_tasks(output_file)

    print(f"🌡️  Temperatures to test: {temperatures}")

    # --- Outer Loop: Temperature ---
    for temp in temperatures:
        print("\n" + "=" * 60)
        print(f"🔄 STARTING BATCH: Temperature {temp}")
        print("=" * 60)

        # Check if this entire batch is already done to avoid initializing LLM unnecessarily
        # (Optional optimization, but good for speed)
        batch_ids = [str(q.get('id')) for q in questions]
        if all((float(temp), qid) in completed_tasks for qid in batch_ids):
            print(f"⏭️  All questions for Temp {temp} are already done. Skipping batch.")
            continue

        # Initialize LLM with current temperature
        try:
            llm = ChatOllama(
                model=MODEL,
                temperature=temp,
                base_url=BASE_URL
            )
            print(f"✅ LLM initialized: {MODEL} @ Temp {temp}")
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            traceback.print_exc()
            continue

        # Connection Test
        try:
            # Only test connection if we actually have work to do in this batch
            print("   Testing connection...")
            test_resp = _invoke_with_timeout(llm, [HumanMessage(content="Reply with exactly: TEST OK")], timeout_sec=min(15.0, request_timeout))
            if test_resp is None:
                print("   ❌ Connection test timed out.")
                print("   ⚠️  Skipping this temperature due to connection timeout.")
                continue
            print("   ✅ Connection successful!")
        except Exception as e:
            print(f"   ❌ Connection test failed: {e}")
            print("   ⚠️  Skipping this temperature due to connection failure.")
            continue

        # --- Inner Loop: Questions ---
        for i, q_data in enumerate(questions):
            # We use the ID from the file, or fallback to index if missing
            q_id = str(q_data.get('id', i + 1))
            q_text = q_data.get('question')
            
            # CHECKPOINT LOGIC: Skip if done
            if (float(temp), q_id) in completed_tasks:
                print(f"   ⏭️  Skipping QID {q_id} (Already exists)")
                continue

            print(f"   Processing QID {q_id}...", end="", flush=True)
            
            # Construct Messages
            messages_payload = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q_text}
            ]
            
            langchain_msgs = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=q_text)
            ]

            # Metadata for the request log
            msg_uuid = f"msg_{uuid.uuid4().hex[:24]}"
            timestamp = datetime.now().isoformat()

            try:
                # Invoke Model with timeout
                response = _invoke_with_timeout(llm, langchain_msgs, timeout_sec=request_timeout)
                if response is None:
                    print(" [Timeout -> Skipped]")
                    continue
                answer_text = response.content.strip()

                # Construct the exact JSON structure requested
                log_entry = {
                    "request": {
                        "id": q_id, # Using the actual ID from file
                        "datetime": timestamp,
                        "messages": messages_payload,
                        "temperature": temp,
                        "max_tokens": 1024
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
                            "input_tokens": response.response_metadata.get('prompt_eval_count', 0),
                            "output_tokens": response.response_metadata.get('eval_count', 0)
                        }
                    }
                }

                save_result(log_entry, output_file)
                
                # Update in-memory checkpoint so we don't repeat if loop logic gets complex
                completed_tasks.add((float(temp), q_id)) 
                print(" [Done]")

            except Exception as e:
                print(f"\n   ❌ Error on QID {q_id}: {e}")
                traceback.print_exc()

def parse_args():
    parser = argparse.ArgumentParser(description="Run zero-shot experiments for directional reasoning.")
    parser.add_argument("--input-file", default=DEFAULT_INPUT_FILE, help="Path to questions JSONL file.")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="Path to output results JSONL file.")
    parser.add_argument("--ids-file", default=DEFAULT_IDS_FILE, help="Optional path to JSONL file containing IDs to evaluate.")
    parser.add_argument("--max-questions", type=int, default=1400, help="Max number of questions to process.")
    parser.add_argument("--temperatures", nargs="*", type=float, default=DEFAULT_TEMPERATURES, help="List of temperatures to run.")
    parser.add_argument("--request-timeout", type=float, default=60.0, help="Per-request timeout in seconds.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_experiments(
        input_file=args.input_file,
        output_file=args.output_file,
        temperatures=args.temperatures,
        ids_file=args.ids_file,
        max_questions=args.max_questions,
        request_timeout=args.request_timeout,
    )