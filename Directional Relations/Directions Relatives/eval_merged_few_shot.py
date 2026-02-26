"""
Few-shot evaluation script for gpt-oss on Merged SpatialEvalLLM datasets.
Samples few-shot examples from the same origin file within the merged dataset.
"""

import json
import os
import random
import traceback
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

from tqdm import tqdm
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# ================= CONFIGURATION =================
MODEL = "gpt-oss"
BASE_URL = "http://ollama.apps.crdig.ulaval.ca"
DATASET_ROOT = "./SpatialEvalLLM"
# Point to the directory containing merged data
GLOBAL_MAP_DIR = os.path.join(DATASET_ROOT, "merged_data")
OUTPUT_DIR = "./evaluation_results"
CHECKPOINT_DIR = "./checkpoints"  # For incremental saves
TEMPERATURE = 0.0  # Use 0 for deterministic results
FEW_SHOT_K = 3  # number of examples to sample per query

SYSTEM_PROMPT = (
    "You are a spatial reasoning expert. Answer the question about spatial navigation "
    "based on the given map and movements. Provide only the final answer, nothing else."
)
# =================================================


class SpatialEvaluatorFewShot:
    """Few-shot evaluator for gpt-oss on spatial reasoning tasks"""

    def __init__(self, model: str, base_url: str, system_prompt: str):
        self.model_name = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.llm = None
        self.results = defaultdict(list)
        self.stats = {}
        self._ensure_checkpoint_dir()

    def _ensure_checkpoint_dir(self):
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)

    def _get_checkpoint_file(self, dataset_name: str) -> str:
        return os.path.join(CHECKPOINT_DIR, f"{dataset_name}_fewshot_checkpoint.jsonl")

    def _load_checkpoint(self, dataset_name: str) -> set:
        checkpoint_file = self._get_checkpoint_file(dataset_name)
        processed = set()
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            processed.add(data.get("index"))
                tqdm.write(f"   📂 Checkpoint loaded: {len(processed)} questions already processed")
            except Exception as e:
                tqdm.write(f"   ⚠️  Error loading checkpoint: {e}")
        return processed

    def _save_checkpoint(self, dataset_name: str, result: Dict, index: int, shot_ids: List[int]):
        checkpoint_file = self._get_checkpoint_file(dataset_name)
        try:
            checkpoint_entry = {
                "index": index,
                "timestamp": datetime.now().isoformat(),
                "few_shot_ids": shot_ids,
                **result,
            }
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(checkpoint_entry) + "\n")
        except Exception as e:
            tqdm.write(f"   ❌ Error saving checkpoint: {e}")

    def initialize_llm(self):
        try:
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=TEMPERATURE,
                base_url=self.base_url,
                timeout=160,
            )
            print(f"✅ LLM initialized: {self.model_name} @ {self.base_url}")
            self.llm.invoke([HumanMessage(content="Say 'OK'")])
            print("✅ Connection test successful!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize LLM: {e}")
            traceback.print_exc()
            return False

    def load_jsonl(self, filepath: str) -> List[Dict]:
        questions = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        questions.append(json.loads(line))
            print(f"📂 Loaded {len(questions)} from {Path(filepath).name}")
            return questions
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return []

    def _normalize_answer(self, answer: str) -> str:
        normalized = answer.strip().lower()
        normalized = normalized.rstrip(".,;:!?")
        for article in ["the ", "a ", "an "]:
            if normalized.startswith(article):
                normalized = normalized[len(article):].strip()
                break
        return normalized.strip()

    def _sample_few_shot(self, dataset: List[Dict], current_idx: int, k: int) -> List[Tuple[str, str, int]]:
        current_item = dataset[current_idx]
        current_origin = current_item.get("origin_file")
        
        # If origin_file is present, only sample from the same origin to maintain consistent distribution
        if current_origin:
            pool = [
                (item.get("question", ""), item.get("answer", ""), idx)
                for idx, item in enumerate(dataset)
                if idx != current_idx and item.get("question") and item.get("answer") and item.get("origin_file") == current_origin
            ]
        else:
            # Fallback if no origin_file
            pool = [
                (item.get("question", ""), item.get("answer", ""), idx)
                for idx, item in enumerate(dataset)
                if idx != current_idx and item.get("question") and item.get("answer")
            ]

        if not pool:
            return []
        k = min(k, len(pool))
        return random.sample(pool, k)

    def _build_messages(self, question_text: str, shots: List[Tuple[str, str, int]]) -> List:
        messages = [SystemMessage(content=self.system_prompt)]
        if shots:
            messages.append(
                HumanMessage(content="Here are a few solved examples (question then answer):")
            )
        for q, a, _idx in shots:
            messages.append(HumanMessage(content=q))
            messages.append(AIMessage(content=a))
        messages.append(HumanMessage(content=question_text))
        return messages

    def evaluate_question(self, question: str, expected_answer: str, dataset: List[Dict], idx: int) -> Dict:
        if not self.llm:
            return {"correct": False, "error": "LLM not initialized", "few_shot_ids": []}
        try:
            shots = self._sample_few_shot(dataset, idx, FEW_SHOT_K)
            shot_ids = [s[2] for s in shots]
            messages = self._build_messages(question, shots)

            response = self.llm.invoke(messages)
            model_answer = response.content.strip()

            expected_norm = self._normalize_answer(expected_answer)
            model_norm = self._normalize_answer(model_answer)
            is_correct = model_norm == expected_norm

            return {
                "correct": is_correct,
                "model_answer": model_answer,
                "expected_answer": expected_answer,
                "error": None,
                "few_shot_ids": shot_ids,
            }
        except Exception as e:
            return {
                "correct": False,
                "model_answer": "",
                "expected_answer": expected_answer,
                "error": str(e),
                "few_shot_ids": [],
            }

    def evaluate_dataset(self, filepath: str, dataset_type: str) -> Dict:
        print(f"\n{'='*70}")
        print(f"📊 Evaluating {dataset_type}: {Path(filepath).name}")
        print(f"{'='*70}")

        questions = self.load_jsonl(filepath)
        if not questions:
            return {"error": "Failed to load questions", "accuracy": 0, "total": 0}

        dataset_name = Path(filepath).stem
        processed_indices = self._load_checkpoint(dataset_name)

        correct = 0
        errors = 0
        results_list = []

        for idx, q_data in tqdm(
            enumerate(questions), total=len(questions), desc=f"Evaluating {dataset_type}", unit="q"
        ):
            if idx in processed_indices:
                tqdm.write(f"   ⏭️  [{idx+1}/{len(questions)}] Skipping (already processed)")
                continue

            question_text = q_data.get("question")
            expected_answer = q_data.get("answer", "")

            result = self.evaluate_question(question_text, expected_answer, questions, idx)
            results_list.append(result)

            self._save_checkpoint(dataset_name, result, idx, result.get("few_shot_ids", []))

            if result.get("error"):
                errors += 1
                tqdm.write(f"   [{idx+1}/{len(questions)}] ❌ Error: {result['error'][:50]}...")
            elif result["correct"]:
                correct += 1
                tqdm.write(f"   [{idx+1}/{len(questions)}] ✅ Correct")
            else:
                tqdm.write(f"   [{idx+1}/{len(questions)}] ❌ Wrong")

        total = len(questions)
        accuracy = (correct / total * 100) if total > 0 else 0

        stats = {
            "dataset_file": Path(filepath).name,
            "dataset_type": dataset_type,
            "total_questions": total,
            "correct": correct,
            "incorrect": total - correct - errors,
            "errors": errors,
            "accuracy": accuracy,
            "results": results_list,
        }

        print(f"\n📈 Results for {dataset_type}:")
        print(f"   Total: {total}")
        print(f"   Correct: {correct}")
        print(f"   Incorrect: {total - correct - errors}")
        print(f"   Errors: {errors}")
        print(f"   Accuracy: {accuracy:.2f}%")

        return stats

    def evaluate_directory(self, directory: str, map_type: str) -> Dict:
        print(f"\n🗂️  Scanning {map_type} directory: {directory}")

        if not os.path.exists(directory):
            print(f"❌ Directory not found: {directory}")
            return {}

        # Modified to look specifically for merged_global_30.jsonl
        jsonl_files = sorted(Path(directory).glob("merged_global_30.jsonl"))
        
        if not jsonl_files:
            print(f"❌ No merged_global_30.jsonl file found in {directory}")
            return {}

        print(f"📁 Found {len(jsonl_files)} dataset files")

        all_results = {}
        for filepath in tqdm(jsonl_files, desc=f"Processing {map_type} datasets", unit="file"):
            dataset_name = filepath.stem
            results = self.evaluate_dataset(str(filepath), map_type)
            all_results[dataset_name] = results

        return all_results

    def run_full_evaluation(self):
        print("🚀 Starting SpatialEvalLLM Few-Shot Evaluation on Merged 30% Data")
        print(f"   Model: {self.model_name}")
        print(f"   Temperature: {TEMPERATURE}")
        print(f"   Few-shot k: {FEW_SHOT_K}")
        print(f"   Timestamp: {datetime.now().isoformat()}")

        if not self.initialize_llm():
            print("❌ Aborting: Cannot connect to LLM")
            return None

        # Only running for global maps as per user request context
        global_results = self.evaluate_directory(GLOBAL_MAP_DIR, "GLOBAL")
        
        # Skipping local maps as we haven't split/merged them in this request
        local_results = {} 

        full_results = {
            "metadata": {
                "model": self.model_name,
                "base_url": self.base_url,
                "temperature": TEMPERATURE,
                "few_shot_k": FEW_SHOT_K,
                "timestamp": datetime.now().isoformat(),
            },
            "global_maps": global_results,
            "local_maps": local_results,
        }

        return full_results

    def save_results(self, results: Dict, output_dir: str = OUTPUT_DIR):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"evaluation_merged_fewshot_{timestamp}.json")

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    evaluator = SpatialEvaluatorFewShot(MODEL, BASE_URL, SYSTEM_PROMPT)
    results = evaluator.run_full_evaluation()
    if results:
        evaluator.save_results(results)
        print("\n✅ Merged Few-shot evaluation complete!")
    else:
        print("\n❌ Evaluation failed!")


if __name__ == "__main__":
    main()
