import json
import os
import re
import random
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID       = "openai/gpt-oss-20b"
DATASET_ROOT   = "./SpatialEvalLLM"
OUTPUT_DIR     = "./evaluation_results"
CHECKPOINT_DIR = "./checkpoints"
FEW_SHOT_K     = 3
EVAL_FILE      = os.path.join(DATASET_ROOT, "merged_global_30.jsonl")
TRAIN_FILE     = os.path.join(DATASET_ROOT, "merged_global_70.jsonl")

# ── Improved system prompts (with "Reasoning: low" to skip analysis channel) ── #
SYSTEM_PROMPT_ZEROSHOT = (
    "You are a strict exact-match spatial reasoning AI. "
    "You track movements on a map or grid and identify the final destination. "
    "CRITICAL: You must respond with ONLY the exact name of the final object. "
    "Never provide steps, reasoning, indices, or punctuation. Just the object name.\n"
    "Reasoning: low"
)

SYSTEM_PROMPT_FEWSHOT = (
    "You are a strict exact-match spatial reasoning AI. "
    "You track movements on a map or grid and identify the final destination. "
    "CRITICAL: Follow the exact format of the examples. Respond with ONLY the name of the final object. "
    "Never provide steps, reasoning, indices, or punctuation.\n"
    "Reasoning: low"
)

class SpatialEvaluator:
    def __init__(self, model_id: str):
        self.model_id  = model_id
        self.tokenizer = None
        self.model     = None
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def initialize_llm(self) -> bool:
        try:
            print(f"\n🔧  Loading tokenizer for {self.model_id} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            print("🔧  Loading model natively directly to GPU (MXFP4) ...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype="auto", 
            )
            self.model.eval()
            print("✅  Model ready.\n")
            return True
        except Exception as e:
            print("\n" + "═"*60)
            traceback.print_exc()
            print("═"*60 + "\n")
            print(f"❌  Model load failed: {e}")
            return False

    def _extract_final_answer(self, raw_output: str) -> str:
        match = re.search(
            r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
            raw_output,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        
        # Fallback: take the first substantive line instead of the last, 
        # ignoring trailing hallucinations.
        lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
        return lines[0] if lines else raw_output.strip()

    def _normalize(self, text: str) -> str:
        return text.lower().strip().replace(".", "").replace(",", "").replace("-", " ")

    def _build_fewshot_messages(
        self,
        question: str,
        idx: int,
        k: int,
        train_questions: List[Dict],
        system_prompt: str,
    ) -> List[Dict]:
        messages = [{"role": "system", "content": system_prompt}]

        if k > 0 and train_questions:
            shot_indices = random.sample(range(len(train_questions)), min(k, len(train_questions)))

            for si in shot_indices:
                shot = train_questions[si]
                # FIX: Structural forcing
                messages.append({"role": "user",      "content": f"{shot['question']}\n\nFinal Object:"})
                messages.append({"role": "assistant", "content": shot["answer"]})

        # FIX: Structural forcing
        messages.append({"role": "user", "content": f"{question}\n\nFinal Object:"})
        return messages

    def _run_inference(
        self,
        question: str,
        expected_answer: str,
        idx: int,
        k: int,
        system_prompt: str,
        train_questions: List[Dict],
    ) -> Dict:
        if not self.model or not self.tokenizer:
            return {"index": idx, "correct": False, "error": "Model not initialized"}
        try:
            messages = self._build_fewshot_messages(
                question, idx, k, train_questions, system_prompt
            )

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.model.device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            raw_with_tokens = self.tokenizer.decode(
                output_ids[0][input_len:], skip_special_tokens=False
            )
            model_answer = self._extract_final_answer(raw_with_tokens)
            is_correct   = self._normalize(expected_answer) in self._normalize(model_answer)

            return {
                "index": idx,
                "correct": is_correct,
                "model_answer": model_answer,
                "model_raw": raw_with_tokens,
                "expected_answer": expected_answer,
                "error": None,
            }
        except torch.cuda.OutOfMemoryError:
            print(f"\n⚠️ CUDA OOM on index {idx}. Clearing cache and skipping...")
            torch.cuda.empty_cache()
            return {
                "index": idx, "correct": False, "model_answer": "",
                "model_raw": "", "expected_answer": expected_answer,
                "error": "CUDA Out of Memory",
            }
        except Exception as e:
            return {
                "index": idx, "correct": False, "model_answer": "",
                "model_raw": "", "expected_answer": expected_answer,
                "error": str(e),
            }

    # ── Checkpoint helpers ──────────────────────────────────────────────── #

    def _ckpt_path(self, mode_tag: str) -> str:
        safe_model = self.model_id.replace("/", "_").replace("-", "_")
        return os.path.join(CHECKPOINT_DIR, f"{safe_model}_{mode_tag}_checkpoint_v2.jsonl")

    def _append_checkpoint(self, mode_tag: str, result: Dict):
        with open(self._ckpt_path(mode_tag), "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    def _load_checkpoint(self, mode_tag: str, expected_total: int) -> list:
        path = self._ckpt_path(mode_tag)
        results = []
        if os.path.exists(path):
            print(f"📂 Found checkpoint for {mode_tag}, loading...")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

            if len(results) > expected_total:
                print(
                    f"⚠️  Checkpoint has {len(results)} entries but dataset has "
                    f"{expected_total}. Truncating to {expected_total}."
                )
                results = results[:expected_total]

        return results

    def _snapshot_path(self, mode_tag: str) -> str:
        safe_model = self.model_id.replace("/", "_").replace("-", "_")
        return os.path.join(OUTPUT_DIR, f"results_{safe_model}_{mode_tag}_live_improved.json")

    def _write_snapshot(self, mode_tag: str, snapshot: Dict):
        with open(self._snapshot_path(mode_tag), "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

    # ── Evaluation loop ─────────────────────────────────────────────────── #

    def _evaluate_mode(
        self,
        questions: List[Dict],
        k: int,
        system_prompt: str,
        mode_tag: str,
        train_questions: Optional[List[Dict]] = None,
    ) -> Dict:
        results   = self._load_checkpoint(mode_tag, expected_total=len(questions))
        start_idx = len(results)
        correct   = sum(1 for r in results if r.get("correct", False))

        if start_idx > 0:
            print(
                f"🔄 Resuming [{mode_tag}] from index {start_idx} "
                f"(already processed {start_idx}/{len(questions)})..."
            )

        if start_idx >= len(questions):
            print(f"✅ [{mode_tag}] Already fully evaluated.")
            acc = (correct / len(questions)) * 100 if questions else 0
            return {"accuracy": acc, "correct": correct, "total": len(questions), "results": results}

        for i, q in tqdm(
            enumerate(questions[start_idx:]),
            total=len(questions) - start_idx,
            desc=mode_tag,
        ):
            actual_idx = start_idx + i
            result = self._run_inference(
                q.get("question", ""),
                q.get("answer", ""),
                actual_idx,
                k,
                system_prompt,
                train_questions or [],
            )
            results.append(result)

            if result["correct"]:
                correct += 1
            else:
                tqdm.write(
                    f"❌ idx={actual_idx} | "
                    f"Exp: '{result['expected_answer']}' | "
                    f"Got: '{result['model_answer']}'"
                )

            self._append_checkpoint(mode_tag, result)

            snapshot = {
                "metadata": {
                    "model": self.model_id,
                    "mode": mode_tag,
                    "few_shot_k": k,
                    "last_updated": datetime.now().isoformat(),
                },
                "progress": {
                    "processed": len(results),
                    "total": len(questions),
                    "correct_so_far": correct,
                    "running_accuracy": round(correct / len(results) * 100, 4),
                },
                "results": results,
            }
            self._write_snapshot(mode_tag, snapshot)

        acc = correct / len(questions) * 100
        print(f"✅ [{mode_tag}] Final accuracy: {acc:.2f}% ({correct}/{len(questions)})")
        return {"accuracy": acc, "correct": correct, "total": len(questions), "results": results}

    def run_full_evaluation(self, train_file: Optional[str] = None):
        if not self.initialize_llm():
            return None
        if not os.path.exists(EVAL_FILE):
            print(f"❌  Evaluation file not found: {EVAL_FILE}")
            return None

        print(f"📂  Loading evaluation data from: {EVAL_FILE}")
        with open(EVAL_FILE, "r", encoding="utf-8") as f:
            questions = [json.loads(line) for line in f if line.strip()]
        print(f"     {len(questions)} test questions loaded.\n")

        train_questions = None
        if train_file and os.path.exists(train_file):
            with open(train_file, "r", encoding="utf-8") as f:
                train_questions = [json.loads(line) for line in f if line.strip()]
            print(f"     {len(train_questions)} training examples loaded for few-shot.\n")
        else:
            print("⚠️  No training file provided. Few-shot examples will be disabled.\n")

        zs_result = self._evaluate_mode(
            questions, 0, SYSTEM_PROMPT_ZEROSHOT, "zeroshot_improved",
            train_questions=train_questions,
        )
        fs_result = self._evaluate_mode(
            questions, FEW_SHOT_K, SYSTEM_PROMPT_FEWSHOT, f"fewshot_k{FEW_SHOT_K}_improved",
            train_questions=train_questions,
        )

        delta = fs_result["accuracy"] - zs_result["accuracy"]

        print("═" * 65)
        print(f"Zero-shot : {zs_result['accuracy']:.2f}% ({zs_result['correct']}/{zs_result['total']})")
        print(f"Few-shot  : {fs_result['accuracy']:.2f}% ({fs_result['correct']}/{fs_result['total']})")
        print(f"Delta     : {delta:+.2f}%")
        print("═" * 65)

        return {
            "comparison": {
                "zeroshot_accuracy": zs_result["accuracy"],
                "fewshot_accuracy":  fs_result["accuracy"],
                "delta": delta,
            },
            "zero_shot": zs_result,
            "few_shot":  fs_result,
        }

    def save_final_results(self, results: Dict):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUTPUT_DIR, f"evaluation_gptoss_improved_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾  Final results saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS 20B Base Model — Spatial Relative Direction Evaluation")
    parser.add_argument("--train-file", default=TRAIN_FILE,
                        help="Path to training JSONL (for few-shot examples, avoids test-set leakage)")
    args = parser.parse_args()

    evaluator = SpatialEvaluator(MODEL_ID)
    results   = evaluator.run_full_evaluation(train_file=args.train_file)
    if results:
        evaluator.save_final_results(results)


if __name__ == "__main__":
    main()