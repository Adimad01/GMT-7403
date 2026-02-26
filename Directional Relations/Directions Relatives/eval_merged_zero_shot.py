"""
Evaluation script for gpt-oss on Merged SpatialEvalLLM datasets.
Zero-shot evaluation on the merged 30% data.
"""

import json
import os
import traceback
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ================= CONFIGURATION =================
MODEL = "gpt-oss"
BASE_URL = "http://ollama.apps.crdig.ulaval.ca"
DATASET_ROOT = "./SpatialEvalLLM"  # Adjust path as needed
# Point to the directory containing merged data
GLOBAL_MAP_DIR = os.path.join(DATASET_ROOT, "merged_data")
OUTPUT_DIR = "./evaluation_results"
CHECKPOINT_DIR = "./checkpoints"  # For incremental saves
TEMPERATURE = 0.0  # Use 0 for deterministic results

# System prompt for spatial reasoning
SYSTEM_PROMPT = (
    "You are a spatial reasoning expert. Answer the question about spatial navigation "
    "based on the given map and movements. Provide only the final answer, nothing else."
)
# =================================================

class SpatialEvaluator:
    """Evaluates gpt-oss on spatial reasoning tasks"""
    
    def __init__(self, model: str, base_url: str, system_prompt: str):
        self.model_name = model
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.llm = None
        self.results = defaultdict(list)
        self.stats = {}
        self._ensure_checkpoint_dir()
        
    def _ensure_checkpoint_dir(self):
        """Ensure checkpoint directory exists"""
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
    
    def _get_checkpoint_file(self, dataset_name: str) -> str:
        """Get checkpoint file path for a dataset"""
        return os.path.join(CHECKPOINT_DIR, f"{dataset_name}_zeroshot_checkpoint.jsonl")
    
    def _load_checkpoint(self, dataset_name: str) -> set:
        """Load already-processed question indices from checkpoint"""
        checkpoint_file = self._get_checkpoint_file(dataset_name)
        processed = set()
        
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            processed.add(data.get('index'))
                tqdm.write(f"   📂 Checkpoint loaded: {len(processed)} questions already processed")
            except Exception as e:
                tqdm.write(f"   ⚠️  Error loading checkpoint: {e}")
        
        return processed
    
    def _save_checkpoint(self, dataset_name: str, result: Dict, index: int):
        """Save individual result to checkpoint"""
        checkpoint_file = self._get_checkpoint_file(dataset_name)
        
        try:
            checkpoint_entry = {
                "index": index,
                "timestamp": datetime.now().isoformat(),
                **result
            }
            with open(checkpoint_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(checkpoint_entry) + "\n")
        except Exception as e:
            tqdm.write(f"   ❌ Error saving checkpoint: {e}")
        
    def initialize_llm(self):
        """Initialize LLM connection"""
        try:
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=TEMPERATURE,
                base_url=self.base_url,
                timeout=160  # Increase from default ~30 second
            )
            print(f"✅ LLM initialized: {self.model_name} @ {self.base_url}")
            
            # Test connection
            self.llm.invoke([HumanMessage(content="Say 'OK'")])
            print("✅ Connection test successful!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize LLM: {e}")
            traceback.print_exc()
            return False
    
    def load_jsonl(self, filepath: str) -> List[Dict]:
        """Load questions from JSONL file"""
        questions = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        questions.append(json.loads(line))
            print(f"📂 Loaded {len(questions)} from {Path(filepath).name}")
            return questions
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return []
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer by removing articles, punctuation, and extra whitespace"""
        # Convert to lowercase and strip whitespace
        normalized = answer.strip().lower()
        
        # Remove trailing punctuation
        normalized = normalized.rstrip('.,;:!?')
        
        # Remove leading articles (a, an, the) with word boundary
        articles = ['the ', 'a ', 'an ']
        for article in articles:
            if normalized.startswith(article):
                normalized = normalized[len(article):].strip()
                break
        
        return normalized.strip()
    
    def evaluate_question(self, question: str, expected_answer: str) -> Dict:
        """Evaluate model on a single question"""
        if not self.llm:
            return {"correct": False, "error": "LLM not initialized"}
        
        try:
            # Prepare messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]
            
            # Get response
            response = self.llm.invoke(messages)
            model_answer = response.content.strip()
            
            # Normalize both answers for comparison
            expected_normalized = self._normalize_answer(expected_answer)
            model_normalized = self._normalize_answer(model_answer)
            
            is_correct = model_normalized == expected_normalized
            
            return {
                "correct": is_correct,
                "model_answer": model_answer,
                "expected_answer": expected_answer,
                "error": None
            }
        except Exception as e:
            return {
                "correct": False,
                "model_answer": "",
                "expected_answer": expected_answer,
                "error": str(e)
            }
    
    def evaluate_dataset(self, filepath: str, dataset_type: str) -> Dict:
        """Evaluate model on entire dataset"""
        print(f"\n{'='*70}")
        print(f"📊 Evaluating {dataset_type}: {Path(filepath).name}")
        print(f"{'='*70}")
        
        # Load questions
        questions = self.load_jsonl(filepath)
        if not questions:
            return {"error": "Failed to load questions", "accuracy": 0, "total": 0}
        
        # Load checkpoint to skip already-processed questions
        dataset_name = Path(filepath).stem
        processed_indices = self._load_checkpoint(dataset_name)
        
        # Evaluate each question
        correct = 0
        errors = 0
        results_list = []
        
        for idx, q_data in tqdm(enumerate(questions), total=len(questions), desc=f"Evaluating {dataset_type}", unit="q"):
            # Skip if already processed
            if idx in processed_indices:
                tqdm.write(f"   ⏭️  [{idx+1}/{len(questions)}] Skipping (already processed)")
                continue
            
            question_text = q_data.get('question')
            expected_answer = q_data.get('answer', '')
            
            result = self.evaluate_question(question_text, expected_answer)
            results_list.append(result)
            
            # Save checkpoint immediately
            self._save_checkpoint(dataset_name, result, idx)
            
            if result.get("error"):
                errors += 1
                tqdm.write(f"   [{idx+1}/{len(questions)}] ❌ Error: {result['error'][:50]}...")
            elif result["correct"]:
                correct += 1
                tqdm.write(f"   [{idx+1}/{len(questions)}] ✅ Correct")
            else:
                tqdm.write(f"   [{idx+1}/{len(questions)}] ❌ Wrong")
                
        # Calculate statistics
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
            "results": results_list
        }
        
        print(f"\n📈 Results for {dataset_type}:")
        print(f"   Total: {total}")
        print(f"   Correct: {correct}")
        print(f"   Incorrect: {total - correct - errors}")
        print(f"   Errors: {errors}")
        print(f"   Accuracy: {accuracy:.2f}%")
        
        return stats
    
    def evaluate_directory(self, directory: str, map_type: str) -> Dict:
        """Evaluate all datasets in a directory"""
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
        """Run complete evaluation on merged data"""
        print("🚀 Starting SpatialEvalLLM Zero-Shot Evaluation on Merged 30% Data")
        print(f"   Model: {self.model_name}")
        print(f"   Temperature: {TEMPERATURE}")
        print(f"   Timestamp: {datetime.now().isoformat()}")
        
        if not self.initialize_llm():
            print("❌ Aborting: Cannot connect to LLM")
            return None
        
        # Evaluate global maps
        global_results = self.evaluate_directory(GLOBAL_MAP_DIR, "GLOBAL")
        
        # Skipping local maps as we haven't split/merged them in this request
        local_results = {}
        
        # Compile final results
        full_results = {
            "metadata": {
                "model": self.model_name,
                "base_url": self.base_url,
                "temperature": TEMPERATURE,
                "timestamp": datetime.now().isoformat()
            },
            "global_maps": global_results,
            "local_maps": local_results
        }
        
        return full_results
    
    def save_results(self, results: Dict, output_dir: str = OUTPUT_DIR):
        """Save evaluation results to JSON"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"evaluation_merged_zeroshot_{timestamp}.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_file}")
            
            # Also save summary
            summary_file = os.path.join(output_dir, f"summary_merged_zeroshot_{timestamp}.txt")
            self._save_summary(results, summary_file)
            print(f"📄 Summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def _save_summary(self, results: Dict, filepath: str):
        """Save human-readable summary"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SPATIAL EVALUATION SUMMARY (MERGED ZERO-SHOT)\n")
            f.write("=" * 70 + "\n\n")
            
            metadata = results.get("metadata", {})
            f.write(f"Model: {metadata.get('model')}\n")
            f.write(f"Base URL: {metadata.get('base_url')}\n")
            f.write(f"Temperature: {metadata.get('temperature')}\n")
            f.write(f"Timestamp: {metadata.get('timestamp')}\n\n")
            
            # Global maps summary
            f.write("-" * 70 + "\n")
            f.write("GLOBAL MAPS\n")
            f.write("-" * 70 + "\n")
            global_total_correct = 0
            global_total_questions = 0
            for dataset_name, stats in results.get("global_maps", {}).items():
                if "accuracy" in stats:
                    f.write(f"{dataset_name}:\n")
                    f.write(f"  Accuracy: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total_questions']})\n")
                    global_total_correct += stats['correct']
                    global_total_questions += stats['total_questions']
            
            if global_total_questions > 0:
                global_acc = (global_total_correct / global_total_questions) * 100
                f.write(f"\nGlobal Average: {global_acc:.2f}% ({global_total_correct}/{global_total_questions})\n")


def main():
    """Main execution"""
    evaluator = SpatialEvaluator(MODEL, BASE_URL, SYSTEM_PROMPT)
    results = evaluator.run_full_evaluation()
    
    if results:
        evaluator.save_results(results)
        print("\n✅ Merged Zero-shot evaluation complete!")
    else:
        print("\n❌ Evaluation failed!")


if __name__ == "__main__":
    main()
