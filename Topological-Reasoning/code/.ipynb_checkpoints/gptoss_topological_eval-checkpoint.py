import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

class GPTOSSSpatialExperiment:
    def __init__(self, tokenizer, model, temperature: float = 0.1, max_new_tokens: int = 50):
        self.tokenizer = tokenizer
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.valid_predicates = {
            "disjoint", "touches", "crosses", "within", "contains", "overlaps", "equals"
        }

    # -------------------------------------------------
    # PROMPT TEMPLATE (GPT-OSS 20B Architecture)
    # -------------------------------------------------
    def create_prompt_template(self, text: str, context_type: str, entity_dict: Dict[str, Any], 
                               df_examples: Optional[pd.DataFrame] = None) -> str:
        """
        Build a prompt using the structured framework optimized for GPT-OSS 20B.
        df_examples: A separate dataframe (e.g. the training set) to draw few-shot examples from.
                     This avoids label leakage from the test set.
        """
        place_type_subject = entity_dict.get("placetype_subject", "unknown place type")
        place_type_object = entity_dict.get("placetype_object", "unknown place type")
        geom_type_subject = entity_dict.get("geometry_type_subject", "unknown")
        geom_type_object = entity_dict.get("geometry_type_object", "unknown")
        relation_predicate = text
        
        # 1. SYSTEM PROMPT
        system_prompt = "### SYSTEM PROMPT ###\nYou are an expert in geospatial topological reasoning using the DE-9IM model."

        # 2. TASK
        task = """### TASK ###
Given a vernacular spatial relation between two geographic entities A and B, determine the correct DE-9IM topological predicate."""

        # 3. VALID PREDICATES (explicit constraint)
        valid_preds = """### VALID PREDICATES ###
contains, within, touches, crosses, disjoint, overlaps, equals"""

        # 4. RULES
        rules = """### RULES ###
1. The relation is DIRECTED from entity A to entity B.
2. Consider the geometry types (Point, LineString, Polygon, MultiPolygon) when reasoning about topological validity.
3. You must pick EXACTLY ONE predicate from the list above."""

        # 5. OUTPUT FORMAT — Reasoning first (brief), then answer
        output_format = """### OUTPUT FORMAT ###
Reasoning: <brief one-sentence justification>
Answer: A [<PREDICATE>] B"""

        # 6. EXAMPLES (Optional — sampled from a separate training set to avoid leakage)
        examples_section = ""
        if context_type == "few_shot" and df_examples is not None:
            try:
                n_samples = min(3, len(df_examples))
                samples = df_examples.sample(n_samples)
                
                examples_section = "### EXAMPLES ###\n"
                for _, row in samples.iterrows():
                    ex_text = str(row.get("relation_predicate", row.get("Sentence", "")))
                    ex_subj_type = row.get("placetype_subject", "place")
                    ex_obj_type = row.get("placetype_object", "place")
                    ex_geom_subj = row.get("geometry_type_subject", "unknown")
                    ex_geom_obj = row.get("geometry_type_object", "unknown")
                    ex_truth = str(row.get("spatial_relation", "disjoint")).lower().strip()
                    
                    examples_section += f"""[INPUT]
Relation: A {ex_text} B
Entity A: {ex_subj_type} (geometry: {ex_geom_subj})
Entity B: {ex_obj_type} (geometry: {ex_geom_obj})

[OUTPUT]
Reasoning: The relation '{ex_text}' between a {ex_subj_type} ({ex_geom_subj}) and a {ex_obj_type} ({ex_geom_obj}) maps to '{ex_truth}'.
Answer: A [{ex_truth}] B
"""
            except Exception as e:
                print(f"Warning: Could not generate few-shot examples: {e}")
                examples_section = ""

        # 7. INPUT
        user_query = f"""### INPUT ###
Relation: A {relation_predicate} B
Entity A: {place_type_subject} (geometry: {geom_type_subject})
Entity B: {place_type_object} (geometry: {geom_type_object})"""
        
        # Assemble the final prompt
        parts = [system_prompt, task, valid_preds, rules, output_format]
        if examples_section:
            parts.append(examples_section)
        parts.append(user_query)
        
        # Add the generation trigger
        parts.append("### OUTPUT ###\nReasoning:")
        
        full_prompt = "\n\n".join(parts)
        return full_prompt

    # -------------------------------------------------
    # NATIVE GPU INFERENCE
    # -------------------------------------------------
    def query_llm(self, prompt: str) -> str:
        """Run prompt through the native Hugging Face model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_len:]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

        # Parse logic
        if "answer:" in content:
            answer_section = content.rpartition("answer:")[2].strip()
            if "[" in answer_section and "]" in answer_section:
                bracket_content = answer_section.split("[")[1].split("]")[0]
                predicates = [p.strip() for p in bracket_content.split("/")]
                for pred in predicates:
                    if "is within" in pred: pred = "within"
                    if pred in self.valid_predicates:
                        return pred

        # Fallback parsing
        for pred in self.valid_predicates:
            if pred in content: return pred
            if pred == "within" and "is within" in content: return "within"
                
        return "invalid"

    # -------------------------------------------------
    # EVALUATION FUNCTION
    # -------------------------------------------------
    def evaluate_model(self, dataframe: pd.DataFrame, context_type: str, save_path: Optional[str] = None,
                       df_examples: Optional[pd.DataFrame] = None):
        """
        df_examples: A separate dataframe (e.g. the training set) used for few-shot examples.
                     Prevents label leakage from the test set.
        """
        if save_path: os.makedirs(save_path, exist_ok=True)
        log_file = os.path.join(save_path or ".", f"gptoss_topological_log_{context_type}_improved.txt")
        
        processed_indices = set()
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                next(f) # Skip header
                for line in f:
                    if line.strip(): processed_indices.add(int(line.split(",")[0]))

        if not os.path.exists(log_file):
             with open(log_file, "w", encoding="utf-8") as f:
                f.write("index,expected,predicted,match,description\n")

        print(f"Starting evaluation with {context_type} prompting...")

        results = []
        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Evaluating"):
            if index in processed_indices: continue

            description = row.get("Sentence", "")
            # Use the clean 'relation_predicate' column for the prompt
            relation_pred = str(row.get("relation_predicate", row.get("Sentence", "")))
            expected = str(row.get("spatial_relation", "")).lower().strip()
            
            entity_dict = {
                "placetype_subject": row.get("placetype_subject", ""),
                "placetype_object": row.get("placetype_object", ""),
                "geometry_type_subject": row.get("geometry_type_subject", "unknown"),
                "geometry_type_object": row.get("geometry_type_object", "unknown"),
            }

            prompt = self.create_prompt_template(
                text=relation_pred, 
                context_type=context_type, 
                entity_dict=entity_dict,
                df_examples=df_examples,
            )
            
            predicted = self.query_llm(prompt)
            is_match = expected == predicted if predicted in self.valid_predicates else False

            with open(log_file, "a", encoding="utf-8") as f:
                 f.write(f"{index},{expected},{predicted},{is_match},{description.replace(',', ' ')}\n")
            
            results.append({"match": is_match})

        print(f"Finished. Results saved to {log_file}")
        return pd.DataFrame(results)

# -------------------------------------------------
# MAIN RUNNER
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-OSS Topological Evaluation")
    parser.add_argument("--dataset", required=True, help="Path to the test dataset CSV file")
    parser.add_argument("--train-dataset", default=None, help="Path to the training dataset CSV (used for few-shot examples to avoid leakage)")
    parser.add_argument("--context-type", choices=["zero_shot", "few_shot"], required=True, help="Prompt strategy to use")
    parser.add_argument("--model-id", default="openai/gpt-oss-20b", help="Hugging Face model ID or local path")
    parser.add_argument("--output-dir", default="./results", help="Directory to save the log files")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for model sampling")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="Max new tokens to generate")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit the number of rows evaluated (for testing)")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"❌ Error: Dataset not found at {args.dataset}")
        exit(1)

    # 1. Load Data
    print(f"Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)
    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"Truncated dataset to {args.max_rows} rows.")

    # Load training dataset for few-shot examples (avoids test-set label leakage)
    df_examples = None
    if args.context_type == "few_shot":
        if args.train_dataset and os.path.exists(args.train_dataset):
            df_examples = pd.read_csv(args.train_dataset)
            print(f"Loaded {len(df_examples)} training examples for few-shot from: {args.train_dataset}")
        else:
            print("⚠️  Warning: --train-dataset not provided for few_shot mode. Few-shot examples will be disabled.")

    # 2. Load Model natively to GPU
    print(f"Loading tokenizer from: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model on {DEVICE} with dtype {DTYPE} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # 3. Run Experiment
    gptoss = GPTOSSSpatialExperiment(
        tokenizer=tokenizer, 
        model=model, 
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    gptoss.evaluate_model(df, args.context_type, save_path=args.output_dir, df_examples=df_examples)