import os
import json
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, Optional

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
BASE_URL = "http://ollama.apps.crdig.ulaval.ca"
MODEL_NAME = "gpt-oss"  # Adjust if model is named differently

class GPTOSSSpatialExperiment:
    def __init__(self, base_url: str = BASE_URL, model: str = MODEL_NAME, temperature: float = 0.1):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.valid_predicates = {
            "disjoint", "touches", "crosses", "within", "contains", "overlaps", "equals"
        }

    # -------------------------------------------------
    # PROMPT TEMPLATE (Updated for Few-Shot)
    # -------------------------------------------------
    def create_prompt_template(self, text: str, context_type: str, entity_dict: Dict[str, Any], 
                               df_dataset: Optional[pd.DataFrame] = None, exclude_index: int = -1) -> str:
        """
        Build a prompt using the GPT-4 style spatial reasoning approach.
        If context_type is 'few_shot', it samples 3 random examples from df_dataset.
        """
        
        # Extract place types
        place_type_subject = entity_dict.get("placetype_subject", "unknown place type")
        place_type_object = entity_dict.get("placetype_object", "unknown place type")
        
        # The text is the vernacular relation (e.g., "is inside", "next to")
        relation_predicate = text
        
        # 1. Base System Prompt
        system_prompt = """The given seven PREDICATES: contains, is within, touches, crosses, disjoint, overlaps, equals. 
Given a sentence that include a vernacular spatial relation term: please give the corresponding PREDICATES. 
- Indicate spatial relation from A to B only.
- The vernacular spatial relation term should consider its typical meanings of the terms and the general geographical relationship it represent.
- MAKE SURE the [PREDICATE] satisfies the dimensions defined by DE-9IM.
- MAKE SURE that the output includes all plausible predicates.
OUTPUT FORMAT: 
    Analysis: Provide a detailed examination of the spatial relation.
    Answer: A [PREDICATE1/PREDICATE2...] B."""

        # 2. Build Few-Shot Examples (if requested)
        few_shot_section = ""
        if context_type == "few_shot" and df_dataset is not None:
            # Drop the current row to avoid data leakage (cheating)
            # We use .sample(3) to pick 3 random examples
            try:
                # Filter out the current index so we don't use the test case as an example
                candidates = df_dataset.drop(exclude_index)
                
                # Pick 3 random samples (or fewer if dataset is small)
                n_samples = min(3, len(candidates))
                samples = candidates.sample(n_samples)
                
                few_shot_section = "\n\n--- EXAMPLES ---\n"
                
                for _, row in samples.iterrows():
                    # Extract data for the example
                    ex_text = row.get("Sentence", "")
                    ex_subj_type = row.get("placetype_subject", "place")
                    ex_obj_type = row.get("placetype_object", "place")
                    ex_truth = row.get("spatial_relation", "disjoint") # The correct answer
                    
                    # Format the example to look exactly like a completed turn
                    few_shot_section += f"""
User query: given the sentence: A {ex_text} B and A is {ex_subj_type}, B is {ex_obj_type}.
Analysis: Based on the spatial context of '{ex_text}' between a {ex_subj_type} and a {ex_obj_type}, the standard topological relation matches.
Answer: A [{ex_truth}] B
"""
                few_shot_section += "--- END EXAMPLES ---\n"
                
            except Exception as e:
                print(f"Warning: Could not generate few-shot examples: {e}")
                few_shot_section = ""

        # 3. The Actual Target Query
        user_prompt = f"""given the sentence: A {relation_predicate} B and A is {place_type_subject}, B is {place_type_object}."""
        
        # Combine everything
        full_prompt = f"""{system_prompt}
{few_shot_section}
User query: {user_prompt}

Please provide your response in the specified format."""
        
        # Optional: Print first prompt to verify formatting
        if exclude_index == 0: 
            print("DEBUG: First Generated Prompt with Few-Shot:\n", full_prompt)
            
        return full_prompt

    # -------------------------------------------------
    # OLLAMA QUERY FUNCTION
    # -------------------------------------------------
    def query_llm(self, prompt: str) -> str:
        """Send prompt to Ollama endpoint and extract predicate from response."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": 4096 # Increased context window for few-shot
                }
            }
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=300) 
            response.raise_for_status()
            result = response.json()

            content = result.get("response") or result.get("output", "")
            content = content.strip().lower()

            # Parse logic (unchanged)
            if "answer:" in content:
                # Split by last occurrence of 'answer:' to avoid picking up the few-shot examples
                # This is crucial because the model might repeat the examples
                answer_section = content.rpartition("answer:")[2].strip()
                
                if "[" in answer_section and "]" in answer_section:
                    bracket_content = answer_section.split("[")[1].split("]")[0]
                    predicates = [p.strip() for p in bracket_content.split("/")]
                    for pred in predicates:
                        if "is within" in pred: pred = "within"
                        if pred in self.valid_predicates:
                            return pred

            # Fallback
            for pred in self.valid_predicates:
                if pred in content: return pred
                if pred == "within" and "is within" in content: return "within"
                    
            return "invalid"

        except Exception as e:
            print(f"❌ Error querying Ollama: {e}")
            return "error"

    # -------------------------------------------------
    # EVALUATION FUNCTION
    # -------------------------------------------------
    def evaluate_model(self, dataframe: pd.DataFrame, context_type: str, save_path: Optional[str] = None):
        # Setup logging (Shortened for brevity, similar to original)
        if save_path: os.makedirs(save_path, exist_ok=True)
        log_file = os.path.join(save_path or ".", f"gptoss_live_log_{context_type}.txt")
        
        processed_indices = set()
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                next(f) # Skip header
                for line in f:
                    if line.strip(): processed_indices.add(int(line.split(",")[0]))

        # Create header if new file
        if not os.path.exists(log_file):
             with open(log_file, "w", encoding="utf-8") as f:
                f.write("index,expected,predicted,match,description,model\n")

        print(f"Starting evaluation with {context_type} prompting...")

        results = []
        # Main Loop
        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Evaluating"):
            if index in processed_indices: continue

            description = row.get("Sentence", "")
            expected = str(row.get("spatial_relation", "")).lower().strip()
            
            # Extract entity dict
            entity_dict_raw = row.get("entity_dict", "{}")
            try:
                entity_dict = json.loads(entity_dict_raw) if isinstance(entity_dict_raw, str) else {}
            except: entity_dict = {}
            entity_dict["placetype_subject"] = row.get("placetype_subject", "")
            entity_dict["placetype_object"] = row.get("placetype_object", "")

            # --- KEY CHANGE: PASS DATAFRAME AND INDEX FOR FEW-SHOT SAMPLING ---
            prompt = self.create_prompt_template(
                text=description, 
                context_type=context_type, 
                entity_dict=entity_dict,
                df_dataset=dataframe,     # Pass full data for sampling
                exclude_index=index       # Exclude current row
            )
            
            predicted = self.query_llm(prompt)
            is_match = expected == predicted if predicted in self.valid_predicates else False

            # Logging
            with open(log_file, "a", encoding="utf-8") as f:
                 f.write(f"{index},{expected},{predicted},{is_match},{description.replace(',', ' ')},{MODEL_NAME}\n")
            
            results.append({"match": is_match})

        return pd.DataFrame(results)

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    # Use 30% split
    DATA_PATH = "C:\\Users\\imadl\\OneDrive\\Documents\\Session Autmn 2025\\IFT-7026\\Topological-Reasoning\\dataset\\triplet_update_v3_30.csv"
    
    # Original: DATA_PATH = "C:\\Users\\imadl\\OneDrive\\Documents\\Session Autmn 2025\\IFT-7026\\Topological-Reasoning\\dataset\\triplet_update_v3.csv"
    SAVE_DIR = ".\\results"

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        gptoss = GPTOSSSpatialExperiment(temperature=0.1) # Low temp is better for few-shot
        
        # Run with "few_shot" context type
        gptoss.evaluate_model(df, "few_shot", save_path=SAVE_DIR)
    else:
        print("Dataset not found.")