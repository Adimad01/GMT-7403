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
# NOTE: Using a placeholder URL/model for runnable code context
# Please ensure these match your actual environment settings
BASE_URL = "http://ollama.apps.crdig.ulaval.ca" 
#MODEL_NAME = "gpt-oss"   # Adjust if model is named differently in your Ollama list

MODEL_NAME = "deepseek-r1:14b"   

class DeepSeekSpatialExperiment:
    def __init__(self, base_url: str = BASE_URL, model: str = MODEL_NAME, temperature: float = 0.1):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.valid_predicates = {
            "disjoint", "touches", "crosses", "within", "contains", "overlaps", "equals"
        }

    # -------------------------------------------------
    # PROMPT TEMPLATE (Updated with new GPT-4 style prompt)
    # -------------------------------------------------
    def create_prompt_template(self, text: str, context_type: str, entity_dict: Dict[str, Any]) -> str:
        """Build a prompt using the GPT-4 style spatial reasoning approach."""
        
        # Extract place types from entity_dict if available
        place_type_subject = entity_dict.get("placetype_subject", "unknown place type")
        place_type_object = entity_dict.get("placetype_object", "unknown place type")
        
        # The text should contain the relation predicate
        relation_predicate = text
        
        system_prompt = """The given seven PREDICATES: contains, is within, touches, crosses, disjoint, overlaps, equals. 
Given a sentence that include a vernacular spatial relation term: please give the corresponding PREDICATES. 
- Indicate spatial relation from A to B only.
- The vernacular spatial relation term should consider its typical meanings of the terms and the general geographical relationship it represent. Also pay attention to any other context that you can get from the sentence. 
- MAKE SURE the [PREDICATE] satisfies the the dimension requirements defined by DE-9IM and OGC given Geometry Type A and Geometry Type B.
- MAKE SURE that the output includes all plausible predicates and excludes any that are impossible or irrelevant in the given context. 
OUTPUT FORMAT: 
    Analysis: Provide a detailed, professional, and coherent examination of the spatial relation from A to B.
    Answer: A [PREDICATE1/PREDICATE2...] B. For example: A [overlaps/is within] B"""

        user_prompt = f"""given the sentence: A {relation_predicate} B and A is {place_type_subject}, B is {place_type_object}."""
        
        # Combine system and user prompts for Ollama
        full_prompt = f"""{system_prompt}

User query: {user_prompt}

Please provide your response in the specified format."""
        
        return full_prompt

    # -------------------------------------------------
    # OLLAMA QUERY FUNCTION (Updated to parse new response format)
    # -------------------------------------------------
    def query_llm(self, prompt: str) -> str:
        """Send prompt to Ollama endpoint and extract predicate from response."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature}
            }
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=300) 
            response.raise_for_status()
            result = response.json()

            content = result.get("response") or result.get("output", "")
            content = content.strip().lower()

            # Look for "Answer:" section and extract predicates between brackets
            if "answer:" in content:
                answer_section = content.split("answer:")[1].strip()
                # Extract text between brackets [predicate1/predicate2/...]
                if "[" in answer_section and "]" in answer_section:
                    bracket_content = answer_section.split("[")[1].split("]")[0]
                    # Split by / to get individual predicates
                    predicates = [p.strip() for p in bracket_content.split("/")]
                    # Return the first valid predicate found
                    for pred in predicates:
                        # Handle "is within" -> "within"
                        if "is within" in pred:
                            pred = "within"
                        if pred in self.valid_predicates:
                            return pred

            # Fallback: check if any valid predicate appears in the response
            for pred in self.valid_predicates:
                if pred in content:
                    return pred
                # Check for "is within" variant
                if pred == "within" and "is within" in content:
                    return "within"
                    
            return "invalid"

        except requests.exceptions.Timeout:
            print(f"❌ Error querying Ollama: Request timed out after 300 seconds.")
            return "error"
        except requests.exceptions.RequestException as e:
            print(f"❌ Error querying Ollama: {e}")
            return "error"
        except Exception as e:
            print(f"❌ Unexpected error querying Ollama: {e}")
            return "error"

    # -------------------------------------------------
    # EVALUATION FUNCTION
    # -------------------------------------------------
    def evaluate_model(self, dataframe: pd.DataFrame, context_type: str, save_path: Optional[str] = None):
        """Evaluate model predictions over dataset and save progress incrementally."""
        results = []
        total = len(dataframe)
        log_file = os.path.join(save_path or ".", f"deepseek_live_log_{context_type}.txt")

        # Create log file if not exists
        if not os.path.exists(log_file):
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("index,expected,predicted,match,description,model,placetype_subject,geometry_type_subject,placetype_object,geometry_type_object\n")
            print(f"🆕 Created new log file: {log_file}")

        # Read already processed indices if any
        processed_indices = set()
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                next(f)  # skip header
                for line in f:
                    if line.strip():
                        processed_indices.add(int(line.split(",")[0]))
        except Exception:
            pass

        print(f"🔁 Resuming from row {max(processed_indices) + 1 if processed_indices else 0} (already processed {len(processed_indices)} rows)")

        # Ensure results folder exists
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"📁 Created directory: {save_path}")

        for index, row in tqdm(dataframe.iterrows(), total=total, desc=f"Evaluating ({context_type})"):
            # Skip already processed rows
            if index in processed_indices:
                continue

            description = row.get("Sentence", "")
            entity_dict_raw = row.get("entity_dict", "{}") 
            try:
                entity_dict = json.loads(entity_dict_raw) if isinstance(entity_dict_raw, str) else {}
            except json.JSONDecodeError:
                entity_dict = {}

            # Add place types to entity_dict from dataframe columns
            entity_dict["placetype_subject"] = row.get("placetype_subject", "")
            entity_dict["placetype_object"] = row.get("placetype_object", "")

            expected = str(row.get("spatial_relation", "")).lower().strip()
            if not description or not expected:
                predicted = "skipped"
                is_match = False
            else:
                prompt = self.create_prompt_template(description, context_type, entity_dict)
                predicted = self.query_llm(prompt)
                is_match = expected == predicted if predicted in self.valid_predicates else False

            placetype_subject = str(row.get("placetype_subject", ""))
            geometry_subject = str(row.get("geometry_type_subject", ""))
            placetype_object = str(row.get("placetype_object", ""))
            geometry_object = str(row.get("geometry_type_object", ""))

            # Save to results list (for accuracy summary)
            results.append({
                "description": description,
                "expected": expected,
                "predicted": predicted,
                "match": is_match
            })

            # Write incremental log
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{index},{expected},{predicted},{is_match},{description.replace(',', ' ')},{MODEL_NAME},{placetype_subject},{geometry_subject},{placetype_object},{geometry_object}\n")

            print(f"Row {index}, Expected: {expected}, Predicted: {predicted}, Match: {is_match}, Description: {description}, Model: {MODEL_NAME}, Placetype subject: {placetype_subject}, Geometry type subject: {geometry_subject}, Placetype object: {placetype_object}, Geometry type object: {geometry_object}")

        # Build DataFrame from all predictions for summary
        df_results = pd.DataFrame(results)
        df_valid = df_results[df_results["predicted"] != "skipped"]

        accuracy = df_valid["match"].mean() * 100 if not df_valid.empty else 0
        print(f"\n✅ Evaluation complete! Accuracy: {accuracy:.2f}%")
        print(f"📄 Log file saved incrementally at: {log_file}")

        # Optional final CSV
        if save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_csv = os.path.join(save_path, f"deepseek_results_{context_type}_{timestamp}.csv")
            df_results.to_csv(final_csv, index=False)
            print(f"📁 Final summary CSV saved to: {final_csv}")

        return df_results

# -------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------
if __name__ == "__main__":
    # Ensure the path to the CSV exists
    DATA_PATH = "/Users/imadeddinelassakeur/Documents/PhD Journey/Projet expérimental IFT-7026/Projet/task3/triplet_update_v3.csv"
    SAVE_DIR = "./results"

    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Input CSV file not found at {DATA_PATH}")
    else:
        try:
            # Initialize experimenter with temperature 0.1 (matching GPT-4 prompt)
            experimenter = DeepSeekSpatialExperiment(base_url=BASE_URL, model=MODEL_NAME, temperature=0.1)
            
            # Load the data
            df = pd.read_csv(DATA_PATH)
            
            # Run the evaluation
            experimenter.evaluate_model(df, "few_shot", save_path=SAVE_DIR)

        except Exception as e:
            print(f"An unexpected error occurred during execution: {e}")