#!/usr/bin/env python3
"""
run_geospatial_evals.py
-----------------------
Unified evaluation suite for Geographic KG Reasoning (Coordinate Prediction).
Runs against the 30% holdout subset of worldcities100k.pkl.
Supports Chain-of-Thought (CoT), Tree-of-Thoughts (ToT), and Graph-of-Thoughts (GoT).
"""

import argparse
import json
import os
import sys
import time
import math
import re
import pandas as pd
from typing import Optional, Any, Tuple

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

EARTH_RADIUS_KM = 6371.0
ACCURACY_THRESHOLD_KM = 100.0

# =============================================================================
# 1. GRAPH SERIALIZATION
# =============================================================================

def serialize_knowledge_graph(graph_data: dict, max_items: int = 500) -> str:
    """Formats the NetworkX JSON graph into a highly readable format for the LLM."""
    if not isinstance(graph_data, dict):
        return "Knowledge Graph Context: Error - Invalid graph format."

    nodes = graph_data.get("nodes", [])
    links = graph_data.get("links", graph_data.get("edges", []))
    lines = ["Knowledge Graph Context:\n--- Entities ---"]
    
    node_count = 0
    for n in nodes:
        if node_count >= max_items: break
        node_id = n.get("id")
        n_type = n.get("type", "Unknown")
        if n_type == 'City' and 'lat' in n and 'lon' in n:
            lines.append(f"Entity: {node_id} ({n_type}) | Coordinates: {n['lat']}, {n['lon']}")
        else:
            lines.append(f"Entity: {node_id} ({n_type})")
        node_count += 1
        
    if len(nodes) > max_items:
        lines.append(f"...and {len(nodes) - max_items} more entities omitted...")

    lines.append("\n--- Relationships ---")
    link_count = 0
    for link in links:
        if link_count >= max_items: break
        lines.append(f"({link.get('source')}) --[{link.get('relation', 'LOCATED_IN')}]--> ({link.get('target')})")
        link_count += 1
        
    if len(links) > max_items:
        lines.append(f"...and {len(links) - max_items} more relationships omitted...")
        
    return "\n".join(lines)


# =============================================================================
# 2. PROMPT BUILDERS
# =============================================================================

def build_cot_prompt(question: str, graph_text: str) -> str:
    return f"""You are an expert geospatial reasoning assistant.
Use the provided knowledge graph to deduce the geo-coordinates.
---
{graph_text}
---
Task: Answer the question step-by-step using Chain of Thought.
Provide short bullet points for your reasoning, then conclude with a final line starting exactly with 'Answer:'.
Question: {question}

Format Requirement:
- Step 1
- Step 2
Answer: Latitude, Longitude
The answer MUST be exactly numerical coordinates. Example: Answer: 45.5017, -73.5673"""

def build_tot_prompt(question: str, graph_text: str) -> str:
    return f"""You are an expert geospatial AI using a Tree-of-Thoughts strategy.
Explore reasoning paths through the knowledge graph to find the geo-coordinates.
---
{graph_text}
---
Question: {question}

Instructions:
1. Generate EXACTLY 3 reasoning branches. Keep each branch EXTREMELY SHORT.
2. Evaluate them in 1 short sentence.
3. Return ONLY a valid JSON object matching this exact schema:
{{
  "branches": ["short path 1...", "short path 2..."],
  "evaluation": "brief evaluation",
  "final_answer": "Latitude, Longitude"
}}
Note: 'final_answer' MUST be exactly numerical coordinates (e.g., '45.5, -73.5')."""

def build_got_prompt(question: str, graph_text: str) -> str:
    return f"""You are an expert geospatial reasoning agent operating using a Graph-of-Thoughts (GoT) paradigm.
Your objective is to compute the geo-coordinates asked in the question by reasoning over a geographic knowledge graph.
---
## INPUTS
Question: {question}

{graph_text}
---
## REASONING PRINCIPLE — GRAPH OF THOUGHTS
You must construct an INTERNAL reasoning graph.
• Create reasoning nodes dynamically  
• Explore multiple reasoning paths and combine evidence (e.g., triangulating via sibling cities)
• Use spatial consistency to converge toward one answer  
---
## OUTPUT RULE (STRICT)
Return ONLY:
Final Answer: Latitude, Longitude

No explanations. No intermediate reasoning. No additional text.
Example: Final Answer: 48.2333, -79.0167

Begin reasoning internally."""


# =============================================================================
# 3. PARSERS & EXTRACTORS
# =============================================================================

def extract_coordinates(text: str) -> Optional[Tuple[float, float]]:
    if not isinstance(text, str) or not text: return None
    m_std = re.search(r'(-?\d{1,3}\.\d+)[^\d\-a-zA-Z]*?,\s*(-?\d{1,3}\.\d+)', text)
    if m_std:
        try:
            lat, lon = float(m_std.group(1)), float(m_std.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180: return lat, lon
        except ValueError: pass
    return None

def parse_cot(raw_resp: str) -> Optional[Tuple[float, float]]:
    text_flat = re.sub(r"\s+", " ", raw_resp)
    m = re.search(r"Answer\s*:\s*(.*)", text_flat, re.IGNORECASE)
    ans_block = m.group(1).strip() if m else None
    if not ans_block:
        lines = [l.strip() for l in raw_resp.strip().splitlines() if l.strip()]
        ans_block = lines[-1] if lines else None
    return extract_coordinates(ans_block) or extract_coordinates(raw_resp)

def parse_tot(raw_resp: str) -> Optional[Tuple[float, float]]:
    m = re.search(r"(\{.*?\})", raw_resp, re.IGNORECASE | re.DOTALL)
    if m:
        try:
            cleaned = m.group(1).replace("'", '"').replace("\n", " ").replace("\r", "")
            cleaned = re.sub(r",\s*\}", "}", cleaned)
            parsed_json = json.loads(cleaned)
            if "final_answer" in parsed_json:
                return extract_coordinates(str(parsed_json["final_answer"]))
        except Exception: pass
    return extract_coordinates(raw_resp)

def parse_got(raw_resp: str) -> Optional[Tuple[float, float]]:
    flat = re.sub(r"\s+", " ", raw_resp)
    m = re.search(r"Final\s+Answer\s*:\s*(.*)", flat, re.IGNORECASE)
    ans_block = m.group(1).strip() if m else None
    if not ans_block:
        lines = [l.strip() for l in raw_resp.strip().splitlines() if l.strip()]
        ans_block = lines[-1] if lines else None
    return extract_coordinates(ans_block) or extract_coordinates(raw_resp)


# =============================================================================
# 4. RUNNER UTILS
# =============================================================================

def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: records.append(json.loads(line))
    return records

def append_record(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def haversine(lat1, lon1, lat2, lon2):
    """Calculates physical distance on a sphere between points."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2 - lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1)/2)**2
    return 2 * math.asin(math.sqrt(a)) * EARTH_RADIUS_KM

def call_model(client, model_name, prompt, max_tokens, temperature, retries=3, delay=2.0):
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [!] API error (attempt {attempt}/{retries}): {e}")
            if attempt < retries: time.sleep(delay)
    return ""

def evaluate_file(path, out_dir, strategy):
    records = load_jsonl(path)
    valid_records = [r for r in records if r.get("error_km") is not None]
    
    correct  = sum(1 for r in valid_records if r.get("correct"))
    total    = len(records)
    valid    = len(valid_records)
    accuracy = correct / valid if valid else 0
    avg_error = sum(r["error_km"] for r in valid_records) / valid if valid else float('inf')

    print(f"\n=== {strategy.upper()} Final Geospatial Evaluation ===")
    print(f"Valid Parsing = {valid}/{total} ({(valid/total)*100 if total else 0:.1f}%)")
    print(f"Accuracy (≤{ACCURACY_THRESHOLD_KM}km) = {accuracy:.3f} ({correct}/{valid})")
    print(f"Average Error = {avg_error:.2f} km")

    metrics = {
        "strategy": strategy.upper(),
        "accuracy_100km": accuracy, 
        "correct": correct, 
        "valid_outputs": valid,
        "total": total,
        "average_error_km": avg_error
    }
    with open(os.path.join(out_dir, f"metrics_eval_{strategy}_outputs.json"), "w") as f:
        json.dump(metrics, f, indent=2)


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation runner for Geospatial Coordinates")
    parser.add_argument("--strategy",    choices=['cot', 'tot', 'got'], required=True)
    parser.add_argument("--data",        default="worldcities100k.pkl")
    parser.add_argument("--subset",      default="outputs/subset_ids_gptoss_30pct.json")
    parser.add_argument("--kg",          default="outputs/geospatial_knowledge_graph.json")
    parser.add_argument("--out-dir",     default="eval_outputs_real")
    parser.add_argument("--model-name",  default="gpt-oss")
    parser.add_argument("--base-url",    default="http://ollama.apps.crdig.ulaval.ca/v1")
    parser.add_argument("--api-key",     default="none")
    parser.add_argument("--limit",       type=int,   default=0)
    parser.add_argument("--reset",       action="store_true")
    args = parser.parse_args()

    if not HAS_OPENAI:
        print("[ERROR] 'openai' package not found. Run: pip install openai")
        sys.exit(1)

    configs = {
        "cot": {"builder": build_cot_prompt, "parser": parse_cot, "temp": 0.0, "tokens": 512},
        "tot": {"builder": build_tot_prompt, "parser": parse_tot, "temp": 0.2, "tokens": 1024},
        "got": {"builder": build_got_prompt, "parser": parse_got, "temp": 0.0, "tokens": 1024},
    }
    config = configs[args.strategy]

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "_no_region" 
    out_path = os.path.join(args.out_dir, f"eval_{args.strategy}{suffix}_outputs.jsonl")
    # 1. Load Data
    full_df = pd.read_pickle(args.data)
    with open(args.subset, 'r') as f:
        holdout_ids = set(json.load(f))
    eval_df = full_df[full_df.index.isin(holdout_ids)].copy()

    # 2. Load KG
    with open(args.kg, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    graph_text = serialize_knowledge_graph(kg_data, max_items=800)
    
    # 3. Resume Logic
    done_ids = set()
    if args.reset and os.path.exists(out_path): open(out_path, "w").close()
    elif os.path.exists(out_path):
        existing = load_jsonl(out_path)
        done_ids = {str(r["id"]) for r in existing}

    # 4. Filter done questions
    eval_df = eval_df[~eval_df.index.astype(str).isin(done_ids)]
    if args.limit > 0: 
        eval_df = eval_df.head(args.limit)
    
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    print(f"Starting {args.strategy.upper()} Evaluation. Samples to process: {len(eval_df)}")

    # 5. Inference Loop
    # --- INSIDE THE main() LOOP ---
    for i, (idx, row) in enumerate(eval_df.iterrows(), start=1):
        qid = str(idx)
        city_name = row['AccentCity']
        country_code = row['Country'].upper() # Get the country code
        
        # NEW: Create the unique ID to help the LLM find the node
        city_unique_id = f"{city_name}_{country_code}"
        
        gold_coords = (float(row['Latitude']), float(row['Longitude']))
        gold_raw = f"{gold_coords[0]}, {gold_coords[1]}"
        
        # UPDATE the question text so the LLM knows exactly which node to look for
        qtext = f"What are the geo-coordinates of the entity {city_unique_id}?"

        prompt = config["builder"](qtext, graph_text)
        raw_resp = call_model(client, args.model_name, prompt, config["tokens"], config["temp"])

        pred_coords = config["parser"](raw_resp)

        error_km, correct = None, False
        if pred_coords and gold_coords:
            error_km = haversine(gold_coords[0], gold_coords[1], pred_coords[0], pred_coords[1])
            correct = bool(error_km <= ACCURACY_THRESHOLD_KM)

        record = {
            "id": qid, "question": qtext, "gold": gold_raw, "raw": raw_resp,
            "pred": f"{pred_coords[0]}, {pred_coords[1]}" if pred_coords else "Invalid",
            "pred_norm": pred_coords, "gold_norm": gold_coords,
            "error_km": error_km, "correct": correct
        }
        append_record(out_path, record)

        # Print detailed progress
        status = '✔' if correct else '✘'
        p_coords = f"{pred_coords[0]:.4f}, {pred_coords[1]:.4f}" if pred_coords else "FAILED"
        g_coords = f"{gold_coords[0]:.4f}, {gold_coords[1]:.4f}"
        err_str = f"{error_km:.2f}km" if error_km is not None else "N/A"

        print(f"[{i:>4}/{len(eval_df)}] {status} City: {city_name:<20} | Pred: {p_coords:<20} | Gold: {g_coords:<20} | Err: {err_str}")

    evaluate_file(out_path, args.out_dir, args.strategy)

if __name__ == "__main__":
    main()