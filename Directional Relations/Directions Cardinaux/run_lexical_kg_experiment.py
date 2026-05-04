"""
run_lexical_kg_experiment.py — External Lexical KG Evaluation
========================================================================
Queries ConceptNet and WordNet dynamically to build an external 
semantic knowledge graph for abstract spatial reasoning.
"""

import os
import json
import re
import time
import requests
import argparse
from tqdm import tqdm
from nltk.corpus import wordnet

# Use your existing local Ollama setup
BASE_URL = "http://ollama.apps.crdig.ulaval.ca"
MODEL_NAME = "gpt-oss"

# 1. KNOWLEDGE FETCHERS
def fetch_conceptnet(word: str) -> list:
    """Fetches Antonyms, PartOf, and Spatial relationships from ConceptNet."""
    url = f"http://api.conceptnet.io/c/en/{word.replace(' ', '_')}?limit=10"
    time.sleep(1.1)  # Respect ConceptNet's 1 req/sec rate limit
    edges_found = []
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            for edge in data.get('edges', []):
                rel = edge['rel']['label']
                # Filter to relations that matter for spatial geometry
                if rel in ['Antonym', 'PartOf', 'RelatedTo', 'HasA', 'AtLocation']:
                    target = edge['end']['label'] if edge['start']['label'].lower() == word else edge['start']['label']
                    edges_found.append(f"({word}) --[{rel}]--> ({target})")
    except Exception as e:
        pass
    return list(set(edges_found))[:5] # Keep it concise

def fetch_wordnet(word: str) -> str:
    """Fetches the literal definition of movement/verbs from WordNet."""
    synsets = wordnet.synsets(word.replace(' ', '_'))
    if synsets:
        return f"Definition of '{word}': {synsets[0].definition()}"
    return ""

def build_lexical_evidence(question: str) -> str:
    """Scans the question for keywords and builds the External KG block."""
    keywords_spatial = ["north", "south", "east", "west", "shore", "lake", "island", "sea", "ocean"]
    keywords_verbs = ["turn around", "head back", "racing", "running"]
    
    q_lower = question.lower()
    evidence_lines = ["=== EXTERNAL CONCEPTNET & WORDNET EVIDENCE ==="]
    
    # ConceptNet for topology and directions
    for word in keywords_spatial:
        if word in q_lower:
            concepts = fetch_conceptnet(word)
            if concepts:
                evidence_lines.extend(concepts)
                
    # WordNet for movement semantics
    for verb in keywords_verbs:
        if verb in q_lower:
            wn_def = fetch_wordnet(verb)
            if wn_def:
                evidence_lines.append(wn_def)
                
    if len(evidence_lines) == 1:
        return "=== EXTERNAL CONCEPTNET & WORDNET EVIDENCE ===\n(No external lexical data found)"
    return "\n".join(evidence_lines)

# 2. LLM REASONING & PARSING
def clean_direction(text: str) -> str:
    if not text: return "invalid"
    a = text.lower().strip()
    a = re.sub(r"[\s_]+", "-", a)
    a = re.sub(r"-+", "-", a).strip("-")
    valid_directions = {"north", "south", "east", "west", "north-east", "north-west", "south-east", "south-west"}
    for d in valid_directions:
        if d in a: return d
    return "invalid"

def extract_answer(text: str) -> str:
    text_flat = re.sub(r"\s+", " ", text)
    m = re.search(r"Answer\s*:\s*(.*)", text_flat, re.IGNORECASE)
    if m: return clean_direction(m.group(1))
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return clean_direction(lines[-1]) if lines else "invalid"

def llm_generate(prompt: str) -> str:
    url = f"{BASE_URL.rstrip('/')}/api/generate"
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.1, "num_predict": 1024}}
    try:
        resp = requests.post(url, json=payload, timeout=300)
        return resp.json().get("response", "").strip()
    except Exception:
        return ""

def run_experiment(sample_size=10):
    print(f"🚀 Loading {sample_size} Abstract Spatial Logic questions...")
    
    # Load your local JSONL files
    questions, answers = {}, {}
    with open("data/questions.jsonl", "r") as f:
        for line in f:
            rec = json.loads(line)
            questions[rec["id"]] = rec["question"]
    with open("data/answers.jsonl", "r") as f:
        for line in f:
            rec = json.loads(line)
            answers[rec["id"]] = rec["absoluteAnswer"]
            
    # Take first 10
    test_ids = list(questions.keys())[:sample_size]
    
    correct_count = 0
    os.makedirs("evaluation_results", exist_ok=True)
    
    with open("evaluation_results/lexical_kg_eval.txt", "w") as log:
        for q_id in tqdm(test_ids):
            q_text = questions[q_id]
            gold = answers[q_id]
            
            def logger(msg):
                log.write(msg + "\n"); log.flush()
                
            logger(f"\n--- Question ID: {q_id} ---")
            logger(f"Q: {q_text}")
            logger(f"Gold Answer: {gold}")
            
            # Fetch External DB Definitions!
            evidence = build_lexical_evidence(q_text)
            logger(f"\n{evidence}\n")
            
            prompt = f"""You are an expert in geospatial logic.
Use the external definitions below to help deduce the correct direction.

{evidence}

Question: {q_text}
Task: Answer the question step-by-step using Chain of Thought.
Conclude with a final line exactly formatted as:
Answer: <direction>"""
            
            raw_output = llm_generate(prompt)
            pred = extract_answer(raw_output)
            
            is_correct = (pred == gold.lower())
            if is_correct: correct_count += 1
            
            status = "✅ Correct" if is_correct else "❌ Incorrect"
            logger(f"LLM Prediction: {pred} | {status}\n")
            logger("-" * 50)
            
    accuracy = (correct_count / sample_size) * 100
    print(f"\n📊 EXTERNAL LEXICAL KG EVALUATION COMPLETE")
    print(f"   Accuracy: {accuracy:.1f}% ({correct_count}/{sample_size})")

if __name__ == "__main__":
    run_experiment(10)