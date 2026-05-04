import os
import re
import time
import json
import argparse
import pandas as pd
import torch
import requests
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import Dict, Any, Optional


# ==========================================================
# HOTFIX — HF MXFP4 bug (Torch 2.5+)
# ==========================================================
if not hasattr(torch, "accelerator"):
    import types

    torch.accelerator = types.SimpleNamespace(
        current_accelerator=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ==========================================================
# LOGGER
# ==========================================================
def log_and_print(msg, log_file=None):
    tqdm.write(msg)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


# ==========================================================
# 1 — GEOGRAPHIC KNOWLEDGE GRAPH (OSM) — OSM DATA IS NOW MANDATORY
# ==========================================================
class GeographicKnowledgeGraph:

    @staticmethod
    def get_nominatim(place):
        time.sleep(1)  # respect Nominatim rate limit

        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": place,
            "format": "json",
            "addressdetails": 1,
            "limit": 1,
        }

        headers = {"User-Agent": "ULaval-Geomatics-Research"}

        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            data = r.json()
            return data[0] if data else None
        except:
            return None

    # -----------------------------
    # ADMIN HIERARCHY (OSM)
    # -----------------------------
    @staticmethod
    def get_administrative_hierarchy(place):

        data = GeographicKnowledgeGraph.get_nominatim(place)

        if not data or "address" not in data:
            return f"Node '{place}' not found in OSM."

        hierarchy = []
        keys = [
            "village",
            "town",
            "city",
            "county",
            "state",
            "country",
        ]

        for k in keys:
            if k in data["address"]:
                val = data["address"][k]
                if val not in hierarchy:
                    hierarchy.append(val)

        return f"OSM Administrative Hierarchy for '{place}': {hierarchy} (lat/lon: {data.get('lat')},{data.get('lon')})"

    # -----------------------------
    # ADJACENCY (OSM Overpass)
    # -----------------------------
    @staticmethod
    def get_adjacent_entities(place):

        data = GeographicKnowledgeGraph.get_nominatim(place)
        if not data:
            return "Adjacency unavailable — OSM lookup failed"

        lat = data["lat"]
        lon = data["lon"]

        overpass_url = "http://overpass-api.de/api/interpreter"

        query = f"""
        [out:json];
        node["place"](around:20000,{lat},{lon});
        out tags;
        """

        try:
            r = requests.post(overpass_url, data={"data": query}, timeout=20)
            js = r.json()

            neighbors = []
            for e in js.get("elements", []):
                name = e.get("tags", {}).get("name")
                if name and name.lower() not in place.lower():
                    neighbors.append(name)

            return f"OSM Nodes within 20 km of '{place}': {neighbors[:15]}"
        except Exception as e:
            return f"Adjacency unavailable — Overpass error: {str(e)[:100]}"


# ==========================================================
# 2 — STOPPING CRITERIA
# ==========================================================
class StopOnObservation(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0][-60:])  # increased window
        return "Observation:" in text or "Final Answer:" in text


# ==========================================================
# 3 — PREDICATE EXTRACTION
# ==========================================================
VALID_PREDICATES = {
    "disjoint",
    "touches",
    "crosses",
    "within",
    "contains",
    "overlaps",
    "equals",
}


def extract_predicate(text):
    match = re.search(
        r"Final Answer:\s*\[?\s*(\w+)\s*\]?",
        text,
        re.IGNORECASE,
    )
    if match:
        pred = match.group(1).lower()
        if pred in VALID_PREDICATES:
            return pred
    return None


# ==========================================================
# 4 — GRAPH-OF-THOUGHT AGENT (FIXED PROMPT + EXPLICIT TOOL FORMAT)
# ==========================================================
class GPTOSSGraphOfThoughtExperiment:
    def __init__(self, tokenizer, model):

        self.tokenizer = tokenizer
        self.model = model
        self.stopper = StoppingCriteriaList(
            [StopOnObservation(tokenizer)]
        )

    # -----------------------------
    # FIXED PROMPT — CLEAR INSTRUCTIONS + EXAMPLE + OSM MANDATORY
    # -----------------------------
    def build_prompt(self, ent):

        return f"""
You are a geospatial reasoning agent that uses ONLY OpenStreetMap (OSM) data via tools.
You are NOT allowed to use any internal knowledge — you MUST call a tool first.

Valid topological predicates (choose exactly ONE):
- contains
- within
- touches
- crosses
- disjoint
- overlaps
- equals

Available tools (OSM only):
- get_administrative_hierarchy(place_name)
- get_adjacent_entities(place_name)

You MUST follow this exact format:

Thought: [your reasoning]
Action: [tool_name](exact_place_name)

You will receive:
Observation: [OSM data]

After at least ONE tool call, you may conclude:

Final Answer: [chosen_predicate]

Example (do not copy — just follow the format):

Thought: I need the administrative hierarchy for both places to check containment.
Action: [get_administrative_hierarchy](Baraboo)
Observation: OSM Administrative Hierarchy for 'Baraboo': ['village', 'Sauk County', 'Wisconsin', 'United States']
Thought: Now check the museum.
Action: [get_adjacent_entities](Circus World Museum)
Observation: OSM Nodes within 20 km of 'Circus World Museum': ['Baraboo', 'Portage', ...]
Thought: The museum is inside Baraboo according to hierarchy and proximity.
Final Answer: [contains]

Entity A (subject): {ent['place_name_subject']}
Entity B (object): {ent['place_name_object']}

Thought:
"""

    # -----------------------------
    def run_agent(self, prompt, log_file):

        current_prompt = prompt
        tools_used = 0
        max_steps = 10  # increased from 6

        for step in range(max_steps):

            inputs = self.tokenizer(
                current_prompt, return_tensors="pt"
            ).to(self.model.device)

            input_len = inputs["input_ids"].shape[-1]

            output = self.model.generate(
                **inputs,
                max_new_tokens=400,          # increased
                do_sample=False,
                repetition_penalty=1.15,
                stopping_criteria=self.stopper,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.0,
            )

            new_tokens = output[0][input_len:]
            text = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            log_and_print(f"Step {step+1}:\n{text}\n{'-'*80}", log_file)

            # Parse Action first (tool call)
            action = re.search(
                r"Action:\s*\[(.*?)\]\((.*?)\)", text, re.IGNORECASE
            )

            if action:
                tools_used += 1

                tool = action.group(1).strip().lower()
                arg = action.group(2).strip()

                if tool == "get_administrative_hierarchy":
                    obs = GeographicKnowledgeGraph.get_administrative_hierarchy(arg)
                elif tool == "get_adjacent_entities":
                    obs = GeographicKnowledgeGraph.get_adjacent_entities(arg)
                else:
                    obs = f"Unknown tool '{tool}' — only OSM tools allowed"

                current_prompt += f"{text}\nObservation: {obs}\nThought: "
                continue

            # Check for Final Answer (only after at least one tool)
            pred = extract_predicate(text)
            if pred and tools_used > 0:
                log_and_print(f"✅ FINAL ANSWER EXTRACTED: {pred}", log_file)
                return pred

            # Force next step if no action and no valid final answer
            current_prompt += (
                text
                + "\nObservation: You MUST use a tool or give Final Answer (after using at least one tool).\nThought: "
            )

        log_and_print("⏰ TIMEOUT — no valid Final Answer after max steps", log_file)
        return "timeout"

    # -----------------------------
    def evaluate(self, df, save_path):

        os.makedirs(save_path, exist_ok=True)

        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):

            log_file = f"{save_path}/log_{idx:04d}.txt"

            ent = {
                "place_name_subject": str(row["place_name_subject"]),
                "place_name_object": str(row["place_name_object"]),
            }

            expected = str(row["spatial_relation"]).lower().strip()

            prompt = self.build_prompt(ent)

            pred = self.run_agent(prompt, log_file)

            results.append(
                {
                    "index": idx,
                    "subject": ent["place_name_subject"],
                    "object": ent["place_name_object"],
                    "expected": expected,
                    "predicted": pred,
                    "match": expected == pred,
                }
            )

        return pd.DataFrame(results)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-id", default="openai/gpt-oss-20b")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit dataset rows",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print("❌ Dataset not found")
        exit(1)

    df = pd.read_csv(args.dataset)

    if args.max_rows is not None:
        df = df.head(args.max_rows)
        print(f"🔬 Running only first {len(df)} rows")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        device_map="auto",   # better for large models
    )

    model.eval()

    experiment = GPTOSSGraphOfThoughtExperiment(tokenizer, model)

    results_df = experiment.evaluate(df, args.output_dir)

    print("\n" + "="*80)
    print(results_df[["index", "subject", "object", "expected", "predicted", "match"]])
    print("="*80)

    acc = results_df["match"].mean() * 100
    print(f"\nFINAL ACCURACY: {acc:.2f}%")