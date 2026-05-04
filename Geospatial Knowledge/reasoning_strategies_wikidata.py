"""
reasoning_strategies_wikidata.py — Volet C Coordinate Prediction
=========================================================================
Implements CoT, ToT, and GoT to predict (Lat, Lon) coordinates 
relying ONLY on Wikidata's semantic hierarchy.
Includes full prompt logging.
"""

import re
import time
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from geographic_kg_wikidata import DynamicWikidataCoordGraph

BASE_URL = "http://ollama.apps.crdig.ulaval.ca"
MODEL_NAME = "gpt-oss"

def extract_coordinates(text: str) -> Optional[Tuple[float, float]]:
    """Extracts a pair of coordinates (Lat, Lon) from the LLM text."""
    if not text: return None
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    match = re.search(r"(?:final\s+)?(?:estimate|answer|coordinates)\s*[:=]\s*(-?\d+\.\d+)[\s,]+(-?\d+\.\d+)", text_clean, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
        
    nums = re.findall(r"(-?\d+\.\d+)", text_clean)
    if len(nums) >= 2:
        return float(nums[-2]), float(nums[-1])
        
    return None

def llm_generate(prompt: str, max_new_tokens: int = 2048, temperature: float = 0.1) -> str:
    """Send a prompt to the Ollama API and return the generated text, retrying up to 5 times."""
    url = f"{BASE_URL.rstrip('/')}/api/generate"
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_predict": max_new_tokens}}
    for attempt in range(1, 6):
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception:
            time.sleep(3 * attempt)
    return ""

TASK_DESCRIPTION = """You are an expert geographer.
Task: Predict the exact geographic coordinates (Latitude and Longitude) of the target city.
You must output the latitude first, then the longitude.
End your response EXACTLY with: Final estimate: <latitude>, <longitude>"""

RULES_BLOCK = """Rules:
1. Use the provided Wikidata semantic hierarchy (e.g., 'located in', state, county) to mentally bound the geographic region.
2. Infer the approximate latitude and longitude based on that region.
3. Your final answer must be two decimal numbers."""

class ReasoningStrategy(ABC):
    def __init__(self, kg: DynamicWikidataCoordGraph, temp: float = 0.1, max_tokens: int = 2048):
        """Initialise the reasoning strategy with a KG, sampling temperature, and token limit."""
        self.kg, self.temp, self.max_tokens = kg, temp, max_tokens
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def reason(self, city_name: str, evidence: str, log_fn=None) -> Tuple[Optional[Tuple[float, float]], Dict]: ...
    def _generate(self, prompt: str) -> str:
        """Delegate prompt generation to the module-level llm_generate helper."""
        return llm_generate(prompt, self.max_tokens, self.temp)

class ChainOfThought(ReasoningStrategy):
    @property
    def name(self) -> str:
        """Return the strategy identifier string."""
        return "CoT"

    def reason(self, city_name: str, evidence: str, log_fn=None) -> Tuple[Optional[Tuple[float, float]], Dict]:
        """Predict coordinates via a single step-by-step Chain-of-Thought prompt."""
        trace = {"strategy": "CoT", "steps": []}
        def _log(step_name, content):
            trace["steps"].append({"step": step_name, "content": content})
            if log_fn: log_fn(f"\n  [CoT] ── {step_name} ──\n{content}")

        prompt = f"""{TASK_DESCRIPTION}\n\n{RULES_BLOCK}\n\n{evidence}\n\nThink step-by-step. Analyze the administrative hierarchy to determine the bounding region, then predict the specific coordinates.\n\nReasoning:"""
        
        # LOG FULL PROMPT
        _log("FULL_PROMPT", prompt)
        
        response = self._generate(prompt)
        _log("LLM_REASONING", response)
        coords = extract_coordinates(response)
        
        if coords is None: 
            fallback_prompt = prompt + "\n" + response + "\nAnswer with just the two numbers.\nFinal estimate:"
            _log("FALLBACK_PROMPT", fallback_prompt)
            coords = extract_coordinates(self._generate(fallback_prompt))
            
        trace["prediction"] = coords
        if log_fn: log_fn(f"\n  [CoT] ✅ FINAL PREDICTION: {coords}")
        return coords, trace

class TreeOfThought(ReasoningStrategy):
    @property
    def name(self) -> str: return "ToT"
    def reason(self, city_name: str, evidence: str, log_fn=None) -> Tuple[Optional[Tuple[float, float]], Dict]:
        trace = {"strategy": "ToT", "branches": []}
        def _log(step, content):
            if log_fn: log_fn(f"\n  [ToT] ── {step} ──\n{content}")

        branch_prompt = f"""{TASK_DESCRIPTION}\n\n{RULES_BLOCK}\n\n{evidence}\n
Explore THREE different reasoning branches for estimating the coordinates of {city_name}.

Format exactly as:
BRANCH 1: [approach]
[reasoning]
Suggested estimate: [lat], [lon]

... (up to BRANCH 3)

Begin:"""

        # LOG FULL PROMPT
        _log("FULL_PROMPT", branch_prompt)

        resp = self._generate(branch_prompt)
        _log("BRANCHES_RAW", resp)
        branches = re.findall(r"BRANCH\s+\d+\s*:\s*(.*?)(?=BRANCH\s+\d+|$)", resp, re.DOTALL | re.IGNORECASE)

        valid_lats, valid_lons = [], []
        for i, b_text in enumerate(branches):
            coords = extract_coordinates(b_text)
            if coords:
                valid_lats.append(coords[0])
                valid_lons.append(coords[1])
            _log(f"BRANCH_{i+1}", f"Predicted: {coords}\n{b_text}")

        if not valid_lats:
            fallback_prompt = f"{evidence}\nWhat are the coordinates?\nFinal estimate:"
            _log("FALLBACK_PROMPT", fallback_prompt)
            final_coords = extract_coordinates(self._generate(fallback_prompt))
        else:
            valid_lats.sort()
            valid_lons.sort()
            final_coords = (valid_lats[len(valid_lats)//2], valid_lons[len(valid_lons)//2])

        trace["prediction"] = final_coords
        if log_fn: log_fn(f"\n  [ToT] ✅ FINAL PREDICTION: {final_coords}")
        return final_coords, trace

class GraphOfThought(ReasoningStrategy):
    @property
    def name(self) -> str: return "GoT"
    def reason(self, city_name: str, evidence: str, log_fn=None) -> Tuple[Optional[Tuple[float, float]], Dict]:
        trace = {"strategy": "GoT"}
        def _log(step, content):
            if log_fn: log_fn(f"\n  [GoT] ── {step} ──\n{content}")

        phase1_prompt = f"""{TASK_DESCRIPTION}\n\n{RULES_BLOCK}\n\n{evidence}\n
Generate FOUR distinct thought nodes predicting the coordinates of {city_name}.
Format:
THOUGHT 1: [angle]
[reasoning]
Estimate: [lat], [lon]

... (up to THOUGHT 4)

Begin:"""

        # LOG FULL PHASE 1 PROMPT
        _log("PHASE1_FULL_PROMPT", phase1_prompt)

        phase1_resp = self._generate(phase1_prompt)
        _log("PHASE1_RAW", phase1_resp)
        thoughts = re.findall(r"THOUGHT\s+\d+\s*:\s*(.*?)(?=THOUGHT\s+\d+|$)", phase1_resp, re.DOTALL | re.IGNORECASE)

        lats, lons = [], []
        for t_text in thoughts:
            coords = extract_coordinates(t_text)
            if coords:
                lats.append(coords[0])
                lons.append(coords[1])

        if len(lats) >= 2:
            summaries = "\n".join([f"Node {j+1}: {lats[j]}, {lons[j]}" for j in range(len(lats))])
            merge_prompt = f"Generated coordinate predictions:\n{summaries}\nWhich pair is the most accurate? Final estimate: "
            
            # LOG FULL MERGE PROMPT
            _log("PHASE2_MERGE_PROMPT", merge_prompt)
            
            final_coords = extract_coordinates(self._generate(merge_prompt))
        elif len(lats) == 1:
            final_coords = (lats[0], lons[0])
        else:
            final_coords = None

        trace["prediction"] = final_coords
        if log_fn: log_fn(f"\n  [GoT] ✅ FINAL PREDICTION: {final_coords}")
        return final_coords, trace

STRATEGY_MAP = {"cot": ChainOfThought, "tot": TreeOfThought, "got": GraphOfThought}
def get_strategy(name: str, kg: DynamicWikidataCoordGraph, **kwargs) -> ReasoningStrategy:
    return STRATEGY_MAP[name.lower()](kg, **kwargs)