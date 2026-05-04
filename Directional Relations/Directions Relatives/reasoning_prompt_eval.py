"""Prompt builders and tolerant parsers for CoT/ToT/GoT evaluation.
Refactored for Topological Graph Traversal (Finding Objects).
Optimized for local LLMs with strict output constraints and isolated Knowledge Graphs.
"""

import json
import re
from typing import Optional, Any, List

def serialize_graph_lines(triples: List[dict], max_lines: int = 150) -> str:
    """Formats the KG into a highly readable format for the LLM."""
    lines = ["Knowledge Graph Structural Examples:"]
    for t in triples[:max_lines]:
        s = t.get('head')
        p = t.get('relation')
        o = t.get('tail')
        lines.append(f"({s}) --[{p}]--> ({o})")
    if len(triples) > max_lines:
        lines.append(f"...and {len(triples)-max_lines} more triples omitted...")
    return "\n".join(lines)


def build_cot_prompt(question: str, graph_text: str, few_shot: Optional[str] = None) -> str:
    parts = [
        "You are an expert spatial reasoning assistant.",
        "CRITICAL INSTRUCTION: The specific objects mentioned in the question below likely DO NOT EXIST in the provided Knowledge Graph.",
        "Do NOT attempt to search the graph for the exact objects.",
        "Instead, use the provided Knowledge Graph text as a blueprint to learn the underlying topological rules (e.g., how grids, rings, or trees are structured, and how directions work in this environment).",
        "Inspire your reasoning from these structural rules and apply them logically to the objects in the question to deduce the final object.",
        "---",
        graph_text,
        "---",
        "Task: Answer the question step-by-step using Chain of Thought.",
        "Provide short bullet points tracing the path through the graph, then conclude with a final line starting exactly with 'Answer:'.",
        f"Question: {question}",
    ]
    if few_shot:
        parts.insert(0, f"Examples:\n{few_shot}\n---")
    
    parts.append("\nFormat Requirement:\n- Step 1: Start at X\n- Step 2: Move to Y\nAnswer: <exact_object_name>")
    parts.append("The final answer MUST be exactly the name of the object found, nothing else.")
    
    # 🔥 THE FIX: Explicitly tell the model to start generating right now
    parts.append("\nBegin your step-by-step reasoning below:")
    
    return "\n".join(parts)


def build_tot_prompt(question: str, graph_text: str, k_branches: int = 2, few_shot: Optional[str] = None) -> str:
    parts = [
        "You are an expert spatial AI using a Tree-of-Thoughts strategy.",
        "CRITICAL INSTRUCTION: The specific objects in the question will NOT be found in the Knowledge Graph.",
        "Use the Knowledge Graph strictly to understand the structural logic and spatial rules of the environment.",
        "Apply these inferred rules to the new objects in the question to explore reasoning paths.",
        "---",
        graph_text,
        "---",
        f"Question: {question}",
        "",
        "Instructions:",
        f"1. Generate EXACTLY {k_branches} reasoning branches tracing the path. Keep each branch EXTREMELY SHORT.",
        "2. Evaluate them in 1 short sentence.",
        "3. Return ONLY a valid JSON object matching this exact schema:",
        "{",
        '  "branches": ["short path 1...", "short path 2..."],',
        '  "evaluation": "brief evaluation",',
        '  "final_answer": "<exact_object_name>"',
        "}"
    ]
    if few_shot:
        parts.insert(0, f"Examples:\n{few_shot}\n---")
        
    # 🔥 THE FIX: Explicitly prompt for the JSON output
    parts.append("\nOutput your valid JSON response below:")
    
    return "\n".join(parts)


def build_got_prompt(question: str, graph_text: str, few_shot: Optional[str] = None) -> str:
    parts = [
        "You are an expert spatial AI using a Graph-of-Thoughts strategy.",
        "CRITICAL INSTRUCTION: Do not search the Knowledge Graph for the exact objects in the question; they are not there.",
        "Instead, study the Knowledge Graph to extract the topological principles of the environment.",
        "Map out the logical nodes and edges of your traversal by applying those structural principles to the question.",
        "---",
        graph_text,
        "---",
        f"Question: {question}",
        "",
        "Instructions:",
        "Keep all descriptions EXTREMELY SHORT (1-5 words per node/edge).",
        "Return ONLY a valid JSON object matching this exact schema:",
        "{",
        '  "thought_nodes": ["start node", "intermediate node"],',
        '  "thought_edges": ["node 1 to node 2"],',
        '  "final_answer": "<exact_object_name>"',
        "}"
    ]
    if few_shot:
        parts.insert(0, f"Examples:\n{few_shot}\n---")
        
    # 🔥 THE FIX: Explicitly prompt for the JSON output
    parts.append("\nOutput your valid JSON response below:")
    
    return "\n".join(parts)
def extract_json_like(text: str) -> Optional[Any]:
    """Aggressively hunts for a JSON dictionary inside a messy string."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    
    m = re.search(r"(\{.*?\})", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    js = m.group(1)
    
    try:
        cleaned = js.replace("'", '"')
        cleaned = re.sub(r",\s*\}", "}", cleaned)
        cleaned = re.sub(r",\s*\]", "]", cleaned)
        cleaned = cleaned.replace("\n", " ").replace("\r", "")
        return json.loads(cleaned)
    except Exception:
        return None


def extract_answer_from_cot(text: str) -> Optional[str]:
    """Finds the answer block even if separated by strange whitespace."""
    if not text:
        return None
    
    text_flat = re.sub(r"\s+", " ", text)
    m = re.search(r"Answer\s*:\s*(.*)", text_flat, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else None


def normalize_answer(ans: Optional[str]) -> str:
    """Standardizes object names by removing trailing punctuation and lowering case."""
    if not ans:
        return ""
    # Convert to lower case, strip whitespace, and remove trailing periods/quotes
    cleaned = ans.strip().lower()
    cleaned = re.sub(r'[\.\"\']+$', '', cleaned)
    return cleaned