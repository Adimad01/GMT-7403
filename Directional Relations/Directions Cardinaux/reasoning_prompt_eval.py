"""Prompt builders and tolerant parsers for CoT/ToT/GoT evaluation.
Refactored for Volet C: Geographic KG Reasoning.
Optimized for local LLMs with strict output constraints to prevent token limits.
"""

import json
import re
from typing import Optional, Any, List


def serialize_graph_lines(triples: List[dict], max_lines: int = 150) -> str:
    """Formats the KG into a highly readable format for the LLM."""
    lines = ["Knowledge Graph Context:"]
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
        "You are an expert geospatial reasoning assistant.",
        "Use the provided knowledge graph as your primary evidence to deduce the correct direction.",
        "Crucially, look at the local neighborhood of the nodes mentioned in the question. Understand the relationships of surrounding entities to inspire your reasoning and help you find the best answer.",
        "---",
        graph_text,
        "---",
        "Task: Answer the question step-by-step using Chain of Thought.",
        "Provide short bullet points for your reasoning, then conclude with a final line starting exactly with 'Answer:'.",
        f"Question: {question}",
    ]
    if few_shot:
        parts.insert(0, f"Examples:\n{few_shot}\n---")
    
    parts.append("\nFormat Requirement:\n- Step 1\n- Step 2\nAnswer: <direction>")
    parts.append("The direction MUST be exactly one of: north, south, east, west, north-east, north-west, south-east, south-west.")
    return "\n".join(parts)


def build_tot_prompt(question: str, graph_text: str, k_branches: int = 2, max_depth: int = 3, few_shot: Optional[str] = None) -> str:
    parts = [
        "You are an expert geospatial AI using a Tree-of-Thoughts strategy.",
        "Use the provided knowledge graph text as hard evidence.",
        "To build your reasoning paths, explicitly explore the neighborhood of the nodes being treated. Analyze their relationships and connections to surrounding entities to inspire different branches of thought.",
        "---",
        graph_text,
        "---",
        f"Question: {question}",
        "",
        "Instructions:",
        f"1. Generate EXACTLY {k_branches} reasoning branches. Keep each branch EXTREMELY SHORT (maximum 1 sentence).",
        "2. Evaluate them in 1 short sentence.",
        "3. Return ONLY a valid JSON object matching this exact schema:",
        "{",
        '  "branches": ["short path 1...", "short path 2..."],',
        '  "evaluation": "brief evaluation",',
        '  "final_answer": "<direction>"',
        "}"
    ]
    if few_shot:
        parts.insert(0, f"Examples:\n{few_shot}\n---")
    
    parts.append("Note: 'final_answer' MUST be exactly one of: north, south, east, west, north-east, north-west, south-east, south-west.")
    return "\n".join(parts)

def build_got_prompt(question: str, graph_text: str, few_shot=None):
    
    GOT_PROMPT = f"""
You are an expert geospatial reasoning agent operating using a
Graph-of-Thoughts (GoT) reasoning paradigm.

Your objective is to compute the spatial direction asked in the question
by reasoning over a geographic knowledge graph.

---

## INPUTS

Question:
{question}

Geographic Knowledge Graph Evidence:
{graph_text}

The graph contains entities connected through spatial relations.

---

## REASONING PRINCIPLE — GRAPH OF THOUGHTS

You must construct an INTERNAL reasoning graph.

You are NOT given predefined reasoning steps.

Instead, use the provided graph text as evidence and do the following:

• Look deeply into the neighborhood of the target nodes.  
• Analyze the relationships of surrounding entities to inspire your logic.  
• Create reasoning nodes (thoughts) dynamically based on these local connections.  
• Explore multiple reasoning paths when useful.  
• Expand, revise, merge, or discard thoughts as you traverse the knowledge graph.  
• Combine evidence from different reasoning paths.  
• Use spatial consistency to converge toward one answer.  

The structure, number, and type of thoughts are entirely up to you.

Do NOT follow fixed templates or predefined reasoning angles.

---

## OBJECTIVE

Determine the SINGLE most consistent spatial direction supported
by the knowledge graph evidence and the neighborhoods you explored.

---

## OUTPUT RULE (STRICT)

Return ONLY:

Final Answer: <north|south|east|west|north-east|north-west|south-east|south-west>

No explanations.  
No intermediate reasoning.  
No additional text.

---

Begin reasoning internally.
"""

    return GOT_PROMPT.strip()

def extract_json_like(text: str) -> Optional[Any]:
    """Aggressively hunts for a JSON dictionary inside a messy string."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Aggressive JSON extraction targeting dictionaries
    m = re.search(r"(\{.*?\})", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    js = m.group(1)
    
    try:
        # Clean up common local model JSON errors
        cleaned = js.replace("'", '"')
        cleaned = re.sub(r",\s*\}", "}", cleaned)
        cleaned = re.sub(r",\s*\]", "]", cleaned)
        # Flatten newlines that might break basic parsers
        cleaned = cleaned.replace("\n", " ").replace("\r", "")
        return json.loads(cleaned)
    except Exception:
        return None


def extract_answer_from_cot(text: str) -> Optional[str]:
    """Finds the answer block even if separated by strange whitespace."""
    if not text:
        return None
    
    # Flatten the text to handle aggressive newlines like "Answer \n : \n north \n -east"
    text_flat = re.sub(r"\s+", " ", text)
    m = re.search(r"Answer\s*:\s*(.*)", text_flat, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    # Fallback to the last non-empty line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else None


def normalize_direction(ans: Optional[str]) -> str:
    """Cleans up direction strings, reconnecting broken tokens like 'north -east'."""
    if not ans:
        return ""
    a = ans.lower().strip()
    
    # Replace all whitespace, underscores, and multiple hyphens with a single hyphen.
    # This correctly reconnects "north \n -east" into "north-east".
    a = re.sub(r"[\s_]+", "-", a)
    a = re.sub(r"-+", "-", a)
    a = a.strip("-")
    
    valid_directions = {
        "north-east", "north-west", "south-east", "south-west",
        "north", "south", "east", "west"
    }
    
    # Extract the valid direction if it's buried in a longer string.
    # We check compound directions first so "north-east" doesn't just trigger "north".
    for d in valid_directions:
        if d in a:
            return d
            
    return ""