"""
reasoning_prompt_eval.py
------------------------
Prompt builders and tolerant parsers for CoT/ToT/GoT evaluation.
Refactored for Volet C: Geographic KG Reasoning (Coordinate Prediction).
Optimized for local LLMs with strict output constraints to prevent token limits.
"""

import json
import re
from typing import Optional, Any, List, Tuple

def serialize_knowledge_graph(graph_data: dict, max_items: int = 500) -> str:
    """
    Formats the NetworkX JSON graph into a highly readable format for the LLM.
    Correctly handles composite node IDs (e.g., 'Paris_FR') by displaying 
    friendly names and coordinates while maintaining the unique ID for relationships.
    """
    if not isinstance(graph_data, dict):
        return "Knowledge Graph Context: Error - Invalid graph format."

    # NetworkX node_link_data standard uses 'nodes' and 'links'
    nodes = graph_data.get("nodes", [])
    links = graph_data.get("links", graph_data.get("edges", []))
    
    lines = ["Knowledge Graph Context:"]
    
    # 1. Serialize Entities (Nodes)
    lines.append("--- Entities ---")
    node_count = 0
    for n in nodes:
        if node_count >= max_items:
            break
            
        node_id = n.get("id")
        n_type = n.get("type", "Unknown")
        
        # Access the 'display_name' attribute created during KG construction.
        # This ensures the LLM sees "Paris" even if the ID is "Paris_FR".
        display_name = n.get("display_name", node_id)
        
        # Differentiate output based on node type
        if n_type == 'City' and 'lat' in n and 'lon' in n:
            # Cities include coordinates for geospatial reasoning
            lines.append(
                f"Entity: {node_id} (Name: {display_name}, Type: {n_type}) | "
                f"Coordinates: {n['lat']}, {n['lon']}"
            )
        else:
            # Countries/Regions/Other nodes
            lines.append(f"Entity: {node_id} (Name: {display_name}, Type: {n_type})")
        
        node_count += 1
        
    if len(nodes) > max_items:
        lines.append(f"...and {len(nodes) - max_items} more entities omitted...")

    # 2. Serialize Relationships (Edges)
    lines.append("\n--- Relationships ---")
    link_count = 0
    for link in links:
        if link_count >= max_items:
            break
        
        # Source and target represent the unique node IDs
        source = link.get("source")
        target = link.get("target")
        relation = link.get("relation", "LOCATED_IN")
        
        lines.append(f"({source}) --[{relation}]--> ({target})")
        link_count += 1
        
    if len(links) > max_items:
        lines.append(f"...and {len(links) - max_items} more relationships omitted...")
        
    return "\n".join(lines)


def build_cot_prompt(question: str, graph_text: str, few_shot: Optional[str] = None) -> str:
    """Build a Chain-of-Thought prompt asking the LLM to reason step-by-step over the KG."""
    parts = [
        "You are an expert geospatial reasoning assistant.",
        "Use the provided knowledge graph to deduce the geo-coordinates.",
        "---",
        graph_text,
        "---",
        "Task: Answer the question step-by-step using Chain of Thought.",
        "Provide short bullet points for your reasoning, then conclude with a final line starting exactly with 'Answer:'.",
        f"Question: {question}",
    ]
    if few_shot:
        parts.insert(0, f"Examples:\n{few_shot}\n---")
    
    parts.append("\nFormat Requirement:\n- Step 1\n- Step 2\nAnswer: Latitude, Longitude")
    parts.append("The answer MUST be exactly numerical coordinates. Example: Answer: 45.5017, -73.5673")
    return "\n".join(parts)


def build_tot_prompt(question: str, graph_text: str, k_branches: int = 2, max_depth: int = 3, few_shot: Optional[str] = None) -> str:
    """Build a Tree-of-Thoughts prompt that requests multiple reasoning branches and a JSON answer."""
    parts = [
        "You are an expert geospatial AI using a Tree-of-Thoughts strategy.",
        "Explore reasoning paths through the knowledge graph to find the geo-coordinates.",
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
        '  "final_answer": "Latitude, Longitude"',
        "}"
    ]
    if few_shot:
        parts.insert(0, f"Examples:\n{few_shot}\n---")
    
    parts.append("Note: 'final_answer' MUST be exactly numerical coordinates (e.g., '45.5, -73.5').")
    return "\n".join(parts)


def build_got_prompt(question: str, graph_text: str, few_shot=None):
    """Build a Graph-of-Thoughts prompt instructing the LLM to construct an internal reasoning graph."""
    GOT_PROMPT = f"""
You are an expert geospatial reasoning agent operating using a
Graph-of-Thoughts (GoT) reasoning paradigm.

Your objective is to compute the geo-coordinates (latitude and longitude) 
asked in the question by reasoning over a geographic knowledge graph.

---

## INPUTS

Question:
{question}

Geographic Knowledge Graph:
{graph_text}

The graph contains entities connected through spatial relations.

---

## REASONING PRINCIPLE — GRAPH OF THOUGHTS

You must construct an INTERNAL reasoning graph.
You are NOT given predefined reasoning steps.

Instead:
• Create reasoning nodes (thoughts) dynamically  
• Explore multiple reasoning paths when useful  
• Expand, revise, merge, or discard thoughts  
• Traverse the knowledge graph using any valid strategy  
• Combine evidence from different reasoning paths (e.g., triangulating via sibling cities)
• Use spatial consistency to converge toward one answer  

The structure, number, and type of thoughts are entirely up to you.

---

## OBJECTIVE

Determine the SINGLE most accurate geo-coordinate supported
by the knowledge graph.

---

## OUTPUT RULE (STRICT)

Return ONLY:

Final Answer: Latitude, Longitude

No explanations.  
No intermediate reasoning.  
No additional text.
Example: Final Answer: 48.2333, -79.0167

---

Begin reasoning internally.
"""
    return GOT_PROMPT.strip()


def extract_coordinates(text: str) -> Optional[Tuple[float, float]]:
    """Robustly extract Latitude and Longitude from raw text."""
    if not isinstance(text, str) or not text:
        return None

    # Standard float comma-separated (handles negative signs and spaces)
    m_std = re.search(r'(-?\d{1,3}\.\d+)[^\d\-a-zA-Z]*?,\s*(-?\d{1,3}\.\d+)', text)
    if m_std:
        try:
            lat, lon = float(m_std.group(1)), float(m_std.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass
    return None


def extract_got_answer(text: str) -> Optional[str]:
    """Extract 'Final Answer: <Lat, Lon>' from GoT verbose output."""
    if not text:
        return None
    flat = re.sub(r"\s+", " ", text)
    m = re.search(r"Final\s+Answer\s*:\s*(.*)", flat, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback to the last line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else None

def extract_answer_from_cot(text: str) -> Optional[str]:
    """Finds the answer block for Chain of Thought."""
    if not text:
        return None
    
    # Flatten the text to handle aggressive newlines
    text_flat = re.sub(r"\s+", " ", text)
    m = re.search(r"Answer\s*:\s*(.*)", text_flat, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    # Fallback to the last non-empty line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else None

def extract_json_like(text: str) -> Optional[Any]:
    """Aggressively hunts for a JSON dictionary inside a messy ToT output string."""
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
        cleaned = cleaned.replace("\n", " ").replace("\r", "")
        return json.loads(cleaned)
    except Exception:
        return None