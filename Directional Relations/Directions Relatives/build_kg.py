#!/usr/bin/env python3
import json
import re
import os

# --- Configuration ---
INPUT_FILE = "SpatialEvalLLM/merged_global_70.jsonl"
OUTPUT_FILE = "global_kg_from_70.json"

def clean_entity(name):
    """Removes leading articles 'a' or 'an' from an entity name."""
    return re.sub(r'^(a|an)\s+', '', name.strip())

def parse_tree(text):
    """Extracts parent-child triples from a tree-type map description."""
    triples = set()
    pattern = r"The ([\w\s\-\']+) has \d+ children?: (.*?)\."
    for match in re.finditer(pattern, text):
        parent = clean_entity(match.group(1))
        children_str = re.sub(r',? and ', ',', match.group(2))
        for child in children_str.split(','):
            if child := clean_entity(child):
                triples.add((parent, "has_child", child))
    return triples

def parse_ring(text):
    """Extracts clockwise-ordering triples from a ring-type map description."""
    triples = set()
    start_match = re.search(r"where you find (?:a|an) ([\w\s\-\']+)\. Moving", text)
    elements_match = re.search(r"the elements on the path are (.*?)\. Starting", text)
    
    if start_match and elements_match:
        start_node = clean_entity(start_match.group(1))
        elements_str = re.sub(r',? and ', ',', elements_match.group(1))
        elements = [clean_entity(e) for e in elements_str.split(',') if e.strip()]

        current = start_node
        for el in elements:
            triples.add((current, "clockwise_next", el))
            current = el
        triples.add((current, "clockwise_next", start_node))
    return triples

def parse_grid(text):
    """Extracts spatial adjacency triples from a grid-type map description."""
    triples = set()
    grid = {}

    if "In the 1st row" in text:
        for i, row_name in enumerate(["1st", "2nd", "3rd"]):
            y = 3 - i
            match = re.search(f"In the {row_name} row, from left to right, we have (?:a|an) (.*?), (?:a|an) (.*?), and (?:a|an) (.*?)\\.", text)
            if match:
                grid[(1, y)] = clean_entity(match.group(1))
                grid[(2, y)] = clean_entity(match.group(2))
                grid[(3, y)] = clean_entity(match.group(3))
    elif "bottom-left corner" in text:
        setup_text = text.split("Now you have all the information")[0]
        coord_matches = re.findall(r"find (?:a |an )([\w\s\-\']+?) at index \((\d),(\d)\)", setup_text)
        if coord_matches:
            for item, x, y in coord_matches:
                grid[(int(x), int(y))] = clean_entity(item)
        else:
            items = re.findall(r"find (?:a |an )([\w\s\-\']+?)(?:\,|\.|$)", setup_text)
            if len(items) == 9:
                snake_coords = [(1,1), (2,1), (3,1), (3,2), (2,2), (1,2), (1,3), (2,3), (3,3)]
                for coord, item in zip(snake_coords, items):
                    grid[coord] = clean_entity(item)

    for (x, y), item in grid.items():
        if (x + 1, y) in grid:
            triples.add((item, "right_of", grid[(x + 1, y)]))
            triples.add((grid[(x + 1, y)], "left_of", item))
        if (x, y + 1) in grid:
            triples.add((grid[(x, y + 1)], "above", item))
            triples.add((item, "below", grid[(x, y + 1)]))

    return triples

def main():
    """Reads the merged JSONL file, parses KG triples by map type, and saves the result."""
    print(f"Extracting KGs from: {INPUT_FILE}")
    kg_dict = {}

    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip(): continue
            data = json.loads(line)
            origin = data.get("origin_file", "")
            text = data.get("question", "")

            # Initialize set for this origin file if it doesn't exist
            if origin not in kg_dict:
                kg_dict[origin] = set()

            # Parse and union the triples
            if "type-tree" in origin:
                kg_dict[origin] |= parse_tree(text)
            elif "type-ring" in origin:
                kg_dict[origin] |= parse_ring(text)
            elif "type-square" in origin:
                kg_dict[origin] |= parse_grid(text)

    # Convert sets to dicts for JSON serialization
    final_kg = {}
    for origin, triples in kg_dict.items():
        final_kg[origin] = [{"head": h, "relation": r, "tail": t} for h, r, t in triples]

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        json.dump(final_kg, outfile, indent=2)
        
    print(f"Successfully generated Knowledge Graphs for {len(final_kg)} unique maps.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()