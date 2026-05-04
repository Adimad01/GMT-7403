#!/usr/bin/env python3
"""Rule-based KG builder for the Directional Relations finetune subset.

This script reads question and answer JSONL files (matched by id) and
produces a small deterministic knowledge graph following the example's
structure (shore_of, oriented_as, performs, has_direction, along,
instance_of, part_of).

UPDATED: Explicitly excludes any IDs found in the evaluation subset to prevent data leakage.
"""

import argparse
import json
import re
from typing import List, Dict, Any, Optional, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
    except FileNotFoundError as e:
        print(f"⚠️ Error loading {path}: {e}")
    return out


def merge_qa(questions: List[Dict[str, Any]], answers: List[Dict[str, Any]], eval_answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ans_map = {str(a.get('id')): a for a in answers}
    
    # 🔥 The Blacklist: Create a set of all eval IDs for fast lookup
    eval_ids_to_exclude = {str(e.get('id')) for e in eval_answers}
    
    merged = []
    for q in questions:
        qid = str(q.get('id'))
        
        # 🔥 EXPLICIT EXCLUSION: If the ID is in the eval set, skip it entirely
        if qid in eval_ids_to_exclude:
            continue
            
        # Only add it if it exists in our finetune answers map
        if qid in ans_map:
            a = ans_map[qid]
            merged.append({'id': qid, 'question': q.get('question',''), 'answer': a.get('absoluteAnswer', '')})
            
    return merged


VERB_RX = re.compile(r"\b(running|walking|cycling|driving|jogging|riding|hiking|perambulating|unicycling|racing)\b", re.IGNORECASE)
SHORE_RX = re.compile(r"along the ([\w\-\s]+? shore)\b", re.IGNORECASE)
SHORE_RX_ALT = re.compile(r"along the ([\w\-\s]+?) of the lake", re.IGNORECASE)
ORIENT_RX = re.compile(r"\b(north(?:-east|-west)?|south(?:-east|-west)?|east|west)\b", re.IGNORECASE)


def extract_shore_and_orientation(question: str) -> Tuple[Optional[str], Optional[str]]:
    m = SHORE_RX.search(question)
    shore = None
    if m:
        shore = m.group(1).strip()
    else:
        m2 = SHORE_RX_ALT.search(question)
        if m2:
            shore = m2.group(1).strip() + ' shore'

    orientation = None
    if shore:
        mo = ORIENT_RX.search(shore)
        if mo:
            orientation = mo.group(1).lower()
    if not orientation:
        mo2 = ORIENT_RX.search(question)
        if mo2:
            orientation = mo2.group(1).lower()

    if shore:
        shore = re.sub(r"\s+", " ", shore).strip()
    return shore, orientation


def extract_activity(question: str) -> str:
    m = VERB_RX.search(question)
    if m:
        return m.group(1).lower()
    return 'moving'


def normalize_direction(text: str) -> str:
    if not text:
        return ''
    return text.strip().lower()


def build_triples_for_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    q = item.get('question','')
    ans = item.get('answer','')
    shore, orient = extract_shore_and_orientation(q)
    activity = extract_activity(q)
    activity_label = activity.capitalize()

    triples = []
    if shore:
        triples.append({'head': shore, 'head_type': 'shore', 'relation': 'shore_of', 'tail': 'lake', 'tail_type': 'lake'})
        if orient:
            triples.append({'head': shore, 'head_type': 'shore', 'relation': 'oriented_as', 'tail': orient, 'tail_type': 'place'})
        triples.append({'head': shore, 'head_type': 'shore', 'relation': 'instance_of', 'tail': 'shore', 'tail_type': 'type'})
        triples.append({'head': shore, 'head_type': 'shore', 'relation': 'part_of', 'tail': 'lake', 'tail_type': 'lake'})

    triples.append({'head': 'I', 'head_type': 'agent', 'relation': 'performs', 'tail': activity_label, 'tail_type': 'activity'})
    if ans:
        triples.append({'head': activity_label, 'head_type': 'activity', 'relation': 'has_direction', 'tail': normalize_direction(ans), 'tail_type': 'direction'})
    
    mo = ORIENT_RX.search(q)
    if mo:
        dirword = mo.group(1).lower()
        triples.append({'head': 'I', 'head_type': 'agent', 'relation': 'heading', 'tail': dirword, 'tail_type': 'place'})

    if shore:
        triples.append({'head': activity_label, 'head_type': 'activity', 'relation': 'along', 'tail': shore, 'tail_type': 'shore'})

    return triples


def dedupe(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for t in triples:
        key = (t['head'].lower(), t['head_type'], t['relation'], str(t['tail']).lower(), t['tail_type'])
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def write_jsonl(path: str, items: List[Dict[str, Any]]):
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Rule-based KG builder for finetune subset')
    parser.add_argument('--input-questions', default='data/questions.jsonl')
    parser.add_argument('--input-answers', default='evaluation_results/answers_finetune_subset.jsonl')
    
    # 🔥 New argument specifically for the evaluation subset
    parser.add_argument('--input-eval-answers', default='evaluation_results/answers_eval_subset.jsonl')
    
    parser.add_argument('--out', default='kg_from_finetune.jsonl')
    parser.add_argument('--nodes-out', default='kg_from_finetune_nodes.jsonl')
    parser.add_argument('--sample-size', type=int, default=0, help='Process only first N items (0=all)')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Load all three files
    questions = load_jsonl(args.input_questions)
    answers = load_jsonl(args.input_answers)
    eval_answers = load_jsonl(args.input_eval_answers) 

    # Pass all three to the merge function so it can exclude the eval data
    data = merge_qa(questions, answers, eval_answers)
    
    if args.sample_size and args.sample_size > 0:
        data = data[:args.sample_size]

    all_triples = []
    for item in data:
        triples = build_triples_for_item(item)
        all_triples.extend(triples)

    deduped = dedupe(all_triples)
    write_jsonl(args.out, deduped)

    nodes = {}
    for t in deduped:
        nodes.setdefault((t['head'], t['head_type']), True)
        nodes.setdefault((t['tail'], t['tail_type']), True)
    node_list = [{'id': n[0], 'type': n[1]} for n in nodes.keys()]
    write_jsonl(args.nodes_out, node_list)

    print(f'✅ Wrote {len(deduped)} triples to {args.out} (Evaluation IDs successfully excluded)')
    print(f'✅ Wrote {len(node_list)} nodes to {args.nodes_out}')


if __name__ == '__main__':
    main()