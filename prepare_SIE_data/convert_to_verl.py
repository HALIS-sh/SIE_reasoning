#!/usr/bin/env python3
"""
Convert step4 SIE data to verl training format.
Generates prompt with structured in-context and ground truth answer.
"""
import json
import argparse
from pathlib import Path


def format_facts(triples):
    """Format knowledge graph triples into numbered list."""
    lines = []
    for i, (h, r, t) in enumerate(triples, 1):
        lines.append(f"{i}. ({h}, {r}, {t})")
    return "\n".join(lines)


TEMPLATE_BASE = """You are given a question and a set of knowledge graph facts (triples).
Each triple is in the form (head, relation, tail). Use ONLY these facts to reason and answer the question.

Question:
{question}

Facts:
{facts}

Please think step by step inside <think></think> and give the final answer entity inside <answer></answer>.
<think>
</think>
<answer>
</answer>"""

TEMPLATE_INCOMPLETE = """You are given a question and a set of knowledge graph facts (triples).
Each triple is in the form (head, relation, tail). 

Note: The provided facts may be incomplete. If the facts do not contain enough information, you may use your own world knowledge.

Question:
{question}

Facts:
{facts}

Please think step by step inside <think></think> and give the final answer entity inside <answer></answer>.
<think>
</think>
<answer>
</answer>"""


def convert_single_example(ex):
    """Convert one step4 example to verl format."""
    question = ex["question"]
    
    # step4 生成的字段是 structured_context，兼容旧的 context_triples
    triples = ex.get("structured_context") or ex.get("context_triples", [])
    if not triples:
        print(f"Warning: No triples found for question: {question[:50]}...")
        triples = []
    
    facts_str = format_facts(triples)
    
    # Choose template based on whether facts are complete
    if ex.get("no_support", False):
        prompt = TEMPLATE_INCOMPLETE.format(question=question, facts=facts_str)
    else:
        prompt = TEMPLATE_BASE.format(question=question, facts=facts_str)
    
    # Extract ground truth answer - 兼容多种字段名
    ans = ex.get("answers") or ex.get("answer") or ex.get("a_entity") or []
    if isinstance(ans, list):
        gt_answers = ans  # Keep all valid answers
        gt_answer = ans[0] if ans else ""
    else:
        gt_answers = [ans]
        gt_answer = ans
    
    # Build verl data format
    out = {
        "data_source": f"sie_reasoning_{ex.get('sie_ratio', 'unknown')}",
        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "ability": "knowledge_reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": gt_answer,
            "all_answers": gt_answers,  # For multi-answer support
        },
        "extra_info": {
            "id": ex.get("id"),
            "sie_ratio": ex.get("sie_ratio"),
            "no_support": ex.get("no_support", False),
            "question": question,
            "num_facts": len(triples),
            "used_support_count": ex.get("used_support_count", 0),
            "used_distract_count": ex.get("used_distract_count", 0),
        }
    }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert step4 SIE data to verl training format"
    )
    parser.add_argument(
        "--infile",
        required=True,
        help="Input step4 jsonl file (e.g., RoG-cwq-step3-step4-sie100.jsonl)"
    )
    parser.add_argument(
        "--outfile",
        required=True,
        help="Output verl format jsonl file"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to convert (for testing)"
    )
    parser.add_argument(
        "--show_first",
        type=int,
        default=3,
        help="Show first N examples for verification"
    )
    args = parser.parse_args()
    
    inpath = Path(args.infile)
    outpath = Path(args.outfile)
    
    if not inpath.exists():
        raise FileNotFoundError(f"Input file not found: {inpath}")
    
    outpath.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(inpath, "r", encoding="utf-8") as fin, \
         open(outpath, "w", encoding="utf-8") as fout:
        
        for line in fin:
            if not line.strip():
                continue
            
            ex = json.loads(line)
            out = convert_single_example(ex)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            
            # Show first few examples
            if count < args.show_first:
                print(f"\n--- Example {count + 1} ---")
                print(f"Question: {ex['question'][:80]}...")
                print(f"SIE Ratio: {ex.get('sie_ratio', 'N/A')}")
                print(f"Num Facts: {out['extra_info']['num_facts']}")
                print(f"Ground Truth: {out['reward_model']['ground_truth']}")
                print(f"No Support: {out['extra_info']['no_support']}")
            
            count += 1
            if args.max_examples and count >= args.max_examples:
                break
    
    print(f"\n✅ Converted {count} examples from {inpath.name} to {outpath.name}")
    print(f"   Output: {outpath}")


if __name__ == "__main__":
    main()