#!/usr/bin/env python3
"""
Batch convert all SIE ratio files to verl format.
Processes multiple step4 output files (sie0, sie25, sie50, sie75, sie100) in one go.
"""
import json
import argparse
from pathlib import Path
from convert_to_verl import convert_single_example


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert all SIE step4 files to verl format"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing step4 output files"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for verl format files"
    )
    parser.add_argument(
        "--dataset",
        default="cwq",
        choices=["cwq", "webqsp"],
        help="Dataset name (default: cwq)"
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=["0", "25", "50", "75", "100"],
        help="SIE ratios to convert (default: 0 25 50 75 100)"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum examples per file (for testing)"
    )
    parser.add_argument(
        "--show_first",
        type=int,
        default=2,
        help="Show first N examples per file for verification"
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting SIE data from {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Ratios: {args.ratios}\n")
    
    total_converted = 0
    results = []
    
    for ratio in args.ratios:
        infile = input_dir / f"RoG-{args.dataset}-step3-step4-sie{ratio}.jsonl"
        outfile = output_dir / f"verl_{args.dataset}_sie{ratio}.jsonl"
        
        if not infile.exists():
            print(f"⚠️  Warning: {infile.name} not found, skipping...")
            continue
        
        print(f"{'='*70}")
        print(f"Processing SIE-{ratio}%: {infile.name}")
        print(f"{'='*70}")
        
        count = 0
        with open(infile, "r", encoding="utf-8") as fin, \
             open(outfile, "w", encoding="utf-8") as fout:
            
            for line in fin:
                if not line.strip():
                    continue
                
                try:
                    ex = json.loads(line)
                    out = convert_single_example(ex)
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    
                    # Show first few examples for verification
                    if count < args.show_first:
                        print(f"\n--- Example {count + 1} ---")
                        print(f"Question: {ex['question'][:80]}...")
                        print(f"SIE Ratio: {ex.get('sie_ratio', 'N/A')}")
                        print(f"Num Facts: {out['extra_info']['num_facts']}")
                        print(f"  - Support: {out['extra_info']['used_support_count']}")
                        print(f"  - Distract: {out['extra_info']['used_distract_count']}")
                        print(f"Ground Truth: {out['reward_model']['ground_truth']}")
                        print(f"All Answers: {out['reward_model']['all_answers']}")
                        print(f"No Support: {out['extra_info']['no_support']}")
                    
                    count += 1
                    if args.max_examples and count >= args.max_examples:
                        break
                
                except Exception as e:
                    print(f"Error processing line {count + 1}: {e}")
                    continue
        
        total_converted += count
        results.append({
            "ratio": ratio,
            "input": infile.name,
            "output": outfile.name,
            "count": count
        })
        
        print(f"\n✅ Converted {count} examples to {outfile.name}\n")
    
    # Print summary
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}")
    for res in results:
        print(f"  SIE-{res['ratio']:>3}%: {res['count']:>6} examples → {res['output']}")
    print(f"{'='*70}")
    print(f"Total: {total_converted} examples converted")
    print(f"Output directory: {output_dir}\n")
    
    # Print usage instructions
    print("📝 To train with verl GRPO, use:")
    print(f"\n  # Train on SIE-100% (full support)")
    print(f"  bash run_sie_grpo.sh \\")
    print(f"    --train_file {output_dir}/verl_{args.dataset}_sie100.jsonl \\")
    print(f"    --val_file {output_dir}/verl_{args.dataset}_sie100.jsonl\n")
    
    print(f"  # Train on SIE-0% (no support, world knowledge only)")
    print(f"  bash run_sie_grpo.sh \\")
    print(f"    --train_file {output_dir}/verl_{args.dataset}_sie0.jsonl \\")
    print(f"    --val_file {output_dir}/verl_{args.dataset}_sie0.jsonl\n")
    
    print(f"  # Or mix multiple ratios:")
    print(f"  bash run_sie_grpo.sh \\")
    print(f"    --train_file {output_dir}/verl_{args.dataset}_sie50.jsonl,{output_dir}/verl_{args.dataset}_sie75.jsonl\n")


if __name__ == "__main__":
    main()