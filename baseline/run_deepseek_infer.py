#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Call DeepSeek (OpenAI-style) API on SIE-style jsonl datasets.

Example:
python run_deepseek_infer.py \
  --input-jsonl dump_sie/kgqa/cwq_0_test.jsonl \
  --output-file results/cwq_deepseek_direct.txt \
  --api-key sk-xxxxx \
  --model deepseek-chat

# CoT:
python run_deepseek_infer.py \
  --input-jsonl dump_sie/kgqa/cwq_0_test.jsonl \
  --output-file results/cwq_deepseek_cot.txt \
  --api-key sk-xxxxx \
  --model deepseek-chat \
  --cot
"""

import argparse
import json
import time
from pathlib import Path

import openai  # pip install openai>=1.40.0


def load_data(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            q = (
                item.get("question")
                or item.get("query")
                or item.get("input")
                or item.get("extra_info", {}).get("question")
            )

            si = item.get("structured_context")
            if si is None and "prompt" in item:
                # SIE 原来那种 list-of-turns
                si = "\n".join(
                    f"{turn.get('role', 'user')}: {turn.get('content', '')}"
                    for turn in item["prompt"]
                )

            gt = (
                item.get("answer")
                or item.get("label")
                or item.get("extra_info", {}).get("answer")
            )

            # reward_model 兜底
            if gt is None and "reward_model" in item:
                rm = item["reward_model"]
                if isinstance(rm, dict):
                    gt_list = rm.get("ground_truth", {}).get("target")
                    if isinstance(gt_list, list) and gt_list:
                        gt = gt_list[0]

            if q is None:
                continue

            data.append(
                {
                    "question": q,
                    "structured_context": si or "",
                    "answer": gt or "",
                }
            )
    return data


def build_prompt_direct(question: str, si: str, use_si: bool = True) -> str:
    if use_si and si.strip():
        return (
            "You are a QA assistant.\n"
            f"Question: {question}\n"
            "You are given the following information:\n"
            f"{si}\n"
            "Give the final answer only."
        )
    else:
        return (
            "You are a QA assistant.\n"
            f"Question: {question}\n"
            "Give the final answer only."
        )


def build_prompt_cot(question: str, si: str, use_si: bool = True) -> str:
    if use_si and si.strip():
        return (
            "You are a QA assistant.\n"
            "Please reason step by step using the given information and then give the final answer.\n"
            f"Question: {question}\n"
            "Information:\n"
            f"{si}\n"
            "First think, then answer."
        )
    else:
        return (
            "You are a QA assistant.\n"
            "Please reason step by step and then give the final answer.\n"
            f"Question: {question}\n"
            "First think, then answer."
        )


def extract_answer(text: str) -> str:
    """
    极简答案抽取：取第一行，如果有 'Answer:' 就去掉。
    你也可以换成你SIE的 <answer> 版本。
    """
    if not text:
        return ""
    line = text.strip().splitlines()[0]
    if "Answer:" in line:
        line = line.split("Answer:", 1)[1].strip()
    return line.strip()


def compute_accuracy(preds, trues):
    correct = 0
    total = len(trues)
    for p, t in zip(preds, trues):
        if p is not None and t is not None and p.strip() == t.strip():
            correct += 1
    return correct / total if total else 0.0


def call_deepseek(client, model: str, prompt: str, max_tokens: int = 128):
    # DeepSeek 是 OpenAI 格式的 chat
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, help="测试集jsonl路径")
    parser.add_argument("--output-file", required=True, help="结果输出路径")
    parser.add_argument("--api-key", required=True, help="DeepSeek API key")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model name")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="DeepSeek base url")
    parser.add_argument("--cot", action="store_true", help="是否使用CoT提示")
    parser.add_argument("--no-si", action="store_true", help="不拼接structured context")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sleep", type=float, default=0.4, help="每条之间sleep，防止限频")
    args = parser.parse_args()

    data = load_data(args.input_jsonl)
    print(f"[INFO] loaded {len(data)} samples.")

    # 配置 client
    client = openai.OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preds, trues = [], []

    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(data):
            q = ex["question"]
            si = ex["structured_context"]
            gt = ex["answer"]

            if args.cot:
                prompt = build_prompt_cot(q, si, use_si=not args.no_si)
            else:
                prompt = build_prompt_direct(q, si, use_si=not args.no_si)

            try:
                resp_text = call_deepseek(
                    client,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_new_tokens,
                )
            except Exception as e:
                print(f"[ERR] {i}: API error: {e}")
                resp_text = ""

            pred = extract_answer(resp_text)
            preds.append(pred)
            trues.append(gt)

            f.write(f"Q: {q}\n")
            if not args.no_si and si.strip():
                f.write(f"SI: {si}\n")
            f.write(f"Pred: {pred}\n")
            f.write(f"GT: {gt}\n")
            f.write("-" * 50 + "\n")

            if (i + 1) % 20 == 0:
                print(f"[INFO] processed {i+1}/{len(data)}")

            # 简单限频
            time.sleep(args.sleep)

    acc = compute_accuracy(preds, trues)
    print(f"[RESULT] Accuracy (strict): {acc:.4f}")
    print(f"[RESULT] saved to {out_path}")


if __name__ == "__main__":
    main()