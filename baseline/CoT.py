#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

import torch

# 尝试导入 vllm，用于高效推理
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_data(jsonl_path):
    """读取我们导出的 jsonl，兼容几种字段名"""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            # 1) question
            q = (
                item.get("question")
                or item.get("query")
                or item.get("input")
                or item.get("extra_info", {}).get("question")
            )

            # 2) structured context / KG facts
            # 你的原脚本里叫 structured_context，我们导出的 SIE 里一般是 prompt: [{role, content}, ...]
            si = item.get("structured_context")
            if si is None and "prompt" in item:
                # 把 messages 合成一段
                si = "\n".join(
                    f"{turn.get('role', 'user')}: {turn.get('content', '')}"
                    for turn in item["prompt"]
                )

            # 3) answer / ground truth
            gt = (
                item.get("answer")
                or item.get("label")
                or item.get("extra_info", {}).get("answer")
            )

            # 再兜底一下我们 SIE 格式的 reward_model
            if gt is None and "reward_model" in item:
                rm = item["reward_model"]
                # 有的写成 {'ground_truth': {'target': [...]}}
                if isinstance(rm, dict):
                    gt = (
                        rm.get("ground_truth", {})
                        .get("target", [None])[0]
                    )

            if q is None:
                # 实在没问题就跳过
                continue

            data.append(
                {
                    "question": q,
                    "structured_context": si or "",
                    "answer": gt or "",
                }
            )
    return data


def construct_cot_prompt(question, si):
    # 你之前的 prompt 我基本照搬，只是把 question 放到上面一点
    return (
        f"You are given a question and some structured information (SI). "
        f"Please reason step by step inside <think></think> and output final answer inside <answer></answer>.\n"
        f"Question: {question}\n"
        f"SI:\n{si}\n"
        f"<think>\n"
        f"</think>\n"
        f"<answer>"
    )


def build_vllm(model_path: str):
    llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16")  # 根据你机器调
    return llm


def build_hf(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def vllm_generate(llm, prompt: str, max_tokens: int = 128) -> str:
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )
    outputs = llm.generate([prompt], params)
    return outputs[0].outputs[0].text


def hf_generate(tokenizer, model, prompt: str, max_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text[len(prompt):]  # 只返回新增部分，方便后面切 <answer>


def extract_answer(text: str) -> str:
    if "<answer>" in text:
        part = text.split("<answer>", 1)[1]
        if "</answer>" in part:
            part = part.split("</answer>", 1)[0]
        return part.strip()
    return text.strip()


def compute_accuracy(preds, trues):
    correct = 0
    for p, t in zip(preds, trues):
        if p is None and t is None:
            correct += 1
        elif p is not None and t is not None and p.strip() == t.strip():
            correct += 1
    return correct / len(trues) if trues else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="本地RL/SFT模型目录，或者HF模型名，比如 Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--input-jsonl", required=True,
                        help="要跑推理的jsonl，比如 dump_sie/kgqa/cwq_0_test.jsonl")
    parser.add_argument("--output-file", required=True,
                        help="把推理结果写到这个文件")
    parser.add_argument("--use-vllm", action="store_true",
                        help="优先用vLLM推理")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    data = load_data(args.input_jsonl)
    print(f"[INFO] loaded {len(data)} examples from {args.input_jsonl}")

    # 构建模型
    use_vllm = args.use_vllm and VLLM_AVAILABLE
    if use_vllm:
        print(f"[INFO] using vLLM with model={args.model-path if hasattr(args, 'model-path') else args.model_path}")
        llm = build_vllm(args.model_path)
        tokenizer = None
        model = None
    else:
        print(f"[INFO] using transformers with model={args.model_path}")
        tokenizer, model = build_hf(args.model_path)
        llm = None

    preds, trues = [], []
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(data):
            q = ex["question"]
            si = ex["structured_context"]
            gt = ex["answer"]

            prompt = construct_cot_prompt(q, si)

            if use_vllm:
                gen = vllm_generate(llm, prompt, max_tokens=args.max_new_tokens)
            else:
                gen = hf_generate(tokenizer, model, prompt, max_tokens=args.max_new_tokens)

            pred = extract_answer(gen)
            preds.append(pred)
            trues.append(gt)

            f.write(f"Q: {q}\n")
            f.write(f"SI: {si}\n")
            f.write(f"Predicted: {pred}\n")
            f.write(f"Ground Truth: {gt}\n")
            f.write("-" * 50 + "\n")

            if (i + 1) % 50 == 0:
                print(f"[INFO] processed {i+1}/{len(data)}")

    acc = compute_accuracy(preds, trues)
    print(f"[RESULT] Accuracy: {acc:.4f}")
    print(f"[RESULT] saved to {out_path}")


if __name__ == "__main__":
    main()