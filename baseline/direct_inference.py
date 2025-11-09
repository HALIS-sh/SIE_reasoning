#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
import torch

# 可选的 vllm
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_data(jsonl_path):
    """读取我们之前导出的 SIE jsonl，做字段兜底"""
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
                # prompt 里是多轮结构的情况
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
                    gt = (
                        rm.get("ground_truth", {})
                        .get("target", [None])[0]
                    )

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


def build_prompt(question: str, si: str, use_si: bool = True) -> str:
    """最简单的 QA prompt，不加 CoT，不加标签"""
    if use_si and si.strip():
        return f"Question: {question}\nGiven the following information:\n{si}\nAnswer:"
    else:
        return f"Question: {question}\nAnswer:"


def build_vllm(model_path: str):
    llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16")
    return llm


def build_hf(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    return tokenizer, model


def vllm_generate(llm, prompt: str, max_tokens: int = 64) -> str:
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    out = llm.generate([prompt], params)[0].outputs[0].text
    return out


def hf_generate(tokenizer, model, prompt: str, max_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    # 把原 prompt 切掉，只看模型新生成的部分
    return text[len(prompt):].strip()


def extract_simple_answer(text: str) -> str:
    """不加标签的话，就取第一行 / 第一段"""
    if not text:
        return ""
    # 先按换行切
    line = text.strip().splitlines()[0]
    # 有时候模型会说 "Answer: xxx"
    if "Answer:" in line:
        line = line.split("Answer:", 1)[1].strip()
    return line.strip()


def compute_accuracy(preds, trues):
    correct = 0
    for p, t in zip(preds, trues):
        if p is not None and t is not None and p.strip() == t.strip():
            correct += 1
    return correct / len(trues) if trues else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="本地SFT/RL模型目录，或HF模型名")
    parser.add_argument("--input-jsonl", required=True,
                        help="测试集jsonl路径")
    parser.add_argument("--output-file", required=True,
                        help="推理结果写到这里")
    parser.add_argument("--use-vllm", action="store_true",
                        help="使用vLLM推理")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-si", action="store_true",
                        help="不拼接structured context，只问question")
    args = parser.parse_args()

    data = load_data(args.input_jsonl)
    print(f"[INFO] loaded {len(data)} samples from {args.input_jsonl}")

    use_vllm = args.use_vllm and VLLM_AVAILABLE
    if use_vllm:
        print(f"[INFO] using vLLM, model={args.model_path}")
        llm = build_vllm(args.model_path)
        tokenizer = model = None
    else:
        print(f"[INFO] using transformers, model={args.model_path}")
        tokenizer, model = build_hf(args.model_path)
        llm = None

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preds, trues = [], []

    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(data):
            q = ex["question"]
            si = ex["structured_context"]
            gt = ex["answer"]

            prompt = build_prompt(q, si, use_si=not args.no_si)

            if use_vllm:
                gen = vllm_generate(llm, prompt, max_tokens=args.max_new_tokens)
            else:
                gen = hf_generate(tokenizer, model, prompt, max_tokens=args.max_new_tokens)

            pred = extract_simple_answer(gen)

            preds.append(pred)
            trues.append(gt)

            f.write(f"Q: {q}\n")
            if not args.no_si and si.strip():
                f.write(f"SI: {si}\n")
            f.write(f"Pred: {pred}\n")
            f.write(f"GT: {gt}\n")
            f.write("-" * 50 + "\n")

            if (i + 1) % 50 == 0:
                print(f"[INFO] processed {i+1}/{len(data)}")

    acc = compute_accuracy(preds, trues)
    print(f"[RESULT] Accuracy (strict string match): {acc:.4f}")
    print(f"[RESULT] saved to {out_path}")


if __name__ == "__main__":
    main()