#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次跑多个 SIE/MATH 测试集的推理脚本（只用本地 base 模型，不加载 VERL/RL 分片权重）
- 直接传一个 HF 本地模型目录，比如 /data/models/Qwen/Qwen2.5-7B-Instruct
- 一次加载模型，循环跑多个 jsonl
- 每个输入文件会生成一个同名的 .pred.jsonl
"""
import argparse
import json
from pathlib import Path
import re
import os
import torch

# 可选 vllm
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM


# =============== 1. 数据读取 =============== #
def load_data(jsonl_path):
    """兼容 SIE、KGQA、logic、math500 这几种结构"""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            # 先试常规字段
            q = (
                item.get("question")
                or item.get("query")
                or item.get("input")
                or item.get("extra_info", {}).get("question")
            )

            # MATH-500 这种只有 prompt 列表的，从最后一条 user 拿题面
            if q is None and "prompt" in item and isinstance(item["prompt"], list) and item["prompt"]:
                q = item["prompt"][-1].get("content", "").strip()

            si = item.get("structured_context") or ""

            gt = (
                item.get("answer")
                or item.get("label")
                or item.get("extra_info", {}).get("answer")
            )

            # SIE 的 reward_model 结构
            if gt is None and "reward_model" in item:
                rm = item["reward_model"]
                if isinstance(rm, dict):
                    gt_struct = rm.get("ground_truth") or {}
                    if isinstance(gt_struct, dict):
                        tgt = gt_struct.get("target")
                        if isinstance(tgt, list) and tgt:
                            gt = tgt[0]
                        elif isinstance(tgt, str):
                            gt = tgt

            if q is None:
                continue

            data.append(
                {
                    "question": q,
                    "structured_context": si,
                    "answer": gt or "",
                }
            )
    return data


# =============== 2. prompt =============== #
def build_prompt(question: str, si: str, use_si: bool = True) -> str:
    parts = []
    parts.append("You are a QA model.")
    parts.append(f"Question: {question}")
    if use_si and si.strip():
        parts.append("Given the following information:")
        parts.append(si)
    parts.append(
        "Output the final answer ONLY inside <answer>...</answer>.\n"
        "If there are multiple correct items, output them separated by comma inside <answer>.\n"
        "Do NOT add anything else."
    )
    parts.append("<think>")
    parts.append("</think>")
    parts.append("<answer>")
    return "\n".join(parts)


# =============== 3. 构建模型（只用 base 模型） =============== #
def build_vllm(model_path: str):
    llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16")
    return llm


def build_hf_from_dir(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    return tokenizer, model


# =============== 4. 生成 =============== #
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
    return text[len(prompt):].strip()


# =============== 5. 答案解析 & 指标 =============== #
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def normalize_text(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    s = s.rstrip(" .")
    return s.lower()


def parse_answer_to_list(raw: str):
    if not raw:
        return []

    # 去掉 <think>...</think>
    raw = THINK_TAG_RE.sub("", raw)

    m = ANSWER_TAG_RE.search(raw)
    if m:
        ans_part = m.group(1).strip()
    else:
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        ans_part = lines[-1] if lines else ""
        for prefix in ["final answer:", "answer:", "prediction:", "pred:"]:
            if ans_part.lower().startswith(prefix):
                ans_part = ans_part[len(prefix):].strip()
                break

    if not ans_part:
        return []

    ans_part = ans_part.replace("|", ",")
    items = [normalize_text(x) for x in ans_part.split(",") if x.strip()]

    seen, out = set(), []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def pick_best_for_display(pred_list, gold):
    if isinstance(gold, list):
        gold_list = [normalize_text(x) for x in gold]
    else:
        gold_list = [normalize_text(str(gold))]

    for g in gold_list:
        for p in pred_list:
            pn = normalize_text(p)
            if g == pn or g in pn or pn in g:
                return re.sub(r"</answer>\s*$", "", p, flags=re.IGNORECASE)

    if pred_list:
        return re.sub(r"</answer>\s*$", "", pred_list[0], flags=re.IGNORECASE)
    return ""


def is_match(gold_list, pred_list):
    # 多项 gold：每项都要能在 pred 里找到（互为子串也行）
    if len(gold_list) > 1:
        for g in gold_list:
            ok = False
            for p in pred_list:
                if g == p or g in p or p in g:
                    ok = True
                    break
            if not ok:
                return False
        return True

    # 单项 gold：有一个 pred 跟它互为子串就行
    if len(gold_list) == 1:
        g = gold_list[0]
        for p in pred_list:
            if g == p or g in p or p in g:
                return True
        return False

    return False


def compute_accuracy(preds, trues):
    correct = 0
    for p_raw, t_raw in zip(preds, trues):
        if isinstance(t_raw, list):
            gold_list = [normalize_text(x) for x in t_raw]
        else:
            gold_list = [normalize_text(str(t_raw))]

        pred_list = parse_answer_to_list(p_raw)

        if is_match(gold_list, pred_list):
            correct += 1

    return correct / len(trues) if trues else 0.0


# =============== 6. main =============== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        required=True,
        help="本地 HF 模型目录，比如 /data/models/Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="要评测的一批 jsonl 路径，空格分开",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出目录，每个输入会在这里生成一个 .pred.jsonl",
    )
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-si", action="store_true")
    args = parser.parse_args()

    # 1) 直接加载 base 模型
    use_vllm = args.use_vllm and VLLM_AVAILABLE
    if use_vllm:
        print(f"[INFO] using vLLM, model={args.base-model}")
        llm = build_vllm(args.base_model)
        tokenizer = model = None
    else:
        print(f"[INFO] using transformers, model={args.base_model}")
        tokenizer, model = build_hf_from_dir(args.base_model)
        llm = None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) 依次跑每个数据集
    for input_path in args.input_files:
        input_path = Path(input_path)
        data = load_data(str(input_path))
        print(f"[INFO] ===== dataset: {input_path.name} ({len(data)} samples) =====")

        preds, trues = [], []
        out_file = out_dir / f"{input_path.stem}.pred.jsonl"
        with out_file.open("w", encoding="utf-8") as wf:
            for i, ex in enumerate(data):
                q = ex["question"]
                si = ex["structured_context"]
                gt = ex["answer"]

                prompt = build_prompt(q, si, use_si=not args.no_si)

                if use_vllm:
                    gen = vllm_generate(llm, prompt, max_tokens=args.max_new_tokens)
                else:
                    gen = hf_generate(
                        tokenizer, model, prompt, max_tokens=args.max_new_tokens
                    )

                # 给准确率用原始输出
                preds.append(gen)
                trues.append(gt)

                # 给文件写一个好看的
                parsed_list = parse_answer_to_list(gen)
                display_pred = pick_best_for_display(parsed_list, gt)

                wf.write(
                    json.dumps(
                        {
                            "question": q,
                            "structured_context": si,
                            "gold": gt,
                            "pred": display_pred,
                            "raw_pred": gen,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                if (i + 1) % 50 == 0:
                    print(f"[INFO] {input_path.name}: processed {i+1}/{len(data)}")

        acc = compute_accuracy(preds, trues)
        print(f"[RESULT] {input_path.name}: Accuracy = {acc:.4f}")
        print(f"[RESULT] saved to {out_file}")

    print("[INFO] all datasets done.")


if __name__ == "__main__":
    main()