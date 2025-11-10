# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 推理脚本（适配你的目录）：
# - 传一个 VERL 的 run 根目录，比如 /exp/my_verl_run
# - 我们会自动找最新的 global_step_xxx/actor
# - 在 actor/ 下面找 model_world_size_4_rank_*.pt（找不到再去 actor/huggingface/）
# - 如果只有分片，没有 HF 的 config，就用 --base-model 来撑起结构
# - 支持 transformers / vLLM
# """
# import argparse
# import json
# from pathlib import Path
# import re
# import os
# import glob
# import torch

# # 可选 vllm
# try:
#     from vllm import LLM, SamplingParams
#     VLLM_AVAILABLE = True
# except Exception:
#     VLLM_AVAILABLE = False

# from transformers import AutoTokenizer, AutoModelForCausalLM


# # ===================== 1. 数据读取 ===================== #
# def load_data(jsonl_path):
#     data = []
#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)

#             q = (
#                 item.get("question")
#                 or item.get("query")
#                 or item.get("input")
#                 or item.get("extra_info", {}).get("question")
#             )

#             si = item.get("structured_context")
#             if si is None and "prompt" in item:
#                 si = "\n".join(
#                     f"{turn.get('role', 'user')}: {turn.get('content', '')}"
#                     for turn in item["prompt"]
#                 )

#             gt = (
#                 item.get("answer")
#                 or item.get("label")
#                 or item.get("extra_info", {}).get("answer")
#             )

#             if gt is None and "reward_model" in item:
#                 rm = item["reward_model"]
#                 if isinstance(rm, dict):
#                     gt_struct = rm.get("ground_truth") or {}
#                     if isinstance(gt_struct, dict):
#                         tgt = gt_struct.get("target")
#                         if isinstance(tgt, list) and tgt:
#                             gt = tgt[0]
#                         elif isinstance(tgt, str):
#                             gt = tgt

#             if q is None:
#                 continue

#             data.append(
#                 {
#                     "question": q,
#                     "structured_context": si or "",
#                     "answer": gt or "",
#                 }
#             )
#     return data


# # ===================== 2. prompt 构建 ===================== #
# def build_prompt(question: str, si: str, use_si: bool = True) -> str:
#     if use_si and si.strip():
#         return f"Question: {question}\nGiven the following information:\n{si}\nAnswer:"
#     else:
#         return f"Question: {question}\nAnswer:"


# # ===================== 3. VERL 目录解析 ===================== #
# def find_latest_actor_dir(path_str: str) -> str:
#     """
#     传进来一个 VERL run 的根，我们去下面找最大的 global_step_xxx/actor
#     如果本来就是一个 HF 模型目录，就原样返回
#     """
#     root = Path(path_str)
#     if not root.exists():
#         return path_str

#     candidates = []
#     for child in root.iterdir():
#         if child.is_dir() and child.name.startswith("global_step_"):
#             try:
#                 step = int(child.name.split("_")[-1])
#             except ValueError:
#                 continue
#             actor_dir = child / "actor"
#             if actor_dir.is_dir():
#                 candidates.append((step, actor_dir))

#     if not candidates:
#         # 说明给的就是普通 HF 模型目录
#         return path_str

#     candidates.sort(key=lambda x: x[0], reverse=True)
#     best_actor_dir = candidates[0][1]
#     print(f"[INFO] detected VERL-style checkpoint, using latest actor dir: {best_actor_dir}")
#     return str(best_actor_dir)


# def _to_plain_tensor(t):
#     # 把 DTensor 拆成普通 tensor
#     if hasattr(t, "to_local"):  # DTensor 通常有这个方法
#         return t.to_local()
#     return t

# def detect_sharded_verl_ckpt(actor_dir: str):
#     """
#     在 actor_dir 下面找分片；
#     如果没有，再去 actor_dir/huggingface/ 下面找
#     返回：(按 rank 排好序的文件列表, 实际所在目录)
#     """
#     pat1 = os.path.join(actor_dir, "model_world_size_*_rank_*.pt")
#     files = glob.glob(pat1)
#     if files:
#         return _sort_by_rank(files), actor_dir

#     pat2 = os.path.join(actor_dir, "huggingface", "model_world_size_*_rank_*.pt")
#     files = glob.glob(pat2)
#     if files:
#         return _sort_by_rank(files), os.path.join(actor_dir, "huggingface")

#     return [], actor_dir

# def _sort_by_rank(files):
#     """
#     从文件名里把 rank_x 拿出来排序
#     """
#     def get_rank(p):
#         m = re.search(r"rank_(\d+)\.pt$", os.path.basename(p))
#         return int(m.group(1)) if m else 0
#     return sorted(files, key=get_rank)

# def load_verl_sharded_state_dict(actor_dir: str):
#     """
#     读取 VERL 的多 rank ckpt，
#     按 rank 顺序把同名参数在 dim=0 上拼回一个完整的 tensor
#     """
#     shard_paths, real_dir = detect_sharded_verl_ckpt(actor_dir)
#     if not shard_paths:
#         return None, actor_dir  # 不是分片格式

#     print(f"[INFO] found {len(shard_paths)} sharded model file(s) under: {real_dir}")
#     for p in shard_paths:
#         print(f"       - {p}")

#     # 1. 全部load进来
#     shard_sds = []
#     for p in shard_paths:
#         sd = torch.load(p, map_location="cpu")
#         # 有些会是 {"model": {...}}
#         if "model" in sd and isinstance(sd["model"], dict):
#             sd = sd["model"]
#         shard_sds.append(sd)

#     # 2. 以第0份的 key 为基准，逐个key去每一份里拿，对应维度上拼
#     merged = {}
#     first_keys = shard_sds[0].keys()
#     world_size = len(shard_sds)

#     for k in first_keys:
#         parts = []
#         for sd in shard_sds:
#             v = sd[k]
#             v = _to_plain_tensor(v).cpu()
#             parts.append(v)

#         # 如果只有一份，或者这个参数是标量，就直接取第一份
#         if world_size == 1 or parts[0].ndim == 0:
#             merged[k] = parts[0]
#             continue

#         # 尝试在 dim=0 上拼接
#         try:
#             merged[k] = torch.cat(parts, dim=0)
#         except Exception:
#             # 有些参数可能根本没被拆（比如单个标量、位置编码等），就拿第0份
#             merged[k] = parts[0]

#     print(f"[INFO] merged {len(merged)} tensors from {world_size} shards (concat on dim=0)")
#     return merged, real_dir



# # ===================== 5. 构建模型 ===================== #
# def build_vllm(model_path: str):
#     llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16")
#     return llm


# def build_hf_from_dir(model_path: str):
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map="auto",
#         torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#         trust_remote_code=True,
#     )
#     return tokenizer, model


# def build_hf_from_base_and_load_shards(base_model: str, shards_state_dict: dict):
#     print(f"[INFO] building base model from: {base_model}")
#     tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         base_model,
#         device_map="auto",
#         torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#         trust_remote_code=True,
#     )

#     missing, unexpected = model.load_state_dict(shards_state_dict, strict=False)
#     if missing:
#         print(f"[WARN] missing keys: {len(missing)} (showing first 20): {missing[:20]}")
#     if unexpected:
#         print(f"[WARN] unexpected keys: {len(unexpected)} (showing first 20): {unexpected[:20]}")

#     model.eval()
#     return tokenizer, model


# # ===================== 6. 生成 ===================== #
# def vllm_generate(llm, prompt: str, max_tokens: int = 64) -> str:
#     params = SamplingParams(
#         max_tokens=max_tokens,
#         temperature=0.0,
#         top_p=1.0,
#     )
#     out = llm.generate([prompt], params)[0].outputs[0].text
#     return out


# def hf_generate(tokenizer, model, prompt: str, max_tokens: int = 64) -> str:
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         out_ids = model.generate(
#             **inputs,
#             max_new_tokens=max_tokens,
#             do_sample=False,
#             temperature=0.0,
#             pad_token_id=tokenizer.eos_token_id,
#         )
#     text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
#     return text[len(prompt):].strip()


# # ===================== 7. 答案 & 指标 ===================== #
# ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)

# def extract_simple_answer(text: str) -> str:
#     if not text:
#         return ""
#     m = ANSWER_RE.search(text)
#     if m:
#         return m.group(1).strip()
#     line = text.strip().splitlines()[0]
#     if "Answer:" in line:
#         line = line.split("Answer:", 1)[1].strip()
#     return line.strip()


# def compute_accuracy(preds, trues):
#     correct = 0
#     for p, t in zip(preds, trues):
#         if p is not None and t is not None and p.strip() == t.strip():
#             correct += 1
#     return correct / len(trues) if trues else 0.0


# # ===================== 8. main ===================== #
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", required=True,
#                         help="可以是 HF 模型目录，也可以是 VERL 的 run 根目录")
#     parser.add_argument("--base-model", default=None,
#                         help="当 actor 目录里只有分片时，用这个 HF 模型名/路径作骨架")
#     parser.add_argument("--input-jsonl", required=True)
#     parser.add_argument("--output-file", required=True)
#     parser.add_argument("--use-vllm", action="store_true")
#     parser.add_argument("--max-new-tokens", type=int, default=64)
#     parser.add_argument("--no-si", action="store_true")
#     args = parser.parse_args()

#     data = load_data(args.input_jsonl)
#     print(f"[INFO] loaded {len(data)} samples from {args.input_jsonl}")

#     # 1) 把 VERL 根 → 最新的 actor 目录
#     actor_dir = find_latest_actor_dir(args.model_path)

#     # 2) 看看 actor 目录里是不是分片
#     shard_sd, real_dir = load_verl_sharded_state_dict(actor_dir)

#     # 3) 构建模型
#     use_vllm = args.use_vllm and VLLM_AVAILABLE
#     if use_vllm:
#         if shard_sd is not None:
#             raise RuntimeError("检测到是分片checkpoint，暂不直接支持 vLLM，请先合并或用 --base-model 用 transformers 跑。")
#         print(f"[INFO] using vLLM, model={real_dir}")
#         llm = build_vllm(real_dir)
#         tokenizer = model = None
#     else:
#         if shard_sd is not None:
#             if args.base_model is None:
#                 raise RuntimeError(
#                     "actor/ 下面只有分片（model_world_size_...），没有完整 HF 结构，"
#                     "请加 --base-model <hf_model_name_or_path>"
#                 )
#             tokenizer, model = build_hf_from_base_and_load_shards(args.base_model, shard_sd)
#         else:
#             # 说明 actor_dir 本身就是 HF 模型目录（或者有 actor/huggingface/{config.json,...} 这种）
#             print(f"[INFO] using transformers, model={real_dir}")
#             tokenizer, model = build_hf_from_dir(real_dir)
#         llm = None

#     # 4) 真正推理
#     out_path = Path(args.output_file)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     preds, trues = [], []

#     with out_path.open("w", encoding="utf-8") as f:
#         for i, ex in enumerate(data):
#             q = ex["question"]
#             si = ex["structured_context"]
#             gt = ex["answer"]

#             prompt = build_prompt(q, si, use_si=not args.no_si)

#             if use_vllm:
#                 gen = vllm_generate(llm, prompt, max_tokens=args.max_new_tokens)
#             else:
#                 gen = hf_generate(tokenizer, model, prompt, max_tokens=args.max_new_tokens)

#             pred = extract_simple_answer(gen)

#             preds.append(pred)
#             trues.append(gt)

#             f.write(f"Q: {q}\n")
#             if not args.no_si and si.strip():
#                 f.write(f"SI: {si}\n")
#             f.write(f"Pred: {pred}\n")
#             f.write(f"GT: {gt}\n")
#             f.write("-" * 50 + "\n")

#             if (i + 1) % 50 == 0:
#                 print(f"[INFO] processed {i+1}/{len(data)}")

#     acc = compute_accuracy(preds, trues)
#     print(f"[RESULT] Accuracy (strict string match): {acc:.4f}")
#     print(f"[RESULT] saved to {out_path}")


# if __name__ == "__main__":
#     main()





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次跑多个 SIE 测试集的推理脚本
- 适配 VERL 的分片 checkpoint (model_world_size_4_rank_*.pt)
- 一次加载模型，循环跑多个 jsonl
- 每个输入文件会生成一个同名的 .pred.jsonl，便于后续分析
"""
import argparse
import json
from pathlib import Path
import re
import os
import glob
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
                si = "\n".join(
                    f"{turn.get('role', 'user')}: {turn.get('content', '')}"
                    for turn in item["prompt"]
                )

            gt = (
                item.get("answer")
                or item.get("label")
                or item.get("extra_info", {}).get("answer")
            )

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
                    "structured_context": si or "",
                    "answer": gt or "",
                }
            )
    return data

# def load_data(jsonl_path):
#     data = []
#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line)

#             # 1) 先按常规字段取
#             q = (
#                 item.get("question")
#                 or item.get("query")
#                 or item.get("input")
#                 or item.get("extra_info", {}).get("question")
#             )

#             # 2) 如果还没有，就看看是不是 MATH 这种只给了 prompt 的
#             if q is None and "prompt" in item and isinstance(item["prompt"], list) and item["prompt"]:
#                 # 通常最后一条 user 就是要问的问题
#                 q = item["prompt"][-1].get("content", "").strip()

#             # structured info 默认空
#             si = item.get("structured_context") or ""

#             # 3) 答案
#             gt = (
#                 item.get("answer")
#                 or item.get("label")
#                 or item.get("extra_info", {}).get("answer")
#             )

#             # SIE reward_model 结构
#             if gt is None and "reward_model" in item:
#                 rm = item["reward_model"]
#                 if isinstance(rm, dict):
#                     gt_struct = rm.get("ground_truth") or {}
#                     if isinstance(gt_struct, dict):
#                         tgt = gt_struct.get("target")
#                         if isinstance(tgt, list) and tgt:
#                             gt = tgt[0]
#                         elif isinstance(tgt, str):
#                             gt = tgt

#             if q is None:
#                 # 到这还没有 question，就真没法做了
#                 continue

#             data.append(
#                 {
#                     "question": q,
#                     "structured_context": si,
#                     "answer": gt or "",
#                 }
#             )
#     return data


# =============== 2. prompt =============== #
def build_prompt(question: str, si: str, use_si: bool = True) -> str:
    parts = []
    parts.append("You are a QA model.")
    parts.append(f"Question: {question}")
    if use_si and si.strip():
        parts.append("Given the following information:")
        parts.append(si)
    parts.append(
        "Please think step by step, and put ALL your reasoning ONLY inside <think>...</think>.\n"
        "Then output the final answer ONLY inside <answer>...</answer>.\n"
        "If there are multiple correct items, output them separated by comma inside <answer>.\n"
        "Do NOT add anything else."
    )
    parts.append("<think>")
    # 模型会在这里写想法
    parts.append("</think>")
    parts.append("<answer>")
    return "\n".join(parts)


# =============== 3. VERL 目录解析 & 分片合并 =============== #
def find_latest_actor_dir(path_str: str) -> str:
    root = Path(path_str)
    if not root.exists():
        return path_str

    # 如果直接给的是 global_step_xxx
    if root.is_dir() and root.name.startswith("global_step_"):
        actor_dir = root / "actor"
        if actor_dir.is_dir():
            print(f"[INFO] detected single step dir, using actor dir: {actor_dir}")
            return str(actor_dir)
        return path_str

    candidates = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("global_step_"):
            try:
                step = int(child.name.split("_")[-1])
            except ValueError:
                continue
            actor_dir = child / "actor"
            if actor_dir.is_dir():
                candidates.append((step, actor_dir))

    if not candidates:
        return path_str

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_actor_dir = candidates[0][1]
    print(f"[INFO] detected VERL-style checkpoint, using latest actor dir: {best_actor_dir}")
    return str(best_actor_dir)


def _to_plain_tensor(t):
    if hasattr(t, "to_local"):
        return t.to_local()
    return t


def _sort_by_rank(files):
    def get_rank(p):
        m = re.search(r"rank_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(files, key=get_rank)


def detect_sharded_verl_ckpt(actor_dir: str):
    pat1 = os.path.join(actor_dir, "model_world_size_*_rank_*.pt")
    files = glob.glob(pat1)
    if files:
        return _sort_by_rank(files), actor_dir

    pat2 = os.path.join(actor_dir, "huggingface", "model_world_size_*_rank_*.pt")
    files = glob.glob(pat2)
    if files:
        return _sort_by_rank(files), os.path.join(actor_dir, "huggingface")

    return [], actor_dir


def load_verl_sharded_state_dict(actor_dir: str):
    shard_paths, real_dir = detect_sharded_verl_ckpt(actor_dir)
    if not shard_paths:
        return None, actor_dir

    print(f"[INFO] found {len(shard_paths)} sharded model file(s) under: {real_dir}")
    for p in shard_paths:
        print(f"       - {p}")

    shard_sds = []
    for p in shard_paths:
        sd = torch.load(p, map_location="cpu")
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        shard_sds.append(sd)

    merged = {}
    first_keys = shard_sds[0].keys()
    world_size = len(shard_sds)

    for k in first_keys:
        parts = []
        for sd in shard_sds:
            v = sd[k]
            v = _to_plain_tensor(v).cpu()
            parts.append(v)

        if world_size == 1 or parts[0].ndim == 0:
            merged[k] = parts[0]
            continue

        try:
            merged[k] = torch.cat(parts, dim=0)
        except Exception:
            merged[k] = parts[0]

    print(f"[INFO] merged {len(merged)} tensors from {world_size} shards (concat on dim=0)")
    return merged, real_dir


# =============== 4. 构建模型 =============== #
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


def build_hf_from_base_and_load_shards(base_model: str, shards_state_dict: dict):
    print(f"[INFO] building base model from: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    missing, unexpected = model.load_state_dict(shards_state_dict, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)} (showing first 20): {missing[:20]}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)} (showing first 20): {unexpected[:20]}")

    model.eval()
    return tokenizer, model


# =============== 5. 生成 =============== #
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


# =============== 6. 答案 & 指标 =============== #
import re

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

def normalize_text(s: str) -> str:
    s = s.strip()
    # 去外层引号
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    s = s.rstrip(" .")
    return s.lower()

def parse_answer_to_list(raw: str):
    if not raw:
        return []

    # 1) 干掉 <think>...</think>
    raw = THINK_TAG_RE.sub("", raw)

    # 2) 抓 <answer>...</answer>
    m = ANSWER_TAG_RE.search(raw)
    if m:
        ans_part = m.group(1).strip()
    else:
        # 没有 tag，就取最后一行再去前缀
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        ans_part = lines[-1] if lines else ""
        for prefix in ["final answer:", "answer:", "prediction:", "pred:"]:
            if ans_part.lower().startswith(prefix):
                ans_part = ans_part[len(prefix):].strip()
                break

    if not ans_part:
        return []

    # 3) 拆成多项
    ans_part = ans_part.replace("|", ",")
    items = [normalize_text(x) for x in ans_part.split(",") if x.strip()]

    # 去重
    seen, out = set(), []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out

def pick_best_for_display(pred_list, gold):
    """
    pred_list: 解析出来的 ['timor-leste', 'mozambique</answer>']
    gold: 原始的 gold，可以是 str 或 list
    返回：最合适拿来写到文件里的那个字符串
    """
    # 统一 gold
    if isinstance(gold, list):
        gold_list = [normalize_text(x) for x in gold]
    else:
        gold_list = [normalize_text(str(gold))]

    # 先看有没有正好包含的
    for g in gold_list:
        for p in pred_list:
            pn = normalize_text(p)
            if g == pn or g in pn:
                return p  # 用原样的 p，别把用户看到的全变小写

    # 实在没有，就用第一个
    return pred_list[0] if pred_list else ""

def is_match(gold_list, pred_list):
    """
    gold_list/pred_list 都是小写的字符串列表
    规则：
    1. 如果 gold 的每一项都在 pred_list 里 → ✅
    2. 如果 gold 只有一项，且 pred_list 里有一项包含它 → ✅
    """
    if gold_list and all(g in pred_list for g in gold_list):
        return True

    if len(gold_list) == 1:
        g = gold_list[0]
        for p in pred_list:
            if g in p:
                return True

    return False

def extract_simple_answer(text: str) -> str:
    """
    给文件里写一个干净的字段，用第一个解析出来的答案。
    真正算分还是用 compute_accuracy 里的那套更宽松逻辑。
    """
    items = parse_answer_to_list(text)
    return items[0] if items else ""

def compute_accuracy(preds, trues):
    """
    preds: list[str]  —— 模型生成的原始文本
    trues: list[str or list] —— 数据里的答案
    """
    correct = 0
    for p_raw, t_raw in zip(preds, trues):
        # gold 也转成 list
        if isinstance(t_raw, list):
            gold_list = [normalize_text(x) for x in t_raw]
        else:
            gold_list = [normalize_text(str(t_raw))]

        pred_list = parse_answer_to_list(p_raw)

        if is_match(gold_list, pred_list):
            correct += 1

    return correct / len(trues) if trues else 0.0


# =============== 7. main =============== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="可以是 HF 模型目录，也可以是 VERL 的 run 根目录")
    parser.add_argument("--base-model", default=None,
                        help="当 actor 目录里只有分片时，用这个 HF 模型名/路径作骨架")
    parser.add_argument("--input-files", nargs="+", required=True,
                        help="要评测的一批 jsonl 路径，空格分开")
    parser.add_argument("--output-dir", required=True,
                        help="输出目录，每个输入会在这里生成一个 .pred.jsonl")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-si", action="store_true")
    args = parser.parse_args()

    # 1) 加载模型（只做一次）
    actor_dir = find_latest_actor_dir(args.model_path)
    shard_sd, real_dir = load_verl_sharded_state_dict(actor_dir)

    use_vllm = args.use_vllm and VLLM_AVAILABLE
    if use_vllm:
        if shard_sd is not None:
            raise RuntimeError("检测到是分片checkpoint，暂不直接支持 vLLM，请先合并或用 --base-model 用 transformers 跑。")
        print(f"[INFO] using vLLM, model={real_dir}")
        llm = build_vllm(real_dir)
        tokenizer = model = None
    else:
        if shard_sd is not None:
            if args.base_model is None:
                raise RuntimeError(
                    "actor/ 下面只有分片（model_world_size_...），没有完整 HF 结构，"
                    "请加 --base-model <hf_model_name_or_path>"
                )
            tokenizer, model = build_hf_from_base_and_load_shards(args.base_model, shard_sd)
        else:
            print(f"[INFO] using transformers, model={real_dir}")
            tokenizer, model = build_hf_from_dir(real_dir)
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
                    gen = hf_generate(tokenizer, model, prompt, max_tokens=args.max_new_tokens)

                # 1) 原始输出，给 compute_accuracy 用
                preds.append(gen)
                trues.append(gt)

                # 2) 解析成list
                parsed_list = parse_answer_to_list(gen)

                # 3) 根据 gold 选一个最好看的来写文件
                display_pred = pick_best_for_display(parsed_list, gt)

                wf.write(json.dumps({
                    "question": q,
                    "structured_context": si,
                    "gold": gt,
                    "pred": display_pred,       # 文件里的是干净结果
                    "raw_pred": gen,          # 可选：把模型全量输出也存下来
                }, ensure_ascii=False) + "\n")

                if (i + 1) % 50 == 0:
                    print(f"[INFO] {input_path.name}: processed {i+1}/{len(data)}")

        acc = compute_accuracy(preds, trues)
        print(f"[RESULT] {input_path.name}: Accuracy = {acc:.4f}")
        print(f"[RESULT] saved to {out_file}")

    print("[INFO] all datasets done.")


if __name__ == "__main__":
    main()