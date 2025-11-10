#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path

# PRED_PATH = "/data/sunwenhe/SIE_reasoning/baseline/preds_dump_sie_grpo_r100_base/gsm8k_test.pred.jsonl"
PRED_PATH = "/data/sunwenhe/SIE_reasoning/baseline/preds_dump_sie_grpo_r100_cot/gsm8k_test.pred.jsonl"
PRED_PATH = "/data/sunwenhe/SIE_reasoning/baseline/preds_dump_sie_grpo_r100_base/kk_test_easy.pred.jsonl"

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
HASH_ANSWER_RE = re.compile(r"####\s*([^\n#]+)")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_str(s: str) -> str:
    return s.strip().lower()


def try_to_number(s: str):
    s = s.strip().replace("$", "").replace(",", "")
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return None


def parse_gold_answer(gold_value):
    if isinstance(gold_value, list):
        text = "\n".join(str(x) for x in gold_value)
    else:
        text = str(gold_value)

    m = HASH_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()

    m = ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip()

    nums = NUMBER_RE.findall(text)
    if nums:
        return nums[-1].strip()

    return text.strip()


def parse_pred_answer(pred_value):
    text = str(pred_value).strip()

    m = ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip()

    text = re.sub(r"</answer>\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def is_same(gold_ans: str, pred_ans: str) -> bool:
    g_num = try_to_number(gold_ans)
    p_num = try_to_number(pred_ans)
    if g_num is not None and p_num is not None:
        return g_num == p_num

    g_norm = normalize_str(gold_ans)
    p_norm = normalize_str(pred_ans)

    if g_norm == p_norm:
        return True
    if g_norm in p_norm:
        return True
    if p_norm in g_norm:
        return True
    return False


def main():
    path = Path(PRED_PATH)
    total = 0
    correct = 0
    bad_lines = 0

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                bad_lines += 1
                # 打印出问题的行号和前 200 个字符，方便你用 sed 去看原始行
                print(f"[JSON ERROR] line {lineno}: {e}")
                print(f"[JSON ERROR] content head: {line[:200]!r}")
                continue

            gold_raw = item.get("gold", "")
            pred_raw = item.get("pred") or item.get("raw_pred") or ""

            gold_ans = parse_gold_answer(gold_raw)
            pred_ans = parse_pred_answer(pred_raw)

            total += 1
            if is_same(gold_ans, pred_ans):
                correct += 1

    acc = correct / total if total else 0.0
    print(f"total(valid json lines): {total}")
    print(f"correct: {correct}")
    print(f"accuracy: {acc:.4f}")
    print(f"bad lines (not json): {bad_lines}")


if __name__ == "__main__":
    main()