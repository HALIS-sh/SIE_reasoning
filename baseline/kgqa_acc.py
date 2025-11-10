#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path

# 改成你的kgqa预测结果文件
KGQA_PRED_PATH = "/data/sunwenhe/SIE_reasoning/baseline/preds_dump_sie_grpo_r100_base_2/cwq_0_test.pred.jsonl"

# 去掉 <answer> ... </answer>
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)

def normalize(s: str) -> str:
    # 全小写、去空格、去末尾的句号
    return s.strip().lower().rstrip(" .")

def strip_answer_tags(text: str) -> str:
    text = text.strip()
    m = ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    # 常见情况：xxx</answer>
    text = re.sub(r"</answer>\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()

def parse_gold(gold_val):
    """
    gold 可能是 ["Judaism"] 也可能是 "Judaism"
    我们最后统一成一个 list[str]
    """
    if isinstance(gold_val, list):
        return [normalize(str(x)) for x in gold_val if str(x).strip()]
    else:
        return [normalize(str(gold_val))] if str(gold_val).strip() else []

def parse_pred(pred_val):
    """
    pred 一般是 "Orthodox Judaism</answer>"
    先去掉标签，然后normalize
    """
    text = strip_answer_tags(str(pred_val))
    return normalize(text)

def is_match(gold_list, pred_str):
    """
    规则：只要 pred 中包含 gold，或 gold 中包含 pred 就对
    gold_list 里只要有一个能匹配就行
    """
    if not gold_list:
        return False
    if not pred_str:
        return False

    for g in gold_list:
        if g == pred_str:
            return True
        if g in pred_str:
            return True
        if pred_str in g:
            return True
    return False

def main():
    path = Path(KGQA_PRED_PATH)
    total = 0
    correct = 0
    bad_json = 0

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] bad json at line {lineno}: {e}")
                bad_json += 1
                continue

            gold_raw = item.get("gold", "")
            pred_raw = item.get("pred") or item.get("raw_pred") or ""

            gold_items = parse_gold(gold_raw)
            pred_item = parse_pred(pred_raw)

            total += 1
            if is_match(gold_items, pred_item):
                correct += 1

    acc = correct / total if total else 0.0
    print(f"total(valid): {total}")
    print(f"correct: {correct}")
    print(f"accuracy: {acc:.4f}")
    if bad_json:
        print(f"bad json lines: {bad_json}")

if __name__ == "__main__":
    main()