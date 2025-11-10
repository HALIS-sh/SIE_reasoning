#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path

# 你的 kk 结果文件
KK_PRED_PATH = "/data/sunwenhe/SIE_reasoning/baseline/preds_dump_sie_grpo_r100/kk_test_easy.pred.jsonl"
# 如果还有 hard 的，就再跑一次或者改成 glob，这里先写死一个

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
# 形如 "Amelia is a knight" / "Jacob is a knave"
ROLE_SENT_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*)\s+is\s+a\s+(knight|knave)\b",
    re.IGNORECASE,
)


def clean_text(s: str) -> str:
    return s.strip()


def strip_answer_tag(text: str) -> str:
    m = ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    # 去掉尾巴的 </answer>
    text = re.sub(r"</answer>\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def parse_roles_from_gold(gold_val):
    """
    gold 可能是 list 也可能是 str
    目标：返回 {name_lower: "knight"/"knave"}
    """
    if isinstance(gold_val, list):
        text = "\n".join(str(x) for x in gold_val)
    else:
        text = str(gold_val)

    roles = {}
    for m in ROLE_SENT_RE.finditer(text):
        name = m.group(1).lower()
        role = m.group(2).lower()
        roles[name] = role
    return roles


def parse_roles_from_pred(pred_val):
    """
    pred 可能是：
    - "Amelia,knave,Jacob,knave</answer>"
    - "<answer>Amelia is a knight; Jacob is a knave</answer>"
    - "Amelia is a knight"
    尽量都吃掉，返回 {name: role}
    """
    text = strip_answer_tag(str(pred_val))

    roles = {}

    # 1) 先抓句式的
    for m in ROLE_SENT_RE.finditer(text):
        name = m.group(1).lower()
        role = m.group(2).lower()
        roles[name] = role

    # 2) 如果句式没抓到，再试逗号配对的形式
    if not roles:
        # 用 , ; 换成统一分隔
        tmp = re.sub(r"[;|]", ",", text)
        parts = [p.strip().lower() for p in tmp.split(",") if p.strip()]
        # 两两成对：name,role,name,role,...
        if len(parts) >= 2:
            for i in range(0, len(parts) - 1, 2):
                name = parts[i]
                role = parts[i + 1]
                # role 里有可能带 </answer> 已经被上面去掉了
                # 只接受 knight/knave
                if role.startswith("knight"):
                    roles[name] = "knight"
                elif role.startswith("knave"):
                    roles[name] = "knave"

    return roles


def is_correct(gold_roles: dict, pred_roles: dict) -> bool:
    """
    gold 里的每个人，pred 都要给出同样的角色才算对
    """
    for name, role in gold_roles.items():
        pred_role = pred_roles.get(name)
        if pred_role is None:
            return False
        if pred_role != role:
            return False
    return True


def main():
    path = Path(KK_PRED_PATH)
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

            gold_val = item.get("gold", "")
            pred_val = item.get("pred") or item.get("raw_pred") or ""

            gold_roles = parse_roles_from_gold(gold_val)
            pred_roles = parse_roles_from_pred(pred_val)

            total += 1
            if is_correct(gold_roles, pred_roles):
                correct += 1

    acc = correct / total if total else 0.0
    print(f"total(valid): {total}")
    print(f"correct: {correct}")
    print(f"accuracy: {acc:.4f}")
    if bad_json:
        print(f"bad json lines: {bad_json}")


if __name__ == "__main__":
    main()