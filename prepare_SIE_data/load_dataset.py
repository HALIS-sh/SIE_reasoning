#!/usr/bin/env python3
"""
load_and_dump_rog.py

加载：
- rmanluo/RoG-cwq
- rmanluo/RoG-webqsp

并把各自的 train split 存到 /data/wenhesun/datasets 目录下：
- /data/wenhesun/datasets/RoG-cwq-train.jsonl
- /data/wenhesun/datasets/RoG-webqsp-train.jsonl
"""

import os
import json
import argparse
from datasets import load_dataset

CWQ_NAME = "rmanluo/RoG-cwq"
WEBQSP_NAME = "rmanluo/RoG-webqsp"
DEFAULT_OUTDIR = "/data/sunwenhe/datasets"


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def dump_jsonl(dataset_split, out_path: str):
    """把 HuggingFace Dataset split 按行写成 JSONL"""
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in dataset_split:
            # 注意：ex 本身就是一个 dict，可以直接 dump
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")


def peek_dataset(ds, name: str, k: int = 3):
    print(f"\n===== {name} =====")
    print(ds)
    print(f"showing first {k} examples from 'train':")
    for i in range(min(k, len(ds["train"]))):
        ex = ds["train"][i]
        print(f"\n--- {name} train[{i}] ---")
        for key, val in ex.items():
            s = str(val)
            # 避免 seed 图太长刷屏
            if len(s) > 500:
                s = s[:500] + " ... <truncated>"
            print(f"{key}: {s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help="保存数据的目录，默认 /data/wenhesun/datasets",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=3,
        help="控制打印多少条样本看看结构",
    )
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # 1) CWQ
    cwq = load_dataset(CWQ_NAME)
    peek_dataset(cwq, CWQ_NAME, k=args.show)
    cwq_out = os.path.join(args.outdir, "RoG-cwq-train.jsonl")
    dump_jsonl(cwq["train"], cwq_out)
    print(f"[saved] CWQ train -> {cwq_out}")

    # 2) WebQSP
    webqsp = load_dataset(WEBQSP_NAME)
    peek_dataset(webqsp, WEBQSP_NAME, k=args.show)
    webqsp_out = os.path.join(args.outdir, "RoG-webqsp-train.jsonl")
    dump_jsonl(webqsp["train"], webqsp_out)
    print(f"[saved] WebQSP train -> {webqsp_out}")


if __name__ == "__main__":
    main()