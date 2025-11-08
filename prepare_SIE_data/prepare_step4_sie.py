#!/usr/bin/env python3
"""
prepare_step4_sie.py

Step 4: Constructing Partial SIEs
参考论文公式 (5):
    SIE-ratio = Shuffle( Retain(G_support, ratio) ∪ G_distract )
其中 ratio ∈ {1.0, 0.75, 0.5, 0.25, 0.0}
paper: "we set a series of retention ratios at {100%, 75%, 50%, 25%, 0%}"  [oai_citation:1‡LEARNING TO REASON IN STRUCTURED IN-CONTEXTENVIRONMENTS WITH REINFORCEMENT LEARNING.pdf](sediment://file_00000000814471f78a095019f905366e)

输入：step3 产出的 jsonl，每条里至少有：
    question
    answer / answers / a_entity (任选其一)
    support_triples: [...]
    distract_triples: [...]
    no_support: 0/1     # 在 step3 里我们已经加上了

输出：会生成多个 jsonl，每个对应一个 ratio
    xxx-step4-sie100.jsonl
    xxx-step4-sie75.jsonl
    ...
"""

import os
import json
import argparse
import random

# 为了复现
random.seed(42)

DEFAULT_RATIOS = [1.0, 0.75, 0.5, 0.25, 0.0]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def retain_support(support_triples, ratio):
    """随机保留 support 中的一个比例"""
    if ratio >= 0.999:
        return list(support_triples)
    if not support_triples:
        return []
    k = max(1, int(len(support_triples) * ratio))
    # 打乱后取前 k 个
    idx = list(range(len(support_triples)))
    random.shuffle(idx)
    picked = [support_triples[i] for i in idx[:k]]
    return picked


def truncate_to_max(triples, max_facts):
    """如果想要控制总事实数，可以截一下"""
    if max_facts is None or max_facts <= 0:
        return triples
    if len(triples) <= max_facts:
        return triples
    return triples[:max_facts]


def process_step4(
    infile: str,
    outdir: str,
    ratios,
    limit: int = 0,
    max_facts: int = 0,
    show: int = 5,
):
    os.makedirs(outdir, exist_ok=True)

    # 打开多个输出文件，每个 ratio 一个
    writers = {}
    for r in ratios:
        # 例如 RoG-cwq-step3.jsonl -> RoG-cwq-step4-sie100.jsonl
        base = os.path.basename(infile)
        name, ext = os.path.splitext(base)
        r_int = int(r * 100)
        out_path = os.path.join(outdir, f"{name}-step4-sie{r_int}.jsonl")
        writers[r] = open(out_path, "w", encoding="utf-8")
        print(f"[info] will write ratio={r} -> {out_path}")

    for idx, sample in enumerate(load_jsonl(infile)):
        if limit and idx >= limit:
            break

        support = sample.get("support_triples", []) or []
        distract = sample.get("distract_triples", []) or []
        no_support = int(sample.get("no_support", 0))

        # 对每个 ratio 单独构造一份
        for r in ratios:
            # 如果本身就没有 support，就强制当成 0%
            if no_support == 1:
                chosen_support = []
            else:
                chosen_support = retain_support(support, r)

            # 合并后打乱
            merged = chosen_support + distract
            random.shuffle(merged)

            # 控制总事实数（可选）
            merged = truncate_to_max(merged, max_facts)

            # 构造一条新的样本
            # 保留原字段，额外加 structured_context / sie_ratio
            new_sample = dict(sample)
            new_sample["sie_ratio"] = r
            new_sample["structured_context"] = merged

            # 也可以保留一下实际用了多少 support
            new_sample["used_support_count"] = len(chosen_support)
            new_sample["used_distract_count"] = len(merged) - len(chosen_support)

            writers[r].write(json.dumps(new_sample, ensure_ascii=False) + "\n")

        if idx < show:
            print(f"\n--- sample {idx} ---")
            print("question:", sample.get("question"))
            print("support_total:", len(support))
            print("distract_total:", len(distract))
            print("no_support:", no_support)

    # 关文件
    for f in writers.values():
        f.close()
    print("[done] step4 files are ready.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, required=True,
                    help="step3 产出的 jsonl，比如 /.../RoG-cwq-step3.jsonl")
    ap.add_argument("--outdir", type=str, required=True,
                    help="step4 各个 ratio 输出目录")
    ap.add_argument("--ratios", type=str, default="1.0,0.75,0.5,0.25,0.0",
                    help="要生成的支持比例，逗号分隔，默认就是论文那五个")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 条，0 表示全部")
    ap.add_argument("--max_facts", type=int, default=0,
                    help="上下文中最多保留多少条 triple，0 表示不限制")
    ap.add_argument("--show", type=int, default=5,
                    help="打印前多少条看看结构")
    args = ap.parse_args()

    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]

    process_step4(
        infile=args.infile,
        outdir=args.outdir,
        ratios=ratios,
        limit=args.limit,
        max_facts=args.max_facts,
        show=args.show,
    )


if __name__ == "__main__":
    main()