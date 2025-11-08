#!/usr/bin/env python3
"""
prepare_step3_distract.py

实现论文 SIE pipeline 的 Step 3: Distractor Subgraph Filtering
基于 Step 2 生成的 jsonl（每条里已有 support_triples），做两阶段语义过滤：

1) Relation filtering:
    - 从 (G_seed \ G_support) 里收集所有关系
    - 用 cross-encoder "cross-encoder/ms-marco-MiniLM-L12-v2"
      跟 question 做相似度，取 top-k 关系，得到 rel_retain

2) Triple filtering:
    - 只保留关系在 rel_retain 里的那些三元组
    - 再次用同一个 cross-encoder 跟 question 打分
    - 取 top-n 三元组，就是最终的 G_distract
"""

import os
import json
import argparse
from typing import List, Any

from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L12-v2"


# ------------------- IO ------------------- #
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def as_triplet_list(tri):
    """把 dict 或 list 的 triple 统一成 [h, r, t]"""
    if isinstance(tri, dict):
        return [tri["head"], tri["rel"], tri["tail"]]
    return tri


# ------------------- core steps ------------------- #
def build_candidate_triples(seed_graph: List[Any],
                            support_triples: List[Any]) -> List[List[str]]:
    """
    seed_graph: 原始 G_seed (sample["graph"])
    support_triples: step2 识别出的 G_support (sample["support_triples"])
    返回: G_seed \ G_support
    """
    support_set = {tuple(as_triplet_list(t)) for t in support_triples}
    candidates = []
    for tri in seed_graph:
        tri_std = as_triplet_list(tri)
        if tuple(tri_std) not in support_set:
            candidates.append(tri_std)
    return candidates


def relation_filtering(question: str,
                       candidates: List[List[str]],
                       model: CrossEncoder,
                       topk_rel: int = 10) -> List[str]:
    """
    第一阶段：对所有 candidate triple 的 relation 做打分，取 topk
    """
    rels = []
    rel_set = set()
    for h, r, t in candidates:
        if r not in rel_set:
            rel_set.add(r)
            rels.append(r)

    if not rels:
        return []

    pairs = [(question, r) for r in rels]
    scores = model.predict(pairs)

    scored = list(zip(rels, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    kept = [r for (r, s) in scored[:topk_rel]]
    return kept


def triple_filtering(question: str,
                     candidates: List[List[str]],
                     kept_relations: List[str],
                     model: CrossEncoder,
                     topn_triples: int = 50) -> List[List[str]]:
    """
    第二阶段：只保留关系在 kept_relations 里的三元组，再按语义得分取 topn
    """
    kept_relations = set(kept_relations)
    filtered = [tri for tri in candidates if tri[1] in kept_relations]
    if not filtered:
        return []

    pairs = []
    for h, r, t in filtered:
        txt = f"{h} [{r}] {t}"
        pairs.append((question, txt))

    scores = model.predict(pairs)
    scored = list(zip(filtered, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    final_triples = [tri for (tri, s) in scored[:topn_triples]]
    return final_triples


# ------------------- main process ------------------- #
def process_file(in_path: str,
                 out_path: str,
                 model_name: str,
                 topk_rel: int,
                 topn_triples: int,
                 show: int,
                 limit: int):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[info] loading cross-encoder: {model_name}")
    model = CrossEncoder(model_name)

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, sample in enumerate(load_jsonl(in_path)):
            if limit and idx >= limit:
                break

            question = sample.get("question", "")
            seed_graph = sample.get("graph", [])
            support_triples = sample.get("support_triples", [])

            # 标记：这条是不是“没有支持子图”的样本
            no_support = int(len(support_triples) == 0)

            # 1) seed \ support
            candidates = build_candidate_triples(seed_graph, support_triples)

            # 2) relation filtering
            kept_rel = relation_filtering(
                question,
                candidates,
                model,
                topk_rel=topk_rel,
            )

            # 3) triple filtering
            distract_triples = triple_filtering(
                question,
                candidates,
                kept_rel,
                model,
                topn_triples=topn_triples,
            )

            sample["distract_triples"] = distract_triples
            sample["rel_retain"] = kept_rel  # 调试方便，看完可以删
            sample["no_support"] = no_support  # ★ 新增字段

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

            if idx < show:
                print(f"\n--- sample {idx} ---")
                print("question:", question)
                print("support_triples:", len(support_triples))
                print("no_support:", no_support)
                print("candidate_count:", len(candidates))
                print("kept_rel:", kept_rel)
                print("distract_triples_count:", len(distract_triples))

    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    # 新增：直接指定输入输出文件，方便你测 tmp 那份
    parser.add_argument(
        "--infile",
        type=str,
        help="step2 的 jsonl 文件路径，比如 /.../RoG-cwq-step2.jsonl"
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="step3 输出的 jsonl 文件路径"
    )

    # 保留老的写法：indir + dataset
    parser.add_argument(
        "--indir",
        type=str,
        default="/data/wenhesun/datasets",
        help="如果不指定 infile，就用这个目录 + 数据集名来推文件名"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/data/wenhesun/datasets",
        help="如果不指定 outfile，就写到这个目录"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cwq",
        choices=["cwq", "webqsp"],
        help="处理哪个数据集（只在你没给 infile/outfile 时起作用）"
    )

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--topk_rel", type=int, default=10)
    parser.add_argument("--topn_triples", type=int, default=50)
    parser.add_argument("--show", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条，0 表示全部")
    args = parser.parse_args()

    # 决定输入输出路径
    if args.infile:
        in_path = args.infile
    else:
        if args.dataset == "cwq":
            in_path = os.path.join(args.indir, "RoG-cwq-step2.jsonl")
        else:
            in_path = os.path.join(args.indir, "RoG-webqsp-step2.jsonl")

    if args.outfile:
        out_path = args.outfile
    else:
        if args.dataset == "cwq":
            out_path = os.path.join(args.outdir, "RoG-cwq-step3.jsonl")
        else:
            out_path = os.path.join(args.outdir, "RoG-webqsp-step3.jsonl")

    process_file(
        in_path=in_path,
        out_path=out_path,
        model_name=args.model,
        topk_rel=args.topk_rel,
        topn_triples=args.topn_triples,
        show=args.show,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()