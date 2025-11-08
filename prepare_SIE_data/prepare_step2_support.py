#!/usr/bin/env python3
"""
prepare_step2_support.py

按论文《Learning to Reason in Structured In-context Environments with RL》的 Step 2 来做：
- 给定 seed subgraph
- 把“所有从问题实体集 E_Q 到答案实体集 E_A 的最短路径（不超过 nhop）”找出来
- 合并这些路径上的三元组，得到 G_support

输出样本会多一个字段：
    "support_triples": [
        ["head", "rel", "tail"],
        ...
    ]
"""

import os
import json
import argparse
from collections import deque, defaultdict
from datasets import load_dataset


CWQ_NAME = "rmanluo/RoG-cwq"
WEBQSP_NAME = "rmanluo/RoG-webqsp"
DEFAULT_OUTDIR = "/data/wenhesun/datasets"

# 为了防 seed 里出现极端高出度，给每个 node 的“同深度路径数量”做个软上限
MAX_PATHS_PER_NODE = 50


# ---------- graph utils ----------

def build_adj_from_graph(graph_list):
    """
    graph_list: 来自 RoG 的 graph 字段，大多数情况下是一个 3 元组列表：
        [
          ["m.film_1", "film.director", "m.person_x"],
          ["m.film_1", "film.actor", "m.person_y"],
          ...
        ]
    返回 adjacency: head -> list[(rel, tail)]
    """
    adj = defaultdict(list)
    for tri in graph_list:
        if isinstance(tri, dict):
            h, r, t = tri["head"], tri["rel"], tri["tail"]
        else:
            if len(tri) != 3:
                continue
            h, r, t = tri
        adj[h].append((r, t))
    return adj


def collect_all_shortest_paths(adj, q_nodes, a_nodes, max_hop=4):
    """
    按论文的意思：
    - 多源 BFS，从所有 q_nodes 一起出发
    - 记录每个节点的“最短深度”
    - 对于同一深度到达同一节点的多条路径全部保留（上限 MAX_PATHS_PER_NODE）
    - 节点在 a_nodes 且深度>0，就把这些路径都加入结果

    返回：list[list[(h, r, t)]]
    """
    if not q_nodes or not a_nodes:
        return []

    a_nodes = set(a_nodes)

    # queue 元素： (current_node, path_triples)
    # 初始化多源：每个问题实体都是起点，路径为空
    queue = deque()
    paths_at_node = dict()  # node -> list[path_triples]，这些 path 都是到达该 node 的最短路径
    depth_at_node = dict()  # node -> depth(int)

    for q in q_nodes:
        queue.append((q, []))
        paths_at_node[q] = [[]]   # 到起点的路径就是空
        depth_at_node[q] = 0

    collected_paths = []

    while queue:
        node, path = queue.popleft()
        depth = len(path)
        if depth > max_hop:
            continue

        # 如果这是答案节点，收集所有到达这个节点的最短路径
        if node in a_nodes and depth > 0:
            for p in paths_at_node.get(node, []):
                collected_paths.append(p)
            # 注意：不能 return，要继续跑同层的别的节点，因为论文要的是“all shortest paths”

        # 展开下一层
        for (rel, nxt) in adj.get(node, []):
            new_path = path + [(node, rel, nxt)]
            new_depth = depth + 1
            if new_depth > max_hop:
                continue

            if nxt not in depth_at_node:
                # 第一次到这个节点
                depth_at_node[nxt] = new_depth
                paths_at_node[nxt] = [new_path]
                queue.append((nxt, new_path))
            else:
                # 已经到过这个节点
                old_depth = depth_at_node[nxt]
                if new_depth < old_depth:
                    # 找到更短的路径，刷新
                    depth_at_node[nxt] = new_depth
                    paths_at_node[nxt] = [new_path]
                    queue.append((nxt, new_path))
                elif new_depth == old_depth:
                    # 找到同样短的路径，也要保留（论文的 all shortest paths）
                    if len(paths_at_node[nxt]) < MAX_PATHS_PER_NODE:
                        paths_at_node[nxt].append(new_path)
                        queue.append((nxt, new_path))
                else:
                    # 更长的就不管了
                    pass

    return collected_paths


def extract_support_triples(sample, max_hop=4, topk_answers=10):
    """
    给一条 RoG 样本，返回支持三元组列表（可能为空）

    和论文对齐的点：
    - 答案实体取前 topk_answers 个
    - 把所有最短路径合并
    """
    graph = sample.get("graph", [])

    # 问题实体
    q_nodes = sample.get("q_entity") or sample.get("question_entities") or []
    # 答案实体 / 答案节点（取前10个）
    a_nodes = sample.get("a_entity") or sample.get("answer_entities") or []

    if not a_nodes:
        # 有些数据集答案在 "answers" 里
        ans = sample.get("answer") or sample.get("answers")
        if isinstance(ans, list):
            a_nodes = ans
        elif isinstance(ans, str):
            a_nodes = [ans]

    # 和论文一致：只取前 topk_answers 个
    if len(a_nodes) > topk_answers:
        a_nodes = a_nodes[:topk_answers]

    adj = build_adj_from_graph(graph)
    all_paths = collect_all_shortest_paths(adj, q_nodes, a_nodes, max_hop=max_hop)

    # 合并去重
    support_set = {(h, r, t) for path in all_paths for (h, r, t) in path}
    support_triples = [[h, r, t] for (h, r, t) in support_set]
    return support_triples


# ---------- main pipeline ----------

def process_one_dataset(hf_name: str, out_path: str,
                        max_hop: int = 4,
                        topk_answers: int = 10,
                        show: int = 0):
    ds = load_dataset(hf_name)
    train = ds["train"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cnt_no_path = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(train):
            if args.limit and i >= args.limit:
                break
            
            support_triples = extract_support_triples(
                ex, max_hop=max_hop, topk_answers=topk_answers
            )
            ex = dict(ex)
            ex["support_triples"] = support_triples
            if not support_triples:
                cnt_no_path += 1
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            if show and i < show:
                print(f"\n--- {hf_name} train[{i}] ---")
                print("question:", ex.get("question"))
                print("q_entity:", ex.get("q_entity") or ex.get("question_entities"))
                print("a_entity:", ex.get("a_entity") or ex.get("answer_entities"))
                print("support_triples:", ex["support_triples"])

    print(f"[saved] {hf_name} train with step2 -> {out_path}")
    print(f"[info] {hf_name}: {cnt_no_path} samples have EMPTY support_triples")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str,
                    default=DEFAULT_OUTDIR,
                    help="输出目录，默认 /data/wenhesun/datasets")
    ap.add_argument("--max_hop", type=int, default=4,
                    help="最短路径搜索的最大 hop 数（要和 seed 的设置一致）")
    ap.add_argument("--topk_answers", type=int, default=10,
                    help="论文里说的“前10个正确答案”，这里可以改")
    ap.add_argument("--show", type=int, default=2,
                    help="打印前多少条看看结构")
    ap.add_argument("--which", type=str, default="both",
                    choices=["cwq", "webqsp", "both"],
                    help="处理哪个数据集")
    ap.add_argument("--limit", type=int, default=0,
                help="只处理前 N 条，0 表示处理全部")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    if args.which in ("cwq", "both"):
        cwq_out = os.path.join(outdir, "RoG-cwq-step2.jsonl")
        process_one_dataset(
            CWQ_NAME,
            cwq_out,
            max_hop=args.max_hop,
            topk_answers=args.topk_answers,
            show=args.show,
        )

    if args.which in ("webqsp", "both"):
        webqsp_out = os.path.join(outdir, "RoG-webqsp-step2.jsonl")
        process_one_dataset(
            WEBQSP_NAME,
            webqsp_out,
            max_hop=args.max_hop,
            topk_answers=args.topk_answers,
            show=args.show,
        )