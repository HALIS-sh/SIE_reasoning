# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# 有些 parquet schema 不一致（尤其 math500），不能一次性 load。
# 这个脚本只加载我们要的几个测试/验证集，并全部存成本地 jsonl。

# 用法：
#     python load_selected_sie.py
# 默认会把文件存到 ./dump_sie/ 下面
# """

# import os
# import json
# from pathlib import Path

# from datasets import load_dataset
# from huggingface_hub import hf_hub_download

# import pyarrow.parquet as pq

# HF_REPO = "pursuitYP/SIE_data"
# OUT_DIR = Path("dump_sie")

# # 我们要的文件清单
# FILES_TO_LOAD = {
#     # 分类只是标注，真正加载是按文件名走的
#     "kgqa": [
#         "webqsp_0_test.parquet",
#         "cwq_0_test.parquet",
#         "grailqa_0_test.parquet",
#         # 你说也要存 validation
#         "cwq_0_validation.parquet",
#         "webqsp_0_validation.parquet",
#     ],
#     "math": [
#         "gsm8k_test.parquet",
#         "math500_test.parquet",  # 这个特殊处理
#     ],
#     "logic": [
#         "kk_test_easy.parquet",
#         "kk_test_hard.parquet",
#     ],
# }


# def save_hf_dataset_to_jsonl(ds, out_path: Path):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with out_path.open("w", encoding="utf-8") as f:
#         for ex in ds:
#             f.write(json.dumps(ex, ensure_ascii=False) + "\n")
#     print(f"[OK] saved {len(ds)} rows to {out_path}")


# def load_normal_parquet(fname: str):
#     """
#     大部分 parquet 都能这样读：
#     load_dataset("pursuitYP/SIE_data", data_files="xxx.parquet", split="train")
#     """
#     print(f"[INFO] loading (datasets) {fname}")
#     ds = load_dataset(
#         HF_REPO,
#         data_files=fname,
#         split="train",  # HF parquet 默认就是一个 train split
#     )
#     print(f"[INFO] loaded {fname}, num_rows={len(ds)}")
#     return ds


# def load_math500_parquet(fname: str):
#     """
#     math500_test.parquet 在这个仓库里 schema 和别的有点不一致，
#     我们直接下载文件 -> pyarrow 读 -> 写 jsonl
#     """
#     print(f"[INFO] math500 detected, downloading via hf_hub_download: {fname}")
#     local_path = hf_hub_download(repo_id=HF_REPO, filename=fname)
#     print(f"[INFO] downloaded to {local_path}, reading with pyarrow...")
#     table = pq.read_table(local_path)
#     # 转成 list-of-dicts
#     records = table.to_pylist()
#     return records


# def main():
#     OUT_DIR.mkdir(parents=True, exist_ok=True)

#     for category, file_list in FILES_TO_LOAD.items():
#         print(f"\n=== Category: {category} ===")
#         cat_dir = OUT_DIR / category
#         cat_dir.mkdir(parents=True, exist_ok=True)

#         for fname in file_list:
#             out_path = cat_dir / (fname.replace(".parquet", ".jsonl"))

#             # 特殊的 math500
#             if fname == "math500_test.parquet":
#                 records = load_math500_parquet(fname)
#                 with out_path.open("w", encoding="utf-8") as f:
#                     for ex in records:
#                         # pyarrow 出来的就是普通 dict
#                         f.write(json.dumps(ex, ensure_ascii=False) + "\n")
#                 print(f"[OK] saved {len(records)} rows to {out_path}")
#                 continue

#             # 其它正常 parquet
#             ds = load_normal_parquet(fname)
#             save_hf_dataset_to_jsonl(ds, out_path)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from datasets import load_dataset
import pyarrow.parquet as pq
import glob
import os

HF_REPO = "pursuitYP/SIE_data"
OUT_DIR = Path("dump_sie")

FILES_TO_LOAD = {
    "kgqa": [
        "webqsp_0_test.parquet",
        "cwq_0_test.parquet",
        "grailqa_0_test.parquet",
        # 也要存 validation
        "cwq_0_validation.parquet",
        "webqsp_0_validation.parquet",
    ],
    "math": [
        "gsm8k_test.parquet",
        "math500_test.parquet",  # 特殊处理：从本地 cache 找
    ],
    "logic": [
        "kk_test_easy.parquet",
        "kk_test_hard.parquet",
    ],
}


def save_hf_dataset_to_jsonl(ds, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[OK] saved {len(ds)} rows to {out_path}")


def load_normal_parquet(fname: str):
    print(f"[INFO] loading (datasets) {fname}")
    ds = load_dataset(
        HF_REPO,
        data_files=fname,
        split="train",
    )
    print(f"[INFO] loaded {fname}, num_rows={len(ds)}")
    return ds


def find_local_math500() -> str:
    """
    尝试在本地 HF cache 里找到 math500_test.parquet
    典型路径：
    ~/.cache/huggingface/hub/datasets--pursuitYP--SIE_data/snapshots/<hash>/math500_test.parquet
    """
    home = os.path.expanduser("~")
    base = os.path.join(
        home,
        ".cache",
        "huggingface",
        "hub",
        "datasets--pursuitYP--SIE_data",
        "snapshots",
    )
    pattern = os.path.join(base, "**", "math500_test.parquet")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"math500_test.parquet not found in {base}, run a dataset load first.")
    # 取最新的一个
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_math500_from_local() -> list:
    local_path = find_local_math500()
    print(f"[INFO] found local math500_test.parquet: {local_path}")
    table = pq.read_table(local_path)
    records = table.to_pylist()
    print(f"[INFO] loaded math500_test.parquet locally, rows={len(records)}")
    return records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for category, file_list in FILES_TO_LOAD.items():
        print(f"\n=== Category: {category} ===")
        cat_dir = OUT_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for fname in file_list:
            out_path = cat_dir / (fname.replace(".parquet", ".jsonl"))

            if fname == "math500_test.parquet":
                # 不走远端，直接从本地 cache 找
                records = load_math500_from_local()
                with out_path.open("w", encoding="utf-8") as f:
                    for ex in records:
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                print(f"[OK] saved {len(records)} rows to {out_path}")
                continue

            # 其它正常 parquet
            ds = load_normal_parquet(fname)
            save_hf_dataset_to_jsonl(ds, out_path)


if __name__ == "__main__":
    main()