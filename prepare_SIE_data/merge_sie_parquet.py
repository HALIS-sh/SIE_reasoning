#!/usr/bin/env python3
import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def merge_one_ratio(data_dir: Path, ratio: str, out_prefix: str = "sie"):
    """
    把同一个 ratio 的 cwq + webqsp 拼起来
    """
    cwq_path = data_dir / f"verl_cwq_sie{ratio}.parquet"
    web_path = data_dir / f"verl_webqsp_sie{ratio}.parquet"

    if not cwq_path.exists():
        raise FileNotFoundError(f"not found: {cwq_path}")
    if not web_path.exists():
        raise FileNotFoundError(f"not found: {web_path}")

    cwq_tbl = pq.read_table(cwq_path)
    web_tbl = pq.read_table(web_path)

    # 简单纵向拼接
    merged = pa.concat_tables([cwq_tbl, web_tbl], promote=True)

    out_path = data_dir / f"{out_prefix}_{ratio}.parquet"
    pq.write_table(merged, out_path)
    print(f"[OK] {ratio}: {cwq_path.name} + {web_path.name} -> {out_path.name} "
          f"(rows={merged.num_rows})")


def main():
    parser = argparse.ArgumentParser(
        description="merge cwq/webqsp parquet of the same ratio into one parquet"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data/sunwenhe/SIE_reasoning/verl_data",
        help="folder that contains verl_cwq_sie*.parquet and verl_webqsp_sie*.parquet",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0,25,50,75,100",
        help="comma separated ratios to merge, e.g. 0,25,50,75,100",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="sie",
        help="output file prefix, final name will be <prefix>_<ratio>.parquet",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ratios = [r.strip() for r in args.ratios.split(",") if r.strip()]

    for r in ratios:
        merge_one_ratio(data_dir, r, args.out_prefix)


if __name__ == "__main__":
    main()