#!/usr/bin/env python3
import json
import pyarrow as pa
import pyarrow.parquet as pq
import sys

in_path = sys.argv[1]
out_path = sys.argv[2]

rows = []
with open(in_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

# pyarrow 要求所有字段能对齐成一个表，SIE 里都是 dict，能直接表化
table = pa.Table.from_pylist(rows)
pq.write_table(table, out_path)
print(f"saved {len(rows)} rows -> {out_path}")