#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入后校验：库里到底有什么，以及标注集的金标是否还全部有效。

换语料后必跑这一条。它回答两个问题：
  1. 规模对不对——多少份文档、多少切片、覆盖多少个主体；
  2. 标注集有没有失效——ground_truth_chunk_ids 里的 id 是否仍能在库里查到。
     Milvus 的 chunk_id 是自增主键，重导入后全部换新，标注集不跟着更新
     就会整份失效，而评测不会报错、只会悄悄给出偏低的数。

用法：
    python scripts/verify_lib.py                       # 校验默认标注集
    python scripts/verify_lib.py --gold 其他.jsonl     # 校验指定标注集
    python scripts/verify_lib.py --no-gold             # 只看库规模
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=False)

# 基础设施都在局域网里，全局代理会把 gRPC 也劫走
for host in ("192.168.217.3", "localhost", "127.0.0.1"):
    os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + f",{host}"

from pymilvus import MilvusClient  # noqa: E402

COLLECTION = "kb_chunks"
DEFAULT_GOLD = "eval/datasets/qa_hard.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description="校验知识库规模与标注集金标有效性")
    ap.add_argument("--gold", default=DEFAULT_GOLD, help="标注集路径")
    ap.add_argument("--no-gold", action="store_true", help="只打印库规模")
    args = ap.parse_args()

    client = MilvusClient(uri=os.environ["MILVUS_URL"])
    res = client.query(collection_name=COLLECTION, filter="",
                       output_fields=["file_title", "chunk_id", "item_name"], limit=16384)

    ids = set(str(r["chunk_id"]) for r in res)
    titles = sorted(set(r["file_title"] for r in res if r.get("file_title")))
    items = sorted({(r.get("item_name") or "").strip() for r in res} - {""})

    print("=== 库规模 ===")
    print(f"  切片总数：{len(res)}")
    print(f"  文档数（file_title 去重）：{len(titles)}")
    print(f"  主体数（item_name 去重）：{len(items)}")

    # 主体为空是最容易被忽略的一类脏数据：这批切片在检索时会被
    # item_name 过滤表达式排除，等于「导进去了但永远搜不到」。
    empty = [r for r in res if not (r.get("item_name") or "").strip()]
    if empty:
        print(f"  [警告] {len(empty)} 条切片主体名为空，需检查主体识别环节")

    if args.no_gold:
        return 0

    gold_path = PROJECT_ROOT / args.gold
    if not gold_path.exists():
        print(f"\n[跳过] 未找到标注集 {args.gold}")
        return 0

    bad, total, qids = [], 0, 0
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            d = json.loads(line)
            qids += 1
            for cid in d.get("ground_truth_chunk_ids", []):
                total += 1
                if str(cid) not in ids:
                    bad.append((d.get("qid"), cid))

    print(f"\n=== 标注集校验（{args.gold}）===")
    print(f"  题数：{qids}")
    print(f"  金标 chunk_id：{total} 个，失效 {len(bad)} 个")
    if bad:
        print("  [失败] 存在失效金标，评测指标会失真：")
        for qid, cid in bad[:10]:
            print(f"    {qid} -> {cid}")
        return 1
    print("  全部有效 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
