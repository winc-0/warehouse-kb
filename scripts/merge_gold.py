"""
合并扩充标注集并校验 chunk_id 有效性
====================================
把 qa_hard.jsonl（原 29 题）与 qa_hard_extra.jsonl（新增）合并：
  1) 备份原 qa_hard.jsonl -> qa_hard_pre_expand_YYYYMMDD.jsonl
  2) 统一重编号 qid 为 gen0001..gen00NN（全局唯一）
  3) 校验每个 ground_truth_chunk_id 在 Milvus 中真实存在
  4) 写回 qa_hard.jsonl

用法：
  python scripts/merge_gold.py --extra eval/datasets/qa_hard_extra.jsonl --tag expand
  python scripts/merge_gold.py --extra eval/datasets/qa_hard_extra2.jsonl --tag coverall

说明：备份名为 qa_hard_pre_{tag}_{日期}.jsonl。每次合并请用不同的 tag，
否则同名备份已存在时脚本会跳过备份，导致回滚点丢失。
"""
import os
import sys
import json
import shutil
import datetime
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from pymilvus import MilvusClient

COLLECTION = "kb_chunks"
BASE = "eval/datasets/qa_hard.jsonl"
EXTRA = "eval/datasets/qa_hard_extra.jsonl"


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default=EXTRA, help="待合并的新增标注文件")
    ap.add_argument("--tag", default="expand", help="备份名标签，避免不同批次互相覆盖")
    args = ap.parse_args()

    today = datetime.date.today().strftime("%Y%m%d")
    base_items = load(BASE)
    extra_items = load(args.extra)
    print(f"原集: {len(base_items)} 题 | 新增: {len(extra_items)} 题（来自 {args.extra}）")

    # 1) 备份
    backup = f"eval/datasets/qa_hard_pre_{args.tag}_{today}.jsonl"
    if not os.path.exists(backup):
        shutil.copy(BASE, backup)
        print(f"已备份原集 -> {backup}")
    else:
        print(f"[注意] 备份 {backup} 已存在，跳过备份（如需新的回滚点请换 --tag）")

    # 2) 合并 + 重编号
    merged = base_items + extra_items
    for idx, it in enumerate(merged, start=1):
        it["qid"] = f"gen{idx:04d}"
    print(f"合并后: {len(merged)} 题")

    # 3) 校验 chunk_id 存在
    client = MilvusClient(uri=os.environ["MILVUS_URL"])
    all_ids = sorted({cid for it in merged for cid in it.get("ground_truth_chunk_ids", [])})
    print(f"待校验 chunk_id: {len(all_ids)} 个")
    existing = set()
    # Milvus `in` 一次最多 16384 个，这里远小于此
    rows = client.query(
        COLLECTION, filter=f"chunk_id in {all_ids}",
        output_fields=["chunk_id"], limit=16384,
    )
    for r in rows:
        existing.add(int(r["chunk_id"]))
    missing = [cid for cid in all_ids if cid not in existing]
    if missing:
        print(f"[warn] 有 {len(missing)} 个 chunk_id 在 Milvus 中不存在：{missing[:20]}")
    else:
        print("全部 chunk_id 校验通过 ✓")

    # 4) 写回
    with open(BASE, "w", encoding="utf-8") as f:
        for it in merged:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # 覆盖统计
    cov = collections.Counter()
    for it in merged:
        for nm in it.get("item_names", []):
            cov[nm] += 1
    print(f"\n最终覆盖文档数: {len(cov)}")
    for nm, c in sorted(cov.items()):
        print(f"  {nm}: {c} 题")


if __name__ == "__main__":
    main()
