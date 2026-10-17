"""盘点 kb_chunks：列出每个 item_name 的切片数、含图切片数、平均长度。"""
import os
import sys
import json
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from pymilvus import MilvusClient

COLLECTION = "kb_chunks"
client = MilvusClient(uri=os.environ["MILVUS_URL"])

rows = client.query(
    COLLECTION,
    filter="",
    output_fields=["chunk_id", "item_name", "content", "title", "file_title", "image_urls"],
    limit=16384,
)

by_item = collections.defaultdict(list)
for r in rows:
    name = (r.get("item_name") or "").strip()
    by_item[name].append(r)

stats = []
for name, chunks in by_item.items():
    n = len(chunks)
    with_img = sum(1 for c in chunks if (c.get("image_urls") or "").strip())
    avg_len = int(sum(len((c.get("content") or "")) for c in chunks) / max(n, 1))
    stats.append((name, n, with_img, avg_len))

stats.sort(key=lambda x: x[0])
print(f"总切片数: {len(rows)}  文档数: {len(stats)}")
print(f"{'item_name':<32s} {'切片':>5s} {'含图':>5s} {'均长':>5s}")
for name, n, wi, al in stats:
    print(f"{name:<32s} {n:>5d} {wi:>5d} {al:>5d}")

# 已覆盖（来自 qa_hard.jsonl）
covered = set()
qa_path = "eval/datasets/qa_hard.jsonl"
if os.path.exists(qa_path):
    for line in open(qa_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        for nm in obj.get("item_names") or []:
            covered.add(nm)

print("\n已覆盖文档：")
for nm in sorted(covered):
    print("  ", nm)
print("\n未覆盖文档：")
for name, n, wi, al in stats:
    if name not in covered:
        print(f"  {name}  ({n} 切片, {wi} 含图)")
