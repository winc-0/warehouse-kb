#!/usr/bin/env python3
# 从 Milvus 导出当前库内所有文档标题(file_title)与商品名(item_name)，供去重比对
import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from pymilvus import MilvusClient

ROOT = Path(__file__).resolve().parent.parent

client = MilvusClient(uri=os.environ["MILVUS_URL"])
res = client.query(collection_name="kb_chunks", filter="", output_fields=["file_title", "item_name"], limit=10000)
file_titles = sorted(set(r["file_title"] for r in res if r.get("file_title")))
item_names = sorted(set(r["item_name"] for r in res if r.get("item_name")))
out = ROOT / "logs" / "lib_titles.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump({"file_titles": file_titles, "item_names": item_names,
               "n_chunks": len(res)}, f, ensure_ascii=False, indent=2)
print(f"导出 {len(file_titles)} 个 file_title / {len(item_names)} 个 item_name / {len(res)} 切片 -> {out}")
print("\n=== file_title 列表 ===")
for t in file_titles:
    print(" -", t)
