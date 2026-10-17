"""
金标 chunk_id 重映射工具（content 锚定版）
=========================================
重导（--force）后 Milvus 切片被重建，旧的 ground_truth_chunk_ids 全部失效。

本脚本不再依赖「旧 id -> 备份 content」的桥接（旧备份 id 与当前库 id 属于不同导入
批次，无法直接对应），而是改用每题自带的锚文本找回当前库的 chunk_id：

  1) 优先用 reference_answer（+ question 兜底）作为锚，在对应 item 的候选切片中
     做 BGE-M3 dense 余弦 top1 检索；
  2) 余弦相似度 < 阈值 的，标记为「需人工复核」，不硬塞；
  3) 兜底：若 reference_answer/question 皆空，则回退到 backup 精确匹配（兼容同代场景）。

用法：
  python scripts/remap_gold.py
依赖：eval/datasets/qa_hard.jsonl、当前 Milvus 库
"""
import os
import sys
import json
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from pymilvus import MilvusClient
import numpy as np
from utils.embedding_utils import generate_embeddings

COLLECTION = "kb_chunks"
GOLD = "eval/datasets/qa_hard.jsonl"
BACKUP = "eval/datasets/gold_content_backup.json"
LOW_SIM_THRESHOLD = 0.75  # 低于此相似度视为不可信，标记人工复核


def main():
    mc = MilvusClient(uri=os.environ["MILVUS_URL"])
    try:
        backup = json.load(open(BACKUP, encoding="utf-8"))
    except FileNotFoundError:
        backup = {}
    qa = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]

    rows = mc.query(
        COLLECTION,
        filter="",
        output_fields=["chunk_id", "item_name", "content", "title"],
        limit=16384,
    )
    by_item = collections.defaultdict(list)  # item_name -> [(chunk_id, content, title)]
    for r in rows:
        by_item[r["item_name"]].append(
            (r["chunk_id"], (r.get("content") or "").strip(), (r.get("title") or ""))
        )

    # 预编码每个 item 的候选切片
    item_emb = {}
    for item, chunks in by_item.items():
        texts = [c for _, c, _ in chunks]
        if texts:
            dense = np.array(generate_embeddings(texts)["dense"], dtype=np.float32)
            item_emb[item] = (chunks, dense)

    report = []
    mapped = 0
    total = 0
    for item in qa:
        new_ids = []
        qid = item.get("qid", "?")
        item_names = item.get("item_names") or []
        # 锚文本：优先 reference_answer，太短则拼接 question
        anchor = (item.get("reference_answer") or "").strip()
        if len(anchor) < 20:
            anchor = (anchor + "\n" + (item.get("question") or "")).strip()
        total += 1

        # 候选 item：优先题面声明的 item_names，否则全局 top1
        cand_items = item_names if item_names else list(by_item.keys())
        best = None
        best_sim = -1.0
        best_item = None
        for it in cand_items:
            chunks, emb = item_emb.get(it, ([], None))
            if not chunks or emb is None:
                continue
            q = np.array(generate_embeddings([anchor])["dense"][0], dtype=np.float32)
            sims = emb @ q / (np.linalg.norm(emb, axis=1) * np.linalg.norm(q) + 1e-9)
            j = int(np.argmax(sims))
            if sims[j] > best_sim:
                best_sim = float(sims[j])
                best = chunks[j][0]
                best_item = it

        # 兜底：backup 精确匹配（同代 id 场景）
        if best is None:
            for cid in item.get("ground_truth_chunk_ids", []):
                info = backup.get(str(cid))
                if not info:
                    continue
                it = info["item_name"]
                content = info["content"].strip()
                chunks, _ = item_emb.get(it, ([], None))
                hit = next((c for c, c2, _ in chunks if c2 == content), None)
                if hit is not None:
                    best = hit
                    best_sim = 1.0
                    best_item = it
                    break

        if best is None:
            report.append((qid, "未匹配", None, ""))
        else:
            new_ids.append(int(best))
            mapped += 1
            tag = "ok" if best_sim >= LOW_SIM_THRESHOLD else "需人工复核"
            hit_title = ""
            for c, _, t in by_item.get(best_item, []):
                if c == best:
                    hit_title = t
                    break
            report.append((qid, f"{tag} sim={best_sim:.3f}", best, hit_title))
        item["ground_truth_chunk_ids"] = new_ids

    with open(GOLD, "w", encoding="utf-8") as f:
        for it in qa:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"金标题数: {total}  成功映射: {mapped}  待复核/失败: {total - mapped}")
    print("-" * 70)
    for r in report:
        print(f"  {r[0]}  {r[1]:<24s} -> {r[2]}  [{r[3][:30]}]")


if __name__ == "__main__":
    main()
