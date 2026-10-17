"""
全主体覆盖标注集生成器
======================
自动发现「知识库里有切片、但标注集里没题」的设备主体，按切片厚度分配题量并生成
hard 风格问题，使评测覆盖全部 36 个设备主体。

本脚本自动比对 Milvus 主体清单与现有标注集，只补缺口，题量按切片数分档。

（早期有过一个 gen_extra.py，主体清单写死在代码里，换语料后即失效，已删除；
  现在一律从 Milvus 实时发现主体，换语料不用改代码。）

题量分档（切片越多，可出的独立问题越多，也越能拉开区分度）：
  >= 200 片 → 6 题   100-199 → 5 题   50-99 → 4 题   < 50 → 3 题

用法：
  python scripts/gen_cover_all.py               # 正常生成
  python scripts/gen_cover_all.py --dry-run     # 只打印计划，不调用 LLM
"""
import os
import sys
import json
import random
import collections
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from pymilvus import MilvusClient
from utils.llm_utils import get_llm_client

COLLECTION = "kb_chunks"
GOLD = "eval/datasets/qa_hard.jsonl"
OUT = "eval/datasets/qa_hard_extra2.jsonl"

SEED = 20261017
MIN_CONTENT_CHARS = 80
MAX_CONTENT_CHARS = 2000

# 目标题量。一轮「每个主体按切片数出题」只能出 145 题（等于切片总数），
# 剩下的差额用 easy 风格补齐——同一段切片换一套提示词，问法自然不同，
# 不会像重复调用同一提示词那样得到一模一样的题。
# 顺带也让题库里既有「不带型号、口语化」的难题，也有「带型号」的正常题。
TARGET_TOTAL = 164

GEN_SYSTEM_PROMPT = "你是企业知识库的测试工程师，擅长把技术资料反推成真实用户会提出的问题。"

# easy 风格：问题里带完整型号，检索几乎不可能失手。
# 它单独存在的意义是当「对照组」——题库里如果只有难题，指标高低只能说明
# 「语义匹配行不行」，看不出「型号对不对」这一环的独立贡献。
GEN_USER_PROMPT_EASY = """下面是一段仓储物流装备手册的切片内容。

【资料切片】
{content}

【所属设备】
{item_name}

请生成一个**真实用户**会向设备运维同事提出的问题，该问题的答案完全包含在上面这段资料中。
要求：
- 用口语化、简短的自然提问方式（15~30 字），像在微信里打字那样；
- 必须带上设备型号或明确指代，让问题脱离上下文也成立；
- 不要出现"根据上述资料""这段文字"之类的元表述；
- 不要照抄资料里的句子，要换成用户的问法。

再给一个标准答案（严格摘自资料，不编造）。

只返回 JSON，格式：{{"question": "...", "answer": "..."}}"""

GEN_USER_PROMPT_HARD = """下面是一段仓储物流装备手册的切片内容。

【资料切片】
{content}

【所属设备】
{item_name}

请生成一个**难倒检索系统**的真实用户问题，要求：

1. **不要出现完整型号**。用"这台车""那台分拣机""我们冷库那台机组""它"之类的指代，
   顶多带一个系列名或品类词。
2. **口语化、有省略、带冗余**。像客户在微信里随口一问，允许语序颠倒、语气词、
   错误说法、少量错别字。
3. **必须依赖语义理解才能对上这段资料**，不能靠关键词直接命中。
   优先从"用户遇到的具体现象/困惑"出发，而不是复述资料里的术语。
4. **加上干扰信息**。可以在问题里提到一个跟本段资料无关的方面，
   让检索系统必须自己判断哪段才真正相关。

再给一个标准答案（严格摘自资料，不编造）。

只返回 JSON，格式：{{"question": "...", "answer": "..."}}"""


def quota(n_chunks: int) -> int:
    """按切片厚度分配题量：切片越多，越能出互不重复的独立问题。"""
    if n_chunks >= 200:
        return 6
    if n_chunks >= 100:
        return 5
    if n_chunks >= 50:
        return 4
    # 语料换成自建的 36 份仓储物流手册后，单个主体的切片数普遍只有 4~5 片，
    # 沿用原来的 3 题只能凑出 108 题，达不到 164 题的样本量要求。
    # 这里改成「有几片出几题，最多 5 题」：每题仍落在不同切片上，
    # 互不重复，同时把总量顶到 160+。
    return min(n_chunks, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不调用 LLM")
    args = ap.parse_args()

    client = MilvusClient(uri=os.environ["MILVUS_URL"])
    rows = client.query(
        COLLECTION, filter="",
        output_fields=["chunk_id", "content", "item_name", "title", "file_title"],
        limit=16384,
    )

    by_item = collections.defaultdict(list)
    for r in rows:
        name = (r.get("item_name") or "").strip()
        content = (r.get("content") or "").strip()
        if not name or not content:
            continue
        if not (MIN_CONTENT_CHARS <= len(content) <= MAX_CONTENT_CHARS):
            continue
        by_item[name].append(r)

    # 现有标注集已覆盖的主体
    covered = collections.Counter()
    try:
        for line in open(GOLD, encoding="utf-8"):
            d = json.loads(line)
            for n in d.get("item_names", []):
                covered[n] += 1
    except FileNotFoundError:
        print(f"[warn] 未找到 {GOLD}，视为全部未覆盖")

    gaps = [(n, len(c)) for n, c in by_item.items() if n and n not in covered]
    gaps.sort(key=lambda x: -x[1])

    print(f"主体总数 {len(by_item)}，已覆盖 {len(covered)}，待补 {len(gaps)}")
    print(f"{'主体':<40}{'可用切片':>8}{'计划':>6}")
    print("-" * 58)
    plan = []
    total = 0
    for name, n in gaps:
        q = quota(n)
        plan.append((name, n, q))
        total += q
        print(f"{name[:38]:<40}{n:>8}{q:>6}")
    print("-" * 58)
    print(f"计划新增 {total} 题（现有 76 题，合并后约 {76 + total} 题）")

    if args.dry_run:
        print("\n[dry-run] 未调用 LLM，未写出文件")
        return

    llm = get_llm_client(json_mode=True)
    rng = random.Random(SEED)

    items = []
    seq = 0
    for item_name, n_chunks, target in plan:
        chunks = by_item[item_name][:]
        rng.shuffle(chunks)
        picked = chunks[:target]
        made = 0
        for row in picked:
            seq += 1
            content = row["content"].strip()
            file_title = row.get("file_title") or row.get("title") or ""
            try:
                resp = llm.invoke([
                    ("system", GEN_SYSTEM_PROMPT),
                    ("user", GEN_USER_PROMPT_HARD.format(
                        content=content[:MAX_CONTENT_CHARS], item_name=item_name)),
                ])
                data = json.loads(resp.content)
                question = (data.get("question") or "").strip()
                answer = (data.get("answer") or "").strip()
            except Exception as exc:
                print(f"    [skip] {item_name} #{row['chunk_id']} 生成失败：{type(exc).__name__}: {exc}")
                continue
            if not question:
                print(f"    [skip] {item_name} #{row['chunk_id']} 模型未返回问题")
                continue
            items.append({
                "qid": f"cov{seq:04d}",
                "question": question,
                "item_names": [item_name],
                "ground_truth_chunk_ids": [int(row["chunk_id"])],
                "reference_answer": answer,
                "source": "generated",
                "note": f"hard | 全主体覆盖补全 | {file_title}",
            })
            made += 1
        print(f"  {item_name}: 计划 {target}，实出 {made}")

    # 补足到 TARGET_TOTAL：换 easy 提示词，在已用过的切片上再问一次。
    # 只在这时候才允许同一 chunk 出两题——提示词不同，问法不同，
    # 不会出现两条一模一样的标注。
    if len(items) < TARGET_TOTAL:
        print(f"\n已出 {len(items)} 题，未达目标 {TARGET_TOTAL}，用 easy 风格补足")
        pool = [r for rows_ in by_item.values() for r in rows_]
        rng.shuffle(pool)
        for row in pool:
            if len(items) >= TARGET_TOTAL:
                break
            seq += 1
            content = row["content"].strip()
            file_title = row.get("file_title") or row.get("title") or ""
            try:
                resp = llm.invoke([
                    ("system", GEN_SYSTEM_PROMPT),
                    ("user", GEN_USER_PROMPT_EASY.format(
                        content=content[:MAX_CONTENT_CHARS], item_name=row["item_name"])),
                ])
                data = json.loads(resp.content)
                question = (data.get("question") or "").strip()
                answer = (data.get("answer") or "").strip()
            except Exception as exc:
                print(f"    [skip] 补足 #{row['chunk_id']} 生成失败：{type(exc).__name__}: {exc}")
                continue
            if not question:
                continue
            items.append({
                "qid": f"cov{seq:04d}",
                "question": question,
                "item_names": [row["item_name"]],
                "ground_truth_chunk_ids": [int(row["chunk_id"])],
                "reference_answer": answer,
                "source": "generated",
                "note": f"easy | 题量补足 | {file_title}",
            })

    with open(OUT, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\n已写入 {OUT}：{len(items)} 条")


if __name__ == "__main__":
    main()
