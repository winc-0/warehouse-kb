# eval/build_dataset.py
"""
标注集构建工具 —— 解决"评测没有 ground truth"这个最先卡住的问题。

三条命令对应两条标注路径：

  generate   半自动：从 Milvus 抽样切片 → 让 LLM 反向生成"这段资料能回答的问题"
             → 该切片自动成为正确答案。几分钟就能产出几十条，缺点是题目偏"好答"，
             指标会偏乐观。建议与人工题混用（报告里会按 source 分组统计）。

  candidates 人工：给一批你手写的真实问题，脚本跑一遍检索，把 Top-N 候选切片
             导成 CSV，你只要在 included 列填 1 就算标完。质量最高，但要人工看。

  merge      把 candidates 产出的、已勾选的 CSV 转成标准 JSONL 数据集。

用法示例
--------
# 1) 反向生成（每台设备抽 8 条切片，共约 50 题）
python eval/build_dataset.py generate --per-item 8 --max-items 6 \\
    --out eval/datasets/qa_generated.jsonl

# 2) 人工标注：准备 questions.txt（每行一个问题，可写 "问题 ||| 设备名"）
python eval/build_dataset.py candidates --questions questions.txt --top-n 8 \\
    --out eval/datasets/to_label.csv

# 3) 勾选完成后合并
python eval/build_dataset.py merge --labeled eval/datasets/to_label.csv \\
    --out eval/datasets/qa_manual.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval._boot import init as _boot_init  # noqa: E402

PROJECT_ROOT = _boot_init()

from eval.dataset import QAItem, describe, load_dataset, save_dataset  # noqa: E402
from eval.strategies import EvalContext  # noqa: E402

# --------------------------------------------------------------------------- #
# 反向生成问题的提示词
# --------------------------------------------------------------------------- #
GEN_SYSTEM_PROMPT = "你是企业知识库的测试工程师，擅长把技术资料反推成真实用户会提出的问题。"

# easy：问题里带完整型号，检索几乎不可能失手 —— 适合先验证链路是否通
GEN_USER_PROMPT_EASY = """下面是一段仓储物流装备手册的切片内容。

【资料切片】
{content}

【所属设备】
{item_name}

请完成两件事：
1. 生成一个**真实用户**会向客服提出的问题，该问题的答案完全包含在上面这段资料中。
   要求：
   - 用口语化、简短的自然提问方式（15~30 字），像客户在微信里打字那样；
   - 必须带上设备型号或明确指代，让问题脱离上下文也成立；
   - 不要出现"根据上述资料""这段文字"之类的元表述；
   - 不要照抄资料里的句子，要换成用户的问法。
2. 给出该问题的标准答案（直接摘自资料，不要编造资料里没有的信息）。

只返回 JSON，格式：{{"question": "...", "answer": "..."}}"""

# hard：去掉型号、加入口语省略与干扰词，逼检索系统真正依赖语义匹配
GEN_USER_PROMPT_HARD = """下面是一段仓储物流装备手册的切片内容。

【资料切片】
{content}

【所属设备】
{item_name}

请生成一个**难倒检索系统**的真实用户问题，要求：

1. **不要出现完整型号**。用"这台车""那台分拣机""我们冷库那台机组""它"之类的指代，
   顶多带一个系列名（如只说"CPD30"里的"30"或只说设备品类）。
2. **口语化、有省略、带冗余**。像客户在微信里随口一问，允许语序颠倒、语气词、
   错误说法、少量错别字。
3. **必须依赖语义理解才能对上这段资料**，不能靠关键词直接命中。
   优先从"用户遇到的具体现象/困惑"出发，而不是复述资料里的术语。
4. **加上干扰信息**。可以在问题里提到一个跟本段资料无关的方面，
   让检索系统必须自己判断哪段才真正相关。

再给一个标准答案（严格摘自资料，不编造）。

只返回 JSON，格式：{{"question": "...", "answer": "..."}}"""

# 兼容旧引用
GEN_USER_PROMPT = GEN_USER_PROMPT_EASY

GEN_PROMPTS = {"easy": GEN_USER_PROMPT_EASY, "hard": GEN_USER_PROMPT_HARD}

# 抽样时跳过的切片：太短没信息量，太长会让问题发散
MIN_CONTENT_CHARS = 80
MAX_CONTENT_CHARS = 2000


# --------------------------------------------------------------------------- #
# Milvus 读取
# --------------------------------------------------------------------------- #
def iter_chunks(ctx: EvalContext, batch: int = 500, max_rows: int = 100000) -> Iterator[Dict[str, Any]]:
    """
    游标式翻页读取切片，避免依赖 Milvus 的 offset 支持。

    主键 chunk_id 是自增 INT64，所以用 `chunk_id > last` 逐批推进既安全又高效。
    """
    collection = ctx.collection
    last_id = -1
    fetched = 0
    while fetched < max_rows:
        rows = ctx.client.query(
            collection_name=collection,
            filter=f"chunk_id > {last_id}",
            output_fields=["chunk_id", "content", "item_name", "title", "file_title"],
            limit=batch,
        )
        if not rows:
            return
        rows = sorted(rows, key=lambda r: int(r["chunk_id"]))
        for row in rows:
            yield row
        fetched += len(rows)
        last_id = int(rows[-1]["chunk_id"])
        if len(rows) < batch:
            return


def sample_chunks_by_item(
    ctx: EvalContext,
    per_item: int,
    max_items: int,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """按 item_name 分组，每组随机抽 per_item 条可用切片。"""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in iter_chunks(ctx):
        item_name = (row.get("item_name") or "").strip()
        content = (row.get("content") or "").strip()
        if not item_name or not content:
            continue
        if not (MIN_CONTENT_CHARS <= len(content) <= MAX_CONTENT_CHARS):
            continue
        grouped.setdefault(item_name, []).append(row)

    rng = random.Random(seed)
    picked: Dict[str, List[Dict[str, Any]]] = {}
    for item_name in sorted(grouped)[:max_items]:
        rows = grouped[item_name]
        rng.shuffle(rows)
        picked[item_name] = rows[:per_item]
    return picked


# --------------------------------------------------------------------------- #
# 子命令：generate
# --------------------------------------------------------------------------- #
def cmd_generate(args: argparse.Namespace) -> int:
    from utils.llm_utils import get_llm_client

    ctx = EvalContext(use_cache=False)
    print("[0/3] 环境自检")
    _check_env_or_warn()
    print(f"[1/3] 扫描 Milvus 集合 {ctx.collection} ...")
    try:
        picked = sample_chunks_by_item(ctx, per_item=args.per_item, max_items=args.max_items, seed=args.seed)
    except Exception as exc:
        print(f"[x] 读取 Milvus 失败：{type(exc).__name__}: {exc}")
        print("    请先执行 `python eval/run_eval.py --preflight` 排查环境问题。")
        return 1

    total = sum(len(v) for v in picked.values())
    if total == 0:
        print("[x] 没有抽到可用切片。请确认集合已有数据，且切片长度在 "
              f"{MIN_CONTENT_CHARS}~{MAX_CONTENT_CHARS} 字符之间。")
        return 1
    print(f"      抽到 {total} 条切片，覆盖 {len(picked)} 个设备")

    plan_calls = total
    style_prompt = GEN_PROMPTS[args.style]
    print(f"[2/3] 逐条反向生成问题（风格 {args.style}，预计 {plan_calls} 次 LLM 调用）...")

    llm = get_llm_client(json_mode=True)
    items: List[QAItem] = []
    seq = 0
    for item_name, rows in picked.items():
        for row in rows:
            seq += 1
            content = row["content"].strip()
            try:
                resp = llm.invoke(
                    [
                        ("system", GEN_SYSTEM_PROMPT),
                        ("user", style_prompt.format(content=content[:MAX_CONTENT_CHARS], item_name=item_name)),
                    ]
                )
                data = json.loads(resp.content)
                question = (data.get("question") or "").strip()
                answer = (data.get("answer") or "").strip()
            except Exception as exc:
                print(f"      [skip] {item_name} #{row['chunk_id']} 生成失败：{type(exc).__name__}: {exc}")
                continue

            if not question:
                print(f"      [skip] {item_name} #{row['chunk_id']} 模型未返回问题")
                continue

            items.append(
                QAItem(
                    qid=f"gen{seq:04d}",
                    question=question,
                    # 过滤条件仍用准确的设备名：难的是"问题措辞"，不是"过滤表达式"
                    item_names=[item_name],
                    ground_truth_chunk_ids=[int(row["chunk_id"])],
                    reference_answer=answer,
                    source="generated",
                    note=f"{args.style} | 由切片自动生成 | {row.get('file_title') or row.get('title') or ''}",
                )
            )
            if seq % 10 == 0:
                print(f"      已生成 {len(items)} 条 ...")

    if not items:
        print("[x] 一条都没生成成功，请检查 LLM 配置")
        return 1

    out = Path(args.out)
    save_dataset(items, out)
    print(f"[3/3] 已写入 {out}")
    for key, value in describe(items).items():
        print(f"      {key}: {value}")

    if args.style == "easy":
        print("\n提示：easy 风格问题都带完整型号，检索几乎不会失手，指标容易顶到 100% 失去区分度。")
        print("      如果跑完评测触发了有效性预警，改用 --style hard 重新生成一遍。")
    else:
        print("\n提示：hard 风格虽更有区分度，但题目仍由切片反向生成，与真实用户问法有差距。")
        print("      追求可信结论时，请再用 candidates 子命令补一批自己写的问题。")
    return 0


# --------------------------------------------------------------------------- #
# 子命令：candidates
# --------------------------------------------------------------------------- #
def cmd_candidates(args: argparse.Namespace) -> int:
    from eval.strategies import STRATEGIES

    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"[x] 问题文件不存在：{questions_path}")
        return 1

    entries: List[tuple] = []
    for lineno, line in enumerate(questions_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|||" in line:
            question, names = line.split("|||", 1)
            item_names = [n.strip() for n in names.split(",") if n.strip()]
        else:
            question, item_names = line, []
        if not item_names:
            print(f"[x] 第 {lineno} 行缺少设备名（格式：问题 ||| 设备名1,设备名2）")
            return 1
        entries.append((f"man{lineno:04d}", question.strip(), item_names))

    if not entries:
        print("[x] 没有解析到任何问题")
        return 1

    ctx = EvalContext(use_cache=True)
    strategy = STRATEGIES[args.strategy]
    print("[0/2] 环境自检")
    _check_env_or_warn()
    print(f"[1/2] 使用策略 [{strategy.label}] 为 {len(entries)} 个问题召回候选切片 ...")

    rows_out: List[Dict[str, Any]] = []
    for qid, question, item_names in entries:
        try:
            ids = strategy.retrieve(ctx, question, item_names)
        except Exception as exc:
            print(f"      [skip] {qid} 检索失败：{type(exc).__name__}: {exc}")
            continue

        for chunk_id in ids[: args.top_n]:
            entity = _fetch_chunk(ctx, int(chunk_id))
            rows_out.append(
                {
                    "qid": qid,
                    "question": question,
                    "item_names": ",".join(item_names),
                    "chunk_id": int(chunk_id),
                    "item_name": (entity.get("item_name") if entity else "") or "",
                    "preview": ((entity.get("content") if entity else "") or "").replace("\n", " ")[:160],
                    "included": "",  # ← 人工勾选：填 1 表示"这段能回答问题"
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[2/2] 已写入 {out}（{len(rows_out)} 行）")
    print("\n下一步：用 Excel/WPS 打开，在 included 列给正确答案填 1，保存后执行：")
    print(f"  python eval/build_dataset.py merge --labeled {out} --out eval/datasets/qa_manual.jsonl")
    return 0


def _fetch_chunk(ctx: EvalContext, chunk_id: int) -> Dict[str, Any]:
    """按主键取单条切片（用于给候选补上可读预览）。"""
    try:
        rows = ctx.client.query(
            collection_name=ctx.collection,
            filter=f"chunk_id == {chunk_id}",
            output_fields=["chunk_id", "content", "item_name"],
            limit=1,
        )
        return rows[0] if rows else {}
    except Exception:
        return {}


def _check_env_or_warn() -> None:
    """跑 Milvus 相关操作前先做一次轻量自检，避免用户对着异常栈猜原因。"""
    from eval.strategies import check_environment

    ok, lines = check_environment()
    for line in lines:
        print("  " + line)
    if not ok:
        print("  [warn] 自检未全部通过，继续执行可能失败。")


# --------------------------------------------------------------------------- #
# 子命令：merge
# --------------------------------------------------------------------------- #
def cmd_merge(args: argparse.Namespace) -> int:
    path = Path(args.labeled)
    if not path.exists():
        print(f"[x] 标注文件不存在：{path}")
        return 1

    grouped: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            ticked = (row.get("included") or "").strip().lower()
            if ticked not in ("1", "y", "yes", "true", "是", "√"):
                continue
            qid = row["qid"].strip()
            entry = grouped.setdefault(
                qid,
                {"question": row["question"].strip(), "item_names": [], "chunk_ids": []},
            )
            names = [n.strip() for n in (row.get("item_names") or "").split(",") if n.strip()]
            entry["item_names"] = list(dict.fromkeys(entry["item_names"] + names))
            try:
                entry["chunk_ids"].append(int(row["chunk_id"]))
            except (TypeError, ValueError):
                continue

    items = [
        QAItem(
            qid=qid,
            question=entry["question"],
            item_names=entry["item_names"],
            ground_truth_chunk_ids=sorted(set(entry["chunk_ids"])),
            source="manual",
        )
        for qid, entry in grouped.items()
        if entry["chunk_ids"]
    ]

    if not items:
        print("[x] included 列里没有任何勾选（填 1 表示正确答案），未生成数据集")
        return 1

    out = Path(args.out)
    save_dataset(items, out)
    print(f"已写入 {out}")
    for key, value in describe(items).items():
        print(f"  {key}: {value}")

    skipped = args.expect_qids - len(items) if args.expect_qids else 0
    if skipped > 0:
        print(f"  提示：有 {skipped} 个问题没有勾选任何切片，已跳过")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python eval/build_dataset.py",
        description="履约知识助手评测集构建工具",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="从 Milvus 抽样切片，让 LLM 反向生成问答对")
    p_gen.add_argument("--per-item", type=int, default=8, help="每台设备抽多少条切片（默认 8）")
    p_gen.add_argument("--max-items", type=int, default=6, help="最多覆盖多少台设备（默认 6）")
    p_gen.add_argument("--seed", type=int, default=42, help="抽样随机种子（默认 42）")
    p_gen.add_argument(
        "--style",
        choices=("easy", "hard"),
        default="easy",
        help="题目风格：easy=带完整型号（易，指标易饱和）；hard=口语化/去型号/带干扰（有区分度）",
    )
    p_gen.add_argument("--out", default="eval/datasets/qa_generated.jsonl", help="输出 JSONL 路径")
    p_gen.set_defaults(func=cmd_generate)

    p_can = sub.add_parser("candidates", help="为人写的问题召回候选切片，导出待勾选 CSV")
    p_can.add_argument("--questions", required=True, help="问题清单 txt（格式：问题 ||| 设备名1,设备名2）")
    p_can.add_argument("--top-n", type=int, default=8, help="每个问题导出多少条候选（默认 8）")
    p_can.add_argument("--strategy", default="hybrid_hyde_rrf", help="用哪个检索策略召回候选")
    p_can.add_argument("--out", default="eval/datasets/to_label.csv", help="输出 CSV 路径")
    p_can.set_defaults(func=cmd_candidates)

    p_mg = sub.add_parser("merge", help="把勾选后的 CSV 合并成 JSONL 数据集")
    p_mg.add_argument("--labeled", required=True, help="已勾选的 CSV")
    p_mg.add_argument("--out", default="eval/datasets/qa_manual.jsonl", help="输出 JSONL 路径")
    p_mg.add_argument("--expect-qids", type=int, default=0, help="预期问题数，仅用于提示漏标")
    p_mg.set_defaults(func=cmd_merge)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
