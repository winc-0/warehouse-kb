# eval/answer_eval.py
"""
答案质量评估（LLM-as-judge）。

和 run_eval.py 的分工
---------------------
  run_eval.py    只测「找得准不准」→ Recall / MRR / nDCG，客观、可复算、免费
  answer_eval.py 测「答得好不好」→ 忠实度 / 相关性 / 正确性，需要 LLM 打分，要花钱

指标口径对齐 RAGAS（这样将来装了 ragas 库，两套数字可以直接对照）
----------------------------------------------------------------
  faithfulness        忠实度：答案里的每个事实性论断，是否都能从检索到的上下文里推出？
                      这是防幻觉最关键的指标 —— 检索对了但答案自己编，等于白做。
  answer_relevancy    答案相关性：答案是否正面回答了用户的问题（不跑题、不敷衍）。
  answer_correctness  答案正确性：与标准答案在事实层面是否一致（需要数据集带 reference_answer）。

为什么不用 ragas 库
-------------------
1. 项目跑在 Python 3.11 + CUDA 12.4 的固定环境里，ragas 会带来 datasets / langchain 版本约束，
   有把现有环境搞坏的风险；
2. 这里的三个指标用一次 JSON 模式调用就能拿到，实现只需要几十行，行为完全可控；
3. 指标定义是公开的、可复核的，不依赖某个库的内部实现。

用法
----
# 先小样本试跑（2 题，确认 prompt 与解析都正常）
python eval/answer_eval.py --dataset eval/datasets/qa_generated.jsonl --limit 2

# 正式跑
python eval/answer_eval.py --dataset eval/datasets/qa_generated.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval._boot import init as _boot_init  # noqa: E402

PROJECT_ROOT = _boot_init()

from eval.dataset import load_dataset  # noqa: E402
from eval.strategies import EvalContext, STRATEGIES  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

JUDGE_SYSTEM_PROMPT = "你是一个严格的 RAG 系统评测员。你只根据给定材料打分，不臆测、不客套。"

JUDGE_USER_PROMPT = """请评估下面这个问答系统的输出质量。

【用户问题】
{question}

【系统检索到的上下文】
{context}

【系统生成的答案】
{answer}

【参考答案（可能为空）】
{reference}

请从三个维度打分，每项 0~1 的小数，保留两位：
1. faithfulness（忠实度）：答案里的每个事实性论断是否都能在上面的上下文中找到依据？
   完全有依据=1.0；有一处凭空编造=0.5 以下；多处编造=0。
2. answer_relevancy（相关性）：答案是否正面、完整地回答了用户问题？跑题或敷衍要扣分。
3. answer_correctness（正确性）：与参考答案在事实层面是否一致？
   若参考答案为空，此项填 null。

同时指出答案中被上下文支持不了的具体说法（没有就填空数组）。

只返回 JSON：
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "answer_correctness": null,
  "unsupported_claims": [], "comment": "一句话点评"}}"""


def build_context(reranked_docs: Sequence[Dict[str, Any]], budget: int = 12000) -> str:
    """复用生产的上下文拼装格式（与 NodeAnswerOutput._format_reranked_docs 保持一致）。"""
    blocks: List[str] = []
    used = 0
    for idx, doc in enumerate(reranked_docs, start=1):
        tags = [f"[{idx}]"]
        for field_name, template in (("source", "[source={}]"), ("chunk_id", "[chunk_id={}]"),
                                     ("title", "[title={}]")):
            value = str(doc.get(field_name) or "").strip()
            if value:
                tags.append(template.format(value))
        score = doc.get("score")
        if score is not None:
            tags.append(f"[score={float(score):.4f}]")
        entry = " ".join(tags) + "\n" + (doc.get("content") or "")
        if used + len(entry) > budget:
            break
        blocks.append(entry)
        used += len(entry) + 2
    return "\n\n".join(blocks)


def generate_answer(llm, question: str, item_names: Sequence[str], reranked_docs) -> str:
    """
    用生产的 ANSWER_PROMPT 拼装提示词，再交给 LLM 生成。

    这里没有直接调用 NodeAnswerOutput，是因为该节点还会写 MongoDB 历史、推送 SSE、
    登记任务状态 —— 评测场景不需要这些副作用，只需要"同一个提示词 + 同一次生成"。
    """
    from processor.query_processor.prompt.answer_prompt import ANSWER_PROMPT

    prompt = ANSWER_PROMPT.format(
        context=build_context(reranked_docs) or "无参考内容",
        history="暂无历史对话",
        item_names=", ".join(item_names) if item_names else "无指定商品",
        question=question,
    )
    return (llm.invoke(prompt).content or "").strip()


def judge(llm_judge, question: str, context: str, answer: str, reference: str = "") -> Dict[str, Any]:
    """调用裁判模型打分。解析失败时返回 None 值并记录原因。"""
    prompt = JUDGE_USER_PROMPT.format(
        question=question,
        context=context[:12000] or "（无）",
        answer=answer or "（模型未输出任何内容）",
        reference=reference or "（无参考答案）",
    )
    resp = llm_judge.invoke([("system", JUDGE_SYSTEM_PROMPT), ("user", prompt)])
    raw = (resp.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
    data = json.loads(raw)
    for key in ("faithfulness", "answer_relevancy"):
        value = data.get(key)
        data[key] = float(value) if value is not None else None
    value = data.get("answer_correctness")
    data["answer_correctness"] = float(value) if value is not None else None
    return data


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python eval/answer_eval.py", description="答案质量评估")
    parser.add_argument("--dataset", default="eval/datasets/qa_template.jsonl")
    parser.add_argument("--strategy", default="full", help="用哪个检索策略产出上下文（默认 full）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--sleep", type=float, default=0.0, help="每次调用之间的间隔秒数，避开限流")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    items = load_dataset(dataset_path)
    if args.limit > 0:
        items = items[: args.limit]

    if args.strategy not in STRATEGIES:
        print(f"[x] 未知策略：{args.strategy}；可选 {list(STRATEGIES)}")
        return 1
    strategy = STRATEGIES[args.strategy]

    ctx = EvalContext(use_cache=True)
    from utils.llm_utils import get_llm_client

    llm_gen = get_llm_client()
    llm_judge = get_llm_client(json_mode=True)

    print("=" * 78)
    print(" 履约知识助手 Warehouse-KB · 答案质量评估（LLM-as-judge，口径对齐 RAGAS）")
    print("=" * 78)
    print(f" 数据集：{dataset_path}  样本数：{len(items)}")
    print(f" 检索策略：{strategy.label}")
    print(f" 预计消耗：生成 {len(items)} 次 + 裁判 {len(items)} 次 LLM 调用")
    print("-" * 78)

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        started = time.time()
        try:
            ranked = strategy.retrieve(ctx, item.question, item.item_names)
            # 检索策略只返回 chunk_id，生成答案需要正文，这里按 id 批量补回
            docs = _hydrate(ctx, ranked)
            answer = generate_answer(llm_gen, item.question, item.item_names, docs)
            context = build_context(docs)
            verdict = judge(llm_judge, item.question, context, answer, item.reference_answer)
        except Exception as exc:
            print(f"  [{idx}/{len(items)}] {item.qid} 失败：{type(exc).__name__}: {exc}")
            continue

        rows.append(
            {
                "qid": item.qid,
                "question": item.question,
                "faithfulness": verdict.get("faithfulness"),
                "answer_relevancy": verdict.get("answer_relevancy"),
                "answer_correctness": verdict.get("answer_correctness"),
                "unsupported": "; ".join(verdict.get("unsupported_claims") or []),
                "comment": verdict.get("comment") or "",
                "answer_len": len(answer),
                "elapsed": round(time.time() - started, 2),
                "source": item.source,
            }
        )
        print(f"  [{idx}/{len(items)}] {item.qid} 忠实度={_fmt(verdict.get('faithfulness'))} "
              f"相关性={_fmt(verdict.get('answer_relevancy'))} "
              f"正确性={_fmt(verdict.get('answer_correctness'))}")
        if args.sleep:
            time.sleep(args.sleep)

    if not rows:
        print("\n[x] 没有任何样本评估成功，请检查 LLM 配置与检索链路。")
        return 1

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 78)
    print(" 汇总（均值）")
    print("=" * 78)
    summary: Dict[str, Any] = {}
    for key, label in (("faithfulness", "忠实度"), ("answer_relevancy", "答案相关性"),
                       ("answer_correctness", "答案正确性")):
        values = [r[key] for r in rows if r[key] is not None]
        if values:
            summary[key] = round(fmean(values), 4)
            print(f"  {label:<10} {summary[key]:.4f}   （{len(values)} 条有效样本）")
        else:
            summary[key] = None
            print(f"  {label:<10} —        （无有效样本）")

    # ---------------- 落盘 ----------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stamp}_answer_eval.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / f"{stamp}_answer_eval.json"
    json_path.write_text(
        json.dumps({"summary": summary, "strategy": strategy.key, "rows": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n产物：")
    print(f"  明细 {csv_path}")
    print(f"  原始 {json_path}")
    return 0


def _hydrate(ctx: EvalContext, chunk_ids: Sequence[int]) -> List[Dict[str, Any]]:
    """按 chunk_id 批量取回切片正文（检索策略只返回 id，生成答案需要正文）。"""
    if not chunk_ids:
        return []
    expr = f"chunk_id in {json.dumps([int(c) for c in chunk_ids])}"
    try:
        rows = ctx.client.query(
            collection_name=ctx.collection,
            filter=expr,
            output_fields=["chunk_id", "content", "item_name", "title", "file_title"],
        )
    except Exception:
        return [{"chunk_id": int(c), "content": "", "title": "", "source": "local"} for c in chunk_ids]

    by_id = {int(r["chunk_id"]): r for r in rows}
    docs: List[Dict[str, Any]] = []
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        row = by_id.get(int(chunk_id))
        if not row:
            continue
        docs.append(
            {
                "chunk_id": int(chunk_id),
                "content": row.get("content") or "",
                "title": row.get("item_name") or row.get("title") or "",
                "source": "local",
                "score": None,
                "rank": rank,
            }
        )
    return docs


def _fmt(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
