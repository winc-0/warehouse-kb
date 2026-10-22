# eval/run_eval.py
"""
检索链路消融评测主脚本。

它回答的问题只有一个：**「三路混合召回 + RRF 融合 + 重排 + 断崖截断」这套设计，
每一层各自贡献了多少？**

跑完会得到一张「逐层叠加」的对比表（Recall@K / MRR / nDCG@K），
以及相对基线的增量与显著性 —— 这些数字才是简历上"召回率从 X 提升到 Y"的出处。

用法
----
# 0) 先做环境自检（不消耗任何额度）
python eval/run_eval.py --preflight

# 1) 全量消融（6 组策略）
python eval/run_eval.py --dataset eval/datasets/qa_generated.jsonl

# 2) 只跑不花钱的策略（跳过 HyDE / 重排）
python eval/run_eval.py --skip-api

# 3) 先小样本试跑，确认链路通了再全量
python eval/run_eval.py --limit 10

输出（全部落在 eval/results/）
------------------------------
  <ts>_report.md     给人和面试官看的对比报告
  <ts>_summary.csv   汇总指标表
  <ts>_per_query.csv 逐题逐策略的明细（用于排查"哪几题挂了"）
  <ts>_raw.json      原始检索结果（保留 chunk_id 序列，便于复查）
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval._boot import init as _boot_init  # noqa: E402

PROJECT_ROOT = _boot_init()

from eval.dataset import QAItem, describe, load_dataset  # noqa: E402
from eval.metrics import (  # noqa: E402
    DEFAULT_KS,
    HEADLINE_K,
    aggregate,
    aggregate_std,
    evaluate_many,
    fmt_delta,
    fmt_pct,
    paired_bootstrap_p_value,
)
from eval.strategies import (  # noqa: E402
    DEFAULT_BASELINE,
    EvalContext,
    check_environment,
    resolve_strategies,
)

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def _git_commit() -> str:
    """当前代码版本，写进报告头部，便于日后核对「这份数字是哪版代码跑出来的」。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _display_path(path: Path) -> str:
    """报告里优先用仓库内相对路径，别把本机绝对路径带出去。"""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except (ValueError, OSError):
        return Path(path).as_posix()


# --------------------------------------------------------------------------- #
# 评测执行
# --------------------------------------------------------------------------- #
def run_strategy(
    strategy,
    items: Sequence[QAItem],
    ctx: EvalContext,
    ks: Sequence[int],
    headline_k: int,
    verbose: bool,
) -> Dict[str, Any]:
    """
    跑完一个策略的全部问题。

    :return: {"metrics": [逐题指标], "retrieved": [逐题 chunk_id 序列], "failures": [...]}
    """
    rankings: List[List[int]] = []
    failures: List[str] = []
    started = time.time()

    for idx, item in enumerate(items, start=1):
        try:
            ids = strategy.retrieve(ctx, item.question, item.item_names)
        except Exception as exc:  # 单题失败不中断整体，但要记录下来
            failures.append(f"{item.qid}: {type(exc).__name__}: {exc}")
            ids = []
        rankings.append([int(cid) for cid in ids if cid is not None])
        if verbose and idx % 10 == 0:
            print(f"        {idx}/{len(items)} ...", flush=True)

    elapsed = time.time() - started
    metrics = evaluate_many(rankings, [i.relevant for i in items], ks=ks, headline_k=headline_k)
    return {
        "strategy": strategy.key,
        "label": strategy.label,
        "metrics": metrics,
        "retrieved": rankings,
        "failures": failures,
        "elapsed": elapsed,
    }


def describe_failure(reason: str) -> str:
    """把底层异常翻译成可操作的下一步建议。"""
    low = reason.lower()
    if any(k in low for k in ("quota", "exhausted", "arrearage", "balance", "403", "402")):
        return ("重排接口额度已用尽或未开通。三条出路：① 到 DashScope 控制台充值/开通；"
                "② 换成本地 BGE-Reranker（需先下载模型）；"
                "③ 用 --skip-api 或 --strategies dense_only,sparse_only,hybrid 跑不依赖外部接口的部分。")
    if "milvus" in low or "connect" in low:
        return "Milvus 连接异常。先跑 `python eval/run_eval.py --preflight` 排查。"
    if "timeout" in low or "timed out" in low:
        return "调用超时。可用 --limit 缩小样本量重试。"
    if "model" in low and "not" in low:
        return "模型名或模型权限有问题，检查 .env 里的 LLM_DEFAULT_MODEL / TEXT_RERANK_MODEL。"
    return ""


def build_summary(
    results: Dict[str, Dict[str, Any]],
    items: Sequence[QAItem],
    ks: Sequence[int],
    headline_k: int,
    baseline: str,
) -> Dict[str, Any]:
    """把各策略的逐题指标聚合成汇总表 + 增量 + 显著性。"""
    order = [k for k in results]
    metric_names = [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks] + [
        f"precision@{headline_k}",
        f"ndcg@{headline_k}",
        "mrr",
        "map",
    ]

    summary: Dict[str, Any] = {"order": order, "metric_names": metric_names, "rows": {}, "deltas": {}}

    for key in order:
        row = results[key]
        summary["rows"][key] = {
            "label": row["label"],
            "n": len(row["metrics"]),
            "failures": len(row["failures"]),
            "elapsed": row["elapsed"],
            "mean": aggregate(row["metrics"]),
            "std": aggregate_std(row["metrics"]),
        }

    if baseline in results:
        base_metrics = results[baseline]["metrics"]
        for key in order:
            if key == baseline:
                continue
            cur_metrics = results[key]["metrics"]
            delta: Dict[str, float] = {}
            pvalue: Dict[str, float] = {}
            for name in metric_names:
                a = [float(m[name]) for m in base_metrics]
                b = [float(m[name]) for m in cur_metrics]
                delta[name] = (sum(b) - sum(a)) / len(a) if a else 0.0
                # 显著性只对诊断指标算，控制 bootstrap 的计算量
                if name in DIAGNOSTIC_METRICS:
                    pvalue[name] = paired_bootstrap_p_value(a, b, n_boot=2000)
            summary["deltas"][key] = {"delta": delta, "p": pvalue}

    return summary


# --------------------------------------------------------------------------- #
# 报告渲染
# --------------------------------------------------------------------------- #
# 用于判断"这批题有没有区分度"的指标，按优先级排列
DIAGNOSTIC_METRICS = ("recall@1", "ndcg@5", "mrr", "map", "recall@3", "recall@5")
# 区分度门槛：相对基线的最大提升小于这个值，就认为该指标分辨不出链路差异
SPREAD_THRESHOLD = 0.02

_METRIC_LABELS = {"recall": "Recall", "ndcg": "nDCG", "hit": "Hit", "precision": "Precision"}


def pretty_metric(name: str) -> str:
    """recall@1 -> Recall@1；mrr -> MRR。用于正文里的人类可读写法。"""
    if name in ("mrr", "map"):
        return name.upper()
    if "@" in name:
        base, k = name.split("@", 1)
        return f"{_METRIC_LABELS.get(base, base)}@{k}"
    return name


def diagnose_metrics(summary: Dict[str, Any], baseline: str, headline_k: int = HEADLINE_K) -> Dict[str, Any]:
    """
    判断这批数据到底能不能区分链路优劣，返回一个状态机结果。

    status 取值：
        single_strategy     只有一组策略成功执行，没有可比对象
        no_baseline         基线策略未参与本次评测
        no_discrimination   所有诊断指标都饱和，本次数据无效
        headline_saturated  头号指标饱和，但其他指标仍有区分度 —— 换指标即可
        ok                  正常

    为什么要分开判断：单看一个指标会误判。语料小的时候 Recall@5 很容易顶到 100%，
    但 Recall@1 / nDCG / MRR 往往还在明显变化，这组数据依然是有效的，
    只是结论不能挂在 Recall@5 上。
    """
    empty = {"ok": False, "status": "ok", "per_metric": {}, "discriminating": [], "best_metric": None}

    if len(summary.get("order", [])) < 2:
        return {**empty, "status": "single_strategy"}
    if baseline not in summary["rows"]:
        return {**empty, "status": "no_baseline"}

    per_metric: Dict[str, Dict[str, float]] = {}
    for name in DIAGNOSTIC_METRICS:
        if name not in summary["rows"][baseline]["mean"]:
            continue
        base = summary["rows"][baseline]["mean"][name]
        spread = max((summary["rows"][k]["mean"][name] - base) for k in summary["order"])
        per_metric[name] = {"baseline": base, "spread": spread}

    discriminating = [n for n, v in per_metric.items() if v["spread"] >= SPREAD_THRESHOLD]
    headline = f"recall@{headline_k}"
    head_info = per_metric.get(headline)

    if not discriminating:
        status = "no_discrimination"
    elif head_info and head_info["baseline"] >= 0.95 and head_info["spread"] < SPREAD_THRESHOLD:
        status = "headline_saturated"
    else:
        status = "ok"

    return {
        "ok": status in ("ok", "headline_saturated"),
        "status": status,
        "per_metric": per_metric,
        "discriminating": discriminating,
        "best_metric": max(discriminating, key=lambda n: per_metric[n]["spread"]) if discriminating else None,
    }


# 诊断状态 -> 一句话说明，用于控制台标题和报告小标题
STATUS_TITLES = {
    "single_strategy": "只有一组策略成功执行，无法做消融归因",
    "no_baseline": "基线策略未参与本次评测",
    "no_discrimination": "这组数字无法区分链路优劣",
    "headline_saturated": "头号指标已饱和，需换指标下结论",
}


def detect_saturation(
    summary: Dict[str, Any], baseline: str, headline_k: int,
    diag: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """返回预警文案；数据集有区分度时返回 None。"""
    diag = diag or diagnose_metrics(summary, baseline, headline_k)
    status = diag["status"]

    if status == "single_strategy":
        return ("本次只有一组策略成功执行，没有可比对象，**无法做消融归因**。\n\n"
                "请查看下面「未跑完的策略」里失败的原因，处理完后重跑。")

    if status == "no_discrimination":
        return (
            f"基线（{summary['rows'][baseline]['label']}）在所有诊断指标上都已接近上限，"
            f"没有任何策略能拉开超过 {fmt_pct(SPREAD_THRESHOLD)} 的差距。"
            "**这组数字无法区分链路优劣**，不能作为优化成果的依据。\n\n"
            "常见成因与对策：\n"
            "1. **语料太小** —— 单个商品下的切片数接近甚至少于 Top-K，检索几乎不可能失手。"
            "对策：导入更多文档，让每个商品的切片数达到 Top-K 的 5 倍以上。\n"
            "2. **题目由切片反向生成** —— 题目和正确答案出自同一段文字，天然好找。"
            "对策：改用 `build_dataset.py candidates` 走人工标注，题目按真实使用场景自己写。\n"
            "3. **题目全部带完整型号** —— 型号本身就是强过滤条件。"
            "对策：用 `build_dataset.py generate --style hard`，让问题口语化、去掉型号。"
        )

    if status == "headline_saturated":
        headline = f"recall@{headline_k}"
        info = diag["per_metric"][headline]
        others = "、".join(f"`{n}`" for n in diag["discriminating"][:3])
        return (
            f"头号指标 Recall@{headline_k} 在基线上已达 {fmt_pct(info['baseline'])}，"
            "顶到了天花板，**不能**用它下结论。\n\n"
            f"不过同一批数据在 {others} 上仍有明显区分度，"
            f"请改以 `{diag['best_metric']}` 作为结论指标 —— 报告第三节的增量表已按各指标分别列出。"
        )
    return None


def render_report(
    args: argparse.Namespace,
    items: Sequence[QAItem],
    results: Dict[str, Dict[str, Any]],
    summary: Dict[str, Any],
    ks: Sequence[int],
    headline_k: int,
    dataset_path: Path,
    failed: Dict[str, str],
    ctx: EvalContext,
) -> str:
    order = summary["order"]
    metric_names = summary["metric_names"]
    baseline = args.baseline
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    add = lines.append

    add("# 履约知识助手 Warehouse-KB · 检索链路消融评测报告")
    add("")
    add(f"- 评测时间：{now}")
    if _git_commit():
        add(f"- 代码版本：`{_git_commit()}`")
    add(f"- 数据集：`{_display_path(dataset_path)}`")
    add(f"- 样本数：{len(items)}（平均每题 {describe(list(items))['平均正确切片数']} 条正确切片）")
    add(f"- 截断位置 K：{', '.join(str(k) for k in ks)}；头号指标 K={headline_k}")
    add(f"- 基线策略：`{baseline}`（{results[baseline]['label'] if baseline in results else 'N/A'}）")
    add(f"- 资源消耗：向量编码 {ctx.embedding_calls} 次 / LLM 调用 {ctx.llm_calls} 次"
        f"（HyDE 磁盘缓存命中 {ctx.hyde_cache_hits} 次）/ 重排接口 {ctx.rerank_calls} 次")
    add("")

    # ---------------- 有效性预警 ----------------
    diag = diagnose_metrics(summary, baseline, headline_k)
    conclusion_metric = diag["best_metric"] or f"recall@{headline_k}"
    warning = detect_saturation(summary, baseline, headline_k, diag)
    if warning:
        add(f"## ⚠ {STATUS_TITLES.get(diag['status'], '有效性提示')}")
        add("")
        for para in warning.split("\n\n"):
            add(para)
            add("")

    # ---------------- 指标区分度诊断 ----------------
    if diag.get("per_metric"):
        add("### 指标区分度诊断")
        add("")
        add("判断这批题能不能区分链路，不能只看一个指标 —— 逐项检查相对基线的最大提升：")
        add("")
        add("| 指标 | 基线值 | 最大提升 | 是否有区分度 |")
        add("|---|---|---|---|")
        for name, info in diag["per_metric"].items():
            mark = "有" if info["spread"] >= SPREAD_THRESHOLD else "无（已饱和）"
            add(f"| `{name}` | {fmt_pct(info['baseline'])} | {fmt_delta(info['spread'])} | {mark} |")
        add("")
        if diag["ok"]:
            add(f"结论指标取 `{conclusion_metric}`（区分度最大的那个）。")
        else:
            add("所有诊断指标均已饱和，本次数据不具备区分能力。")
        add("")

    # ---------------- 主表 ----------------
    add("## 一、各策略指标对比")
    add("")
    header = ["策略"] + [f"Recall@{k}" for k in ks] + [f"nDCG@{headline_k}", "MRR", "MAP"]
    add("| " + " | ".join(header) + " |")
    add("|" + "---|" * len(header))
    for key in order:
        row = summary["rows"][key]
        mean = row["mean"]
        cells = [f"{row['label']}"]
        cells += [fmt_pct(mean[f"recall@{k}"]) for k in ks]
        cells += [
            fmt_pct(mean[f"ndcg@{headline_k}"]),
            fmt_pct(mean["mrr"]),
            fmt_pct(mean["map"]),
        ]
        add("| " + " | ".join(cells) + " |")
    add("")

    # 若有效样本为 0（例如全部失败），后面不渲染增量
    if baseline in results and args.baseline in summary["rows"]:
        add(f"## 二、相对基线的增量（{results[baseline]['label']}）")
        add("")
        header2 = ["策略"] + [f"ΔRecall@{k}" for k in ks] + [f"ΔnDCG@{headline_k}", "ΔMRR", f"p({conclusion_metric})"]
        add("| " + " | ".join(header2) + " |")
        add("|" + "---|" * len(header2))
        for key in order:
            if key == baseline:
                continue
            delta = summary["deltas"].get(key, {}).get("delta", {})
            pvals = summary["deltas"].get(key, {}).get("p", {})
            cells = [summary["rows"][key]["label"]]
            cells += [fmt_delta(delta.get(f"recall@{k}", 0.0)) for k in ks]
            cells += [
                fmt_delta(delta.get(f"ndcg@{headline_k}", 0.0)),
                fmt_delta(delta.get("mrr", 0.0)),
            ]
            p = pvals.get(conclusion_metric)
            d = delta.get(conclusion_metric, 0.0)
            if p is None:
                cells.append("—")
            elif p < 0.05:
                cells.append(f"p={p:.3f}（显著" + ("提升" if d > 0 else "下降") + "）")
            else:
                cells.append(f"p={p:.3f}（不显著）")
            add("| " + " | ".join(cells) + " |")
        add("")
        add(f"> 显著性用配对 bootstrap 检验（同一批问题、逐题配对，2000 次重采样，双侧），"
            f"检验对象是结论指标 `{conclusion_metric}`。")
        add("> p < 0.05 表示该变化不太可能由抽样波动造成。样本量小时 p 值会偏大，属正常现象。")
        add("")

    # ---------------- 逐层归因 ----------------
    add("## 三、逐层叠加的增量归因")
    add("")
    add(f"按实现顺序逐层启用，观察每一步单独带来的收益（结论指标取 `{pretty_metric(conclusion_metric)}`）：")
    add("")
    add(f"| 步骤 | 策略 | {pretty_metric(conclusion_metric)} | 相对上一步 |")
    add("|---|" + "---|" * 3)
    chain = [k for k in ("dense_only", "hybrid", "hybrid_hyde_rrf", "rerank", "full") if k in results]
    prev_value: Optional[float] = None
    for idx, key in enumerate(chain, start=1):
        value = summary["rows"][key]["mean"][conclusion_metric]
        if prev_value is None:
            delta_text = "—（基线）"
        else:
            delta_text = fmt_delta(value - prev_value)
        add(f"| {idx} | {summary['rows'][key]['label']} | {fmt_pct(value)} | {delta_text} |")
        prev_value = value
    add("")

    # ---------------- 简历表述 ----------------
    add("## 四、可直接使用的成果表述")
    add("")
    if warning and not diag["ok"]:
        add("**本次结果触发了有效性预警，因此不生成成果表述。** "
            "先按上面的建议换一批更有区分度的题，再回来跑一次。")
        add("")
    elif baseline in results and chain:
        best_key = max(order, key=lambda k: summary["rows"][k]["mean"][conclusion_metric])
        base_v = summary["rows"][baseline]["mean"][conclusion_metric]
        best_v = summary["rows"][best_key]["mean"][conclusion_metric]
        add("> 下面这段是根据本次实测数字自动拼出来的，**请核对后再使用**，不要直接照抄。")
        add("")
        # 主句用结论指标；补句只列「不是结论指标」的其他关键指标，避免重复同一组数字
        extras = [
            m for m in ("recall@1", "ndcg@5", "mrr")
            if m != conclusion_metric and m in summary["rows"][baseline]["mean"]
        ]
        extra_text = ""
        if extras:
            bits = [
                f"{pretty_metric(m)} 由 {fmt_pct(summary['rows'][baseline]['mean'][m])} "
                f"提升至 {fmt_pct(summary['rows'][best_key]['mean'][m])}"
                for m in extras
            ]
            extra_text = "；同时 " + "，".join(bits) + "。"
        add(
            f"在自建 {len(items)} 题标注集上完成检索链路消融实验："
            f"以「{summary['rows'][baseline]['label']}」为基线（{pretty_metric(conclusion_metric)} = {fmt_pct(base_v)}），"
            f"引入「三路混合召回 + RRF 融合 + Cross-Encoder 重排 + 断崖检测动态截断」后达到 "
            f"{fmt_pct(best_v)}（{fmt_delta(best_v - base_v)}）{extra_text or '。'}"
        )
        add("")
        if warning and diag["ok"]:
            add(f"（说明：头号指标 Recall@{headline_k} 在基线已达 "
                f"{fmt_pct(summary['rows'][baseline]['mean'][f'recall@{headline_k}'])}、已饱和，"
                f"因此结论以 {pretty_metric(conclusion_metric)} 为准。）")
            add("")

        # 样本量提示：p 值落在临界区时，必须提醒结论还不够稳
        best_p = summary.get("deltas", {}).get(best_key, {}).get("p", {}).get(conclusion_metric)
        if best_p is not None and best_p >= 0.05:
            need = max(len(items) * 2, 60)
            add(f"**样本量提示**：该提升的显著性检验 p={best_p:.3f}，未达到 p<0.05。"
                f"本次只有 {len(items)} 题，单题权重高达 {fmt_pct(1 / len(items))}，"
                f"差异容易被抽样波动掩盖。要把这个结论说稳，建议把标注集扩到 {need} 题以上。")
            add("")
    else:
        add("（基线策略未参与本次评测，无法生成对比表述）")
        add("")

    # ---------------- 失败策略 ----------------
    add("## 五、未跑完的策略")
    add("")
    if failed:
        for key, reason in failed.items():
            add(f"- `{key}`：{reason}")
            hint = describe_failure(reason)
            if hint:
                add(f"  - 建议：{hint}")
    else:
        add("无。本次所有策略均完整执行。")
    add("")

    # ---------------- 复现方式 ----------------
    add("## 六、复现命令")
    add("")
    add("```bash")
    add(f"python eval/run_eval.py --dataset {_display_path(dataset_path)}"
        + (f" --limit {args.limit}" if args.limit else "")
        + (f" --strategies {args.strategies}" if args.strategies else ""))
    add("```")
    add("")

    # ---------------- 局限 ----------------
    add("## 七、结论的适用边界（务必一并阅读）")
    add("")
    add("1. **不含联网检索**：`node_web_search_mcp` 未参与评测，因为结果依赖外部网络、无稳定 chunk_id、不可复现。"
        "本报告只说明「本地知识库检索」这一段的优劣。")
    add("2. **RRF 保留条数放宽**：评测中 RRF 保留 10 条（生产为 5），以便观察 Recall@10；"
        f"在 K≤{HEADLINE_K} 的指标上，重排 + 断崖截断后两者结果一致。")
    add("3. **二值相关性**：标注只区分「能/不能回答问题」，未做 0~3 级相关性分级，"
        "因此 nDCG 是二值版本，数值会略高于分级版本。")
    src = describe(list(items))["来源分布"]
    src_text = "、".join(f"{k}={v}" for k, v in src.items()) or "无"
    generated = src.get("generated", 0)
    add(f"4. **标注来源偏差**：本次来源分布为 {src_text}。"
        + (f"其中 {generated}/{len(items)} 条由切片反向生成（题目与答案同源），"
           "这部分样本的指标会系统性偏乐观；人工标注题越多，结论越可信。" if generated else ""))
    add("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #
def write_outputs(
    out_dir: Path,
    stamp: str,
    items: Sequence[QAItem],
    results: Dict[str, Dict[str, Any]],
    summary: Dict[str, Any],
    report_md: str,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # 1) 报告
    report_path = out_dir / f"{stamp}_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    paths["report"] = report_path

    # 2) 汇总 CSV
    summary_path = out_dir / f"{stamp}_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["策略key", "策略", "样本数", "失败数", "耗时秒"] + summary["metric_names"] + [
            f"{m}_std" for m in summary["metric_names"]
        ])
        for key in summary["order"]:
            row = summary["rows"][key]
            writer.writerow(
                [key, row["label"], row["n"], row["failures"], f"{row['elapsed']:.2f}"]
                + [f"{row['mean'][m]:.4f}" for m in summary["metric_names"]]
                + [f"{row['std'][m]:.4f}" for m in summary["metric_names"]]
            )
    paths["summary"] = summary_path

    # 3) 逐题明细
    per_query_path = out_dir / f"{stamp}_per_query.csv"
    metric_names = summary["metric_names"]
    with per_query_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["qid", "question", "source", "正确切片数", "策略key", "策略", "召回序列"]
                        + metric_names)
        for key in summary["order"]:
            row = results[key]
            for item, retrieved, metrics in zip(items, row["retrieved"], row["metrics"]):
                writer.writerow(
                    [
                        item.qid,
                        item.question,
                        item.source,
                        len(item.ground_truth_chunk_ids),
                        key,
                        row["label"],
                        " ".join(str(c) for c in retrieved),
                    ]
                    + [f"{metrics[m]:.4f}" for m in metric_names]
                )
    paths["per_query"] = per_query_path

    # 4) 原始 JSON
    raw_path = out_dir / f"{stamp}_raw.json"
    payload = {
        "generated_at": stamp,
        "dataset": [{"qid": i.qid, "question": i.question, "item_names": i.item_names,
                     "ground_truth_chunk_ids": i.ground_truth_chunk_ids, "source": i.source} for i in items],
        "strategies": {
            key: {
                "label": results[key]["label"],
                "retrieved": results[key]["retrieved"],
                "failures": results[key]["failures"],
                "mean": summary["rows"][key]["mean"],
            }
            for key in summary["order"]
        },
    }
    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["raw"] = raw_path

    return paths


# --------------------------------------------------------------------------- #
# 控制台输出
# --------------------------------------------------------------------------- #
def print_table(summary: Dict[str, Any], ks: Sequence[int], headline_k: int) -> None:
    cols = [f"R@{k}" for k in ks] + [f"nDCG@{headline_k}", "MRR", "MAP"]
    width_label = max([len(summary["rows"][k]["label"]) for k in summary["order"]] + [12])
    header = "策略".ljust(width_label) + "  " + "  ".join(c.rjust(9) for c in cols)
    print(header)
    print("-" * len(header))
    for key in summary["order"]:
        row = summary["rows"][key]
        mean = row["mean"]
        values = [fmt_pct(mean[f"recall@{k}"]) for k in ks]
        values += [fmt_pct(mean[f"ndcg@{headline_k}"]), fmt_pct(mean["mrr"]), fmt_pct(mean["map"])]
        print(row["label"].ljust(width_label) + "  " + "  ".join(v.rjust(9) for v in values))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def check_ground_truth(ctx: Any, items: Sequence[QAItem]) -> List[Tuple[str, int]]:
    """
    校验标注集的金标 chunk_id 是否还在库里。

    为什么必须做：清理重复导入、重灌文档、改切分参数之后，旧的 chunk_id 会整批消失。
    如果不前置拦截，表现在于"指标莫名掉了十几个点"，得先怀疑模型、再怀疑策略，
    最后才想到是数据没了 —— 前面那次就这么绕了一圈。查一次全量 chunk_id 只要几秒，
    这里直接判死，不许带着失效金标跑出数字。

    返回 [(qid, chunk_id), ...]；空列表表示全部有效。
    """
    wanted = {cid for item in items for cid in item.ground_truth_chunk_ids}
    if not wanted:
        return []

    alive: set = set()
    ids = sorted(wanted)
    try:
        for start in range(0, len(ids), 500):
            batch = ids[start: start + 500]
            rows = ctx.client.query(
                collection_name=ctx.collection,
                filter=f"chunk_id in {batch}",
                output_fields=["chunk_id"],
                limit=len(batch),
            )
            alive.update(int(r["chunk_id"]) for r in rows)
    except Exception as exc:
        print(f"  [warn] 金标校验失败（无法查询 Milvus）：{exc}")
        return []

    return [(item.qid, cid) for item in items for cid in item.ground_truth_chunk_ids if cid not in alive]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python eval/run_eval.py",
        description="履约知识助手检索链路消融评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="eval/datasets/qa_hard.jsonl",
        help="JSONL 数据集路径（默认 164 题标注集；qa_template.jsonl 只是字段模板，只有 14 条）",
    )
    parser.add_argument("--strategies", default="", help="逗号分隔的策略 key（默认全部）")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help=f"基线策略 key（默认 {DEFAULT_BASELINE}）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    parser.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS), help="截断位置，如 1,3,5,10")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR), help="输出目录")
    parser.add_argument("--no-cache", action="store_true", help="禁用 HyDE 磁盘缓存")
    parser.add_argument("--skip-api", action="store_true", help="跳过需要 LLM / 重排接口的策略")
    parser.add_argument("--preflight", action="store_true", help="只做环境自检，不跑评测")
    parser.add_argument("--force", action="store_true", help="自检不通过也继续跑")
    parser.add_argument("--allow-dead-gt", action="store_true",
                        help="金标 chunk_id 失效时仍继续（指标会失真，仅用于排查）")
    parser.add_argument("--verbose", action="store_true", help="打印逐题进度")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    headline_k = min(HEADLINE_K, max(ks)) if ks else HEADLINE_K
    ctx = EvalContext(use_cache=not args.no_cache, verbose=args.verbose)

    # ---------------- 环境自检 ----------------
    print("=" * 78)
    print(" 履约知识助手 Warehouse-KB · 检索链路消融评测")
    print("=" * 78)
    print("[自检] 运行环境")
    env_ok, env_lines = check_environment(ctx)
    for line in env_lines:
        print("  " + line)

    if args.preflight:
        print("\n自检结束（--preflight 模式，未执行评测）")
        return 0 if env_ok else 1

    if not env_ok and not args.force:
        print("\n[x] 环境自检未通过，已中止。修好上面的问题后重试，或用 --force 强行继续。")
        return 1

    # ---------------- 数据集 ----------------
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    try:
        items = load_dataset(dataset_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[x] 数据集不可用：{exc}")
        print("\n提示：评测必须有带 ground truth 的标注集。两种生成方式：")
        print("  ① 半自动反向生成（几分钟出结果，题目偏简单）：")
        print("       python eval/build_dataset.py generate --per-item 8 --max-items 6 \\")
        print("           --out eval/datasets/qa_generated.jsonl")
        print("  ② 人工标注（质量最高）：")
        print("       python eval/build_dataset.py candidates --questions questions.txt \\")
        print("           --out eval/datasets/to_label.csv")
        print("       # 在 Excel 里给 included 列填 1，保存后：")
        print("       python eval/build_dataset.py merge --labeled eval/datasets/to_label.csv \\")
        print("           --out eval/datasets/qa_manual.jsonl")
        return 1
    if args.limit > 0:
        items = items[: args.limit]

    stats = describe(items)
    print(f"\n[数据集] {dataset_path}")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # ---------------- 金标有效性 ----------------
    dead = check_ground_truth(ctx, items)
    if dead:
        by_qid: Dict[str, List[int]] = {}
        for qid, cid in dead:
            by_qid.setdefault(qid, []).append(cid)
        print(f"\n[x] 标注集里有 {len(dead)} 个金标 chunk_id 在库中不存在，涉及 {len(by_qid)} 道题：")
        for qid, cids in list(by_qid.items())[:10]:
            print(f"      {qid}: {cids}")
        if len(by_qid) > 10:
            print(f"      ... 另有 {len(by_qid) - 10} 道题")
        print("\n    常见原因：清理了重复导入的切片、用 --force 重灌导致 chunk_id 全变、改了切分参数。")
        print("    这两种情况下跑出来的指标没有意义（金标永远召回不到，R@1 会假性下跌）。")
        print("    处理办法：用 eval/build_dataset.py 重新生成这批题，或手工把 ground_truth_chunk_ids")
        print("    改指当前库里对应的切片。确认要带着失效金标继续，加 --allow-dead-gt。")
        if not args.allow_dead_gt:
            return 1

    strategies = resolve_strategies([s for s in args.strategies.split(",") if s.strip()] or None)
    if args.skip_api:
        skipped = [s.key for s in strategies if s.needs_llm]
        strategies = [s for s in strategies if not s.needs_llm]
        if skipped:
            print(f"  [--skip-api] 已跳过：{', '.join(skipped)}")

    if args.baseline not in [s.key for s in strategies]:
        print(f"  [warn] 基线 {args.baseline} 不在本次策略列表中，将不生成增量对比")

    llm_calls = len(items) * sum(1 for s in strategies if s.needs_llm)
    rerank_calls = len(items) * sum(1 for s in strategies if s.needs_rerank_api)
    print(f"\n[计划] {len(strategies)} 组策略 × {len(items)} 题")
    print(f"       预计消耗：LLM 约 {llm_calls} 次（HyDE）· 重排接口约 {rerank_calls} 次")
    print("       HyDE 结果会写入 eval/.cache/，同一问题复跑不再重复调用；")
    print("       重排接口每次都会真实调用，可用 --skip-api 或 --strategies 避开。")

    # ---------------- 逐策略评测 ----------------
    print("\n[执行]")
    results: Dict[str, Dict[str, Any]] = {}
    failed: Dict[str, str] = {}
    for idx, strategy in enumerate(strategies, start=1):
        print(f"  [{idx}/{len(strategies)}] {strategy.label} ...", flush=True)
        try:
            outcome = run_strategy(strategy, items, ctx, ks=ks, headline_k=headline_k, verbose=args.verbose)
        except Exception as exc:
            failed[strategy.key] = f"{type(exc).__name__}: {exc}"
            print(f"        [x] 整组失败：{failed[strategy.key]}")
            continue

        # 整组失败（例如接口不可用）时不计入对比
        if len(outcome["failures"]) == len(items) and items:
            failed[strategy.key] = f"全部 {len(items)} 题失败，首个错误：{outcome['failures'][0]}"
            print(f"        [x] {failed[strategy.key]}")
            continue

        results[strategy.key] = outcome
        recall = aggregate(outcome["metrics"]).get(f"recall@{headline_k}", 0.0)
        note = f"，{len(outcome['failures'])} 题失败" if outcome["failures"] else ""
        print(f"        Recall@{headline_k} = {fmt_pct(recall)}  用时 {outcome['elapsed']:.1f}s{note}")

    if not results:
        print("\n[x] 没有任何策略跑出结果，无法生成报告。")
        return 1

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 78)
    print(" 汇总")
    print("=" * 78)
    summary = build_summary(results, items, ks=ks, headline_k=headline_k, baseline=args.baseline)
    print_table(summary, ks=ks, headline_k=headline_k)

    saturation = detect_saturation(summary, args.baseline, headline_k)
    if saturation:
        diag = diagnose_metrics(summary, args.baseline, headline_k)
        title = STATUS_TITLES.get(diag["status"], "有效性提示")
        print("\n" + "!" * 78)
        print(f" ⚠ 有效性预警：{title}")
        print("!" * 78)
        for para in saturation.split("\n\n"):
            for line in para.split("\n"):
                print("  " + line)
            print()

    # ---------------- 报告 ----------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_md = render_report(
        args, items, results, summary, ks, headline_k, dataset_path, failed, ctx
    )
    paths = write_outputs(Path(args.out_dir), stamp, items, results, summary, report_md)

    print("\n" + "=" * 78)
    print(" 产物")
    print("=" * 78)
    for name, path in paths.items():
        print(f"  {name:<10} {path}")

    if failed:
        print("\n[注意] 以下策略未纳入对比：")
        for key, reason in failed.items():
            print(f"  - {key}: {reason}")
            hint = describe_failure(reason)
            if hint:
                print(f"    建议：{hint}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
