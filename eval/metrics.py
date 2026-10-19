# eval/metrics.py
"""
检索评测指标实现（纯标准库，零第三方依赖）。

设计说明
--------
本模块只依赖 Python 标准库（math / random / statistics），原因有三：
1. 指标公式是确定的、可手算验证的，引入第三方库反而让"数字对不对"变得不可控；
2. 项目主力环境是 Python 3.11 + CUDA 12.4，额外的评估库可能带来依赖冲突；
3. 便于单独单元测试（见 eval/selfcheck.py）。

所有函数的输入约定
------------------
retrieved : Sequence[str | int]  检索系统返回的有序结果（按相关性从高到低），元素是 chunk_id
relevant  : Collection[str | int] 标注的正确答案集合（ground truth）

相关性一律按「二值」处理（命中=1，未命中=0）。这是项目当前标注能力的现实选择：
人工只标"这段是不是能回答问题"，不标 0~3 的相关性等级。
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Collection, Dict, List, Sequence

# 默认评测的截断位置
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)
# 报告中的"头号指标"截断位置
HEADLINE_K: int = 5


# --------------------------------------------------------------------------- #
# 单指标
# --------------------------------------------------------------------------- #
def hit_at_k(retrieved: Sequence[Any], relevant: Collection[Any], k: int) -> float:
    """Hit@K（命中率）：前 K 条里是否至少有一条正确。返回 0 或 1。"""
    if not relevant:
        return 0.0
    return 1.0 if any(doc in relevant for doc in retrieved[:k]) else 0.0


def recall_at_k(retrieved: Sequence[Any], relevant: Collection[Any], k: int) -> float:
    """
    Recall@K（召回率）：前 K 条覆盖了多少比例的正确答案。

    注意：当正确答案数量 > K 时，Recall@K 的理论上限就是 < 1，这是定义使然，
    不是系统缺陷。所以报告里必须同时给出标注集的"平均正确答案数"。
    """
    if not relevant:
        return 0.0
    topk = set(retrieved[:k])
    return len(topk & set(relevant)) / len(relevant)


def precision_at_k(retrieved: Sequence[Any], relevant: Collection[Any], k: int) -> float:
    """Precision@K（准确率）：前 K 条里正确的比例，分母固定为 K。"""
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    return sum(1 for doc in topk if doc in relevant) / k


def reciprocal_rank(retrieved: Sequence[Any], relevant: Collection[Any]) -> float:
    """RR@K / MRR 的单条计算：第一条正确答案出现位置倒数的倒数。"""
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(binary_gains: Sequence[float], k: int) -> float:
    """折损累计增益：sum(gain_i / log2(i+1))，i 从 1 开始。"""
    return sum(g / math.log2(idx + 1) for idx, g in enumerate(binary_gains[:k], start=1))


def ndcg_at_k(retrieved: Sequence[Any], relevant: Collection[Any], k: int) -> float:
    """
    nDCG@K：考虑排序位置的加权指标。

    二值相关性下，理想排序是「命中的全部排最前」，因此
    IDCG = DCG([1]*min(|relevant|, K))。
    """
    gains = [1.0 if doc in relevant else 0.0 for doc in retrieved[:k]]
    ideal = [1.0] * min(len(relevant), k)
    idcg = _dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return _dcg(gains, k) / idcg


def average_precision(retrieved: Sequence[Any], relevant: Collection[Any]) -> float:
    """AP（平均精度）：每次命中位置上的 Precision 求均值，再除以正确答案总数。"""
    if not relevant:
        return 0.0
    hits = 0
    acc = 0.0
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            acc += hits / rank
    return acc / len(relevant)


# --------------------------------------------------------------------------- #
# 单条样本 / 整体聚合
# --------------------------------------------------------------------------- #
def evaluate_one(
    retrieved: Sequence[Any],
    relevant: Collection[Any],
    ks: Sequence[int] = DEFAULT_KS,
    headline_k: int = HEADLINE_K,
) -> Dict[str, float]:
    """对单条问题计算全套指标，返回扁平字典，便于直接写进 CSV。"""
    result: Dict[str, float] = {}
    for k in ks:
        result[f"hit@{k}"] = hit_at_k(retrieved, relevant, k)
    for k in ks:
        result[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
    result[f"precision@{headline_k}"] = precision_at_k(retrieved, relevant, headline_k)
    result[f"ndcg@{headline_k}"] = ndcg_at_k(retrieved, relevant, headline_k)
    result["mrr"] = reciprocal_rank(retrieved, relevant)
    result["map"] = average_precision(retrieved, relevant)
    return result


def evaluate_many(
    rankings: Sequence[Sequence[Any]],
    ground_truths: Sequence[Collection[Any]],
    ks: Sequence[int] = DEFAULT_KS,
    headline_k: int = HEADLINE_K,
) -> List[Dict[str, float]]:
    """批量计算，返回逐题的指标列表（顺序与输入一致）。"""
    if len(rankings) != len(ground_truths):
        raise ValueError(
            f"rankings 与 ground_truths 长度不一致：{len(rankings)} vs {len(ground_truths)}"
        )
    return [
        evaluate_one(ranked, relevant, ks=ks, headline_k=headline_k)
        for ranked, relevant in zip(rankings, ground_truths)
    ]


def aggregate(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """对所有题的指标取算术平均。空输入返回空字典。"""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: statistics.fmean(float(row[k]) for row in per_query) for k in keys}


def aggregate_std(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """对所有题的指标取样本标准差（ddof=1），单样本返回 0。"""
    if len(per_query) < 2:
        return {k: 0.0 for k in (per_query[0].keys() if per_query else [])}
    keys = per_query[0].keys()
    return {k: statistics.stdev(float(row[k]) for row in per_query) for k in keys}


# --------------------------------------------------------------------------- #
# 显著性检验（配对 bootstrap）
# --------------------------------------------------------------------------- #
def paired_bootstrap_p_value(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 10000,
    seed: int = 42,
) -> float:
    """
    配对 bootstrap 检验（双侧），H0：两条链路在该指标上的均值差为 0。

    做法：把逐题差值平移到以 0 为中心后重采样，看"极端程度不弱于观测值"的比例。
    相比 t 检验，它不假设差值服从正态分布 —— 指标（如 Hit@1）是 0/1 变量，
    天然偏态，用 bootstrap 更稳妥。

    :param values_a: 基线链路逐题指标
    :param values_b: 对比链路逐题指标（顺序必须与 a 一一对应，即同一批问题）
    :param n_boot: 重采样次数
    :return: 双侧 p 值。p < 0.05 通常认为提升显著。
    """
    if len(values_a) != len(values_b) or not values_a:
        return 1.0
    diffs = [float(b) - float(a) for a, b in zip(values_a, values_b)]
    n = len(diffs)
    observed = statistics.fmean(diffs)

    rng = random.Random(seed)
    centered = [d - observed for d in diffs]
    extreme = 0
    for _ in range(n_boot):
        sample_mean = statistics.fmean(rng.choice(centered) for _ in range(n))
        if abs(sample_mean) >= abs(observed) - 1e-12:
            extreme += 1
    # +1 平滑，避免 p 值报 0
    return (extreme + 1) / (n_boot + 1)


# --------------------------------------------------------------------------- #
# 展示辅助
# --------------------------------------------------------------------------- #
def fmt_pct(value: float, digits: int = 2) -> str:
    """0.8231 -> '82.31%'"""
    return f"{value * 100:.{digits}f}%"


def fmt_delta(value: float, digits: int = 2) -> str:
    """带正负号的百分点增量：0.051 -> '+5.10pt'"""
    return f"{value * 100:+.{digits}f}pt"
