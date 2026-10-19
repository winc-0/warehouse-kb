# eval/selfcheck.py
"""
离线自检 —— 不需要 Milvus、不需要 GPU、不消耗任何 API 额度。

为什么必须有这个文件
--------------------
评测脚本最大的风险不是"跑不通"，而是"跑通了但数字是错的"。指标算错的报告
比没有报告更糟：它会让人在面试里说出一组站不住的数字。

所以这里用两类手段验证 eval/metrics.py：

1. **手算对照**：每个指标都准备了一个能用纸笔算出来的用例，逐一比对。
2. **性质测试**：对随机生成的检索结果批量验证指标应满足的不变式
   （Recall@K 随 K 单调不减、所有指标落在 [0,1]、完美排序下 nDCG=1 等）。

用法
----
    python eval/selfcheck.py
"""

from __future__ import annotations

import math
import random
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval._boot import init as _boot_init  # noqa: E402

PROJECT_ROOT = _boot_init()

from eval.dataset import QAItem, describe, load_dataset, save_dataset  # noqa: E402
from eval.metrics import (  # noqa: E402
    average_precision,
    hit_at_k,
    ndcg_at_k,
    paired_bootstrap_p_value,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    aggregate,
    aggregate_std,
    evaluate_one,
)

TOL = 1e-9
_failures: List[str] = []
_checked = 0


def check(name: str, actual, expected, tol: float = TOL) -> None:
    global _checked
    _checked += 1
    if isinstance(expected, float) or isinstance(actual, float):
        ok = abs(float(actual) - float(expected)) <= tol
    else:
        ok = actual == expected
    if ok:
        print(f"  [v] {name}")
    else:
        print(f"  [x] {name}\n      期望 {expected!r}，实际 {actual!r}")
        _failures.append(name)


def check_true(name: str, condition: bool, detail: str = "") -> None:
    global _checked
    _checked += 1
    if condition:
        print(f"  [v] {name}")
    else:
        print(f"  [x] {name} {detail}")
        _failures.append(name)


# --------------------------------------------------------------------------- #
# 1. 手算对照
# --------------------------------------------------------------------------- #
def test_hand_computed() -> None:
    print("\n[1] 手算对照（retrieved=[1,2,3,4,5]，relevant={3,7}）")
    retrieved = [1, 2, 3, 4, 5]
    relevant = {3, 7}

    check("hit@1 = 0（第 1 条不在答案里）", hit_at_k(retrieved, relevant, 1), 0.0)
    check("hit@2 = 0", hit_at_k(retrieved, relevant, 2), 0.0)
    check("hit@3 = 1（第 3 条命中）", hit_at_k(retrieved, relevant, 3), 1.0)
    check("hit@5 = 1", hit_at_k(retrieved, relevant, 5), 1.0)

    check("recall@1 = 0", recall_at_k(retrieved, relevant, 1), 0.0)
    check("recall@3 = 0.5（2 个答案只捞到 1 个）", recall_at_k(retrieved, relevant, 3), 0.5)
    check("recall@5 = 0.5", recall_at_k(retrieved, relevant, 5), 0.5)

    check("precision@5 = 0.2（前 5 条只有 1 条对）", precision_at_k(retrieved, relevant, 5), 0.2)

    check("MRR = 1/3（第一个命中在第 3 位）", reciprocal_rank(retrieved, relevant), 1.0 / 3.0)

    # DCG  = 1 / log2(4) = 0.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309297536 = 1.6309297536
    expected_ndcg = 0.5 / (1.0 + 1.0 / math.log2(3))
    check("nDCG@5 = 0.306573...", ndcg_at_k(retrieved, relevant, 5), expected_ndcg)

    # AP = (命中位置上的 Precision 之和) / 答案总数 = (1/3) / 2
    check("AP = 1/6", average_precision(retrieved, relevant), 1.0 / 6.0)


def test_edge_cases() -> None:
    print("\n[2] 边界情况")
    retrieved = [1, 2, 3, 4, 5]

    check("空答案集：recall=0", recall_at_k(retrieved, set(), 5), 0.0)
    check("空答案集：hit=0", hit_at_k(retrieved, set(), 5), 0.0)
    check("空答案集：nDCG=0（IDCG=0 时不除零）", ndcg_at_k(retrieved, set(), 5), 0.0)
    check("空答案集：AP=0", average_precision(retrieved, set()), 0.0)

    check("空检索结果：hit=0", hit_at_k([], {1}, 5), 0.0)
    check("空检索结果：MRR=0", reciprocal_rank([], {1}), 0.0)
    check("k=0：precision=0（不除零）", precision_at_k(retrieved, {1}, 0), 0.0)

    # 完美排序：答案全部排在最前面
    perfect_r, perfect_rel = [3, 7, 9], {3, 7}
    check("完美排序：recall@1 = 0.5", recall_at_k(perfect_r, perfect_rel, 1), 0.5)
    check("完美排序：recall@2 = 1.0", recall_at_k(perfect_r, perfect_rel, 2), 1.0)
    check("完美排序：MRR = 1.0", reciprocal_rank(perfect_r, perfect_rel), 1.0)
    check("完美排序：nDCG@2 = 1.0", ndcg_at_k(perfect_r, perfect_rel, 2), 1.0)
    check("完美排序：nDCG@5 = 1.0（答案只有 2 条，IDCG 取 min(|R|,K)）",
          ndcg_at_k(perfect_r, perfect_rel, 5), 1.0)
    check("完美排序：AP = 1.0", average_precision(perfect_r, perfect_rel), 1.0)

    check("检索结果少于 K 条：recall@10 仍按答案总数算",
          recall_at_k([3], {3, 7}, 10), 0.5)


def test_aggregation() -> None:
    print("\n[3] 聚合与标准差")
    per_query = evaluate_one([3, 1, 2], {3})
    check("evaluate_one 含 recall@5 字段", "recall@5" in per_query, True)
    check("evaluate_one 含 mrr 字段", "mrr" in per_query, True)

    rows = [{"x": 0.0}, {"x": 1.0}]
    check("aggregate 均值 = 0.5", aggregate(rows)["x"], 0.5)
    check("aggregate_std 样本标准差 = 0.7071", aggregate_std(rows)["x"], 1.0 / math.sqrt(2))

    check("aggregate 空输入返回空字典", aggregate([]), {})
    check("aggregate_std 单样本返回 0", aggregate_std([{"x": 0.4}])["x"], 0.0)


def test_bootstrap() -> None:
    print("\n[4] 配对 bootstrap 显著性检验")
    same = [0.5] * 40
    p_same = paired_bootstrap_p_value(same, list(same), n_boot=1000)
    check_true("两条相同链路：p 接近 1（无差异）", p_same > 0.9, f"实际 p={p_same:.4f}")

    a = [0.0] * 40
    b = [1.0] * 40
    p_diff = paired_bootstrap_p_value(a, b, n_boot=1000)
    check_true("完全分离的两组：p < 0.05（差异显著）", p_diff < 0.05, f"实际 p={p_diff:.4f}")

    check_true("长度不一致时返回 1.0（不做无意义检验）",
               paired_bootstrap_p_value([1.0], [1.0, 0.0]) == 1.0)

    rng = random.Random(7)
    noisy_a = [rng.gauss(0.5, 0.2) for _ in range(30)]
    noisy_b = [rng.gauss(0.5, 0.2) for _ in range(30)]
    p_noise = paired_bootstrap_p_value(noisy_a, noisy_b, n_boot=1000)
    check_true("同分布噪声：p 不显著", p_noise > 0.05, f"实际 p={p_noise:.4f}")


def test_properties() -> None:
    print("\n[5] 性质测试（随机 200 例，验证指标不变式）")
    rng = random.Random(2026)
    ks = [1, 2, 5, 10]
    monotonic_ok = True
    range_ok = True
    perfect_ok = True

    for _ in range(200):
        pool = list(range(1, 51))
        rng.shuffle(pool)
        n_rel = rng.randint(1, 6)
        relevant = set(pool[:n_rel])
        retrieved = pool[: rng.randint(1, 30)]

        rec = [recall_at_k(retrieved, relevant, k) for k in ks]
        if any(rec[i] > rec[i + 1] + 1e-12 for i in range(len(rec) - 1)):
            monotonic_ok = False

        metrics = evaluate_one(retrieved, relevant, ks=ks, headline_k=5)
        if any(v < -1e-12 or v > 1.0 + 1e-12 for v in metrics.values()):
            range_ok = False

        # 把全部答案按原顺序插到最前面，nDCG 必须为 1
        ideal = sorted(relevant) + [x for x in retrieved if x not in relevant]
        if abs(ndcg_at_k(ideal, relevant, len(ideal)) - 1.0) > 1e-9:
            perfect_ok = False

    check_true("Recall@K 随 K 单调不减", monotonic_ok)
    check_true("所有指标落在 [0,1] 区间", range_ok)
    check_true("理想排序下 nDCG = 1", perfect_ok)


# --------------------------------------------------------------------------- #
# 6. 数据集读写
# --------------------------------------------------------------------------- #
def test_dataset_roundtrip() -> None:
    print("\n[6] 数据集读写与校验")
    items = [
        QAItem(qid="q1", question="怎么调温度？", item_names=["合力叉车 CPD30 三支点电动叉车"],
               ground_truth_chunk_ids=[101, 102], reference_answer="旋钮", source="manual"),
        QAItem(qid="q2", question="怎么换保险丝？", item_names=["合力叉车 CPD30 三支点电动叉车"],
               ground_truth_chunk_ids=[205], source="generated"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ds.jsonl"
        save_dataset(items, path)
        loaded = load_dataset(path)
        check("读回样本数一致", len(loaded), 2)
        check("chunk_id 保持整数类型", loaded[0].ground_truth_chunk_ids, [101, 102])
        check("relevant 集合正确", loaded[0].relevant, {101, 102})
        check("reference_answer 保留", loaded[0].reference_answer, "旋钮")
        check("describe 统计来源分布", describe(loaded)["来源分布"], {"manual": 1, "generated": 1})

        # 缺字段必须报错（这是防止"没标注就开跑"的关键防线）
        bad = Path(tmp) / "bad.jsonl"
        bad.write_text('{"qid": "b1", "question": "缺答案集合"}\n', encoding="utf-8")
        try:
            load_dataset(bad)
            check_true("缺字段时抛错", False, "居然没抛错")
        except ValueError:
            check_true("缺字段时抛错", True)

        # 空答案集合也要拦住
        empty_gt = Path(tmp) / "empty.jsonl"
        empty_gt.write_text(
            '{"qid": "e1", "question": "q", "item_names": ["A"], "ground_truth_chunk_ids": []}\n',
            encoding="utf-8",
        )
        try:
            load_dataset(empty_gt)
            check_true("空答案集合时抛错", False, "居然没抛错")
        except ValueError:
            check_true("空答案集合时抛错", True)

        # 非整数 chunk_id 也要拦住
        badid = Path(tmp) / "badid.jsonl"
        badid.write_text(
            '{"qid": "b2", "question": "q", "item_names": ["A"], "ground_truth_chunk_ids": ["abc"]}\n',
            encoding="utf-8",
        )
        try:
            load_dataset(badid)
            check_true("非整数 chunk_id 抛错", False, "居然没抛错")
        except ValueError:
            check_true("非整数 chunk_id 抛错", True)


# --------------------------------------------------------------------------- #
# 7. 策略注册表形状
# --------------------------------------------------------------------------- #
def test_strategies_shape() -> None:
    print("\n[7] 策略注册与过滤表达式")
    from eval.strategies import DEFAULT_ORDER, STRATEGIES, _item_filter, resolve_strategies

    check("策略数量 = 6", len(STRATEGIES), 6)
    check("默认顺序覆盖全部策略", sorted(DEFAULT_ORDER), sorted(STRATEGIES))
    resolved = resolve_strategies(None)
    check("resolve_strategies(None) 返回 6 组", len(resolved), 6)
    check_true("每个策略都有可调用 retrieve", all(callable(s.retrieve) for s in STRATEGIES.values()))

    check(
        "item_name 过滤表达式与生产写法一致",
        _item_filter(["A", "B"]),
        'item_name in ["A", "B"]',
    )
    check("空商品名返回空表达式（不过滤）", _item_filter([]), "")

    try:
        resolve_strategies(["不存在"])
        check_true("未知策略 key 抛错", False, "居然没抛错")
    except KeyError:
        check_true("未知策略 key 抛错", True)


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 78)
    print(" eval/selfcheck.py —— 离线自检（不需要 Milvus / GPU / API）")
    print("=" * 78)

    test_hand_computed()
    test_edge_cases()
    test_aggregation()
    test_bootstrap()
    test_properties()
    test_dataset_roundtrip()
    test_strategies_shape()

    print("\n" + "=" * 78)
    if _failures:
        print(f" 失败 {len(_failures)}/{_checked} 项：")
        for name in _failures:
            print(f"   - {name}")
        print("=" * 78)
        return 1
    print(f" 全部通过（{_checked} 项）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
