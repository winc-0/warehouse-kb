# tests/test_metrics.py
"""
检索指标的手算校验。

为什么值得单测：这些函数的输出会直接变成 README 里的百分比。一旦某个 K 的分母写错、
或者 nDCG 的 IDCG 用了错误的理想排序，结果是"看起来完全正常"的数字——
没人会从 82.31% 里看出它算错了。所以这里每个指标都用能手算的样例钉住。
"""

import math
import unittest

from eval.metrics import (
    aggregate,
    aggregate_std,
    average_precision,
    evaluate_many,
    evaluate_one,
    hit_at_k,
    ndcg_at_k,
    paired_bootstrap_p_value,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestSingleMetrics(unittest.TestCase):
    def test_hit_at_k(self):
        self.assertEqual(hit_at_k([1, 2, 3], {3}, k=3), 1.0)
        self.assertEqual(hit_at_k([1, 2, 3], {3}, k=2), 0.0)
        # 空标注：没有正确答案，命中率定义上取 0（避免除零）
        self.assertEqual(hit_at_k([1, 2, 3], set(), k=3), 0.0)

    def test_recall_at_k(self):
        # 3 个正确答案，前 2 条命中 1 个
        self.assertAlmostEqual(recall_at_k([1, 9, 2, 3], {1, 2, 3}, k=2), 1 / 3)
        self.assertAlmostEqual(recall_at_k([1, 2, 3], {1, 2, 3}, k=3), 1.0)

    def test_recall_ceiling_when_k_smaller_than_relevant(self):
        # 正确答案 4 条、只取前 2 条 → 上限就是 0.5，这是定义使然
        self.assertAlmostEqual(recall_at_k([1, 2], {1, 2, 3, 4}, k=2), 0.5)

    def test_precision_denominator_is_k(self):
        # 分母固定为 K，不随返回条数变化——这是 precision@k 与 recall@k 的关键区别
        self.assertAlmostEqual(precision_at_k([1], {1}, k=2), 0.5)
        self.assertEqual(precision_at_k([1, 2], {1, 2}, k=2), 1.0)
        self.assertEqual(precision_at_k([1, 2], {1, 2}, k=0), 0.0)

    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank([7, 8, 9], {9}), 1 / 3)
        self.assertEqual(reciprocal_rank([7, 8], {9}), 0.0)
        self.assertEqual(reciprocal_rank([], {9}), 0.0)

    def test_ndcg(self):
        # retrieved=[1,2,3]，relevant={1,3}，k=3
        #   DCG  = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
        #   IDCG = 1/log2(2) + 1/log2(3)
        expected = 1.5 / (1.0 + 1.0 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k([1, 2, 3], {1, 3}, k=3), expected)
        # 完美排序：nDCG = 1
        self.assertAlmostEqual(ndcg_at_k([1, 3, 2], {1, 3}, k=3), 1.0)
        # 无命中：nDCG = 0（IDCG 不为 0，所以是 0 而不是除零）
        self.assertEqual(ndcg_at_k([9, 8, 7], {1, 3}, k=3), 0.0)

    def test_average_precision(self):
        # 命中在 rank1（p=1/1）与 rank3（p=2/3）→ (1 + 2/3) / 2
        self.assertAlmostEqual(average_precision([1, 2, 3], {1, 3}), (1 + 2 / 3) / 2)
        self.assertEqual(average_precision([9, 8], {1}), 0.0)
        self.assertEqual(average_precision([1, 2], set()), 0.0)


class TestAggregation(unittest.TestCase):
    def test_evaluate_one_keys(self):
        row = evaluate_one([1, 2, 3], {1, 2}, ks=(1, 3), headline_k=3)
        for key in ("hit@1", "hit@3", "recall@1", "recall@3", "precision@3", "ndcg@3", "mrr", "map"):
            self.assertIn(key, row)

    def test_evaluate_many_length_mismatch(self):
        with self.assertRaises(ValueError):
            evaluate_many([[1], [2]], [{1}])

    def test_aggregate_uses_mean(self):
        rows = [{"x": 1.0}, {"x": 0.0}, {"x": 0.5}]
        self.assertAlmostEqual(aggregate(rows)["x"], 0.5)

    def test_aggregate_empty(self):
        self.assertEqual(aggregate([]), {})

    def test_aggregate_std(self):
        rows = [{"x": 1.0}, {"x": 1.0}, {"x": 1.0}]
        self.assertEqual(aggregate_std(rows)["x"], 0.0)
        # 单样本无法算样本标准差，约定返回 0 而不是抛异常
        self.assertEqual(aggregate_std([{"x": 3.0}])["x"], 0.0)


class TestBootstrap(unittest.TestCase):
    def test_identical_inputs_are_not_significant(self):
        a = [1, 0, 1, 1, 0, 1] * 5
        self.assertGreater(paired_bootstrap_p_value(a, list(a), n_boot=2000), 0.9)

    def test_large_consistent_shift_is_significant(self):
        # b 每一题都比 a 高 1，观测差值标准差为 0 → p 应很小
        a = [0] * 40
        b = [1] * 40
        self.assertLess(paired_bootstrap_p_value(a, b, n_boot=2000), 0.05)

    def test_length_mismatch_returns_one(self):
        self.assertEqual(paired_bootstrap_p_value([1, 0], [1], n_boot=100), 1.0)

    def test_empty_returns_one(self):
        self.assertEqual(paired_bootstrap_p_value([], [], n_boot=100), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
