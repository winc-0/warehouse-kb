# tests/test_strategies.py
"""
策略层与 RRF 融合的测试（不连 Milvus、不打 API）。

这一层最容易出的错不是"算错"，而是"测的不是线上"：
为了评测方便把召回重写一遍，结果消融结论跟线上行为对不上。
所以这里的重点是**结构约束**——策略表完整、默认顺序固定、参数与生产一致；
再加一个 RRF 的手算校验，因为融合顺序决定了后面所有指标。
"""

import unittest

from eval.strategies import (
    DEFAULT_BASELINE,
    DEFAULT_ORDER,
    PER_REQUEST_LIMIT,
    RANKER_WEIGHTS,
    RRF_K,
    RRF_MAX_RESULTS,
    STRATEGIES,
    resolve_strategies,
)


class TestRegistry(unittest.TestCase):
    def test_six_ablation_groups(self):
        # 六组逐层累加：裸稠密 / 裸稀疏 / 混合 / 混合+HyDE+RRF / 重排 / 完整
        self.assertEqual(
            set(STRATEGIES),
            {"dense_only", "sparse_only", "hybrid", "hybrid_hyde_rrf", "rerank", "full"},
        )

    def test_default_order_covers_all(self):
        # 默认顺序必须把所有策略都排进去，否则「不带参数跑」会漏组
        self.assertEqual(sorted(DEFAULT_ORDER), sorted(STRATEGIES))

    def test_baseline_exists(self):
        self.assertIn(DEFAULT_BASELINE, STRATEGIES)

    def test_every_strategy_is_runnable(self):
        for key, strategy in STRATEGIES.items():
            with self.subTest(key=key):
                self.assertEqual(strategy.key, key)
                self.assertTrue(strategy.label.strip(), "label 不能为空（报告里要展示）")
                self.assertTrue(callable(strategy.retrieve))
                self.assertTrue(strategy.description.strip(), "description 不能为空")


class TestResolve(unittest.TestCase):
    def test_none_returns_all_in_default_order(self):
        got = [s.key for s in resolve_strategies(None)]
        self.assertEqual(got, list(DEFAULT_ORDER))

    def test_subset_keeps_requested_order(self):
        got = [s.key for s in resolve_strategies(["full", "dense_only"])]
        self.assertEqual(got, ["full", "dense_only"])

    def test_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            resolve_strategies(["nope"])


class TestProductionParity(unittest.TestCase):
    """参数必须与线上一致——不一致就等于在评另一个系统。

    这些值改动时通常是"顺手调一下"，但那会让历史报告失去可比性，
    所以钉在这里，改的时候必须连带改注释和报告里的口径说明。
    """

    def test_rrf_k(self):
        self.assertEqual(RRF_K, 60)

    def test_eval_keeps_more_rrf_results_than_production(self):
        # 生产保留 5 条，评测放宽到 10 以便看 Recall@10；K<=5 的指标上两者一致
        self.assertEqual(RRF_MAX_RESULTS, 10)
        self.assertGreaterEqual(RRF_MAX_RESULTS, 5)

    def test_per_request_limit(self):
        self.assertEqual(PER_REQUEST_LIMIT, 10)

    def test_ranker_weights(self):
        self.assertEqual(RANKER_WEIGHTS, (0.8, 0.2))


class TestRRFMerge(unittest.TestCase):
    """RRF 手算校验：score(d) = Σ weight / (k + rank)。"""

    def setUp(self):
        # 延迟导入：NodeRrf 会拉起处理器依赖，模块顶层导入会让纯指标测试也变重
        from processor.query_processor.nodes.node_rrf import NodeRrf

        self.merge = NodeRrf()._rrf_merge

    def test_scores_match_formula(self):
        merged = self.merge(
            [([{"chunk_id": "a"}, {"chunk_id": "b"}], 1.0)],
            k=60,
            max_results=10,
        )
        self.assertEqual([doc["chunk_id"] for doc, _ in merged], ["a", "b"])
        self.assertAlmostEqual(merged[0][1], 1 / 61)
        self.assertAlmostEqual(merged[1][1], 1 / 62)

    def test_doc_hit_by_two_lists_ranks_first(self):
        # b 在两路都出现（rank2 + rank1），得分应高于只在 a 路拿 rank1 的 a
        merged = self.merge(
            [
                ([{"chunk_id": "a"}, {"chunk_id": "b"}], 1.0),
                ([{"chunk_id": "b"}], 1.0),
            ],
            k=60,
            max_results=10,
        )
        self.assertEqual(merged[0][0]["chunk_id"], "b")
        self.assertAlmostEqual(merged[0][1], 1 / 62 + 1 / 61)

    def test_zero_weight_contributes_nothing(self):
        merged = self.merge(
            [
                ([{"chunk_id": "a"}], 1.0),
                ([{"chunk_id": "b"}], 0.0),  # 权重 0 → 该路完全不贡献
            ],
            k=60,
            max_results=10,
        )
        self.assertEqual(merged[0][0]["chunk_id"], "a")
        # 权重为 0 的文档仍在结果里，但得分为 0，排在最后
        self.assertEqual(merged[-1][0]["chunk_id"], "b")
        self.assertAlmostEqual(merged[-1][1], 0.0)

    def test_max_results_truncates(self):
        docs = [{"chunk_id": i} for i in range(10)]
        merged = self.merge([(docs, 1.0)], k=60, max_results=3)
        self.assertEqual(len(merged), 3)

    def test_max_results_none_returns_all(self):
        docs = [{"chunk_id": i} for i in range(5)]
        merged = self.merge([(docs, 1.0)], k=60, max_results=None)
        self.assertEqual(len(merged), 5)

    def test_empty_input(self):
        self.assertEqual(self.merge([([], 1.0)], k=60, max_results=5), [])

    def test_sorted_descending(self):
        merged = self.merge(
            [([{"chunk_id": i} for i in range(8)], 1.0)],
            k=60,
            max_results=8,
        )
        scores = [score for _, score in merged]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
