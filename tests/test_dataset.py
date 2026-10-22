# tests/test_dataset.py
"""
数据集读写与校验的测试。

为什么值得单测：`load_dataset(strict=True)` 是评测的第一道闸门。
它一旦"宽松"了（比如重复 qid 只是打印警告然后继续），指标就会在无人察觉的情况下失真——
同一道题被算两次、或者标注为空的题被当作「模型没答对」计入分母。
这些错误不会报错，只会让 README 里的百分比低几个点，很难反查。
"""

import json
import tempfile
import unittest
from pathlib import Path

from eval.dataset import (
    DEFAULT_DATASET,
    QAItem,
    describe,
    load_dataset,
    save_dataset,
)


def _row(**overrides):
    row = {
        "qid": "q1",
        "question": "CPD30 的额定载荷是多少？",
        "item_names": ["合力CPD30三支点电动叉车"],
        "ground_truth_chunk_ids": [101, 102],
    }
    row.update(overrides)
    return row


class DatasetCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, lines, name="d.jsonl"):
        path = self.dir / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


class TestLoadHappyPath(DatasetCase):
    def test_basic(self):
        path = self.write([json.dumps(_row(), ensure_ascii=False)])
        items = load_dataset(path)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].qid, "q1")
        self.assertEqual(items[0].relevant, {101, 102})

    def test_blank_and_comment_lines_are_skipped(self):
        path = self.write([
            "",
            "// 这是注释",
            json.dumps(_row(), ensure_ascii=False),
            "   ",
        ])
        self.assertEqual(len(load_dataset(path)), 1)

    def test_chunk_ids_accept_numeric_strings(self):
        path = self.write([json.dumps(_row(ground_truth_chunk_ids=["101", 102]))])
        self.assertEqual(load_dataset(path)[0].ground_truth_chunk_ids, [101, 102])

    def test_optional_fields_default(self):
        path = self.write([json.dumps(_row())])
        item = load_dataset(path)[0]
        self.assertEqual(item.reference_answer, "")
        self.assertEqual(item.source, "manual")


class TestLoadRejectsBadInput(DatasetCase):
    def test_missing_required_field(self):
        row = _row()
        row.pop("item_names")
        path = self.write([json.dumps(row, ensure_ascii=False)])
        with self.assertRaises(ValueError) as ctx:
            load_dataset(path)
        self.assertIn("item_names", str(ctx.exception))

    def test_duplicate_qid(self):
        path = self.write([
            json.dumps(_row(), ensure_ascii=False),
            json.dumps(_row(question="另一道题"), ensure_ascii=False),
        ])
        with self.assertRaises(ValueError) as ctx:
            load_dataset(path)
        self.assertIn("qid 重复", str(ctx.exception))

    def test_broken_json_line(self):
        path = self.write(["{not json"])
        with self.assertRaises(ValueError) as ctx:
            load_dataset(path)
        self.assertIn("JSON 解析失败", str(ctx.exception))

    def test_non_integer_chunk_id(self):
        path = self.write([json.dumps(_row(ground_truth_chunk_ids=["abc"]), ensure_ascii=False)])
        with self.assertRaises(ValueError) as ctx:
            load_dataset(path)
        self.assertIn("非整数", str(ctx.exception))

    def test_empty_ground_truth(self):
        # 空列表在「必填字段」这一关就被拦下
        path = self.write([json.dumps(_row(ground_truth_chunk_ids=[]), ensure_ascii=False)])
        with self.assertRaises(ValueError) as ctx:
            load_dataset(path)
        self.assertIn("缺少字段", str(ctx.exception))

    def test_all_null_chunk_ids_is_also_rejected(self):
        # [null] 能过字段非空检查，但清洗后为空 —— 也不能放行，否则这题会被当成"没答对"
        path = self.write([json.dumps(_row(ground_truth_chunk_ids=[None]), ensure_ascii=False)])
        with self.assertRaises(ValueError) as ctx:
            load_dataset(path)
        self.assertIn("正确切片为空", str(ctx.exception))

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset(self.dir / "nope.jsonl")

    def test_non_strict_skips_bad_rows(self):
        path = self.write([
            json.dumps(_row(), ensure_ascii=False),
            json.dumps(_row(ground_truth_chunk_ids=[]), ensure_ascii=False),
        ])
        items = load_dataset(path, strict=False)
        self.assertEqual(len(items), 1)

    def test_all_rows_bad_still_raises(self):
        # 放宽模式也不能返回空集——那会让下游算出一堆 0 分
        path = self.write([json.dumps(_row(ground_truth_chunk_ids=[]), ensure_ascii=False)])
        with self.assertRaises(ValueError):
            load_dataset(path, strict=False)


class TestRoundTrip(DatasetCase):
    def test_save_then_load(self):
        items = [
            QAItem("a", "问题一", ["主体A"], [1, 2], reference_answer="答案", note="备注"),
            QAItem("b", "问题二", ["主体B"], [3]),
        ]
        path = self.dir / "out.jsonl"
        save_dataset(items, path)

        # 文件里必须是 UTF-8 明文，不能是 \uXXXX 转义——否则没法直接看内容
        self.assertIn("问题一", path.read_text(encoding="utf-8"))

        back = load_dataset(path)
        self.assertEqual([i.qid for i in back], ["a", "b"])
        self.assertEqual(back[0].reference_answer, "答案")
        self.assertEqual(back[0].ground_truth_chunk_ids, [1, 2])


class TestDescribe(unittest.TestCase):
    def test_counts(self):
        items = [
            QAItem("a", "q", ["A", "B"], [1, 2], reference_answer="x"),
            QAItem("b", "q", ["A"], [3], source="generated"),
        ]
        info = describe(items)
        self.assertEqual(info["样本数"], 2)
        self.assertEqual(info["覆盖商品数"], 2)
        self.assertEqual(info["平均正确切片数"], 1.5)
        self.assertEqual(info["最大正确切片数"], 2)
        self.assertEqual(info["含标准答案数"], 1)
        self.assertEqual(info["来源分布"], {"manual": 1, "generated": 1})


class TestDefaultDataset(unittest.TestCase):
    """默认数据集必须是那份 164 题的标注集，不能是 14 条模板。

    这条看着无聊，但正是它防止了「不带参数跑评测 → 静默只评 14 题 → 报告看起来很正经」。
    """

    def test_default_is_the_real_dataset(self):
        self.assertTrue(DEFAULT_DATASET.exists(), f"默认数据集不存在：{DEFAULT_DATASET}")
        self.assertEqual(DEFAULT_DATASET.name, "qa_hard.jsonl")
        items = load_dataset(DEFAULT_DATASET)
        self.assertGreaterEqual(len(items), 100)

    def test_default_parser_matches(self):
        # eval.run_eval 会连带拉起项目依赖，没装依赖时跳过而不是报错
        try:
            from eval.run_eval import build_parser
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"缺少项目依赖，跳过：{exc}")

        args = build_parser().parse_args([])
        self.assertEqual(Path(args.dataset).name, DEFAULT_DATASET.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
