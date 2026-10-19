# eval/dataset.py
"""
评测数据集的读写与校验。

数据格式（JSONL，每行一个 JSON 对象）
------------------------------------
{
  "qid": "q001",                                  // 必填，唯一编号
  "question": "CPD30 叉车的额定载荷是多少？",       // 必填，用户原始问法
  "item_names": ["合力CPD30三支点电动叉车"],        // 必填，用于 Milvus 的 item_name 过滤
  "ground_truth_chunk_ids": [10231, 10245],        // 必填，正确切片的主键（Milvus 自增 chunk_id）
  "reference_answer": "……",                        // 选填，标准答案，用于答案质量评估
  "source": "manual",                              // 选填，manual=人工标注 / generated=模型反向生成
  "note": "……"                                     // 选填，备注
}

为什么 chunk_id 必须是真实值
----------------------------
项目里 chunk_id 是 Milvus 的自增 INT64 主键（见 processor/import_processor/nodes/
node_import_milvus.py 的 _create_chunks_collection），由数据库生成、不落原文。
所以标注必须来自"先检索、再勾选"或"从库里读切片、再反问生成"，
不能凭肉眼编造。eval/build_dataset.py 提供了这两条路径。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

REQUIRED_FIELDS = ("qid", "question", "item_names", "ground_truth_chunk_ids")

# 默认数据集路径
DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets" / "qa_template.jsonl"


@dataclass
class QAItem:
    """一条评测样本。"""

    qid: str
    question: str
    item_names: List[str]
    ground_truth_chunk_ids: List[int]
    reference_answer: str = ""
    source: str = "manual"
    note: str = ""
    # 检索系统实际返回的 chunk_id 序列（评测时填充，不参与序列化到数据集文件）
    retrieved_ids: List[int] = field(default_factory=list)

    @property
    def relevant(self) -> set:
        """ground truth 的集合形式，供 metrics 使用。"""
        return set(self.ground_truth_chunk_ids)


def _coerce_chunk_ids(raw: Iterable[Any]) -> List[int]:
    """把 chunk_id 统一成 int（Milvus 主键是 INT64；字符串数字也接受）。"""
    ids: List[int] = []
    for value in raw:
        if value is None or value == "":
            continue
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            raise ValueError(f"ground_truth_chunk_ids 中存在非整数项：{value!r}")
    return ids


def load_dataset(path: str | Path, strict: bool = True) -> List[QAItem]:
    """
    读取 JSONL 数据集。

    :param path: 数据集文件路径
    :param strict: True 时字段缺失/为空直接抛错；False 时跳过并打印警告
    :return: QAItem 列表
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据集不存在：{path}")

    items: List[QAItem] = []
    seen_qids: set = set()
    problems: List[str] = []

    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"第 {lineno} 行 JSON 解析失败：{exc}")
                continue

            missing = [f for f in REQUIRED_FIELDS if not raw.get(f)]
            if missing:
                problems.append(f"第 {lineno} 行缺少字段：{', '.join(missing)}")
                continue

            qid = str(raw["qid"])
            if qid in seen_qids:
                problems.append(f"第 {lineno} 行 qid 重复：{qid}")
                continue
            seen_qids.add(qid)

            try:
                chunk_ids = _coerce_chunk_ids(raw.get("ground_truth_chunk_ids") or [])
            except ValueError as exc:
                problems.append(f"第 {lineno} 行（qid={qid}）：{exc}")
                continue

            if not chunk_ids:
                problems.append(f"第 {lineno} 行（qid={qid}）正确切片为空")
                continue

            items.append(
                QAItem(
                    qid=qid,
                    question=str(raw["question"]).strip(),
                    item_names=[str(n) for n in (raw.get("item_names") or [])],
                    ground_truth_chunk_ids=chunk_ids,
                    reference_answer=str(raw.get("reference_answer") or ""),
                    source=str(raw.get("source") or "manual"),
                    note=str(raw.get("note") or ""),
                )
            )

    if problems:
        message = "数据集校验发现问题：\n  - " + "\n  - ".join(problems)
        if strict:
            raise ValueError(message)
        print(f"[warn] {message}")

    if not items:
        raise ValueError(f"数据集 {path} 中没有可用的样本")

    return items


def save_dataset(items: Iterable[QAItem], path: str | Path) -> Path:
    """把样本写成 JSONL（UTF-8，ensure_ascii=False，方便直接看内容）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            row: Dict[str, Any] = {
                "qid": item.qid,
                "question": item.question,
                "item_names": item.item_names,
                "ground_truth_chunk_ids": item.ground_truth_chunk_ids,
            }
            if item.reference_answer:
                row["reference_answer"] = item.reference_answer
            row["source"] = item.source
            if item.note:
                row["note"] = item.note
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def describe(items: List[QAItem]) -> Dict[str, Any]:
    """给出一份数据集的概况，用于打印和写进报告。"""
    gt_sizes = [len(i.ground_truth_chunk_ids) for i in items]
    all_items = {name for i in items for name in i.item_names}
    sources: Dict[str, int] = {}
    for i in items:
        sources[i.source] = sources.get(i.source, 0) + 1

    return {
        "样本数": len(items),
        "覆盖商品数": len(all_items),
        "平均正确切片数": round(sum(gt_sizes) / len(gt_sizes), 2) if gt_sizes else 0.0,
        "最大正确切片数": max(gt_sizes) if gt_sizes else 0,
        "来源分布": sources,
        "含标准答案数": sum(1 for i in items if i.reference_answer),
    }


def item_to_json(item: QAItem) -> str:
    """调试用：把单条样本序列化为可读 JSON。"""
    return json.dumps(asdict(item), ensure_ascii=False, indent=2)
