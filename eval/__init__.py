# eval/__init__.py
"""
履约知识助手 Warehouse-KB 评测子包。

模块分工：
    _boot.py        启动引导（sys.path / 控制台编码 / .env）
    metrics.py      检索指标（纯标准库，可手算验证）
    dataset.py      JSONL 标注集读写与校验
    strategies.py   6 组检索策略（直接复用生产实现）
    build_dataset.py 标注集构建（反向生成 / 人工勾选）
    run_eval.py     消融评测主脚本
    answer_eval.py  答案质量评估（LLM-as-judge）
    selfcheck.py    离线自检，不需要任何外部服务
"""

__all__ = [
    "metrics",
    "dataset",
    "strategies",
]
