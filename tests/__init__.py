# tests/__init__.py
"""履约知识助手评测层的离线单元测试。

跑法（在仓库根目录）：

    python -m unittest discover -s tests -v

全部纯标准库 + 不连 Milvus / 不打 API，所以任何机器上都能跑。
覆盖的是「数字算得对不对」和「数据集读得对不对」这两件最容易静默出错的事，
而不是检索效果本身——后者得靠 eval/run_eval.py 跑真实链路。
"""
