#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清空知识库（kb_chunks + kb_item_names），为换语料做准备。

为什么要整个清掉而不是按 file_title 删
--------------------------------------
业务域换了：库里原来是 IT 设备手册（H3C 路由器、打印机等），现在换成仓储物流装备手册。
导入链路本身是按 file_title 做幂等覆盖的，只会替掉同名文档，
旧手册的切片会原样留在库里——那样检索时会召回一堆已经不存在的设备，
评测指标也就没法解释了。所以换域必须重建，不能增量。

两个集合一起清：
  kb_chunks      切片本体（向量 + 稀疏向量 + 文本）
  kb_item_names  主体名库（主体确认环节用它做候选召回）
只清 kb_chunks 的话，主体确认还会拿旧设备名去匹配，第一环就错了。

用法
----
    python scripts/reset_library.py           # 打印现状，等确认
    python scripts/reset_library.py --yes     # 直接执行
    python scripts/reset_library.py --stats   # 只看规模，不动数据
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=False)

# 基础设施都在局域网里，全局代理会把 gRPC 也劫走
for host in ("192.168.217.3", "localhost", "127.0.0.1"):
    os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + f",{host}"

from pymilvus import MilvusClient  # noqa: E402

from config.milvus_config import milvus_config  # noqa: E402


def stats(client: MilvusClient) -> dict[str, int]:
    out = {}
    for name in (milvus_config.chunks_collection, milvus_config.item_name_collection):
        if not client.has_collection(name):
            out[name] = -1
            continue
        out[name] = client.get_collection_stats(name).get("row_count", 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="清空履约知识助手知识库集合")
    ap.add_argument("--yes", action="store_true", help="跳过确认直接执行")
    ap.add_argument("--stats", action="store_true", help="只打印规模")
    args = ap.parse_args()

    client = MilvusClient(uri=milvus_config.milvus_url)
    before = stats(client)

    print("当前库规模：")
    for name, n in before.items():
        print(f"  {name}: {'不存在' if n < 0 else f'{n} 条'}")

    if args.stats:
        return 0

    if not args.yes:
        ans = input("\n确认清空以上集合？输入 yes 继续：").strip().lower()
        if ans != "yes":
            print("已取消")
            return 1

    for name in (milvus_config.chunks_collection, milvus_config.item_name_collection):
        if client.has_collection(name):
            client.drop_collection(name)
            print(f"  已删除集合 {name}")

    after = stats(client)
    print("\n清空后：")
    for name, n in after.items():
        print(f"  {name}: {'不存在' if n < 0 else f'{n} 条'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
