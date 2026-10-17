#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量导入脚本 —— 把一个文件夹里的 PDF / MD 直接灌进知识库。

用法
----
    # 先看看会导入哪些文件，不真跑
    python scripts/batch_import.py --src "D:/docs" --dry-run

    # 正式导入（递归子目录，跳过已在库里的同名文档）
    python scripts/batch_import.py --src "D:/docs"

    # 只导文件名含 2608 的
    python scripts/batch_import.py --src "D:/docs" --only 2608

    # 已存在的也强制重跑（注意：会先删旧切片再插入）
    python scripts/batch_import.py --src "D:/docs" --force

    # 先只跑一份，验证环境
    python scripts/batch_import.py --src "D:/docs" --only B530 --limit 1

退出码
------
    0  全部成功
    1  有失败（明细见报告）
    2  环境/参数问题，一条都没跑
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NoReturn, Optional
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_SUFFIXES = {".pdf", ".md"}
# 跳过中间产物：
#   *_new.md      —— 上一轮插图处理后生成的版本，不是用户原始文档
#   *_origin.pdf  —— MinerU 解析时下载回本地的副本，UUID 命名，导入会造出垃圾标题
SKIP_SUFFIX_PATTERNS = ("_new.md", "_origin.pdf")

# UUID 形式的文件名：MinerU 中间产物的特征，导入后 file_title 会是一串无意义字符
import re as _re

_UUID_STEM_RE = _re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", _re.I
)


# --------------------------------------------------------------------------- #
# 环境准备
# --------------------------------------------------------------------------- #
def bootstrap(log_level: str) -> None:
    """Python 环境对齐：UTF-8 输出、加载 .env、绕过代理、压掉调试日志。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as exc:
        _die_on_wrong_python(exc)

    load_dotenv(PROJECT_ROOT / ".env", override=False)

    relax_proxy_for_infra()
    apply_log_level(log_level)


def _die_on_wrong_python(exc: ModuleNotFoundError) -> NoReturn:
    """
    依赖缺失时，最常见的原因是**用了全局 Python 而不是项目的 venv**。

    直接抛 `ModuleNotFoundError: colorlog` 会让人以为是代码坏了，
    实际上全局解释器里根本没装这个项目的依赖。这里把「你用的哪个解释器、
    该用哪个」直接摆在眼前。
    """
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"

    print(f"[x] 当前 Python 缺少依赖：{exc.name}")
    print(f"    正在使用：{sys.executable}")
    print("")
    if venv_python.exists():
        print("    这个脚本必须用项目虚拟环境运行，请改用：")
        print(f'      "{venv_python}" scripts/batch_import.py --src "路径" --dry-run')
        print("")
        print("    或者直接双击项目根目录下的 batch_import.bat（已配好解释器）。")
    else:
        print(f"    没找到虚拟环境：{venv_python}")
        print("    先按 README 创建并安装依赖，再跑这个脚本。")
    raise SystemExit(2)


def _host_of(url: str) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url if "://" in url else f"//{url}")
    return parsed.hostname


def relax_proxy_for_infra() -> None:
    """
    把 Milvus / Mongo / MinIO 的主机名加进 NO_PROXY。

    全局代理（v2rayN 之类）会把 pymilvus 的 gRPC 也劫持走，
    报 `HTTP proxy handshake failed`。这几个服务在局域网里，不能走代理。
    """
    hosts = {"localhost", "127.0.0.1", "::1"}
    for var in ("MILVUS_URL", "MONGO_URL", "MINIO_ENDPOINT"):
        host = _host_of(os.getenv(var) or "")
        if host:
            hosts.add(host)

    value = ",".join(sorted(hosts))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def apply_log_level(level: str) -> None:
    """
    默认压到 WARNING。

    `tool.logger` 把 root 设成 DEBUG，导入流程里每个节点的调试输出、
    三方库的加载信息会刷满屏，反而看不见真正重要的错误。
    """
    import logging

    target = (level or "WARNING").upper()
    numeric = getattr(logging, target, logging.WARNING)

    try:
        import tool.logger  # noqa: F401  触发它把 root 设成 DEBUG
    except ModuleNotFoundError as exc:
        # 走到这里基本是用了全局 Python（缺 colorlog 之类），给出解释器对照
        _die_on_wrong_python(exc)

    logging.getLogger().setLevel(numeric)
    for name in (
        "httpx", "httpcore", "urllib3", "transformers", "tokenizers",
        "sentence_transformers", "FlagEmbedding", "pymilvus", "minio",
        "modelscope", "PIL", "matplotlib",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


# --------------------------------------------------------------------------- #
# 文件扫描
# --------------------------------------------------------------------------- #
def scan_files(src: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    found: List[Path] = []
    for path in sorted(src.glob(pattern)):
        if not path.is_file():
            continue
        name = path.name
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if any(name.endswith(p) for p in SKIP_SUFFIX_PATTERNS):
            continue
        # 忽略 Office 临时文件
        if name.startswith("~$"):
            continue
        # 同目录下已有同名 PDF 的 .md，是 MinerU 的解析产物，不是用户文档
        if path.suffix.lower() == ".md" and path.with_suffix(".pdf").exists():
            continue
        # UUID 命名的文件：MinerU 中间产物，导进去会让 file_title 变成一串乱码
        if _UUID_STEM_RE.match(path.stem):
            continue
        found.append(path)
    return found


def dedupe_by_title(files: List[Path]) -> tuple[List[Path], List[tuple[Path, Path]]]:
    """
    按文件名（不含扩展名）去重。

    为什么要去重：`file_title` 取自文件名，同名文档在库里是**同一个 key**，
    后导入的会删掉先导入的切片。同名不同内容同时导入 = 互相覆盖，必然出错。
    这里保留体积最大的那个，其余在报告里列出来让人工确认。
    """
    best: Dict[str, Path] = {}
    dropped: List[tuple[Path, Path]] = []

    for path in files:
        title = path.stem
        current = best.get(title)
        if current is None:
            best[title] = path
            continue
        if path.stat().st_size > current.stat().st_size:
            dropped.append((current, path))
            best[title] = path
        else:
            dropped.append((path, current))

    return list(best.values()), dropped


# --------------------------------------------------------------------------- #
# Milvus 查询：已存在的文档 / 导入后的切片数
# --------------------------------------------------------------------------- #
def _milvus_client():
    from utils.milvus_utils import get_milvus_client

    return get_milvus_client()


def _read_titles() -> Dict[str, int]:
    """读一次全库，返回 {file_title: 切片数}。"""
    from config.milvus_config import milvus_config

    client = _milvus_client()
    collection = milvus_config.chunks_collection
    counts: Dict[str, int] = {}

    try:
        if not client.has_collection(collection):
            return counts
        rows = client.query(
            collection_name=collection,
            filter="chunk_id >= 0",
            output_fields=["file_title"],
            limit=16384,
        )
        for row in rows:
            title = row.get("file_title")
            if title:
                counts[title] = counts.get(title, 0) + 1
    except Exception as exc:  # 连不上就当作空，不阻断导入
        print(f"  [警告] 读取现有文档失败（将不跳过任何文件）：{exc}")

    return counts


def existing_titles(expect_total: Optional[int] = None, timeout: float = 10.0) -> Dict[str, int]:
    """
    返回 {file_title: 切片数}，用于跳过已导入的文档。

    Milvus 的 delete 是异步生效的：调用返回后马上 query 仍会读到旧数据。
    传 expect_total 时轮询到总数等于它为止（最多 timeout 秒），
    否则清理完立刻打印出来的数字会偏大（实测删了 5 个仍显示 63 而非 59）。
    """
    counts = _read_titles()
    if expect_total is None:
        return counts

    deadline = time.time() + timeout
    while sum(counts.values()) != expect_total and time.time() < deadline:
        time.sleep(0.5)
        counts = _read_titles()
    return counts


def count_chunks(file_title: str, retries: int = 5) -> int:
    """
    数某个 file_title 下有多少切片。

    刚 insert 完立刻 query 可能查不到 —— Milvus 默认是 Bounded 一致性，
    新写入的数据要过一小会儿才可见。所以这里带重试，别把 0 当成导入失败。
    """
    from config.milvus_config import milvus_config
    from utils.milvus_utils import escape_milvus_string

    client = _milvus_client()
    title = escape_milvus_string(file_title)
    for attempt in range(retries):
        try:
            rows = client.query(
                collection_name=milvus_config.chunks_collection,
                filter=f"file_title == '{title}'",
                output_fields=["chunk_id"],
                limit=16384,
            )
            if rows:
                return len(rows)
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(1.0)
    return 0


def report_orphan_titles(counts: Dict[str, int]) -> Dict[str, int]:
    """
    找出 file_title 里带扩展名的记录。

    正常情况下 file_title 应该是 `Path(name).stem`，不带扩展名。
    库里出现 `xxx.pdf` 这种，说明历史上有一批文档走了 `xxx.pdf.md` 的路径，
    于是同一份文档在库里存成了两个 title —— 检索时它们互相当干扰项，
    指标会被拉低。

    :return: {疑似重复的 title: 切片数}，供 `--purge-orphan-titles` 使用
    """
    orphans = {
        t: n for t, n in counts.items()
        if Path(t).suffix.lower() in SUPPORTED_SUFFIXES and Path(t).stem in counts
    }
    if not orphans:
        return {}

    print(f"\n[体检] 发现 {len(orphans)} 个 file_title 带扩展名的历史遗留记录：")
    for title, n in sorted(orphans.items(), key=lambda kv: -kv[1]):
        base = Path(title).stem
        print(f"   {n:4d} 切片  {title}")
        print(f"             对应正常标题 '{base}' 下有 {counts[base]} 切片")
    print("   影响：同一份文档被拆成两个 title，检索时互相干扰。")
    print("   可用 --purge-orphan-titles 清理（会先打印再确认，不会静默删除）。")
    return orphans


def purge_orphan_titles(orphans: Dict[str, int]) -> int:
    """按 file_title 删除遗留记录。只删那些「正常标题也存在」的，避免误删孤儿。返回删除的切片数。"""
    from config.milvus_config import milvus_config
    from utils.milvus_utils import escape_milvus_string

    client = _milvus_client()
    removed = 0
    for title, n in orphans.items():
        try:
            client.delete(
                collection_name=milvus_config.chunks_collection,
                filter=f"file_title == '{escape_milvus_string(title)}'",
            )
            removed += n
            print(f"   已删除 {n:4d} 切片  {title}")
        except Exception as exc:
            print(f"   删除失败 {title}: {exc}")
    return removed


# --------------------------------------------------------------------------- #
# 单个文件导入
# --------------------------------------------------------------------------- #
# 这些错误是网络抖动，不是文档本身有问题，重试通常就能过。
# 尤其 MinerU 走 HTTPS，全局代理偶发 `SSLEOFError: UNEXPECTED_EOF_WHILE_READING`，
# 之前网页端导入失败就是这样，而且任务状态存在内存里，重启后看不到任何错误。
TRANSIENT_MARKERS = (
    "SSLError", "SSLEOFError", "UNEXPECTED_EOF",
    "ConnectionError", "ConnectionReset", "RemoteDisconnected",
    "ProxyError", "Timeout", "timed out", "Max retries exceeded",
    "BrokenPipe", "IncompleteRead",
)

_NODE_IN_ERROR_RE = _re.compile(r"\[(node_[a-z_]+)\]")


def is_transient(error_text: str) -> bool:
    return any(marker.lower() in error_text.lower() for marker in TRANSIENT_MARKERS)


def failing_node(result: dict) -> str:
    """失败发生在哪个节点：优先看异常文本，其次看已完成节点列表。"""
    match = _NODE_IN_ERROR_RE.search(result.get("error") or "")
    if match:
        return match.group(1)
    return result["nodes_done"][-1] if result["nodes_done"] else "(未进入任何节点)"


def import_one(path: Path, data_root: Path, show_traceback: bool,
               retries: int = 2) -> dict:
    """跑一个文件的完整导入图，返回结果字典（不抛异常）。"""
    from processor.import_processor.main_graph import KBImportWorkflow

    task_id = str(uuid.uuid4())
    file_dir = data_root / datetime.now().strftime("%Y%m%d") / task_id
    file_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "task_id": task_id,
        "file_dir": str(file_dir),
        "import_file_path": str(path),
    }

    result = {
        "file": path.name,
        "path": str(path),
        "title": path.stem,
        "task_id": task_id,
        "file_dir": str(file_dir),
        "status": "failed",
        "nodes_done": [],
        "item_name": "",
        "chunks": 0,
        "seconds": 0.0,
        "attempts": 0,
        "error": "",
    }

    started = time.time()

    for attempt in range(1, retries + 2):
        result["attempts"] = attempt
        result["nodes_done"] = []
        try:
            workflow = KBImportWorkflow()
            for event in workflow.run(state, stream=True):
                for node_name, node_result in event.items():
                    result["nodes_done"].append(node_name)
                    if isinstance(node_result, dict) and node_result.get("item_name"):
                        result["item_name"] = node_result["item_name"]
                    print(f"      ✓ {node_name}")
            result["status"] = "completed"
            result["error"] = ""
            break
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            result["status"] = "failed"
            result["error"] = detail
            result["traceback"] = traceback.format_exc()
            can_retry = attempt <= retries and is_transient(detail)
            print(f"      ✗ 第 {attempt} 次失败：{detail[:130]}")
            if can_retry:
                wait = 3 * attempt
                print(f"      ↻ 网络类错误，{wait}s 后重试……")
                time.sleep(wait)
                continue
            if show_traceback:
                print(indent(result["traceback"], 8))
            break

    result["seconds"] = round(time.time() - started, 1)
    if result["status"] == "completed":
        result["chunks"] = count_chunks(result["title"])
    return result


def indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.rstrip().splitlines())


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def render_summary(results: List[dict]) -> str:
    ok = [r for r in results if r["status"] == "completed"]
    bad = [r for r in results if r["status"] != "completed"]

    lines = ["", "=" * 78, " 导入汇总", "=" * 78]
    lines.append(f" 成功 {len(ok)} / 失败 {len(bad)} / 共 {len(results)}")
    total_chunks = sum(r["chunks"] for r in ok)
    lines.append(f" 新增可检索切片：{total_chunks}（按 file_title 统计，含覆盖旧切片）")
    lines.append("")

    if ok:
        lines.append(" [成功]")
        for r in sorted(ok, key=lambda x: -x["chunks"]):
            name = r["item_name"] or "(未识别商品名)"
            lines.append(
                f"   {r['chunks']:5d} 切片  {r['seconds']:6.1f}s  {r['title'][:38]:<40s} {name}"
            )

    if bad:
        lines.append("")
        lines.append(" [失败]")
        for r in bad:
            tries = f"（重试 {r['attempts'] - 1} 次）" if r.get("attempts", 1) > 1 else ""
            lines.append(f"   {r['title'][:44]:<46s} 卡在 {failing_node(r)}{tries}")
            lines.append(f"      {r['error'][:160]}")

    return "\n".join(lines)


def write_report(results: List[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{stamp}_batch_import.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(results),
        "completed": sum(1 for r in results if r["status"] == "completed"),
        "failed": sum(1 for r in results if r["status"] != "completed"),
        "results": results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="批量导入 PDF/MD 到知识库（绕过 HTTP，直接跑导入图）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
常用示例
--------
  # 先看会导入什么，不真跑（强烈建议先跑这个）
  python scripts/batch_import.py --src "D:/我的文档" --recursive --dry-run

  # 正式导入（已导入过的自动跳过）
  python scripts/batch_import.py --src "D:/我的文档" --recursive

  # 只导文件名含关键词的
  python scripts/batch_import.py --src "D:/我的文档" --recursive --only 2608

  # 清理 file_title 带扩展名的历史遗留记录
  python scripts/batch_import.py --src "D:/file" --recursive --purge-orphan-titles

注意：路径里有空格或中文时，两侧要加英文双引号。
""",
    )
    parser.add_argument("--src", help="源文件夹（或单个文件）")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    parser.add_argument("--only", default="", help="只处理文件名包含该关键词的文件")
    parser.add_argument("--limit", type=int, default=0, help="最多处理 N 个文件（0=不限）")
    parser.add_argument("--retries", type=int, default=2,
                        help="网络类错误的重试次数（默认 2；MinerU 走 HTTPS，代理偶发断连）")
    parser.add_argument("--force", action="store_true", help="已导入过的也重跑（会覆盖旧切片）")
    parser.add_argument("--purge-orphan-titles", action="store_true",
                        help="清理 file_title 带扩展名的历史遗留记录（同一文档被存成两个标题）")
    parser.add_argument("--dry-run", action="store_true", help="只列出将要导入的文件，不执行")
    parser.add_argument("--log-level", default="WARNING", help="日志级别（默认 WARNING）")
    parser.add_argument("--traceback", action="store_true", help="失败时打印完整堆栈")
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "logs"), help="报告输出目录")
    args = parser.parse_args(argv)

    # --src 是必需的，但用自定义提示代替 argparse 那句干巴巴的报错。
    # 手动在终端里跑（stdin 是 tty）时直接问路径，省得记参数；
    # 被脚本/CI 调用时（无 tty）就打印用法，不阻塞等待输入。
    if not args.src:
        if sys.stdin and sys.stdin.isatty():
            print("请输入要导入的文件夹路径（也可以直接把文件夹拖进这个窗口）：")
            entered = input("> ").strip().strip('"').strip("'")
            if not entered:
                print("已取消。")
                return 2
            args.src = entered
            if not args.dry_run:
                print("")
                print("提示：本次会【真正导入】。想先看一眼会导入什么，按 Ctrl+C 退出后加 --dry-run。")
        else:
            print("缺少 --src：请告诉我要导入哪个文件夹（或哪个文件）。")
            print("")
            print('  python scripts/batch_import.py --src "D:/我的文档" --recursive --dry-run')
            print("")
            print("先用 --dry-run 看一眼会导入什么，确认无误再去掉它。")
            print("完整参数说明：python scripts/batch_import.py --help")
            return 2

    print("=" * 78)
    print(" 履约知识助手 Warehouse-KB · 批量导入")
    print("=" * 78)

    bootstrap(args.log_level)

    src = Path(args.src).expanduser().resolve()
    if not src.exists():
        print(f"[x] 路径不存在：{src}")
        return 2

    files = [src] if src.is_file() else scan_files(src, args.recursive)
    if args.only:
        files = [f for f in files if args.only.lower() in f.name.lower()]
    if not files:
        print(f"[x] 在 {src} 下没有找到可导入的 PDF/MD 文件")
        print("    提示：子目录里的文件需要加 --recursive")
        return 2

    files, dropped = dedupe_by_title(files)
    if dropped:
        print(f"\n[警告] 有 {len(dropped)} 组同名文档，只保留体积最大的那份：")
        for loser, winner in dropped:
            print(f"   忽略 {loser}")
            print(f"   采用 {winner}")

    print(f"\n[扫描] 共找到 {len(files)} 个文件")

    # 已导入的跳过
    existing = {} if args.force else existing_titles()
    if existing:
        orphans = report_orphan_titles(existing)
        if args.purge_orphan_titles and not args.dry_run:
            if not orphans:
                print("   没有需要清理的记录。")
            else:
                print("\n[清理] 开始删除遗留记录")
                total_before = sum(existing.values())
                removed = purge_orphan_titles(orphans)
                existing = existing_titles(expect_total=total_before - removed)
                print(f"[清理] 完成，库内现有 {sum(existing.values())} 切片")

    pending, skipped = [], []
    for f in files:
        if f.stem in existing:
            skipped.append((f, existing[f.stem]))
        else:
            pending.append(f)

    if skipped:
        print(f"\n[跳过] {len(skipped)} 个文档已在库中（用 --force 可强制重跑）")
        for f, n in skipped:
            print(f"   {f.name}（现有 {n} 切片）")

    if args.limit > 0:
        pending = pending[: args.limit]

    if not pending:
        print("\n没有需要导入的文件。")
        return 0

    print(f"\n[计划] 实际导入 {len(pending)} 个文件")
    for f in pending:
        print(f"   · {f.name}")

    if args.dry_run:
        print("\n[dry-run] 到此为止，没有执行任何导入。")
        return 0

    data_root = Path(os.getenv("DATA_BASED_ROOT_DIR") or PROJECT_ROOT / "data").expanduser()
    print(f"\n[输出目录] {data_root}")

    results: List[dict] = []
    started_all = time.time()

    for idx, path in enumerate(pending, 1):
        print(f"\n[{idx}/{len(pending)}] {path.name}")
        result = import_one(path, data_root, args.traceback, args.retries)
        results.append(result)

    summary = render_summary(results)
    print(summary)

    report_path = write_report(results, Path(args.report_dir))
    print(f"\n总耗时 {time.time() - started_all:.1f}s")
    print(f"报告：{report_path}")

    return 0 if all(r["status"] == "completed" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
