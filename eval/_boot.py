# eval/_boot.py
"""
评测子包的引导模块。

职责（只做三件事，不做业务）：
1. 把项目根目录加入 sys.path，保证 `from utils.xxx import ...` 可用，
   无论脚本是 `python eval/run_eval.py` 还是 `python -m eval.run_eval` 启动。
2. 把 stdout/stderr 强制切换为 UTF-8，避免 Windows 控制台默认 GBK 时中文/符号报 UnicodeEncodeError。
3. 显式加载项目根目录下的 .env（而不是依赖当前工作目录），
   必须在任何 `config.*` 模块被导入之前完成。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# 项目根目录（eval/ 的上一级）
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

_initialized = False
# 已加入 NO_PROXY 的主机名，供自检打印
no_proxy_hosts: list[str] = []


def _host_of(value: str) -> str:
    """从 "http://192.168.217.3:19530" / "mongodb://host:27017" / "host:9000" 里取出主机名。"""
    if not value:
        return ""
    if "//" in value:
        return urlparse(value).hostname or ""
    return value.split(":")[0].strip()


def relax_proxy_for_infra() -> list[str]:
    """
    把项目依赖的基础服务（Milvus / MongoDB / MinIO）加进 NO_PROXY。

    为什么必须做：
        pymilvus 的 gRPC 通道会读取 HTTP_PROXY / HTTPS_PROXY 环境变量。当宿主机上
        装了代理软件（Clash / v2rayN 之类）并设置了全局代理时，连局域网内的 Milvus
        也会被塞进代理，表现为：
            HTTP proxy handshake with ipv4:127.0.0.1:9987 failed: ... 502
            Fail connecting to server on 192.168.x.x:19530
        服务明明是好的，却连不上，非常难排查。这里显式声明"这几个地址不要走代理"。

    :return: 本次追加/确认的主机名列表
    """
    hosts = {"localhost", "127.0.0.1", "::1"}
    for var in ("MILVUS_URL", "MONGO_URL"):
        host = _host_of(os.getenv(var) or "")
        if host:
            hosts.add(host)
    host = _host_of(os.getenv("MINIO_ENDPOINT") or "")
    if host:
        hosts.add(host)

    existing = {h.strip() for h in (os.getenv("NO_PROXY") or "").split(",") if h.strip()}
    merged = sorted(existing | hosts)
    value = ",".join(merged)
    # 大小写两种都设，不同的 HTTP 客户端读的变量名不一样
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value

    global no_proxy_hosts
    no_proxy_hosts = sorted(hosts)
    return no_proxy_hosts


def apply_log_level(level: str | None = None) -> str:
    """
    压住项目里过高等级的日志。

    项目 tool/logger.py 把 **root logger** 设成了 DEBUG，而 dashscope 的 HTTP 客户端
    会在 DEBUG 级别把整个请求体打出来 —— 一次重排调用就会刷屏几百行原始文档。
    评测脚本只需要结果，所以这里把级别提到 WARNING。

    想恢复完整日志：设环境变量 EVAL_LOG_LEVEL=DEBUG，或调用 apply_log_level("DEBUG")。

    :return: 实际生效的级别名
    """
    import logging

    target = (level or os.getenv("EVAL_LOG_LEVEL") or "WARNING").upper()
    numeric = getattr(logging, target, logging.WARNING)

    # 先导入项目的 logger 模块，触发它把 root 设成 DEBUG 的行为，再把级别改回来
    try:
        import tool.logger  # noqa: F401
    except Exception:
        pass

    logging.getLogger().setLevel(numeric)
    # 顺带压掉这些库的 DEBUG/INFO，它们比业务日志还吵
    for name in ("httpx", "httpcore", "urllib3", "dashscope", "pymilvus",
                 "openai", "langchain", "langchain_openai", "modelscope"):
        logging.getLogger(name).setLevel(max(numeric, logging.WARNING))
    # 这几个库即使在 WARNING 级也会刷提示（如 tokenizer 的性能建议），直接压到 ERROR
    for name in ("transformers", "tokenizers", "sentence_transformers",
                 "FlagEmbedding", "huggingface_hub", "filelock"):
        logging.getLogger(name).setLevel(logging.ERROR)
    # transformers 在 import 时读这个环境变量决定 verbosity，比 import 它再改设置便宜得多
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return target


def detect_proxy() -> dict[str, str]:
    """返回当前进程可见的代理设置，供自检提示用。"""
    found: dict[str, str] = {}
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.getenv(name)
        if value:
            found[name] = value
    return found


def init() -> Path:
    """
    初始化评测运行环境。幂等，重复调用无副作用。

    :return: 项目根目录 Path
    """
    global _initialized
    if _initialized:
        return PROJECT_ROOT

    # 1. sys.path —— 让项目内的绝对导入生效
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    # 2. 控制台编码 —— Windows 下默认 GBK，中文表格会崩
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    # 3. .env —— 项目各 config 模块内部调用的是 load_dotenv()（按 cwd 向上找），
    #    这里用绝对路径先加载一次，保证从任意目录启动都能读到配置。
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file, override=False)
        except Exception:
            pass

    # 4. 代理绕过（必须在 pymilvus 建连之前）
    relax_proxy_for_infra()

    # 5. 压掉 DEBUG 刷屏
    apply_log_level()

    _initialized = True
    return PROJECT_ROOT


def ensure_project_cwd() -> None:
    """
    把当前工作目录切到项目根目录。

    仅在使用相对路径配置（如 DATA_BASED_ROOT_DIR=./data-root/）时才需要。
    评测脚本默认不调用它，避免改变调用者的预期。
    """
    os.chdir(PROJECT_ROOT)
