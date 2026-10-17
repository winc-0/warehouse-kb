# utils/reranker_utils.py
"""
重排（Rerank）统一入口 —— 支持云端 API 与本地模型两种后端，本地又分两种模型架构。

为什么要有这一层
----------------
改之前的状态是割裂的：
  * `node_rerank.py` 直接 `from utils.reranker_http_utils import rerank_documents`，
    把「重排」和「HTTP 调用」焊死在一起；
  * `utils/reranker_utils.py` 因为 `from FlagEmbedding import FlagReranker` 那行被注释掉，
    实际是死代码，谁 import 谁炸。

结果就是：想从云端换成本地，必须去改业务节点。换重排模型本该是一行环境变量的事。

后端与模型
----------
    RERANK_BACKEND=dashscope   云端 qwen3-rerank，消耗额度（默认，保持线上行为不变）
    RERANK_BACKEND=local       本地模型，不消耗额度

本地模型支持两种架构，按模型目录里的 `config.json` 自动识别：

  * **qwen3**：`Qwen/Qwen3-Reranker-0.6B` 这类基于因果语言模型的重排器。
    用「yes / no」两个 token 的 logits 做二分类打分，与云端 qwen3-rerank 同族，
    排序行为最接近。**推荐**。
  * **bge**：`BAAI/bge-reranker-v2-m3` 这类 Cross-Encoder。
    走 FlagEmbedding 的 FlagReranker，打分更"极化"，中文长文档上排序能力弱于 qwen3 系。

环境变量
--------
    RERANK_LOCAL_MODEL        本地模型目录（也认 BGE_RERANKER_PATH / BGE_RERANKER_LARGE）
    RERANK_LOCAL_DEVICE       运行设备（也认 BGE_RERANKER_DEVICE），默认 cuda:0
    RERANK_LOCAL_FP16         1/0（也认 BGE_RERANKER_FP16），默认 0
    RERANK_LOCAL_MAX_LENGTH   单条 pair 的最大 token 数，默认 2048
    RERANK_LOCAL_BATCH        推理批大小，默认 8

注意：换后端要重新看断崖阈值
----------------------------
三种后端的分数分布不同，断崖检测（`node_rerank.py` 的 RERANK_GAP_ABS / RERANK_GAP_RATIO）
的松紧会跟着变。这几个阈值已支持环境变量覆盖，换模型后建议观察截断是否过早。
"""

from __future__ import annotations

import json
import os
from typing import List, Sequence

from tool.logger import logger

# 后端
BACKEND_DASHSCOPE = "dashscope"
BACKEND_LOCAL = "local"
_VALID_BACKENDS = (BACKEND_DASHSCOPE, BACKEND_LOCAL)

# 本地模型架构
ENGINE_QWEN3 = "qwen3"
ENGINE_BGE = "bge"

# ---- 本地模型单例 ----
_local_state = {
    "signature": None,   # (engine, model_path, device, fp16)
    "engine": None,
    "tokenizer": None,
    "model": None,       # transformers 模型（qwen3 引擎）
    "reranker": None,    # FlagReranker（bge 引擎）
    "yes_id": None,
    "no_id": None,
    "keep_last_only": False,  # 模型是否支持 logits_to_keep
}


# --------------------------------------------------------------------------- #
# 配置读取
# --------------------------------------------------------------------------- #
def get_backend() -> str:
    """当前生效的重排后端，非法值回退到 dashscope 并告警。"""
    backend = (os.getenv("RERANK_BACKEND") or BACKEND_DASHSCOPE).strip().lower()
    if backend not in _VALID_BACKENDS:
        logger.warning(
            f"RERANK_BACKEND={backend!r} 不合法（可选 {_VALID_BACKENDS}），已回退到 {BACKEND_DASHSCOPE}"
        )
        return BACKEND_DASHSCOPE
    return backend


def get_local_model_path():
    """本地模型目录：优先 RERANK_LOCAL_MODEL，兼容旧的 BGE_RERANKER_PATH / BGE_RERANKER_LARGE。"""
    for name in ("RERANK_LOCAL_MODEL", "BGE_RERANKER_PATH", "BGE_RERANKER_LARGE"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


def _device() -> str:
    return (os.getenv("RERANK_LOCAL_DEVICE") or os.getenv("BGE_RERANKER_DEVICE") or "cuda:0").strip()


def _use_fp16() -> bool:
    raw = os.getenv("RERANK_LOCAL_FP16")
    if raw is None:
        raw = os.getenv("BGE_RERANKER_FP16")
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _max_length() -> int:
    for name in ("RERANK_LOCAL_MAX_LENGTH", "BGE_RERANKER_MAX_LENGTH"):
        raw = os.getenv(name)
        if raw not in (None, ""):
            try:
                return int(raw)
            except ValueError:
                pass
    return 2048


def _batch_size() -> int:
    for name in ("RERANK_LOCAL_BATCH", "BGE_RERANKER_BATCH"):
        raw = os.getenv(name)
        if raw not in (None, ""):
            try:
                return int(raw)
            except ValueError:
                pass
    return 8


# --------------------------------------------------------------------------- #
# 本地模型加载
# --------------------------------------------------------------------------- #
def detect_engine(model_path: str) -> str:
    """按 config.json 的 architectures / model_type 判断该用哪种打分方式。"""
    config_file = os.path.join(model_path, "config.json")
    try:
        with open(config_file, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception:
        return ENGINE_BGE

    archs = config.get("architectures") or []
    if any("Qwen3" in str(a) for a in archs) or config.get("model_type") == "qwen3":
        return ENGINE_QWEN3
    return ENGINE_BGE


def _load_qwen3(model_path: str, device: str, fp16: bool):
    """
    加载 Qwen3 系重排器。

    打分方式与官方一致：把 (指令, 查询, 文档) 渲染成对话模板，
    取最后一个位置 "yes" / "no" 两个 token 的 logits 做 softmax，
    "yes" 的概率即为相关性分数（0~1）。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    dtype = torch.float16 if fp16 else torch.float32
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, trust_remote_code=True)
        .to(device)
        .eval()
    )

    # yes / no 的 token id：优先用模型自带的 LogitScore 配置，取不到就按词表查
    yes_id, no_id = None, None
    logit_cfg = os.path.join(model_path, "1_LogitScore", "config.json")
    if os.path.isfile(logit_cfg):
        try:
            with open(logit_cfg, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            yes_id, no_id = int(cfg["true_token_id"]), int(cfg["false_token_id"])
        except Exception:
            yes_id, no_id = None, None
    if yes_id is None or no_id is None:
        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")

    logger.info(f"本地重排引擎=qwen3，yes_id={yes_id}, no_id={no_id}")
    return tokenizer, model, yes_id, no_id


def _supports_logits_to_keep(model) -> bool:
    """
    是否支持 `logits_to_keep`。

    为什么要这个：Qwen3 的 lm_head 默认把**每个位置**的 logits 都算出来。
    重排只关心最后一个位置，但词表 15 万 × 序列 2048 × batch 8 ≈ 5.7 GiB，
    8G 显存（还和 BGE-M3 共用）直接 OOM。指定 logits_to_keep=1 后只剩最后一格，
    显存占用降到几 MB，是本地跑得起来的前提。
    """
    try:
        import inspect

        params = inspect.signature(model.forward).parameters
        return "logits_to_keep" in params or "num_logits_to_keep" in params
    except (TypeError, ValueError):
        return False


def _ensure_loaded():
    """确保模型与配置匹配（配置变了会自动重载），返回 (engine, 单例字典)。"""
    model_path = get_local_model_path()
    if not model_path:
        raise RuntimeError(
            "RERANK_BACKEND=local 但未配置 RERANK_LOCAL_MODEL（或 BGE_RERANKER_PATH）。"
            "请在 .env 中指向本地重排模型目录。"
        )
    if not os.path.isdir(model_path):
        raise RuntimeError(
            f"本地重排模型目录不存在：{model_path}\n"
            "下载示例：\n"
            '  modelscope download --model Qwen/Qwen3-Reranker-0.6B --local-dir "D:/ai_models/Qwen3-Reranker-0.6B"'
        )

    engine = detect_engine(model_path)
    device = _device()
    fp16 = _use_fp16()
    signature = (engine, model_path, device, fp16)

    if _local_state["signature"] == signature:
        return engine

    if engine == ENGINE_QWEN3:
        tokenizer, model, yes_id, no_id = _load_qwen3(model_path, device, fp16)
        _local_state.update(
            tokenizer=tokenizer, model=model, yes_id=yes_id, no_id=no_id, reranker=None,
            keep_last_only=_supports_logits_to_keep(model),
        )
    else:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError(
                "缺少 FlagEmbedding，无法使用 bge 系本地重排。安装：pip install -U FlagEmbedding"
            ) from exc
        logger.info(f"本地重排引擎=bge，加载 {model_path}（device={device}, fp16={fp16}）")
        _local_state.update(
            reranker=FlagReranker(model_name_or_path=model_path, device=device, use_fp16=fp16),
            tokenizer=None, model=None, yes_id=None, no_id=None, keep_last_only=False,
        )

    _local_state["signature"] = signature
    _local_state["engine"] = engine
    logger.info(f"本地重排模型就绪：{model_path}")
    return engine


# --------------------------------------------------------------------------- #
# 打分
# --------------------------------------------------------------------------- #
_QWEN3_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_DEFAULT_INSTRUCT = "针对给定的查询，检索能够解答该查询的相关段落"


def _score_qwen3(query: str, documents: Sequence[str]) -> List[float]:
    import torch

    tokenizer = _local_state["tokenizer"]
    model = _local_state["model"]
    yes_id = _local_state["yes_id"]
    no_id = _local_state["no_id"]
    keep_last_only = _local_state["keep_last_only"]
    instruct = os.getenv("TEXT_RERANK_INSTRUCT") or _DEFAULT_INSTRUCT
    max_length = _max_length()

    texts = [
        f"{_QWEN3_PREFIX}<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {doc or ''}{_QWEN3_SUFFIX}"
        for doc in documents
    ]

    scores: List[float] = []
    batch_size = max(1, _batch_size())
    start = 0
    while start < len(texts):
        batch = texts[start: start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)
        kwargs = {"logits_to_keep": 1} if keep_last_only else {}
        try:
            with torch.no_grad():
                logits = model(**encoded, **kwargs).logits[:, -1, :]
        except torch.cuda.OutOfMemoryError:
            # 显存不够就减半重试；减到 1 还失败说明单条就放不下，只能调小 RERANK_LOCAL_MAX_LENGTH
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise RuntimeError(
                    f"本地重排显存不足（RERANK_LOCAL_MAX_LENGTH={max_length}）。"
                    "请调小该值（如 1024）或改用更小的模型。"
                )
            batch_size = max(1, batch_size // 2)
            logger.warning(f"本地重排显存不足，批大小降到 {batch_size} 重试")
            continue
        pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
        probs = torch.softmax(pair.float(), dim=1)[:, 1]
        scores.extend(probs.tolist())
        start += len(batch)

    return [float(s) for s in scores]


def _score_bge(query: str, documents: Sequence[str]) -> List[float]:
    reranker = _local_state["reranker"]
    pairs = [[query, doc or ""] for doc in documents]
    scores = reranker.compute_score(
        pairs,
        normalize=True,  # sigmoid 归一化到 0~1，与云端分数口径对齐
        max_length=_max_length(),
        batch_size=_batch_size(),
    )
    if isinstance(scores, (int, float)):
        scores = [float(scores)]
    return [float(s) for s in scores]


def rerank_local(query: str, documents: Sequence[str]) -> List[float]:
    """
    本地重排打分。

    :return: 与 documents 等长、顺序一致的 0~1 相关性分数
    """
    if not documents:
        return []
    engine = _ensure_loaded()
    if engine == ENGINE_QWEN3:
        return _score_qwen3(query, documents)
    return _score_bge(query, documents)


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #
def rerank_documents(query: str, documents: Sequence[str]) -> List[float]:
    """
    重排统一入口 —— 业务节点只调用这个函数。

    :param query: 用户查询（已改写）
    :param documents: 待打分文本列表
    :return: 与 documents 顺序一致的相关性分数（0~1，越高越相关）
    """
    if get_backend() == BACKEND_LOCAL:
        return rerank_local(query, documents)

    # 云端：延迟导入，保证只用本地时不会因为 dashscope 缺失而失败
    from utils.reranker_http_utils import rerank_documents as dashscope_rerank

    return dashscope_rerank(query, list(documents))


def describe_backend() -> str:
    """给自检/日志用的一行说明。"""
    if get_backend() == BACKEND_LOCAL:
        path = get_local_model_path() or "(未配置)"
        engine = detect_engine(path) if os.path.isdir(path) else "未知"
        return f"local / {engine}（{path}，device={_device()}）"
    model = os.getenv("TEXT_RERANK_MODEL") or "(未配置)"
    return f"dashscope（{model}，消耗云端额度）"
