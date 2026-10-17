# utils/llm_utils.py
"""
LLM 客户端工厂 —— 统一走 `config.lm_config.resolve_endpoint(role)`。

改动要点
--------
1. **按角色取端点**：`role="llm"`（回答生成 / HyDE）、`"item"`（商品名识别）、
   `"vl"`（图片描述）各自可以有独立的 base_url / model，支持云端与本地混跑。
2. **`enable_thinking` 只发给云端**：这是百炼 DashScope 的私有字段，
   Ollama 之类的 OpenAI 兼容端点收到未知字段可能直接 400。
3. **本地后端的 JSON 输出不保证**：提供 `extract_json()` 做兜底解析，
   小模型爱加 ```json 围栏或前后带解释。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from config.lm_config import lm_config, resolve_endpoint
from tool.logger import logger

# 键:(role, 模型名, 是否是json模式)
_llm_client_cache: dict[tuple[str, str, bool], ChatOpenAI] = {}


def build_client(role: str = "llm", json_mode: bool = False,
                 model: Optional[str] = None,
                 temperature: Optional[float] = None) -> ChatOpenAI:
    """按角色构建客户端（带缓存）。"""
    endpoint = resolve_endpoint(role)
    m = model or endpoint.model
    if not m:
        env_hint = "LLM_DEFAULT_MODEL" if role == "llm" else f"{role.upper()}_MODEL"
        raise RuntimeError(
            f"角色 {role!r} 未解析出模型名。云端请检查 {env_hint}，"
            "本地请检查 LLM_LOCAL_MODEL（或 ITEM_LOCAL_MODEL / VL_LOCAL_MODEL）。"
        )

    key = (role, m, json_mode)
    if key in _llm_client_cache:
        return _llm_client_cache[key]

    model_kwargs: dict[str, Any] = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    extra_body: dict[str, Any] = {}
    if endpoint.is_local:
        logger.info(f"[LLM] 角色 {role} 使用本地端点：{endpoint.describe()}")
    else:
        # DashScope 私有开关：关掉思考模式，避免 qwen3 系输出推理过程
        extra_body["enable_thinking"] = False

    client = ChatOpenAI(
        model=m,
        temperature=lm_config.llm_temperature if temperature is None else temperature,
        api_key=endpoint.api_key or "EMPTY",
        base_url=endpoint.base_url,
        extra_body=extra_body or None,
        model_kwargs=model_kwargs,
        timeout=120,
        max_retries=1,
    )
    _llm_client_cache[key] = client
    return client


def get_llm_client(model: str | None = None, json_mode: bool = False,
                   role: str = "llm") -> ChatOpenAI:
    """
    获取 LangChain ChatOpenAI 客户端实例。

    :param model: 覆盖模型名；不传则用该角色解析出来的模型
    :param json_mode: True 时要求输出 JSON
    :param role: llm / item / vl
    """
    return build_client(role=role, json_mode=json_mode, model=model)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """
    从模型输出里抠出 JSON。

    本地小模型经常不老实：加 ```json 围栏、前后带解释、或者少个括号。
    这里按「围栏内容 → 原文 → 首尾括号区间」的顺序依次尝试。
    """
    if not text:
        raise ValueError("模型返回为空")

    candidates: list[str] = []
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text.strip())

    for start_ch, end_ch in (("{", "}"), ("[", "]")):
        s, e = text.find(start_ch), text.rfind(end_ch)
        if s != -1 and e > s:
            candidates.append(text[s: e + 1])

    for chunk in candidates:
        try:
            return json.loads(chunk)
        except (json.JSONDecodeError, TypeError):
            continue

    raise ValueError(f"无法从模型输出中解析 JSON：{text[:200]!r}")


if __name__ == "__main__":
    for role in ("llm", "item", "vl"):
        print(resolve_endpoint(role).describe())
    print(get_llm_client())
