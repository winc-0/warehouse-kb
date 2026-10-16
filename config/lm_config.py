# config/lm_config.py
"""
大模型端点配置 —— 按「角色」解析，支持云端 / 本地混合。

为什么要有角色
--------------
项目里有三类 LLM 用途，对模型能力的要求完全不同：

    llm    主回答生成 + HyDE 假设文档     需要较好的生成与理解能力
    item   商品名识别（判断这句话在问哪个产品）  小模型足够
    vl     图片描述（多模态）              必须是视觉模型

以前只有一个 `base_url` / `api_key`，三类用途绑死在同一家云端。
现在每个角色可以独立选择后端，于是可以做到「贵的用云、简单的用本地」，
或者干脆全部本地。

环境变量
--------
    LLM_BACKEND=dashscope|local     默认后端（角色未单独指定时生效）

    # 云端（沿用原有变量，保持兼容）
    OPENAI_API_BASE / OPENAI_API_KEY
    LLM_DEFAULT_MODEL / ITEM_MODEL / VL_MODEL

    # 本地（OpenAI 兼容端点，如 Ollama / vLLM / LM Studio）
    LLM_LOCAL_BASE_URL   默认 http://127.0.0.1:11434/v1
    LLM_LOCAL_API_KEY    本地服务通常不校验，随便填，默认 ollama
    LLM_LOCAL_MODEL      本地文本模型名，如 qwen2.5:3b

    # 角色级覆盖（不写就跟随 LLM_BACKEND / LLM_LOCAL_*）
    LLM_BACKEND / ITEM_BACKEND / VL_BACKEND
    ITEM_LOCAL_MODEL / VL_LOCAL_MODEL
"""

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

BACKEND_CLOUD = "dashscope"
BACKEND_LOCAL = "local"

# 角色 -> 云端模型名所在的环境变量
_CLOUD_MODEL_ENV = {
    "llm": "LLM_DEFAULT_MODEL",
    "item": "ITEM_MODEL",
    "vl": "VL_MODEL",
}

_DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
_DEFAULT_LOCAL_API_KEY = "ollama"


@dataclass(frozen=True)
class Endpoint:
    """一个具体的模型端点。"""

    role: str
    backend: str      # dashscope / local
    base_url: str
    api_key: str
    model: str

    @property
    def is_local(self) -> bool:
        return self.backend == BACKEND_LOCAL

    def describe(self) -> str:
        tag = "本地" if self.is_local else "云端"
        return f"{self.role}: {tag} {self.model} @ {self.base_url}"


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def role_backend(role: str) -> str:
    """该角色走哪个后端；未单独配置时跟随 LLM_BACKEND，再回退到云端。"""
    raw = _env(f"{role.upper()}_BACKEND") or _env("LLM_BACKEND") or BACKEND_CLOUD
    backend = raw.lower()
    if backend not in (BACKEND_CLOUD, BACKEND_LOCAL):
        return BACKEND_CLOUD
    return backend


def resolve_endpoint(role: str = "llm") -> Endpoint:
    """解析某个角色实际要用的端点。"""
    role = (role or "llm").strip().lower()
    backend = role_backend(role)

    if backend == BACKEND_LOCAL:
        base_url = _env("LLM_LOCAL_BASE_URL") or _DEFAULT_LOCAL_BASE_URL
        api_key = _env("LLM_LOCAL_API_KEY") or _DEFAULT_LOCAL_API_KEY
        model = (
            _env(f"{role.upper()}_LOCAL_MODEL")
            or _env("LLM_LOCAL_MODEL")
            or _env(_CLOUD_MODEL_ENV.get(role, "LLM_DEFAULT_MODEL"))
        )
        return Endpoint(role=role, backend=backend, base_url=base_url, api_key=api_key, model=model)

    return Endpoint(
        role=role,
        backend=BACKEND_CLOUD,
        base_url=_env("OPENAI_API_BASE"),
        api_key=_env("OPENAI_API_KEY"),
        model=_env(_CLOUD_MODEL_ENV.get(role, "LLM_DEFAULT_MODEL")),
    )


@dataclass
class LLMConfig:
    """兼容原有用法：属性名不变，但值都经过角色解析。"""

    llm_temperature: float

    # --- 角色解析后的端点（推荐新代码使用） ---
    def endpoint(self, role: str = "llm") -> Endpoint:
        return resolve_endpoint(role)

    # --- 旧属性，等价于 resolve_endpoint("llm") ---
    @property
    def base_url(self) -> str:
        return resolve_endpoint("llm").base_url

    @property
    def api_key(self) -> str:
        return resolve_endpoint("llm").api_key

    @property
    def llm_model(self) -> str:
        return resolve_endpoint("llm").model

    @property
    def item_model(self) -> str:
        return resolve_endpoint("item").model

    @property
    def vl_model(self) -> str:
        return resolve_endpoint("vl").model


lm_config = LLMConfig(
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE") or 0.7),
)


if __name__ == "__main__":
    for r in ("llm", "item", "vl"):
        print(resolve_endpoint(r).describe())
