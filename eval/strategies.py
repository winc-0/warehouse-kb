# eval/strategies.py
"""
检索策略层 —— 消融实验的"实验组"定义。

核心设计原则：**不复制项目代码，直接复用生产实现**
-----------------------------------------------------
评测最容易犯的错，是为了"方便评测"把检索逻辑重写一遍，结果测出来的不是线上系统。
本模块因此坚持：

* 混合检索      → 调用 utils.milvus_utils.create_hybrid_search_requests / hybrid_search
* HyDE          → 复用 processor.query_processor.prompt.search_embedding_hyde.HYDE_PROMPT
* RRF 融合      → 调用 processor.query_processor.nodes.node_rrf.NodeRrf._rrf_merge
* 重排 + 断崖   → 调用 processor.query_processor.nodes.node_rerank.NodeRerank 的三个 _step_*

只有"仅稠密/仅稀疏"这两组裸基线，因为生产链路里不存在，才在本模块内用
pymilvus 原生 search 实现（参数与集合索引保持一致：稠密 COSINE / 稀疏 IP）。

与生产链路的两处已知差异（写进报告，避免数字被误读）
----------------------------------------------------
1. 联网搜索（node_web_search_mcp）不参与评测。原因：结果依赖外部网络、无稳定 chunk_id、
   不可复现。因此评测结论只覆盖"本地知识库检索"这一段。
2. 评测时 RRF 保留条数放宽到 10（生产为 5），以便观察 Recall@10；
   在 K≤5 的指标上，重排 + 断崖截断后两者结果一致。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from eval._boot import PROJECT_ROOT

# --------------------------------------------------------------------------- #
# 与生产一致的参数
# --------------------------------------------------------------------------- #
OUTPUT_FIELDS: Tuple[str, ...] = ("chunk_id", "content", "item_name", "title", "file_title")
# 单路召回候选数（生产 node_search_embedding 中 limit=10）
PER_REQUEST_LIMIT: int = 10
# 混合检索权重（生产 ranker_weights=(0.8, 0.2)）
RANKER_WEIGHTS: Tuple[float, float] = (0.8, 0.2)
# RRF 平滑参数（生产 _rrf_merge 默认 k=60）
RRF_K: int = 60
# RRF 保留条数：生产为 5，评测放宽到 10 以观察 Recall@10
RRF_MAX_RESULTS: int = 10

CACHE_DIR: Path = PROJECT_ROOT / "eval" / ".cache"


# --------------------------------------------------------------------------- #
# 运行时上下文：Milvus 客户端 + 各类缓存
# --------------------------------------------------------------------------- #
class EvalContext:
    """
    一次评测运行的共享上下文。

    为什么需要它：BGE-M3 编码是整条链路里最贵的本地操作（GPU 上也约几十毫秒/条），
    6 组策略会把同一批问题编码 6 次。这里做进程内缓存，把编码次数压到"唯一文本数"。
    """

    def __init__(self, use_cache: bool = True, verbose: bool = False, cache_dir: Path = CACHE_DIR):
        self.verbose = verbose
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self._client: Any = None
        self._llm: Any = None
        self._emb_cache: Dict[str, Dict[str, Any]] = {}
        self._hyde_cache: Dict[str, str] = {}
        self.embedding_calls: int = 0
        self.llm_calls: int = 0
        self.rerank_calls: int = 0
        self.hyde_cache_hits: int = 0
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_hyde_cache()

    # ---------------- Milvus ----------------
    @property
    def client(self) -> Any:
        if self._client is None:
            from utils.milvus_utils import get_milvus_client

            self._client = get_milvus_client()
        return self._client

    @property
    def collection(self) -> str:
        from config.milvus_config import milvus_config

        return milvus_config.chunks_collection

    # ---------------- 向量编码 ----------------
    def embed(self, text: str) -> Tuple[List[float], Dict[int, float]]:
        """返回 (dense_vector, sparse_vector)。逐条编码，命中缓存则复用。"""
        cached = self._emb_cache.get(text)
        if cached is not None:
            return cached["dense"], cached["sparse"]

        from utils.embedding_utils import generate_embeddings

        embeddings = generate_embeddings([text])
        self.embedding_calls += 1
        dense = embeddings["dense"][0]
        sparse = embeddings["sparse"][0]
        self._emb_cache[text] = {"dense": dense, "sparse": sparse}
        return dense, sparse

    # ---------------- HyDE 假设文档 ----------------
    def _hyde_cache_file(self) -> Path:
        return self.cache_dir / "hyde_cache.jsonl"

    def _load_hyde_cache(self) -> None:
        path = self._hyde_cache_file()
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    self._hyde_cache[row["key"]] = row["value"]
                except Exception:
                    continue

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def hyde_doc(self, question: str) -> str:
        """
        生成 HyDE 假设性文档。带磁盘缓存 —— HyDE 每条问题要花一次 LLM 调用，
        调参复跑时不应该重复付费。
        """
        key = self._hash(question)
        if self.use_cache and key in self._hyde_cache:
            self.hyde_cache_hits += 1
            return self._hyde_cache[key]

        from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
        from utils.llm_utils import get_llm_client

        if self._llm is None:
            self._llm = get_llm_client()

        prompt = HYDE_PROMPT.format(rewritten_query=question)
        doc = self._llm.invoke(prompt).content
        self.llm_calls += 1

        self._hyde_cache[key] = doc
        if self.use_cache:
            with self._hyde_cache_file().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "question": question, "value": doc}, ensure_ascii=False) + "\n")
        return doc


# --------------------------------------------------------------------------- #
# 底层检索原语
# --------------------------------------------------------------------------- #
def _hit_id(hit: Dict[str, Any]) -> Optional[int]:
    """从 Milvus 命中结果里取 chunk_id：优先取主键 id，回退到 entity.chunk_id。"""
    value = hit.get("id")
    if value is None:
        entity = hit.get("entity") or {}
        value = entity.get("chunk_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _item_filter(item_names: Sequence[str]) -> str:
    """
    构造 Milvus 过滤表达式，与生产 node_search_embedding 的写法一致：
        item_name in ["A", "B"]

    全局检索消融开关：
        设置环境变量 EVAL_GLOBAL_SEARCH=1 时返回空表达式（即不加过滤，
        在全库范围内检索）。用它跑一版，与默认版对比，即可量化
        「按设备主体收敛召回范围」这一设计的收益：
            EVAL_GLOBAL_SEARCH=1 python eval/run_eval.py --dataset ...
    """
    if os.getenv("EVAL_GLOBAL_SEARCH") == "1":
        return ""
    names = [n for n in item_names if n]
    if not names:
        return ""
    return f'item_name in {json.dumps(list(names), ensure_ascii=False)}'


def dense_search(ctx: EvalContext, question: str, item_names: Sequence[str], limit: int) -> List[int]:
    """仅稠密向量检索（COSINE）。生产链路不含此路，作为消融基线。"""
    dense, _ = ctx.embed(question)
    res = ctx.client.search(
        collection_name=ctx.collection,
        data=[dense],
        anns_field="dense_vector",
        search_params={"metric_type": "COSINE"},
        limit=limit,
        filter=_item_filter(item_names),
        output_fields=list(OUTPUT_FIELDS),
    )
    return [cid for cid in (_hit_id(h) for h in (res[0] if res else [])) if cid is not None]


def sparse_search(ctx: EvalContext, question: str, item_names: Sequence[str], limit: int) -> List[int]:
    """仅稀疏向量检索（IP）。生产链路不含此路，作为消融基线。"""
    _, sparse = ctx.embed(question)
    res = ctx.client.search(
        collection_name=ctx.collection,
        data=[sparse],
        anns_field="sparse_vector",
        search_params={"metric_type": "IP"},
        limit=limit,
        filter=_item_filter(item_names),
        output_fields=list(OUTPUT_FIELDS),
    )
    return [cid for cid in (_hit_id(h) for h in (res[0] if res else [])) if cid is not None]


def hybrid_search_raw(
    ctx: EvalContext,
    question: str,
    item_names: Sequence[str],
    limit: int = PER_REQUEST_LIMIT,
) -> List[Dict[str, Any]]:
    """
    稠密 + 稀疏混合检索，**完全复用生产实现**。
    返回原始命中（含 entity），供 RRF / 重排继续消费。
    """
    from utils.milvus_utils import create_hybrid_search_requests, hybrid_search

    dense, sparse = ctx.embed(question)
    reqs = create_hybrid_search_requests(
        dense_vector=dense,
        sparse_vector=sparse,
        expr=_item_filter(item_names),
        limit=limit,
    )
    res = hybrid_search(
        client=ctx.client,
        collection_name=ctx.collection,
        reqs=reqs,
        ranker_weights=RANKER_WEIGHTS,
        output_fields=list(OUTPUT_FIELDS),
    )
    return list(res[0]) if res else []


def _entities(hits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把命中结果摊平成 entity 字典列表（生产 RRF 节点消费的就是这个形态）。"""
    out: List[Dict[str, Any]] = []
    for hit in hits:
        entity = hit.get("entity")
        if isinstance(entity, dict):
            if entity.get("chunk_id") is None and hit.get("id") is not None:
                entity = {**entity, "chunk_id": hit.get("id")}
            out.append(entity)
    return out


def rrf_merge(
    entity_lists: Sequence[Sequence[Dict[str, Any]]],
    weights: Sequence[float],
    max_results: int = RRF_MAX_RESULTS,
) -> List[Dict[str, Any]]:
    """调用生产的 RRF 实现做融合。"""
    from processor.query_processor.nodes.node_rrf import NodeRrf

    rrf_inputs = [(list(entities), float(w)) for entities, w in zip(entity_lists, weights)]
    merged = NodeRrf()._rrf_merge(rrf_inputs, k=RRF_K, max_results=max_results)
    return [doc for doc, _score in merged]


def rerank_and_cut(
    ctx: EvalContext,
    question: str,
    entities: Sequence[Dict[str, Any]],
    use_cliff: bool,
) -> List[Dict[str, Any]]:
    """
    调用生产的重排 + 断崖截断实现。

    :param use_cliff: False 时只做重排，不做断崖截断 —— 用于单独量化断崖截断的影响
    """
    from processor.query_processor.nodes.node_rerank import NodeRerank

    node = NodeRerank()
    state = {
        "rewritten_query": question,
        "rrf_chunks": list(entities),
        "web_search_docs": [],  # 评测不引入联网结果
    }
    merged_docs = node._step_1_merge_multi_source_docs(state)
    if not merged_docs:
        return []
    ctx.rerank_calls += 1
    ranked = node._step_2_rerank_merged_docs(state, merged_docs)
    if use_cliff:
        ranked = node._step_3_cliff_cutoff(ranked)
    return ranked


# --------------------------------------------------------------------------- #
# 策略定义
# --------------------------------------------------------------------------- #
@dataclass
class Strategy:
    """一个消融实验组。"""

    key: str
    label: str
    retrieve: Callable[[EvalContext, str, Sequence[str]], List[int]]
    needs_llm: bool = False
    needs_rerank_api: bool = False
    description: str = ""


def _s_dense(ctx: EvalContext, q: str, items: Sequence[str]) -> List[int]:
    return dense_search(ctx, q, items, PER_REQUEST_LIMIT)


def _s_sparse(ctx: EvalContext, q: str, items: Sequence[str]) -> List[int]:
    return sparse_search(ctx, q, items, PER_REQUEST_LIMIT)


def _s_hybrid(ctx: EvalContext, q: str, items: Sequence[str]) -> List[int]:
    hits = hybrid_search_raw(ctx, q, items)
    return [cid for cid in (_hit_id(h) for h in hits) if cid is not None]


def _s_hybrid_hyde_rrf(ctx: EvalContext, q: str, items: Sequence[str]) -> List[int]:
    """生产召回阶段：混合检索 ∥（HyDE → 混合检索）→ RRF 融合。"""
    base = _entities(hybrid_search_raw(ctx, q, items))

    hyde_doc = ctx.hyde_doc(q)
    combined = f"{q} {hyde_doc}"
    hyde_hits = hybrid_search_raw(ctx, combined, items)
    hyde = _entities(hyde_hits)

    if not base and not hyde:
        return []
    if not hyde:
        merged = base
    elif not base:
        merged = hyde
    else:
        merged = rrf_merge([base, hyde], weights=(1.0, 1.0), max_results=RRF_MAX_RESULTS)

    ids: List[int] = []
    for doc in merged:
        cid = doc.get("chunk_id")
        if cid is not None:
            ids.append(int(cid))
    return ids


def _recall_stage_entities(ctx: EvalContext, q: str, items: Sequence[str]) -> List[Dict[str, Any]]:
    """抽出"召回阶段结束时的有序切片列表"，给两组重排策略共用。"""
    return _s_hybrid_hyde_rrf_entities(ctx, q, items)


def _s_hybrid_hyde_rrf_entities(ctx: EvalContext, q: str, items: Sequence[str]) -> List[Dict[str, Any]]:
    base = _entities(hybrid_search_raw(ctx, q, items))
    hyde = _entities(hybrid_search_raw(ctx, f"{q} {ctx.hyde_doc(q)}", items))
    if not base and not hyde:
        return []
    if not hyde:
        return base
    if not base:
        return hyde
    return rrf_merge([base, hyde], weights=(1.0, 1.0), max_results=RRF_MAX_RESULTS)


def _rerank_ids(ctx: EvalContext, q: str, items: Sequence[str], use_cliff: bool) -> List[int]:
    entities = _recall_stage_entities(ctx, q, items)
    ranked = rerank_and_cut(ctx, q, entities, use_cliff=use_cliff)
    ids: List[int] = []
    for doc in ranked:
        cid = doc.get("chunk_id")
        if cid is not None:
            ids.append(int(cid))
    return ids


def _s_rerank(ctx: EvalContext, q: str, items: Sequence[str]) -> List[int]:
    return _rerank_ids(ctx, q, items, use_cliff=False)


def _s_full(ctx: EvalContext, q: str, items: Sequence[str]) -> List[int]:
    return _rerank_ids(ctx, q, items, use_cliff=True)


STRATEGIES: Dict[str, Strategy] = {
    "dense_only": Strategy(
        key="dense_only",
        label="仅稠密向量召回",
        retrieve=_s_dense,
        description="BGE-M3 dense 检索（COSINE）。基线。",
    ),
    "sparse_only": Strategy(
        key="sparse_only",
        label="仅稀疏向量召回",
        retrieve=_s_sparse,
        description="BGE-M3 sparse 检索（IP）。对照基线。",
    ),
    "hybrid": Strategy(
        key="hybrid",
        label="稠密+稀疏混合召回",
        retrieve=_s_hybrid,
        description="WeightedRanker(0.8, 0.2)，与生产 node_search_embedding 一致。",
    ),
    "hybrid_hyde_rrf": Strategy(
        key="hybrid_hyde_rrf",
        label="+ HyDE 路 + RRF 融合",
        retrieve=_s_hybrid_hyde_rrf,
        needs_llm=True,
        description="两条混合召回并行后做 RRF 融合，即生产链路的召回阶段。",
    ),
    "rerank": Strategy(
        key="rerank",
        label="+ Cross-Encoder 重排",
        retrieve=_s_rerank,
        needs_llm=True,
        needs_rerank_api=True,
        description="qwen3-rerank 精排，不做断崖截断。",
    ),
    "full": Strategy(
        key="full",
        label="+ 断崖检测动态截断（完整链路）",
        retrieve=_s_full,
        needs_llm=True,
        needs_rerank_api=True,
        description="重排后按相邻得分落差动态截断，与生产 node_rerank 完全一致。",
    ),
}

# 默认基线（用于计算增量）
DEFAULT_BASELINE = "dense_only"

# 默认执行顺序（也是报告中"逐层叠加"的顺序）
DEFAULT_ORDER: Tuple[str, ...] = (
    "dense_only",
    "sparse_only",
    "hybrid",
    "hybrid_hyde_rrf",
    "rerank",
    "full",
)


def resolve_strategies(keys: Optional[Sequence[str]] = None) -> List[Strategy]:
    """按给定 key 顺序返回策略对象；None 表示用默认顺序。"""
    if not keys:
        return [STRATEGIES[k] for k in DEFAULT_ORDER]
    unknown = [k for k in keys if k not in STRATEGIES]
    if unknown:
        raise KeyError(f"未知策略：{unknown}；可选：{list(STRATEGIES)}")
    return [STRATEGIES[k] for k in keys]


# --------------------------------------------------------------------------- #
# 环境自检
# --------------------------------------------------------------------------- #
REQUIRED_ENV = (
    "MILVUS_URL",
    "CHUNKS_COLLECTION",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "LLM_DEFAULT_MODEL",
    "BGE_M3_PATH",
)


def check_environment(ctx: Optional[EvalContext] = None) -> Tuple[bool, List[str]]:
    """
    运行前自检：环境变量 → 本地模型 → Milvus 连通性 → 集合与数据量。

    :return: (是否全部通过, 逐条说明)
    """
    lines: List[str] = []
    ok = True

    # 1. 环境变量
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        ok = False
        lines.append(f"[x] .env 缺少必填项：{', '.join(missing)}")
    else:
        lines.append("[v] .env 必填项齐全")

    # 2. 本地 BGE-M3
    model_path = os.getenv("BGE_M3_PATH") or ""
    if model_path and Path(model_path).exists():
        lines.append(f"[v] BGE-M3 模型存在：{model_path}")
    else:
        ok = False
        lines.append(f"[x] BGE-M3 模型路径不存在：{model_path or '(未配置)'}")

    # 3. 代理绕过提示
    from eval._boot import detect_proxy, no_proxy_hosts

    proxy = detect_proxy()
    if proxy:
        sample = next(iter(proxy.values()))
        lines.append(f"[i] 检测到代理设置：{sample}")
        lines.append(f"    已将以下地址加入 NO_PROXY 绕过代理：{', '.join(no_proxy_hosts)}")
        lines.append("    若仍连不上 Milvus，请确认代理软件未强制接管局域网流量")

    # 4. Milvus
    ctx = ctx or EvalContext(use_cache=False)
    try:
        stats = ctx.client.get_collection_stats(collection_name=ctx.collection)
        row_count = int(stats.get("row_count", 0))
        if row_count > 0:
            lines.append(f"[v] Milvus 集合 {ctx.collection} 可用，切片数 {row_count}")
        else:
            ok = False
            lines.append(f"[x] Milvus 集合 {ctx.collection} 为空，请先导入文档")
    except Exception as exc:
        ok = False
        url = os.getenv("MILVUS_URL")
        lines.append(f"[x] 无法访问 Milvus（{url}）：{type(exc).__name__}: {exc}")
        lines.append("    提示：Milvus 通常以 Docker 方式启动，确认容器已运行、端口 19530 可连通")

    return ok, lines
