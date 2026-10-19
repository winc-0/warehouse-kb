# 履约知识助手 Warehouse-KB · 检索链路评测套件

这个目录回答一个问题：**「三路混合召回 + RRF 融合 + Cross-Encoder 重排 + 断崖检测截断」
这套设计，每一层各自贡献了多少？**

它产出的是数字，不是感觉。跑完你会得到一张逐层叠加的对比表（Recall@K / nDCG@K / MRR），
以及相对基线的增量与显著性检验结果 —— 简历上「召回率从 X 提升到 Y」的出处就在这里。

---

## 快速开始

```bash
# 0) 先自检环境，不消耗任何额度
python eval/run_eval.py --preflight

# 1) 生成标注集（半自动，几分钟，会消耗约 30 次 LLM 调用）
python eval/build_dataset.py generate --style hard --per-item 8 --max-items 6 \
    --out eval/datasets/qa_hard.jsonl

# 2) 小样本试跑，确认链路通了
python eval/run_eval.py --dataset eval/datasets/qa_hard.jsonl --limit 5

# 3) 全量消融评测
python eval/run_eval.py --dataset eval/datasets/qa_hard.jsonl
```

结果落在 `eval/results/`，其中 `*_report.md` 是给人看的报告。

想先验证代码本身没问题，跑离线自检（不需要 Milvus / GPU / 网络）：

```bash
python eval/selfcheck.py     # 53 项指标校验，全部通过才继续
```

---

## 目录结构

| 文件 | 作用 |
|---|---|
| `_boot.py` | 启动引导：sys.path、控制台 UTF-8、.env、代理绕过、日志降噪 |
| `metrics.py` | 检索指标实现（纯标准库，可手算复核） |
| `dataset.py` | 标注集读写与校验 |
| `strategies.py` | 6 组检索策略，**直接复用生产代码**；`check_environment()` 环境自检 |
| `build_dataset.py` | 标注集构建：反向生成 / 人工勾选 / 合并 |
| `run_eval.py` | 消融评测主脚本 |
| `answer_eval.py` | 答案质量评估（LLM-as-judge，口径对齐 RAGAS） |
| `selfcheck.py` | 离线自检，验证指标算得对不对 |
| `datasets/` | 标注集（`qa_template.jsonl` 只是字段模板，不可运行） |
| `results/` | 报告、汇总 CSV、逐题明细、原始检索结果 |
| `.cache/` | HyDE 结果缓存，避免调参复跑时重复调用 LLM |

---

## 标注集格式（JSONL，每行一条）

```json
{
  "qid": "q001",
  "question": "那个2608的网关，配了半天拨号老是不通，是不是得先搞个规则允许IP过？",
  "item_names": ["H3CLA2608室内无线网关"],
  "ground_truth_chunk_ids": [469571489785991047],
  "reference_answer": "配置拨号访问组的拨号控制列表：dialer-rule 1 ip permit",
  "source": "manual",
  "note": ""
}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `qid` | 是 | 唯一编号 |
| `question` | 是 | 用户原始问法 |
| `item_names` | 是 | 商品名，用于 Milvus 的 `item_name in [...]` 过滤；**必须与库中值完全一致** |
| `ground_truth_chunk_ids` | 是 | 正确切片的 `chunk_id`，**必须是 Milvus 真实主键** |
| `reference_answer` | 否 | 标准答案，用于答案质量评估 |
| `source` | 否 | `manual` / `generated`，报告会按来源分组说明偏差 |

> **为什么 chunk_id 不能手编**：项目里 `chunk_id` 是 Milvus 的自增 INT64 主键
> （见 `processor/import_processor/nodes/node_import_milvus.py`），由数据库生成、不落原文。
> 校验会拦住非整数和空值，避免你拿假数据跑出一份全 0 的报告。

### 两条建标注集的路径

**路径 A：反向生成（快，但偏乐观）**

```bash
python eval/build_dataset.py generate --style hard --per-item 8 --max-items 6 \
    --out eval/datasets/qa_hard.jsonl
```

从 Milvus 抽样切片，让 LLM 反推「这段资料能回答什么问题」，该切片自动成为正确答案。

- `--style easy`：问题带完整型号 → 检索几乎不会失手，**指标容易顶到 100% 失去区分度**
- `--style hard`：口语化、去掉完整型号、带干扰信息 → 有区分度，**推荐**

**路径 B：人工标注（慢，但可信）**

```bash
# 1. 准备 questions.txt，每行一个问题，格式：问题 ||| 商品名1,商品名2
# 2. 召回候选切片，导出待勾选 CSV
python eval/build_dataset.py candidates --questions questions.txt --top-n 8 \
    --out eval/datasets/to_label.csv
# 3. 用 Excel/WPS 打开，在 included 列给正确答案填 1，保存
python eval/build_dataset.py merge --labeled eval/datasets/to_label.csv \
    --out eval/datasets/qa_manual.jsonl
```

真实用户问法和反向生成的题目差别很大，条件允许时优先走这条。

---

## 六组消融策略

| key | 策略 | 说明 |
|---|---|---|
| `dense_only` | 仅稠密向量召回 | BGE-M3 dense + COSINE。**默认基线** |
| `sparse_only` | 仅稀疏向量召回 | BGE-M3 sparse + IP。对照基线 |
| `hybrid` | 稠密+稀疏混合召回 | `WeightedRanker(0.8, 0.2)`，与生产 `node_search_embedding` 一致 |
| `hybrid_hyde_rrf` | + HyDE 路 + RRF 融合 | 两条混合召回并行后 RRF 融合 = 生产链路的召回阶段 |
| `rerank` | + Cross-Encoder 重排 | `qwen3-rerank` 精排，不做截断 |
| `full` | + 断崖检测动态截断 | 与生产 `node_rerank` 完全一致 |

```bash
# 只跑不花钱的策略
python eval/run_eval.py --skip-api

# 只跑其中几组
python eval/run_eval.py --strategies dense_only,full

# 换基线
python eval/run_eval.py --baseline hybrid
```

**关键设计**：这些策略不复制项目代码，而是直接调用生产的 `utils/milvus_utils`、
`node_rrf.NodeRrf._rrf_merge`、`node_rerank.NodeRerank._step_*`。
否则测出来的不是线上系统，而是一个「评测专用分支」。

---

## 指标口径

| 指标 | 含义 | 什么时候看它 |
|---|---|---|
| `Recall@K` | 前 K 条覆盖了多少比例的正确答案 | 看「有没有捞到」 |
| `Precision@K` | 前 K 条里正确的比例 | 看「有没有混进垃圾」 |
| `Hit@K` | 前 K 条里是否有至少一条正确 | 只看「有没有命中」 |
| `MRR` | 第一条正确答案出现位置的倒数 | 看「正确答案排得够不够前」 |
| `nDCG@K` | 位置加权的排序质量 | 综合看排序好坏，最接近用户感受 |
| `MAP` | 每次命中位置上的精度均值 | 看整体排序质量 |

相关性按**二值**处理（命中=1，未命中=0），因为人工只标「这段能不能回答问题」。
因此 nDCG 是二值版本，数值会略高于做 0~3 级相关性分级的版本。

显著性用**配对 bootstrap 检验**（同一批问题逐题配对，2000 次重采样，双侧）。
指标是 0/1 变量、天然偏态，用 bootstrap 比 t 检验稳妥。

---

## 报告里的「有效性预警」是什么

评测最危险的陷阱不是跑不通，而是**跑通了但数字没有区分度**：
基线已经 100%，所有策略都是 100%，表格很漂亮，实际上什么都没测出来。

所以 `run_eval.py` 会对一组诊断指标分别计算「相对基线的最大提升」，
只要有一个指标还能拉开 ≥2pt 的差距，就提示你改用它下结论；
如果一个都拉不开，直接判定本次数据无效，并**拒绝生成成果表述**。

典型成因与对策：

1. **语料太小** —— 单个商品下的切片数接近甚至少于 Top-K，检索几乎不可能失手。
   → 导入更多文档，让每个商品的切片数达到 Top-K 的 5 倍以上。
2. **题目由切片反向生成** —— 题目和答案同源，天然好找。
   → 用 `candidates` 走人工标注。
3. **题目都带完整型号** —— 型号本身就是强过滤条件。
   → 用 `generate --style hard` 重新生成。

---

## 常见问题

**Milvus 连不上，但服务明明是好的**

宿主机装了代理软件（Clash / v2rayN 等）并设置了全局代理时，`pymilvus` 的 gRPC 通道
会读取 `HTTP_PROXY` 环境变量，把局域网内的 Milvus 也塞进代理，报
`HTTP proxy handshake ... failed`。

`_boot.init()` 会把 `MILVUS_URL` / `MONGO_URL` / `MINIO_ENDPOINT` 的主机名
自动加进 `NO_PROXY`，通常能直接解决。仍不行的话检查代理软件是否强制接管了局域网流量。

**输出被日志刷屏**

项目 `tool/logger.py` 把 root logger 设成了 DEBUG，dashscope 会把整个请求体打出来。
评测脚本已经压到 WARNING。想看完整日志：

```bash
EVAL_LOG_LEVEL=DEBUG python eval/run_eval.py ...
```

**HyDE 每次都要重新调 LLM 吗**

不用。结果缓存在 `eval/.cache/hyde_cache.jsonl`（按问题 md5 索引），
同一问题复跑不再调用。加 `--no-cache` 可以强制重算。

**大概花多少钱**

- 向量编码：本地 GPU，免费
- HyDE：每问 1 次 LLM 调用，**首次跑才有**，之后命中缓存
- 重排：每问 2 次 `qwen3-rerank` 调用（`rerank` 和 `full` 各一次），每次都真实调用
- 跑之前脚本会打印预计调用次数

想完全不花钱：`--skip-api`（只跑三组本地策略）。

**重排报 403 / Free quota exhausted 怎么办**

DashScope 的 `qwen3-rerank` 有免费额度，用完后会返回 403。脚本会把受影响的两组策略
（`rerank`、`full`）标为「未跑完」并附上建议，其余三组照常出报告。

三条出路：

1. 到 DashScope 控制台充值或开通付费；
2. 改用本地 `bge-reranker-large`（`.env` 里的 `BGE_RERANKER_LARGE` 路径目前指向
   一个并不存在的目录，需要先下载模型，并给 `utils/reranker_utils.py` 补回
   `from FlagEmbedding import FlagReranker` 这行 import）；
3. 用 `--skip-api` 或 `--strategies dense_only,sparse_only,hybrid` 先把不依赖外部接口的部分跑完。

**单策略/少策略时会提示「无法做消融归因」**

这是保护机制：只有一组策略成功时，没有可比对象，硬算增量只会误导。
诊断状态共四种（见 `run_eval.py` 的 `diagnose_metrics`）：

| status | 含义 | 该怎么办 |
|---|---|---|
| `single_strategy` | 只有一组成功 | 看失败原因，处理完重跑 |
| `no_baseline` | 基线没参与 | 用 `--baseline` 指定一个已跑的策略 |
| `no_discrimination` | 所有指标都饱和 | 换更难的题、扩语料 |
| `headline_saturated` | 头号指标饱和但其他指标有效 | 换结论指标即可（报告会自动切） |
| `ok` | 正常 | — |

---

## 结论的适用边界

以下三点会写进每份报告，请连同数字一起看：

1. **不含联网检索**：`node_web_search_mcp` 未参与评测 —— 结果依赖外部网络、
   无稳定 chunk_id、不可复现。本套件只评估「本地知识库检索」这一段。
2. **RRF 保留条数放宽**：评测中 RRF 保留 10 条（生产为 5），以便观察 Recall@10；
   在 K≤5 的指标上，重排 + 断崖截断后两者结果一致。
3. **标注来源偏差**：`source=generated` 占比越高，指标越偏乐观。

---

## 可以继续做的

- **答案质量评估**：`answer_eval.py` 已实现忠实度 / 相关性 / 正确性三个指标，
  可以量化「检索对了但答案自己编」这种幻觉问题。
- **扩充语料**：当前 `kb_chunks` 只有 51 个切片，是限制评测区分度的最大瓶颈。
- **人工标注集**：走 `candidates` 路径，用真实用户问法换掉反向生成的题目。
- **参数搜索**：`PR = 0.8/0.2` 的权重、RRF 的 `k=60`、断崖阈值都是硬编码常量，
  本套件已具备扫参的基础设施（策略可插拔）。
