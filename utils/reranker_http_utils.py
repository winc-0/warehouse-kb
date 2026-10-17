from http import HTTPStatus

import dashscope

from config.reranker_config import reranker_config


def rerank_documents(query: str, documents: list[str]) -> list[float]:
    """
    使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    :param query: 输入查询
    :param documents: 输入文档列表
    :return: 重排后的文档列表
    """

    # 1. 发起请求
    dashscope.api_key = reranker_config.text_rerank_api_key # 设置 API Key
    resp = dashscope.TextReRank.call(
        model=reranker_config.text_rerank_model,
        query=query,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        instruct=reranker_config.text_rerank_instruct
    )

    # 2. 获取相应
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"DashScope qwen3 rerank API 调用失败: {resp.status_code}, 响应消息：{resp.message}")


    # 3. 解析成功的相应
    results = resp.output.get("results", [])

    scores = [0.0] * len(results)
    for result in results:
        score = result.get("relevance_score")
        index = result.get("index")
        scores[index] = score

    return scores