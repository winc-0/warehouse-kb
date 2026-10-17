# processor/query_processor/nodes/node_search_embedding.py
from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase, T
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.json_format_utils import serialize_json
from utils.milvus_utils import create_hybrid_search_requests, hybrid_search, get_milvus_client
from utils.task_utils import add_done_task


class NodeSearchEmbedding(NodeBase):
     """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

     # 覆盖基类的 name 属性，标识节点名称
     name: str = "node_search_embedding"

     def process(self, state: T) -> T:
        """
         节点逻辑
         :param state: 工作流状态对象
         :return: 更新后的状态对象
        """
        try:
             # 1、用户问题和已确认商品名
             query = state.get("rewritten_query")
             item_names = state.get("item_names")

             # 2. 将改写后的问题向量化
             embeddings = generate_embeddings([query])
             dense_vector = embeddings.get("dense")[0]
             sparse_vector = embeddings.get("sparse")[0]

             # 3. 组织查询表达式：匹配已确认商品名
             # 'item_name in ["合力叉车 CPD30 三支点电动叉车","合力叉车CPD30三支点电动叉车"]'
             #
             # global_search=True 时不做主体过滤（expr=None 即全库检索）。
             # 这个开关是给程序化调用准备的：主体确认节点遇到歧义时会返回澄清问句、
             # 不产出 item_names，此时上层若确实需要「先召回一批看看」，
             # 就得能绕开过滤，而不是检索到 0 条。
             if item_names and not state.get("global_search"):
                 expr = f'item_name in {item_names}'
             else:
                 logger.info(f"[{self.name}] 未限定主体，执行全库检索")
                 expr = None

             # quoted = ", ".join(f"'{v}'" for v in item_names)
             # expr = f"item_name in [{quoted}]"

             # 4. 组织混合向量检索的参数
             reqs = create_hybrid_search_requests(
                 dense_vector=dense_vector,
                 sparse_vector=sparse_vector,
                 expr=expr,
                 limit=10
             )

             # 5. 执行混合向量检索
             client = get_milvus_client()
             collection_name = milvus_config.chunks_collection
             search_res = hybrid_search(
                 client = client,
                 collection_name = collection_name,
                 reqs = reqs,
                 ranker_weights=(0.8,0.2),
                 output_fields=["chunk_id", "content", "item_name"]
             )

             add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
             return {"embedding_chunks":  search_res[0] if search_res else []}

        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            return {}

if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于合力 CPD30 叉车，额定载荷是多少？",
        "item_names": ["合力叉车 CPD30 三支点电动叉车", "合力叉车CPD30三支点电动叉车"]
    }
    node_search_embedding = NodeSearchEmbedding()
    result = node_search_embedding(init_state)
    logger.info(serialize_json(result, indent=4))