"""
查询流程的接口定义
"""
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, StreamingResponse

from processor.query_processor.main_graph import KBQueryWorkflow
from utils.mongo_history_utils import clear_history, get_recent_messages
from utils.sse_utils import create_sse_queue, SSEEvent, push_to_session, sse_generator
from utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, get_task_result, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED
from tool.logger import logger
# 1. 创建应用
app = FastAPI(
    title="履约知识助手 Warehouse-KB-查询API",
    description="此文档是履约知识助手 Warehouse-KB 查询流程的 API 接口说明"
)

# 2. 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的源
    allow_credentials=True,  # 允许携带cookie
    allow_methods=["*"],  # 允许的请求方法
    allow_headers=["*"],  # 允许的请求头
)

# 3. 静态页面路由
@app.get("/chat.html")  # 对外访问地址
async def chat():
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    html_path = current_dir_parent_path / "page" / "chat.html"
    # 如果不存在，抛出404异常
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{html_path}")
    return FileResponse(html_path)


# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    """查询请求数据结构"""
    query: str = Field(..., description="查询内容")  # ...必须填写
    session_id: str = Field(None, description="会话ID")
    is_stream: bool = Field(False, description="是否流式返回")

@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    1 解析参数
    2 更新任务状态
    3 调用处理流程图
    4 返回结果
    :param background_tasks:
    :param request:
    :return:
    """
    user_query = request.query
    session_id = request.session_id if request.session_id else str(uuid.uuid4())

    # 处理是不是流式返回结果
    is_stream = request.is_stream

    if is_stream:
        # 创建一个字典 存储对一个session_id : queue 结果队列
        create_sse_queue(session_id)
    # 更新任务状态
    # 当前会话id作为key! 整体装填处于运行中！
    update_task_status(session_id, TASK_STATUS_PROCESSING,is_stream)

    print("开始处理流程... 是否流式:", is_stream, f"其他参数:{user_query}, session_id:{session_id}")

    if is_stream:
        # 如果是流式，则返回一个流式响应，过程不断地推送
        # 运行执行图对象方法
        background_tasks.add_task(run_query_graph, session_id, user_query, is_stream)
        # 返回结果
        print("开始处理结果....")
        return {
            "message":"结果正在处理中...",
            "session_id":session_id
        }
    else:
        # 同步运行
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id,"answer","")
        return {
            "message":"处理完成！",
            "session_id":session_id,
            "answer":answer,
            "done_list":[]
        }

# 只跑检索链路的节点（不生成答案），供物流研究助手 Research-Agent 编排调用
_retrieve_nodes = None


def _get_retrieve_nodes():
    """
    懒加载检索链路上的四个节点。

    这些节点构造时会加载嵌入模型、建立 Milvus / 重排客户端连接，
    放在模块级一次性创建，避免每次请求都重新加载一遍模型。
    """
    global _retrieve_nodes
    if _retrieve_nodes is None:
        from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm
        from processor.query_processor.nodes.node_rerank import NodeRerank
        from processor.query_processor.nodes.node_rrf import NodeRrf
        from processor.query_processor.nodes.node_search_embedding import NodeSearchEmbedding
        _retrieve_nodes = {
            "item_name_confirm": NodeItemNameConfirm(),
            "search_embedding": NodeSearchEmbedding(),
            "rrf": NodeRrf(),
            "rerank": NodeRerank(),
        }
    return _retrieve_nodes


class RetrieveRequest(BaseModel):
    """检索请求：只要切片，不要答案。"""
    query: str = Field(..., description="查询内容")
    top_k: int = Field(8, description="返回切片条数")


@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """
    只执行检索链路（主体确认 → 向量召回 → RRF 融合 → 重排），不生成答案。

    与 /query 的区别：/query 面向终端用户，产出自然语言答案；
    /retrieve 面向上层编排系统，产出结构化切片，让调用方自己决定怎么用。
    物流研究助手 Research-Agent 的知识库证据源就是调这个接口。

    注意：节点是同步实现（内部有模型推理与网络 IO），这里直接调用会占用事件循环，
    该接口定位是内部编排调用、并发量低，暂不引入线程池。
    """
    import uuid

    nodes = _get_retrieve_nodes()
    state = {
        "session_id": f"retrieve-{uuid.uuid4().hex[:12]}",
        "message_id": uuid.uuid4().hex[:12],
        "original_query": request.query,
        "rewritten_query": request.query,
        "is_stream": False,
        "history": [],
        "embedding_chunks": [],
        "hyde_embedding_chunks": [],
        "web_search_docs": [],
        "rrf_chunks": [],
        "reranked_docs": [],
        "item_names": [],
    }

    try:
        state.update(nodes["item_name_confirm"].process(state) or {})
        # 主体确认遇到歧义时不产出 item_names（它会返回一句澄清问句），
        # 此时退化为全库检索：拿不到精确主体，也总比召回 0 条要好
        if not state.get("item_names"):
            state["global_search"] = True
        state.update(nodes["search_embedding"].process(state) or {})
        state.update(nodes["rrf"].process(state) or {})

        # 重排服务要求 documents 非空，召回为空时直接返回，不要去撞它的参数校验
        if state.get("rrf_chunks"):
            state.update(nodes["rerank"].process(state) or {})
    except Exception as e:
        logger.error(f"/retrieve 检索链路异常: {e}")
        raise HTTPException(status_code=500, detail=f"检索失败：{e}")

    docs = state.get("reranked_docs") or state.get("rrf_chunks") or []
    return {
        "query": request.query,
        "item_names": state.get("item_names") or [],
        "count": len(docs[:request.top_k]),
        "chunks": [
            {
                "chunk_id": d.get("chunk_id"),
                "content": d.get("content", ""),
                "item_name": d.get("item_name", ""),
                "score": d.get("score"),
            }
            for d in docs[:request.top_k]
        ],
    }


# 定义查询接口
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True):
    print(f"开始流程图处理...{session_id} {user_query} {is_stream}")

    init_state = {
        "original_query": user_query,
        "session_id": session_id,
        "is_stream": is_stream
    }

    try:
        workflow = KBQueryWorkflow()
        for chunk in workflow.run(init_state, stream=is_stream):
            logger.debug(chunk)
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        print(f"流程执行异常: {e}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})



@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):

    print("调用流式/stream...")
    """
    sse 实时返回结果
    """
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """
    清空指定会话的历史记录
    """
    count = clear_history(session_id)
    return {"message": "历史会话已清空", "deleted_count": count}

@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """
    查询当前会话历史记录
    """
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append({
                "_id": str(r.get("_id")) if r.get("_id") is not None else "",
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts")
            })
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history error: {e}")


# 证明服务器启动即可
@app.get("/health")
async def health():
    """
    检查服务是否正常
    """
    return {"ok": True}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)