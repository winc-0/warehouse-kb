# processor/query_processor/nodes/node_web_search_mcp.py
import json

import httpx

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.task_utils import add_done_task


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    # 百炼联网搜索 MCP 暴露的工具名
    tool_name: str = "bailian_web_search"

    # MCP 协议版本：使用服务端已支持的 initialize 握手版本
    protocol_version: str = "2024-11-05"

    # 单次联网搜索的读写超时（秒）
    timeout: float = 10.0

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        query = state.get("rewritten_query")
        web_docs = []

        try:
            # 1. 安全调用 MCP 接口，直接拿到解析后的网页列表
            web_docs = self._search(query)

        except Exception as e:
            # 捕获所有异常（包括 HTTP 500、连接超时、JSON解析失败等），打印警告日志，进行服务降级
            logger.warning(f"[{self.name}] MCP 联网搜索服务调用异常，跳过联网检索，错误信息: {e}")

        finally:
            # 无论成功还是异常，都标记任务完成，避免卡主流程
            add_done_task(state.get("session_id"), self.name, state.get("is_stream"))

        # 统一返回状态，确保后续节点（如 RRF / Fusion 节点）拿到明确的列表
        return {"web_search_docs": web_docs}


    def _search(self, query: str, count: int = 5) -> list:
        """
        调用百炼联网搜索 MCP 服务，返回网页文档列表。

        说明：这里不使用 openai-agents 的 MCPServerStreamableHttp，原因是新版
        mcp SDK（2.x）握手会发送 2026 版协议的 server/discover 方法，而百炼
        MCP 服务端尚未支持该方法，会直接返回 HTTP 500，导致连接阶段就失败。
        本方法用标准 MCP  initialize 握手 + tools/call 两步完成调用，行为更稳定。

        :param query: 检索关键词
        :param count: 期望返回的结果条数
        :return: [{"title":..., "url":..., "snippet":...}, ...]，失败时为空列表
        """
        if not query:
            return []

        headers = {
            "Authorization": f"Bearer {mcp_config.api_key}",
            # 服务端可能返回 JSON，也可能返回 SSE 事件流，两种都要能接收
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout) as client:
            # 1. initialize 握手，确认服务端可用
            initialize_body = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {},
                    "clientInfo": {"name": self.name, "version": "1.0.0"},
                },
            }
            resp = client.post(mcp_config.mcp_base_url, headers=headers, json=initialize_body)
            resp.raise_for_status()

            # 部分服务端要求携带会话标识，存在则透传
            session_id = resp.headers.get("mcp-session-id")
            if session_id:
                headers["Mcp-Session-Id"] = session_id

            # 2. 握手完成通知（失败不影响后续调用）
            try:
                client.post(
                    mcp_config.mcp_base_url,
                    headers=headers,
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                )
            except Exception as e:
                logger.warning(f"[{self.name}] initialized 通知发送失败，忽略: {e}")

            # 3. 调用联网搜索工具
            call_body = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": self.tool_name,
                    "arguments": {"query": query, "count": count},
                },
            }
            resp = client.post(mcp_config.mcp_base_url, headers=headers, json=call_body)
            resp.raise_for_status()

            payload = self._parse_rpc_message(resp.text)

        # 4. 解析 JSON-RPC 结果并做空值保护
        return self._extract_docs(payload)


    @staticmethod
    def _parse_rpc_message(raw_text: str) -> dict:
        """
        把响应文本统一解析成 JSON-RPC 对象，兼容纯 JSON 与 SSE（text/event-stream）两种返回。

        :param raw_text: HTTP 响应正文
        :return: JSON-RPC 对象；无法解析时返回空字典
        """
        if not raw_text:
            return {}

        # SSE 场景：真正的 JSON-RPC 载荷位于某个 data: 行内
        for line in raw_text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data or data == "[DONE]":
                continue
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                continue

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning(f"[node_web_search_mcp] 响应内容无法解析为 JSON，前 200 字符: {raw_text[:200]}")
            return {}

    @classmethod
    def _extract_docs(cls, payload: dict) -> list:
        """
        从 JSON-RPC 响应中提取网页文档列表，对服务端错误与空值都做空保护。

        :param payload: 解析后的 JSON-RPC 对象
        :return: [{"title":..., "url":..., "snippet":...}, ...]
        """
        if not isinstance(payload, dict):
            return []

        # 服务端返回 JSON-RPC 错误（如工具名不匹配、鉴权失效）
        error = payload.get("error")
        if error:
            logger.warning(f"[{cls.name}] MCP 返回错误: {error}")
            return []

        content = (payload.get("result") or {}).get("content") or []
        json_str = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                json_str = item.get("text") or ""
                break

        if not json_str:
            return []

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"[{cls.name}] 工具返回内容不是合法 JSON，前 200 字符: {json_str[:200]}")
            return []

        pages = data.get("pages", []) if isinstance(data, dict) else []

        web_docs = []
        for item in pages:
            if not isinstance(item, dict):
                continue
            web_docs.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            })
        return web_docs


if __name__ == "__main__":

  init_state = {
      "rewritten_query": "关于合力 CPD30 叉车，额定载荷是多少？",
      "session_id": "local_test_session",
  }

  # 执行节点的业务调用
  node_web_search_mcp = NodeWebSearchMcp()
  result = node_web_search_mcp(init_state)
  logger.info(serialize_json(result, indent=4))