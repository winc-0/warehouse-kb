
ITEM_NAME_EXTRACT_SYSTEM_PROMPT = ("你是一个专业的客服助手，"
                                   "擅长理解用户意图和提前关键信息。")

ITEM_NAME_EXTRACT_TEMPLATE = """
历史会话：
{history_text}

当前用户问题：{query}

请根据历史会话和当前问题，提取用户正在询问的商品名称（item_names）。
1. 如果用户明确提到了商品名称，请提取出来。可能有一个或多个，但不能重复。
2. 如果用户使用了代词（如"这个"、"它"），请结合历史会话指代消解，确定商品名称。
3. 如果无法确定商品名称，item_names 返回空列表。
4. 请重新改写用户的问题（rewritten_query），使其成为包含商品名称的独立完整问题。

【输出要求】
仅输出严格的合法 JSON 字符串，不得包含 Markdown 代码块标记（如 ```json），不得包含任何前导或后置的说明性文字。
格式如下：
{{
    "item_names": ["商品A", "商品B"],
    "rewritten_query": "关于商品A和商品B，..."
}}
"""