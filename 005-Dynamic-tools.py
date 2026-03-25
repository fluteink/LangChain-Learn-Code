import os
from dataclasses import dataclass
from typing import Callable
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore

"""
动态工具是指在运行时调整代理可用的工具集，而不是预先定义所有工具。并非每个工具都适用于所有情况。工具过多可能会使模型过载（上下文过载）并增加错误；工具过少则会限制功能。动态工具选择可以根据身份验证状态、用户权限、功能标志或对话阶段来调整可用工具集。
根据工具是否事先已知，有两种处理方法：
(1) 筛选预注册工具：如果在创建代理时已知所有可能的工具，则可以预先注册它们，并根据状态、权限或上下文动态筛选哪些工具可以公开给模型。
(2) 运行时工具注册：当在运行时发现或创建工具时（例如，从 MCP 服务器加载、根据用户数据生成或从远程注册表中获取），您需要注册这些工具并动态处理它们的执行。
这需要两个中间件钩子：wrap_model_call- 将动态工具添加到请求中;wrap_tool_call- 处理动态添加工具的执行
这种方案在以下情况中效果最佳：
所有可能的工具在编译 / 启动阶段就已明确
需要根据权限、功能标记或对话状态对工具进行过滤
工具本身是静态的，但可用性是动态变化的
"""

####基本配置-BEGIN####
load_dotenv()
api_key_DeepSeek = os.getenv("DEEPSEEK_API_KEY")
base_url_DeepSeek = os.getenv("DEEPSEEK_BASE_URL")
api_key_GLM = os.getenv("GLM_API_KEY")
base_url_GLM = os.getenv("GLM_BASE_URL")
GLM_model = ChatOpenAI(
    model="glm-4.7-flash",
    api_key=api_key_GLM,
    base_url=base_url_GLM,
)

model=ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key_DeepSeek,
    base_url=base_url_DeepSeek,
)
####基本配置-END####


"""
筛选预注册工具->按状态过滤：仅在对话达到特定里程碑 (达到特定条件) 后，才启用高级工具。
"""

#工具定义
@tool
def public_search(query: str) -> str:
    """公开搜索工具，任何人都可以使用。"""
    return f"公开搜索结果：{query}很好，阳光明媚"


@tool
def private_search(query: str) -> str:
    """私有搜索工具，仅认证后可用。"""
    return f"私有搜索结果：{query}订单编号：DD2026032008956731，下单时间：2026-03-20 15:42:18，商品：ComfyUI 全套模型整合包，实付¥199.00，收货人：李先生，电话：138****6728，地址：福建省厦门市集美区杏林街道 XX 科技园 B 栋 302 室，状态：待发货"


@tool
def advanced_search(query: str) -> str:
    """高级搜索工具，需要更多上下文后使用。"""
    return f"高级搜索结果：{query}已调用高级工具执行多条件组合检索，筛选条件：下单时间在 2026 年 3 月、支付方式为微信支付、商品含 ComfyUI 相关套件，成功匹配对应订单明细并完成结果汇总。"
#工具定义 END


@wrap_model_call
def state_based_tools(
    request: ModelRequest,  # 模型请求对象，包含当前对话状态、可用工具等信息
    handler: Callable[[ModelRequest], ModelResponse]  # 下游处理函数，接收请求并返回模型响应
) -> ModelResponse:  # 返回模型响应对象
    """根据对话状态过滤工具。"""
    # 优先读取 runtime context（LangChain v1 推荐）
    runtime_ctx = getattr(getattr(request, "runtime", None), "context", {}) or {}
    state = request.state or {}

    # 兼容旧写法：context > state.configurable > state
    configurable = state.get("configurable", {})
    is_authenticated = runtime_ctx.get(
        "authenticated",
        configurable.get("authenticated", state.get("authenticated", False))
    )
    message_count = runtime_ctx.get(
        "message_count",
        configurable.get("message_count", len(state.get("messages", [])))
    )

    if not is_authenticated:
        # 未认证时，只保留名称以 public_ 开头的公开工具
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        # 已认证但上下文不足，禁用高级工具
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)
    else:
        # 已认证且消息数足够，所有工具都可用
        tools = request.tools
        request = request.override(tools=tools)

    return handler(request)

def print_agent_result(title, result):
    """打印工具调用情况和最终结果。"""
    print(f"\n{title}")
    messages = result.get("messages", [])

    # 逐条打印消息，包含工具调用和最终回复
    for msg in messages:
        msg_type = type(msg).__name__
        print(f"- {msg_type}: {getattr(msg, 'content', '')}")

        # 打印模型发起的工具调用
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            print(f"  工具调用：{tool_calls}")

        # 打印工具执行结果
        if msg_type == "ToolMessage":
            print(f"  工具返回：{getattr(msg, 'name', '')} -> {getattr(msg, 'content', '')}")

    # 打印最后一条 AI 回复作为最终结果
    if messages:
        last_msg = messages[-1]
        print(f"最终结果：{getattr(last_msg, 'content', '')}")

agent = create_agent(
    model=model,
    tools=[public_search, private_search, advanced_search],
    middleware=[state_based_tools],
    system_prompt=(
        "你是一个会调用工具的助手。\n"
        "可用工具由中间件按上下文动态过滤。\n"
        "重要规则：\n"
        "1. 当用户请求中提到工具名称（如 public_search、private_search、advanced_search）或工具功能（如公开搜索、私有搜索、高级搜索、多条件组合检索）时，你必须先调用对应工具再回答。\n"
        "2. 不要在工具可用时仅做口头说明，必须实际调用工具。\n"
        "3. 调用工具后，根据工具返回的结果给出结论。\n"
        "4. 如果用户明确请求使用某个工具，直接调用该工具，不要询问更多信息。"
    )
)
# 演示 1：未认证，只允许 public_search
response1 = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "请使用公开工具查询：北京天气"}
        ]
    },
    context={"authenticated": False, "message_count": 1}
)
print_agent_result("=============public_search 演示：=============\n", response1)

# 演示 2：已认证且消息数=5，允许 private_search（advanced 也不再被禁用）
response2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "请使用 private_search 工具查询订单信息。"
            }
        ]
    },
    context={"authenticated": True, "message_count": 5}
)
print_agent_result("=============private_search 演示：=============\n", response2)

# 演示 3：已认证且消息数足够，允许 advanced_search
response3 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "请使用 advanced_search 工具搜索：ComfyUI 多条件组合检索。"
            }
        ]
    },
    context={"authenticated": True, "message_count": 6}
)
print_agent_result("=============advanced_search 演示：=============\n", response3)


"""
筛选预注册工具->按存储过滤：基于存储（Store）中的用户偏好或功能标记，对工具进行过滤
"""

@dataclass
class Context:
    user_id: str

@wrap_model_call
def store_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据存储偏好过滤工具。"""
    user_id = request.runtime.context.user_id

    # 从 Store 中读取：获取用户已启用的功能
    store = request.runtime.store
    feature_flags = store.get(("features",), user_id)

    if feature_flags:
        enabled_features = feature_flags.value.get("enabled_tools", [])
        # 仅包含该用户已启用的工具
        tools = [t for t in request.tools if t.name in enabled_features]
        request = request.override(tools=tools)

    return handler(request)

@tool
def search_tool(query: str) -> str:
    """第1步：执行检索，返回可直接传给 analysis_tool 的 JSON 字符串。"""
    import time, uuid, json
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    results = [
        {
            "id": str(uuid.uuid4()),
            "title": f"{query} - 示例结果 1",
            "snippet": f"这是有关 '{query}' 的摘要信息，用于展示搜索返回片段。",
            "score": 0.94,
            "url": f"https://example.com/search/{uuid.uuid4().hex[:8]}",
            "timestamp": now,
        },
        {
            "id": str(uuid.uuid4()),
            "title": f"{query} - 示例结果 2",
            "snippet": "第二条示例结果，包含更多上下文信息与推荐理由。",
            "score": 0.81,
            "url": f"https://example.com/search/{uuid.uuid4().hex[:8]}",
            "timestamp": now,
        },
    ]
    payload = {
        "tool": "search_tool",
        "step": "search",
        "meta": {"query": query, "total_results": len(results), "queried_at": now},
        "results": results,
    }
    return json.dumps(payload, ensure_ascii=False)

@tool
def analysis_tool(text: str) -> str:
    """第2步：分析文本，优先解析 search_tool 输出，返回可直接传给 export_tool 的 JSON 字符串。"""
    import json, re, time, uuid
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    query = ""
    source_meta = {}
    source_results = []

    # 兼容直接接收 search_tool 的 JSON 输出
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            source_meta = parsed.get("meta", {}) or {}
            query = source_meta.get("query", "")
            source_results = parsed.get("results", []) or []
            if source_results:
                text = "\n".join(
                    f"{item.get('title', '')} {item.get('snippet', '')}" for item in source_results
                )
    except Exception:
        # 非 JSON 输入时回退到原始文本分析
        pass

    # 简单关键词提取示例（非真实 NLP）
    words = re.findall(r"\w+", text.lower())
    freq = {}
    for w in words:
        if len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
    top_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
    sentiment = "neutral"
    if any(x in text.lower() for x in ("好", "优秀", "满意", "喜欢")):
        sentiment = "positive"
    if any(x in text.lower() for x in ("差", "不好", "失望", "糟糕")):
        sentiment = "negative"
    analysis_id = str(uuid.uuid4())
    summary = text if len(text) <= 120 else text[:117] + "..."
    payload = {
        "tool": "analysis_tool",
        "step": "analysis",
        "analysis_id": analysis_id,
        "query": query,
        "summary": summary,
        "sentiment": sentiment,
        "top_keywords": [k for k, _ in top_keywords],
        "keyword_counts": {k: v for k, v in top_keywords},
        "source_meta": source_meta,
        "source_result_count": len(source_results),
        "analyzed_at": now,
        "original_length": len(text),
    }
    return json.dumps(payload, ensure_ascii=False)

@tool
def export_tool(data: str, fmt: str = "json") -> str:
    """第3步：导出结果，接收 analysis_tool 输出并返回导出元信息 JSON。"""
    import time, uuid, json, base64
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    fmt = (fmt or "json").lower()
    if fmt not in {"json", "csv"}:
        fmt = "json"

    export_id = str(uuid.uuid4())
    # 简单估算大小并生成伪下载链接
    raw_bytes = data.encode("utf-8")
    size_bytes = len(raw_bytes)
    download_token = base64.urlsafe_b64encode(export_id.encode()).decode().strip("=")
    file_name = f"{export_id}.{fmt}"
    download_url = f"https://storage.example.com/exports/{file_name}?token={download_token}"
    meta = {
        "tool": "export_tool",
        "step": "export",
        "export_id": export_id,
        "format": fmt,
        "file_name": file_name,
        "size_bytes": size_bytes,
        "download_url": download_url,
        "input_preview": data[:120],
        "exported_at": now,
    }
    return json.dumps(meta, ensure_ascii=False)

agent_store = create_agent(
    model=model,
    tools=[search_tool, analysis_tool, export_tool],
    middleware=[store_based_tools],
    context_schema=Context,
    store=InMemoryStore()
)

# 新增：示例调用（简单演示）
response_store = agent_store.invoke(
    {
        "messages": [
            {"role": "user", "content": "请使用 search_tool 检索并用 analysis_tool 分析，然后用 export_tool 导出结果：" "示例查询"}
        ]
    },
    context={"user_id": "user_123"}
)
print_agent_result("=============store_based_tools 演示：=============\n", response_store)

# 1) 给 user_123 配置只允许 search + analysis，不允许 export
agent_store.store.put(
    ("features",),
    "user_123",
    {"enabled_tools": ["search_tool", "analysis_tool"]}
)

# 2) 再发同样请求
response_store = agent_store.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "请使用search_tool检索并用analysis_tool分析，然后用export_tool导出结果：示例查询"
            }
        ]
    },
    context={"user_id": "user_123"}
)
print_agent_result("=============store_based_tools（限制 export_tool）演示：=============\n", response_store)


"""
筛选预注册工具->运行时上下文，基于运行时上下文用户权限的过滤工具
"""
@dataclass
class Context:
    user_role: str

@wrap_model_call
def context_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据运行时上下文权限过滤工具。"""
    # 从运行时上下文读取：获取用户角色
    if request.runtime is None or request.runtime.context is None:
        # 如果未提供上下文，默认按 viewer（最严格权限）处理
        user_role = "viewer"
    else:
        user_role = request.runtime.context.user_role

    if user_role == "admin":
        # 管理员可使用全部工具
        pass
    elif user_role == "editor":
        # 编辑者不能使用删除工具
        tools = [t for t in request.tools if t.name != "delete_data"]
        request = request.override(tools=tools)
    else:
        # viewer 仅可使用只读工具
        tools = [t for t in request.tools if t.name.startswith("read_")]
        request = request.override(tools=tools)

    return handler(request)


@tool
def read_data(query: str) -> str:
    """只读查询工具（viewer/editor/admin 都可用）。"""
    return f"【模拟只读数据】你查询的是：{query}。示例结果：今日访问量 1234，新增用户 56，转化率 7.8%。"

@tool
def write_data(content: str) -> str:
    """写入工具（editor/admin 可用）。"""
    return f"【模拟写入成功】已接收内容：{content}。示例记录号：REC-20260325-0001。"

@tool
def delete_data(target: str) -> str:
    """删除工具（仅 admin 可用）。"""
    return f"【模拟删除成功】已删除目标：{target}。示例回收站编号：DEL-20260325-0099。"


agent = create_agent(
    model=model,
    tools=[read_data, write_data, delete_data],
    middleware=[context_based_tools],
    context_schema=Context
)


# ============ context_based_tools 演示 ============
# viewer：只能 read_*
resp_viewer = agent.invoke(
    {
        "messages": [
            {
                "role": "system",
                "content": "你必须调用且只能调用一次工具 read_data，然后基于工具结果作答；不要直接回答。"
            },
            {"role": "user", "content": "请调用 read_data 查询：今日报表"}
        ]
    },
    context=Context(user_role="viewer")
)
print_agent_result("=============context_based_tools viewer 演示：=============\n", resp_viewer)

# editor：可 read/write，不可 delete
resp_editor = agent.invoke(
    {
        "messages": [
            {
                "role": "system",
                "content": "你必须调用且只能调用一次工具 write_data，然后基于工具结果作答；不要直接回答。"
            },
            {"role": "user", "content": "请调用 write_data 写入：新增一条测试记录"}
        ]
    },
    context=Context(user_role="editor")
)
print_agent_result("=============context_based_tools editor 演示：=============\n", resp_editor)

# admin：全部可用（含 delete）
resp_admin = agent.invoke(
    {
        "messages": [
            {
                "role": "system",
                "content": "你必须调用且只能调用一次工具 delete_data，然后基于工具结果作答；不要直接回答。"
            },
            {"role": "user", "content": "请调用 delete_data 删除：测试记录-001"}
        ]
    },
    context=Context(user_role="admin")
)
print_agent_result("=============context_based_tools admin 演示：=============\n", resp_admin)

# 无 context：走默认 viewer（最严格）
resp_default = agent.invoke(
    {
        "messages": [
            {
                "role": "system",
                "content": "你必须调用且只能调用一次工具 read_data，然后基于工具结果作答；不要直接回答。"
            },
            {"role": "user", "content": "请调用 read_data 查询：默认权限测试"}
        ]
    }
)
print_agent_result("=============context_based_tools 默认权限演示：=============\n", resp_default)

# viewer 尝试 write_data：应被权限过滤（仅允许 read_*）
resp_viewer_write = agent.invoke(
    {
        "messages": [
            {
                "role": "system",
                "content": "优先调用 write_data；如果该工具不可用，直接说明权限不足且不要改用其他工具。"
            },
            {"role": "user", "content": "请调用 write_data 写入：viewer 权限测试记录"}
        ]
    },
    context=Context(user_role="viewer")
)
print_agent_result("=============context_based_tools viewer 调用 write_data（应被拒绝）演示：=============\n", resp_viewer_write)



