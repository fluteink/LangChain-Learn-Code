"""# 工具调用（Tool Calling）
模型可以请求调用工具，以完成从数据库获取数据、联网搜索、运行代码等任务。工具由两部分配对组成：
1. 一套规范定义：包含工具名称、功能描述、参数定义（通常为 JSON 规范）；
2. 负责实际执行的普通函数或异步协程。

你可能听过**函数调用**这一术语，在本文语境中，它与**工具调用**为同义概念，可互换使用。
"""

"""
若要让你自定义的工具可供模型调用，必须通过 bind_tools 完成绑定。后续发起推理调用时，模型便可按需选择调用任意已绑定的工具。
部分模型服务商提供内置工具，可通过模型参数或调用参数直接启用（例如 ChatOpenAI、ChatAnthropic）。具体细节请查阅对应服务商的官方文档。
更多工具创建的详细说明与其他实现方案，参见工具使用指南。
"""

####基本配置-BEGIN####
import os

import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

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

model = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key_DeepSeek,
    base_url=base_url_DeepSeek,
)

ChatGPT_model = ChatOpenAI(
    model="gpt-5.4",
    api_key=os.getenv("ChatGPT_API_KEY"),
    base_url=os.getenv("ChatGPT_BASE_URL"),
)
def print_model_result(result, title="Model Result"):
    """美观打印模型返回结果（content、metadata、token 使用等）。"""
    print("\n" + "═" * 90)
    print(f" {title}")
    print("═" * 90)

    print("【内容】")
    print(getattr(result, "content", ""))

    print("\n【基础信息】")
    print(f"id: {getattr(result, 'id', '')}")
    print(f"tool_calls: {getattr(result, 'tool_calls', [])}")
    print(f"invalid_tool_calls: {getattr(result, 'invalid_tool_calls', [])}")

    additional_kwargs = getattr(result, "additional_kwargs", {}) or {}
    response_metadata = getattr(result, "response_metadata", {}) or {}
    usage_metadata = getattr(result, "usage_metadata", {}) or {}

    print("\n【additional_kwargs】")
    print(json.dumps(additional_kwargs, ensure_ascii=False, indent=2))

    print("\n【response_metadata】")
    print(json.dumps(response_metadata, ensure_ascii=False, indent=2))

    print("\n【usage_metadata】")
    print(json.dumps(usage_metadata, ensure_ascii=False, indent=2))

    # 额外提取常用 token 字段（如果存在）
    token_usage = response_metadata.get("token_usage", {})
    if token_usage:
        print("\n【Token 使用摘要】")
        print(
            f"prompt={token_usage.get('prompt_tokens', 0)}, "
            f"completion={token_usage.get('completion_tokens', 0)}, "
            f"total={token_usage.get('total_tokens', 0)}"
        )

    print("═" * 90)
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
            print(f"  工具调用：\n" + "*" * 50 + "\n" + f"{tool_calls}" + "\n" + "*" * 50)

        # 打印工具执行结果
        if msg_type == "ToolMessage":
            print(
                f"  工具返回：\n"
                + "*" * 50
                + "\n"
                + f"{getattr(msg, 'name', '')} -> "
                + f"{getattr(msg, 'content', '')}"
                + "\n"
                + "*" * 50
            )

    # 打印最后一条 AI 回复作为最终结果
    if messages:
        last_msg = messages[-1]
        print(f"最终结果：\n" + "*" * 50 + "\n" + f"{getattr(last_msg, 'content', '')}" + "\n" + "*" * 50)
####基本配置-END####
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """获取某个位置的天气。"""
    return f"{location} 天气晴朗。"

model_with_tools = model.bind_tools([get_weather])  # [!code highlight]
response = model_with_tools.invoke("波士顿的天气怎么样？")

for tool_call in response.tool_calls:  # 查看模型进行的工具调用
    print(f"工具：{tool_call['name']}")
    print(f"参数：{tool_call['args']}")
print(f"最终模型输出：{getattr(response, 'content', '')}")
print_model_result(response, title="最终模型输出")

"""

绑定用户自定义工具后，模型返回的结果中会包含执行工具的调用请求。
如果**脱离智能体单独调用模型**，需要你手动执行被请求的工具，并把运行结果返回给模型，供其进行后续推理思考；
如果搭配**智能体（Agent）** 使用，则由智能体循环自动完成整套工具调用与结果回传流程。

下文将介绍工具调用的常见使用方式。
"""

"""
Tool execution loop（工具执行循环）:
当模型返回工具调用请求时，你需要执行对应工具，并将结果回传给模型。由此形成对话循环，模型可借助工具返回的结果生成最终回答。
LangChain 提供了智能体抽象层，能够自动调度、统筹这一整套流程。
以下是实现该流程的简易示例：
"""
# 将（可能多个）工具绑定到模型
model_with_tools = model.bind_tools([get_weather])

# 步骤 1：模型生成工具调用
messages = [{"role": "user", "content": "波士顿的天气怎么样？"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# 步骤 2：执行工具并收集结果（打印中间过程）
print("\n步骤2：开始执行工具并收集结果")
for i, tool_call in enumerate(ai_msg.tool_calls, start=1):
    print(f"[{i}] 工具调用 -> name={tool_call.get('name')} args={tool_call.get('args')}")
    tool_result = get_weather.invoke(tool_call)
    print(
        f"[{i}] 工具返回 -> name={getattr(tool_result, 'name', '')} "
        f"content={getattr(tool_result, 'content', '')}"
    )
    messages.append(tool_result)
    print(f"[{i}] 当前 messages 数量: {len(messages)}")
print("步骤2：工具执行完成\n")
# 步骤 3：将结果传递回模型以获取最终响应
final_response = model_with_tools.invoke(messages)
print(final_response.text)
# "波士顿当前的天气是 72°F，晴朗。"



"""
Forcing tool calls（强制工具调用）:
默认情况下，模型会根据用户输入**自主选择**已绑定的工具调用。  
但你有时需要**强制指定工具**，限制模型只能使用某个特定工具，或从指定工具列表里选择调用：
"""

"""
Force use of any tool（强制使用任意工具）:
"""


@tool
def get_city_time(location: str) -> str:
    """获取某个位置的时间。"""
    return f"{location} 当前时间为 10:00。"


# 示例1：强制使用任意工具（必须调用已绑定工具中的一个）
model_with_tools_any = model.bind_tools([get_weather, get_city_time], tool_choice="any")

"""
Force use of specific tools（强制使用特定工具）:
"""
# 示例2：强制使用指定工具（必须调用 get_weather）
model_with_tools_specific = model.bind_tools([get_weather, get_city_time], tool_choice="get_weather")

from langchain_core.messages import ToolMessage

tool_map = {
    "get_weather": get_weather,
    "get_city_time": get_city_time,
}

# 不强制工具选择，用于拿最终自然语言答案，避免再次被强制走工具
final_model_with_tools = model.bind_tools([get_weather, get_city_time])

# ===== Case 1: Force use of any tool（强制使用任意工具）=====
messages_any = [{"role": "user", "content": "我在波士顿，请任选一个工具来回答。"}]
ai_msg_any = model_with_tools_any.invoke(messages_any)
messages_any.append(ai_msg_any)

print_model_result(ai_msg_any, title="Force use of any tool（强制使用任意工具） - 首轮模型输出")

print("\n步骤2：开始执行工具并收集结果")
for i, tool_call in enumerate(getattr(ai_msg_any, "tool_calls", []), start=1):
    name = tool_call.get("name")
    args = tool_call.get("args", {})
    print(f"[{i}] 工具调用 -> name={name} args={args}")

    tool_fn = tool_map.get(name)
    if tool_fn is None:
        print(f"[{i}] 未找到工具实现，跳过")
        continue

    tool_output = tool_fn.invoke(args)
    tool_msg = ToolMessage(
        content=str(tool_output),
        name=name,
        tool_call_id=tool_call.get("id", ""),
    )
    print(f"[{i}] 工具返回 -> name={name} content={tool_msg.content}")
    messages_any.append(tool_msg)
    print(f"[{i}] 当前 messages 数量: {len(messages_any)}")
print("步骤2：工具执行完成\n")

final_response_any = final_model_with_tools.invoke(messages_any)
print_model_result(final_response_any, title="Force use of any tool（强制使用任意工具） - 最终模型输出")

# ===== Case 2: Force use of specific tools（强制使用特定工具）=====
messages_specific = [{"role": "user", "content": "我在波士顿，请使用天气工具回答。"}]
ai_msg_specific = model_with_tools_specific.invoke(messages_specific)
messages_specific.append(ai_msg_specific)

print_model_result(ai_msg_specific, title="Force use of specific tools（强制使用特定工具） - 首轮模型输出")

print("\n步骤2：开始执行工具并收集结果")
for i, tool_call in enumerate(getattr(ai_msg_specific, "tool_calls", []), start=1):
    name = tool_call.get("name")
    args = tool_call.get("args", {})
    print(f"[{i}] 工具调用 -> name={name} args={args}")

    tool_fn = tool_map.get(name)
    if tool_fn is None:
        print(f"[{i}] 未找到工具实现，跳过")
        continue

    tool_output = tool_fn.invoke(args)
    tool_msg = ToolMessage(
        content=str(tool_output),
        name=name,
        tool_call_id=tool_call.get("id", ""),
    )
    print(f"[{i}] 工具返回 -> name={name} content={tool_msg.content}")
    messages_specific.append(tool_msg)
    print(f"[{i}] 当前 messages 数量: {len(messages_specific)}")
print("步骤2：工具执行完成\n")

final_response_specific = final_model_with_tools.invoke(messages_specific)
print_model_result(final_response_specific, title="Force use of specific tools（强制使用特定工具） - 最终模型输出")


"""
Parallel tool calls（并行工具调用）:
许多模型在合适场景下支持**并行调用多个工具**，能够同时从不同来源收集所需信息。
模型会根据所请求操作的相互独立性，智能判断何时适合执行并行调用。大多数支持工具调用的模型默认启用并行工具调用。部分模型（包括 OpenAI 和 Anthropic）
允许你禁用该功能。如需禁用，请设置：parallel_tool_calls=False
model.bind_tools([get_weather], parallel_tool_calls=False)
术语解释
parallel tool calls：并行工具调用（一次请求里同时调用多个工具）
bind_tools：绑定工具（LangChain 核心方法）
"""

# 重写重写 get_weather 工具，添加多语言支持
@tool
def get_weather(location: str) -> str:
    """获取某个位置的天气（示例静态数据）。"""
    weather_db = {
        "波士顿": {"condition": "晴朗", "temp_c": 22},
        "boston": {"condition": "Sunny", "temp_c": 22},
        "东京": {"condition": "多云", "temp_c": 27},
        "tokyo": {"condition": "Cloudy", "temp_c": 27},
    }

    key = (location or "").strip()
    data = weather_db.get(key) or weather_db.get(key.lower())
    if not data:
        return f"{location} 天气信息暂不可用。"

    return f"{location} 当前天气：{data['condition']}，{data['temp_c']}°C。"

model_with_tools = model.bind_tools([get_weather])

# 步骤 1：模型生成（可能多个）工具调用
messages = [{"role": "user", "content": "波士顿和东京的天气怎么样？"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

print_model_result(ai_msg, title="Parallel tool calls（并行工具调用）- 首轮模型输出")

# 步骤 2：逐个执行工具调用并回填 ToolMessage（可改为异步并行）
results = []
for i, tool_call in enumerate(getattr(ai_msg, "tool_calls", []), start=1):
    name = tool_call.get("name")
    args = tool_call.get("args", {})
    print(f"[{i}] 工具调用 -> name={name} args={args}")

    if name != "get_weather":
        print(f"[{i}] 非目标工具，跳过")
        continue

    tool_output = get_weather.invoke(args)
    tool_msg = ToolMessage(
        content=str(tool_output),
        name=name,
        tool_call_id=tool_call.get("id", ""),
    )
    messages.append(tool_msg)
    results.append(tool_msg.content)
    print(f"[{i}] 工具返回 -> {tool_msg.content}")

# 步骤 3：将工具结果传回模型，获取最终自然语言答案
final_response = model_with_tools.invoke(messages)
print_model_result(final_response, title="Parallel tool calls（并行工具调用）- 最终模型输出")


"""
Streaming tool calls（流式工具调用）:
在流式响应（Streaming）模式下，工具调用信息会通过 ToolCallChunk 逐步构建。这让你能够实时查看正在生成中的工具调用过程，而无需等待完整响应生成完毕。
"""



for chunk in model_with_tools.stream("波士顿和东京的天气怎么样？"):
    # 工具调用块渐进式到达
    for tool_chunk in getattr(chunk, "tool_call_chunks", []):
        if name := tool_chunk.get("name"):
            print(f"工具：{name}")
        if id_ := tool_chunk.get("id"):
            print(f"ID: {id_}")
        if args := tool_chunk.get("args"):
            print(f"参数：{args}")

# 输出示例（参数可能被拆分为多个片段）：
# 工具：get_weather
# ID: call_xxx
# 参数：{"lo
# 参数：catio
# 参数：n":"波
# 参数：士顿"}

# 你可以累积块以构建完整的工具调用
gathered = None
for chunk in model_with_tools.stream("波士顿的天气怎么样？"):
    gathered = chunk if gathered is None else gathered + chunk

if gathered is not None:
    print(gathered.tool_calls)