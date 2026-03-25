import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest
from langchain.tools import tool
from langchain_openai import ChatOpenAI

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

model = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key_DeepSeek,
    base_url=base_url_DeepSeek,
)

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
            print(f"  工具调用：\n"+"*"*50+"\n"+f"{tool_calls}"+"\n"+"*"*50)

        # 打印工具执行结果
        if msg_type == "ToolMessage":
            print(f"  工具返回：\n"+"*"*50+"\n"+f"{getattr(msg, 'name', '')} -> "+f"{getattr(msg, 'content', '')}"+"\n"+"*"*50)

    # 打印最后一条 AI 回复作为最终结果
    if messages:
        last_msg = messages[-1]
        print(f"最终结果：\n"+"*"*50+"\n"+f"{getattr(last_msg, 'content', '')}"+"\n"+"*"*50)
####基本配置-END####


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    mock_weather = {
        "北京": "晴，12 到 20°C",
        "上海": "多云，15 到 22°C",
        "广州": "小雨，19 到 26°C",
        "深圳": "阵雨，20 到 27°C",
    }
    weather = mock_weather.get(city, "天气数据暂不可用")
    return f"{city}：{weather}"

# 将在运行时动态添加的工具
@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> str:
    """计算账单的小费金额。"""
    tip = bill_amount * (tip_percentage / 100)
    return f"Tip: ${tip:.2f}, Total: ${bill_amount + tip:.2f}"

class DynamicToolMiddleware(AgentMiddleware):
    """注册和处理动态工具的中间件。"""

    def wrap_model_call(self, request: ModelRequest, handler):
        # `wrap_model_call` 在模型真正被调用前执行。
        # 意义：可以在这里动态修改本次请求的上下文（例如补充工具、调整参数等）。

        # `request.tools` 是当前请求已可用的工具列表（这里包含静态注册的 `get_weather`）。
        # 通过 `[*request.tools, calculate_tip]` 复制原工具并追加 `calculate_tip`。
        # 意义：让模型在“本轮推理时”感知到额外工具，而不必在 `create_agent` 时静态写死。
        updated = request.override(tools=[*request.tools, calculate_tip])

        # 将修改后的请求继续交给后续处理链（最终会到模型）。
        # 意义：中间件本身不执行模型，只负责“改请求再转发”。
        return handler(updated)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        # `wrap_tool_call` 在工具真正执行前触发。
        # 意义：可对工具调用做路由、鉴权、限流、替换实现等控制。

        # 当模型决定调用名为 `calculate_tip` 的工具时：
        if request.tool_call["name"] == "calculate_tip":
            # 将这次调用绑定到实际的 `calculate_tip` 函数对象后再继续处理。
            # 意义：确保“动态注入的工具名”能被正确解析并执行对应实现。
            return handler(request.override(tool=calculate_tip))

        # 其它工具（如静态工具 `get_weather`）保持默认流程，不做额外干预。
        return handler(request)
agent = create_agent(
    model=model,
    tools=[get_weather],  # 此处仅注册静态工具
    middleware=[DynamicToolMiddleware()],
)


# 智能体现在可以同时使用get_weather和calculate_tip
result = agent.invoke({
    "messages": [{"role": "user", "content": "请帮我计算 85 美元账单在 20\% 小费下的小费和总金额"}]
})
print_agent_result("="*50+"\n智能体结果\n"+"="*50, result)

#智能体使用get_weather工具获取天气信息
result = agent.invoke({
    "messages": [{"role": "user", "content": "请告诉我北京的天气"}]
})
print_agent_result("="*50+"\n智能体结果\n"+"="*50, result)