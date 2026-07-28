
"""
智能体通过消息状态自动维护对话历史。您还可以配置智能体使用自定义状态模式在对话期间记住额外信息。
代理通过消息状态自动维护对话历史。你还可以配置代理使用自定义状态模式，以便在对话中记住额外信息。
存储在状态中的信息可以理解为短期记忆代理人的：
自定义状态模式必须扩展AgentState作为一个 .TypedDict
定义自定义状态有两种方式：
Via中间件（优先）
Viastate_schema关于create_agent
"""

####基本配置-BEGIN####
import os

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

def print_structured_response(title, result):
    """打印结构化输出对象，方便对照字段是否提取成功。"""
    print(f"\n{title}")
    structured = result.get("structured_response")
    print("*" * 50)
    print(structured)
    print("*" * 50)
####基本配置-END####

"""
通过中间件定义状态
当需要通过特定中间件钩子和附加在中间件上的工具访问自定义状态时，使用中间件来定义自定义状态。
"""
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, dynamic_prompt, ModelRequest
from typing import Any, cast
from typing_extensions import TypedDict


from langchain.agents import create_agent

class CustomState(AgentState):
    """自定义状态：在默认消息历史之外，额外记录用户偏好。"""
    user_preferences: dict


def tool1(text: str) -> str:
    """示例工具1：返回简单处理结果。"""
    return f"工具1已处理：{text}"


def tool2(text: str) -> str:
    """示例工具2：返回简单处理结果。"""
    return f"工具2已处理：{text}"


tools = [tool1, tool2]


class CustomMiddleware(AgentMiddleware):
    # 指定中间件使用的状态结构
    state_schema = CustomState
    # 挂载该中间件可访问的工具
    tools = tools

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        """模型调用前的钩子：可在这里读取/写入自定义状态。"""
        # 示例中不修改状态，直接继续流程
        print("模型调用前的 CustomMiddleware.before_model 钩子被触发。当前状态：")
        print(state)

        return None


# 创建代理：注入模型、工具和中间件
agent = create_agent(
    model=model,
    tools=tools,
    middleware=[CustomMiddleware()]
)

# 代理现在可以在消息历史之外追踪 user_preferences
result = agent.invoke(cast(Any, {
    "messages": [{"role": "user", "content": "我偏好技术性解释"}],
    "user_preferences": {"style": "技术型", "verbosity": "详细"},
}))
print_agent_result("CustomState 通过中间件定义状态演示", result)

"""
通过 state_schema 定义状态
使用 state_schema 参数作为定义自定义状态的快捷方式。
为了直观看到效果，下面通过 dynamic_prompt 读取 state 中的 user_preferences，
并将偏好写入系统提示，让模型在回答中显式体现。
"""

class CustomStateBySchema(TypedDict):
    """通过 state_schema 定义的自定义状态（TypedDict 方式）。"""
    messages: list[dict[str, str]]
    user_preferences: dict[str, str]


@dynamic_prompt
def schema_preferences_prompt(request: ModelRequest) -> str:
    """从 state_schema 定义的 state 读取偏好并动态生成系统提示。"""
    state = getattr(request, "state", {}) or {}
    prefs = state.get("user_preferences", {})
    style = prefs.get("style", "未设置")
    verbosity = prefs.get("verbosity", "未设置")
    return (
        "你是一个中文助手。"
        "请先明确告知你读取到的用户偏好，"
        f"当前偏好：style={style}，verbosity={verbosity}。"
        "然后再给出简短回答。"
    )


agent_by_schema = create_agent(
    model=model,
    tools=[],
    middleware=[schema_preferences_prompt],
    state_schema=cast(Any, CustomStateBySchema),
)

# 代理现在可以在 messages 之外追踪 user_preferences
result_by_schema = agent_by_schema.invoke(cast(Any, {
    "messages": [{"role": "user", "content": "你读取到了我在状态里的偏好吗？请直接说出来。"}],
    "user_preferences": {"style": "技术型", "verbosity": "详细"},
}))
print_agent_result("CustomState 通过 state_schema 定义状态演示", result_by_schema)

