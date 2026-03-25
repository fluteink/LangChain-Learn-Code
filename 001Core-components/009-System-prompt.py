import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
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

"""
动态系统提示:对于需要根据运行时上下文或代理状态修改系统提示符的高级用例，可以使用中间件 。@dynamic_prompt 装饰器创建中间件，根据模型请求生成系统提示：
"""
@tool
def web_search(query: str) -> str:
    """模拟搜索工具：根据关键词返回简短检索结果。"""
    mock_db = {
        "LangChain": "LangChain 是用于构建 LLM 应用的框架，支持工具调用、记忆、检索增强等能力。",
        "ReAct Agent": "ReAct 是“推理 + 行动”范式：模型先思考，再调用工具，再根据工具结果继续推理。",
        "系统提示": "系统提示用于定义助手角色、风格、边界与优先级，对回答行为有全局约束作用。",
    }

    # 简单匹配：命中关键词则返回对应摘要，否则返回默认结果
    for key, value in mock_db.items():
        if key.lower() in query.lower():
            return f"搜索关键词：{query}\n检索结果：{value}"

    return (
        f"搜索关键词：{query}\n"
        "检索结果：未命中知识库精确词条。建议补充更具体关键词（如 LangChain、ReAct Agent）。"
    )


class Context(TypedDict):
    # 用户角色：beginner / expert / user
    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色动态生成系统提示。"""
    # 兼容读取 runtime.context，避免上下文缺失时报错
    runtime_ctx = getattr(getattr(request, "runtime", None), "context", {}) or {}
    user_role = runtime_ctx.get("user_role", "user")

    base_prompt = (
        "你是一个乐于助人的中文助手。\n"
        "当问题涉及事实查询时，优先调用工具再回答。\n"
        "回答要清晰、准确，并结合工具返回结果。"
    )

    if user_role == "expert":
        return (
            f"{base_prompt}\n"
            "当前用户是专家：请使用更专业术语，补充关键技术细节与原理。"
        )
    if user_role == "beginner":
        return (
            f"{base_prompt}\n"
            "当前用户是初学者：请用通俗中文解释，尽量避免术语堆砌。"
        )

    return f"{base_prompt}\n当前用户为普通角色：保持中等详细度即可。"


agent = create_agent(
    model=model,
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context,
)


if __name__ == "__main__":
    # 演示1：初学者角色
    result_beginner = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请搜索 LangChain，然后用简单的话解释它是做什么的。",
                }
            ]
        },
        context={"user_role": "beginner"},
    )
    print_agent_result("=" * 50 + "\nbeginner 角色演示\n" + "=" * 50, result_beginner)

    # 演示2：专家角色
    result_expert = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请搜索 ReAct Agent，并给出技术向说明。",
                }
            ]
        },
        context={"user_role": "expert"},
    )
    print_agent_result("=" * 50 + "\nexpert 角色演示\n" + "=" * 50, result_expert)

    # 演示3：默认角色（不传或传 user）
    result_user = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请搜索 系统提示，并概述它的作用。",
                }
            ]
        },
        context={"user_role": "user"},
    )
    print_agent_result("=" * 50 + "\nuser 角色演示\n" + "=" * 50, result_user)


"""静态系统提示"""
static_agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=(
        "你是一个乐于助人的中文助手。\n"
        "回答风格：简洁、清晰、结构化。\n"
        "当问题涉及事实查询时，优先调用工具再回答。\n"
        "若信息不足，请明确说明不确定性，不要编造。"
    ),
)
# 演示0：静态系统提示
result_static = static_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "请搜索 LangChain，并用两句话说明它的作用。",
            }
        ]
    }
)
print_agent_result("=" * 50 + "\n静态 system_prompt 演示\n" + "=" * 50, result_static)

# 演示1：初学者角色（动态系统提示）
result_beginner = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "请搜索 LangChain，然后用简单的话解释它是做什么的。",
            }
        ]
    },
    context={"user_role": "beginner"},
)
print_agent_result("=" * 50 + "\nbeginner 角色演示\n" + "=" * 50, result_beginner)
