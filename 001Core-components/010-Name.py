import os
from dotenv import load_dotenv
from langchain.agents import create_agent
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
                + f"{getattr(msg, 'name', '')} -> {getattr(msg, 'content', '')}"
                + "\n"
                + "*" * 50
            )

    # 打印最后一条 AI 回复作为最终结果
    if messages:
        last_msg = messages[-1]
        print(f"最终结果：\n" + "*" * 50 + "\n" + f"{getattr(last_msg, 'content', '')}" + "\n" + "*" * 50)


####基本配置-END####


def _extract_final_text(result: dict) -> str:
    """提取子代理的最后一条 AI 文本，便于主代理汇总。"""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if type(msg).__name__ == "AIMessage":
            return getattr(msg, "content", "")
    return "子代理未返回有效内容。"


# 子代理 1：信息检索代理（只负责查资料）
research_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "你是 research_agent。"
        "你的职责是基于常识给出简洁、客观的资料点，不做长篇结论。"
    ),
    name="research_agent",
)


# 子代理 2：总结润色代理（只负责组织表达）
writer_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "你是 writer_agent。"
        "你的职责是把输入内容整理成中文简短总结，格式清晰，语气自然。"
    ),
    name="writer_agent",
)


@tool
def ask_research_agent(topic: str) -> str:
    """调用 research_agent 获取某个主题的要点。"""
    result = research_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"请给出主题“{topic}”的 3 条关键要点。",
                }
            ]
        }
    )
    return _extract_final_text(result)


@tool
def ask_writer_agent(draft: str) -> str:
    """调用 writer_agent 将草稿整理为最终可读答案。"""
    result = writer_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请把以下内容整理成最终回复：\n"
                        f"{draft}\n"
                        "要求：中文、120字内、先结论后补充。"
                    ),
                }
            ]
        }
    )
    return _extract_final_text(result)


"""主代理（orchestrator）通过工具调用两个子代理，实现多 Agent 协作。"""
agent = create_agent(
    model=model,
    tools=[ask_research_agent, ask_writer_agent],
    system_prompt=(
        "你是 orchestrator_agent（主代理）。\n"
        "当用户提问时，必须先调用 ask_research_agent 获取要点，"
        "再调用 ask_writer_agent 生成最终回答。"
    ),
    name="orchestrator_agent",
)


if __name__ == "__main__":
    # 演示：主代理分派给两个子代理，再汇总成最终答案。
    print(f"当前主代理名称：{getattr(agent, 'name', 'orchestrator_agent')}")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请用多 agent 的方式，介绍一下 LangChain Agent 的核心价值。",
                }
            ]
        }
    )
    print_agent_result("=" * 50 + "\n多 Agent 协作演示\n" + "=" * 50, result)
