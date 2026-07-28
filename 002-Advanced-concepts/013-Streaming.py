"""
我们已经看到了如何调用代理以获得最终响应。如果代理执行多个步骤，则这可能需要一段时间。为了显示中间进度，我们可以在消息发生时将其流回。
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
####基本配置-END####

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import AIMessage, HumanMessage


@tool
def search_news(topic: str) -> str:
    """搜索新闻：返回指定主题的模拟新闻列表。"""
    news_items = {
        "AI": [
            "OpenAI 发布 GPT-5，性能大幅提升",
            "Google DeepMind 推出新一代多模态模型",
            "中国发布 AI 发展新规，强调安全与伦理",
        ],
        "科技": [
            "苹果发布新款 iPhone，搭载 A18 芯片",
            "特斯拉推出全自动驾驶 Beta 版",
            "微软收购动视暴雪交易完成",
        ],
    }
    items = news_items.get(topic, [f"暂无关于'{topic}'的新闻"])
    return "\n".join(items)


@tool
def summarize_text(text: str) -> str:
    """总结文本：返回简短摘要（示例实现）。"""
    lines = text.strip().split("\n")
    if len(lines) <= 2:
        return f"摘要：{text}"
    return f"摘要：共{len(lines)}条新闻，主要涉及{lines[0][:20]}..."


# 创建支持工具调用的 Agent
agent = create_agent(
    model=model,
    tools=[search_news, summarize_text],
    system_prompt=(
        "你是一个新闻助手。当用户要求搜索新闻并总结时，\n"
        "必须先调用 search_news 获取新闻，\n"
        "再调用 summarize_text 进行总结，\n"
        "最后基于工具结果给出简洁结论。"
    ),
)

if __name__ == "__main__":
    print("=" * 60)
    print("LangChain Agent 流式输出演示")
    print("=" * 60)

    # 演示 1：使用 stream_mode="values" 流式输出完整状态
    print("\n【演示 1】stream_mode='values' - 流式输出每条消息\n")

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "搜索 AI 新闻并总结发现"}]},
        stream_mode="values",
    ):
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            if isinstance(latest_message, HumanMessage):
                print(f"👤 用户：{latest_message.content}")
            elif isinstance(latest_message, AIMessage):
                print(f"🤖 智能体：{latest_message.content}")
        elif latest_message.tool_calls:
            print(f"🔧 调用工具：{[tc['name'] for tc in latest_message.tool_calls]}")

    # 演示 2：使用 stream_mode="messages" 流式输出 token
    print("\n" + "=" * 60)
    print("【演示 2】stream_mode='messages' - 流式输出 token\n")

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "搜索科技新闻并总结"}]},
        stream_mode="messages",
    ):
        # chunk 是 (message, metadata) 元组或直接是消息对象
        if isinstance(chunk, tuple) and len(chunk) == 2:
            msg, metadata = chunk
        else:
            msg = chunk
        
        if hasattr(msg, 'content') and msg.content:
            print(msg.content, end="", flush=True)
        elif hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"\n🔧 调用工具：{[tc['name'] for tc in msg.tool_calls]}")

    print("\n\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
