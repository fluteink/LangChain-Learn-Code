import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
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


"""结构化输出
ToolStrategy 使用人工工具调用生成结构化输出，适用于支持工具调用的模型。
"""


class ContactInfo(BaseModel):
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱")
    phone: str = Field(description="电话号码")


@tool
def search_tool(query: str) -> str:
    """模拟检索工具：返回待提取的原始文本。"""
    return f"检索到内容：{query}"


tool_strategy_agent = create_agent(
    model=model,
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo),
    system_prompt=(
        "你是一个信息提取助手。"
        "当用户给出包含姓名、邮箱、电话的文本时，"
        "请提取为结构化结果。"
    ),
)


"""ProviderStrategy 使用模型提供方原生结构化输出能力。
注：并非所有模型都支持；本示例用 try-except 做兼容演示。
"""
provider_strategy_agent = create_agent(
    model=ChatGPT_model,
    response_format=ProviderStrategy(ContactInfo),
    system_prompt="你是一个信息提取助手，请返回结构化联系方式。",
)


if __name__ == "__main__":
    user_text = "从以下信息提取联系方式：张三, zhangsan@example.com, 138-0013-8000"

    # 演示 1：ToolStrategy（与前面示例同风格，打印消息链 + 结构化结果）
    result_tool = tool_strategy_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_text,
                }
            ]
        }
    )
    print_agent_result("=" * 50 + "\nToolStrategy 演示\n" + "=" * 50, result_tool)
    print_structured_response("ToolStrategy structured_response", result_tool)

    # 演示 2：ProviderStrategy（如果当前模型不支持，打印可读提示）
    try:
        result_provider = provider_strategy_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_text,
                    }
                ]
            }
        )
        print_agent_result("=" * 50 + "\nProviderStrategy 演示\n" + "=" * 50, result_provider)
        print_structured_response("ProviderStrategy structured_response", result_provider)
    except Exception as exc:
        print("\n" + "=" * 50)
        print("ProviderStrategy 演示")
        print("=" * 50)
        print(f"当前模型暂不支持 ProviderStrategy 或配置不满足，已跳过。错误信息：{exc}")

