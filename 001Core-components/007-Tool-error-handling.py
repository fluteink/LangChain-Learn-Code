from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
import os
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

def search(query: str) -> str:
    """模拟搜索工具：故意抛错用于测试。"""
    raise RuntimeError(f"search 工具故障，query={query}")


def get_weather(city: str) -> str:
    """模拟天气工具：故意抛错用于测试。"""
    raise RuntimeError(f"get_weather 工具故障，city={city}")

@wrap_tool_call
def handle_tool_errors(request, handler):
    """
    工具调用错误处理中间件。

    说明：
    1\) 函数名 `handle_tool_errors` 可以修改；
       但在 `create_agent(..., middleware=[handle_tool_errors])` 中的引用必须同步修改。
    2\) 参数名 `request`、`handler` 也可以改名（例如 `tool_request`、`next_handler`）；
       但必须保留两个参数，且顺序建议保持：
       - 第1个：当前工具调用请求对象（包含 tool_call 信息，如 id）
       - 第2个：下一个处理函数/原始工具执行入口（通过调用它来继续执行）
    """
    try:
        # 正常路径：调用后续处理器（通常会真正执行工具）
        return handler(request)
    except Exception as e:
        # 异常路径：把可读错误包装成 ToolMessage 返回给模型
        # tool_call_id 用于让模型知道是哪一次工具调用失败
        return ToolMessage(
            content=f"工具调用错误：请检查输入后重试。（{str(e)}）",
            tool_call_id=request.tool_call["id"]
        )
agent = create_agent(
    model=model,
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)

# 测试用例：故意触发工具调用错误，观察 middleware 返回的 ToolMessage
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "请先调用 search 查询 LangChain，然后调用 get_weather 查询北京天气。",
            }
        ]
    }
)
print_agent_result("工具报错测试", result)