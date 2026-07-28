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
    model="gpt-5.3-codex-2cx",
    api_key=os.getenv("ChatGPT_API_KEY"),
    base_url=os.getenv("ChatGPT_BASE_URL"),
)

OpenRouter_model = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    api_key=os.getenv("OpenRoute_API_KEY"),
    base_url=os.getenv("OpenRoute_BASE_URL"),
)

ZhiZengZeng_model = ChatOpenAI(

    model="gpt-oss-20b",
    api_key=os.getenv("ZHIZENGZENG_API_KEY"),
    base_url=os.getenv("ZHIZENGZENG_BASE_URL"),
)

JIEKOU_model = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("JIEKOU_API_KEY"),
    base_url=os.getenv("JIEKOU_BASE_URL"),
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
"""
可以要求模型按照指定规范格式输出结果。这能保障输出内容易于解析，并顺利对接后续业务处理。LangChain 支持多种规范类型与强制结构化输出的实现方案。
"""

"""
Pydantic 模型提供最丰富的功能集，包括字段验证、描述和嵌套结构。
"""


from pydantic import BaseModel, Field

class Movie(BaseModel):
    """一部有详细信息的电影。"""
    title: str = Field(description="电影标题")
    year: int = Field(description="电影发行年份")
    director: str = Field(description="电影导演")
    rating: float = Field(description="电影评分（满分 10 分）")

model_with_structure = JIEKOU_model.with_structured_output(Movie)
response = model_with_structure.invoke("提供电影《盗梦空间》的详细信息")
print(response)
# Movie(title="盗梦空间", year=2010, director="克里斯托弗·诺兰", rating=8.8)



"""
Python 的 TypedDict 提供了比 Pydantic 模型更简单的替代方案，适用于不需要运行时验证的情况。
"""


from typing_extensions import TypedDict, Annotated

class MovieDict(TypedDict):
    """一部有详细信息的电影。"""
    title: Annotated[str, ..., "电影标题"]
    year: Annotated[int, ..., "电影发行年份"]
    director: Annotated[str, ..., "电影导演"]
    rating: Annotated[float, ..., "电影评分（满分 10 分）"]

model_with_structure = JIEKOU_model.with_structured_output(MovieDict)
response = model_with_structure.invoke("提供电影《盗梦空间》的详细信息")
print(response)
# {'title': '盗梦空间', 'year': 2010, 'director': '克里斯托弗·诺兰', 'rating': 8.8}


"""
提供 JSON Schema 以获得最大控制和互操作性。
"""
import json

json_schema = {
    "title": "Movie",
    "description": "一部有详细信息的电影",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "电影标题"
        },
        "year": {
            "type": "integer",
            "description": "电影发行年份"
        },
        "director": {
            "type": "string",
            "description": "电影导演"
        },
        "rating": {
            "type": "number",
            "description": "电影评分（满分 10 分）"
        }
    },
    "required": ["title", "year", "director", "rating"]
}

model_with_structure = JIEKOU_model.with_structured_output(
    json_schema,
    method="json_schema",
)
response = model_with_structure.invoke("提供电影《盗梦空间》的详细信息")
print(response)
# {'title': '盗梦空间', 'year': 2010, ...}
