import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")


SYSTEM_PROMPT = """你是一位擅长使用双关语的专业天气预报员。

你可以使用两款工具：
- get_weather_for_location：用于获取指定地点的天气信息
- get_user_location：用于获取用户所在位置

若用户向你询问天气，请务必确认地点。若从问题中能判断对方想了解自身所在地的天气，则使用 get_user_location 工具获取其位置。"""

@dataclass
class Context:
    """智能体的自定义上下文。"""
    user_id: str

@tool
def get_weather_for_location(city : str) ->str:
    """获取给定城市的天气。"""
    return f"{city}总是阳光明媚"


@tool
def get_user_location(runtime: ToolRuntime[Context]) ->str:
    """根据用户ID查询用户信息。"""
    user_id = runtime.context.user_id
    return "北京" if user_id == "1" else "成都"

model = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url=base_url,
    temperature=0
)

@dataclass()
class ResponseFormat:
    """智能体的响应架构。"""
    #一个俏皮回答
    punny_response: str
    # 任何关于天气的有趣信息，如果可用的话
    weather_conditions: str | None = None

# 设置记忆
checkpointer = InMemorySaver()

# 创造一个agent
agent  = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location,get_weather_for_location],
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

#运行智能体
# `thread_id`是给定会话的唯一标识符。
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "外面的天气怎么样？"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
# punny_response='看来北京的天气真是"京"彩夺目啊！阳光明媚得让人想出去"晒"一下存在感。不过要小心，这么晴朗的天气可能会让你"晒"得有点"北"常热哦！',
# weather_conditions='阳光明媚'
# )


#注意，我们可以使用相同的`thread_id`继续对话。
response = agent.invoke(
    {"messages":[{"role":"user","content":"我刚刚说了什么？"}]},
    config=config,
    context=Context(user_id="1")
)
print(response['structured_response'])