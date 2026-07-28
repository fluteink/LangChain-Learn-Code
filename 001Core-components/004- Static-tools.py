import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

####基本配置-BEGIN####
load_dotenv()
api_key_DeepSeek = os.getenv("DEEPSEEK_API_KEY")
base_url_DeepSeek = os.getenv("DEEPSEEK_BASE_URL")

model=ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key_DeepSeek,
    base_url=base_url_DeepSeek,
)
####基本配置-END####

@tool
def search(query: str) -> str:
    """搜索信息。"""
    return f"结果：{query}"

@tool
def get_weather(location: str) -> str:
    """获取某个地点的天气信息。"""
    return f"{location} 天气：晴，72°F"

agent = create_agent(model, tools=[search, get_weather])


if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "请帮我搜索一下'Python编程'，并告诉我北京的天气。"}]}
    )
    print("=== 最终回复 ===")
    print(result["messages"][-1].content)