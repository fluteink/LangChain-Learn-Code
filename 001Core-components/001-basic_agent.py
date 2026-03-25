import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# Load .env variables (DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL)
load_dotenv()


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"冬天总是阳光明媚 {city}!"

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Build a DeepSeek chat model via OpenAI-compatible API
model = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url=base_url,
)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是个有用的助手",
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "旧金山的天气怎么样"}]}
)

print(result)

