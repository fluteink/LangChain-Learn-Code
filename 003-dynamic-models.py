import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

load_dotenv()
api_key_DeepSeek = os.getenv("DEEPSEEK_API_KEY")
base_url_DeepSeek = os.getenv("DEEPSEEK_BASE_URL")
api_key_GLM = os.getenv("GLM_API_KEY")
base_url_GLM = os.getenv("GLM_BASE_URL")

basic_model = ChatOpenAI(
    model="glm-4.7-flash",
    api_key=api_key_GLM,
    base_url=base_url_GLM,
)
advanced_model = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key_DeepSeek,
    base_url=base_url_DeepSeek,
)

tools = []  # 若需要工具，请在此处创建列表，例如 [Tool(...), ...]


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """根据对话复杂度选择模型。"""
    message_count = len(request.state["messages"])

    if message_count > 10:
        model = advanced_model
        model_name = "deepseek-chat"
    else:
        model = basic_model
        model_name = "glm-4.7-flash"

    print(f"[router] message_count={message_count}, chosen_model={model_name}")
    return handler(request.override(model=model))


agent = create_agent(
    model=basic_model,  # 默认模型
    tools=tools,
    middleware=[dynamic_model_selection],
)


if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "你好，简单介绍一下你自己"}]}
    )
    print("=== 最终回复 ===")
    print(result["messages"][-1].content)

# 强制选择 advanced 模型（构造超过 10 条消息以触发路由到 advanced_model）
    long_messages = [{"role": "user", "content": f"历史消息 {i}"} for i in range(11)]
    result_adv = agent.invoke({"messages": long_messages})
    print("=== 最终回复（advanced 示例） ===")
    print(result_adv["messages"][-1].content)
