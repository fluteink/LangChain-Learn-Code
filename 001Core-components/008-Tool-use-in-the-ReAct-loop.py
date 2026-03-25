
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
            print(f"  工具调用：\n"+"*"*50+"\n"+f"{tool_calls}"+"\n"+"*"*50)

        # 打印工具执行结果
        if msg_type == "ToolMessage":
            print(f"  工具返回：\n"+"*"*50+"\n"+f"{getattr(msg, 'name', '')} -> "+f"{getattr(msg, 'content', '')}"+"\n"+"*"*50)

    # 打印最后一条 AI 回复作为最终结果
    if messages:
        last_msg = messages[-1]
        print(f"最终结果：\n"+"*"*50+"\n"+f"{getattr(last_msg, 'content', '')}"+"\n"+"*"*50)
####基本配置-END####


@tool
def search_products(query: str) -> str:
    """搜索商品：返回按热门程度排序的商品列表（示例数据）。"""
    # 这里用固定数据做演示，重点是让 Agent 进入“先搜索再查库存”的工具链路。
    top_products = [
        "WH-1000XM5",
        "AirPods Pro 2",
        "Bose QuietComfort Ultra",
        "Sennheiser Momentum 4",
        "Soundcore Liberty 4 NC",
    ]
    return f'找到 5 个与“{query}”相关的商品。热门前 5：{", ".join(top_products)}'


@tool
def check_inventory(product_id: str) -> str:
    """查询库存：根据商品 ID 返回库存数量（示例数据）。"""
    inventory = {
        "WH-1000XM5": 10,
        "AirPods Pro 2": 6,
        "Bose QuietComfort Ultra": 0,
        "Sennheiser Momentum 4": 3,
        "Soundcore Liberty 4 NC": 18,
    }
    stock = inventory.get(product_id)
    if stock is None:
        return f"商品 {product_id}：未找到"
    return f"商品 {product_id}：库存 {stock} 件"


agent = create_agent(
    model=model,
    tools=[search_products, check_inventory],
    system_prompt=(
        "你是一个会调用工具的购物助手。\n"
        "当用户询问‘最受欢迎且是否有货’时，必须先调用 search_products，"
        "再对第一个结果调用 check_inventory，最后基于工具结果给出简洁结论。"
    ),
)


if __name__ == "__main__":
    # 演示 ReAct 循环：模型先检索热门商品，再查询库存，最后给出答案。
    demo_prompt = "帮我找出当前最受欢迎的无线耳机，并确认是否有库存"
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": demo_prompt,
                }
            ]
        }
    )
    print_agent_result("=" * 50 + "\nReAct 工具调用演示\n" + "=" * 50, result)





