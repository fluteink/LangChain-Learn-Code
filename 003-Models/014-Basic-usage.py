"""
模型
LLM（大语言模型）是强大的 AI 工具，能够像人类一样理解和生成文本。它们用途广泛，无需针对每个任务进行专门训练即可编写内容、翻译语言、摘要和回答问题。

除了文本生成之外，许多模型还支持：

工具调用 - 调用外部工具（如数据库查询或 API 调用）并在响应中使用结果。

结构化输出 - 模型的响应被限制为遵循定义的格式。

多模态 - 处理和返回文本以外的数据，如图像、音频和视频。

推理 - 模型执行多步推理以得出结论。

模型是 智能体 的推理引擎。它们驱动智能体的决策过程，确定调用哪些工具、如何解释结果以及何时提供最终答案。您选择的模型的质量和性能直接影响智能体的基线可靠性和性能。

不同的模型擅长不同的任务——有些更擅长遵循复杂指令，有些更擅长结构化推理，还有一些支持更大的上下文窗口以处理更多信息。LangChain 的标准模型接口让您可以访问许多不同的提供商集成，从而可以轻松地在不同模型之间进行实验和切换，以找到最适合您用例的模型。

有关提供商特定的集成信息和功能，请参阅提供商的 聊天模型页面。
"""


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
    model="gpt-5.4",
    api_key=os.getenv("ChatGPT_API_KEY"),
    base_url=os.getenv("ChatGPT_BASE_URL"),
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
基本用法
模型有两种使用方式：

与智能体一起使用 - 在创建 智能体 时可以动态指定模型。

独立使用 - 直接调用模型（在智能体循环之外），用于文本生成、分类或提取等任务，无需智能体框架。

相同的模型接口在这两种场景下都适用，让您能够灵活地从简单开始，并根据需要扩展到更复杂的基于智能体的工作流。

主要方法
invoke - 模型接收消息作为输入，并在生成完整响应后输出消息。

stream - 调用模型，但在生成时实时流式输出。

batch - 将多个请求批量发送到模型以更高效地处理。
"""


# 独立使用示例
response = model.invoke("为什么鹦鹉会说话？")
print_model_result(response)





"""
参数
聊天模型接受的参数可用于配置其行为。支持的完整参数集因模型和提供商而异，但标准参数包括：

model - 要与提供商一起使用的特定模型的名称或标识符。您还可以使用 '{model_provider}:{model}' 格式在单个参数中指定模型及其提供商，例如 'openai:o1'。

api_key - 与模型提供商进行身份验证所需的密钥。这通常在您注册访问模型时颁发。通常通过设置环境变量来访问。

temperature - 控制模型输出的随机性。较高的数字使响应更具创造性；较低的数字使响应更确定性。

max_tokens - 限制响应中的总 token 数，有效控制输出的长度。

timeout - 在取消请求之前等待模型响应的最长时间（以秒为单位）。

max_retries - 如果请求因网络超时或速率限制等问题失败，系统将尝试重新发送请求的最大次数。重试使用指数退避和抖动。网络错误、速率限制 (429) 和服务器错误 (5xx) 会自动重试。客户端错误（如 401（未授权）或 404）不会重试。对于在不可靠网络上运行的长时间 智能体 任务，请考虑将此值增加到 10-15。
"""
# 参数演示示例
demo_model = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key_DeepSeek,
    base_url=base_url_DeepSeek,
    temperature=0.3,
    max_tokens=120,
    timeout=30,
    max_retries=3,
)
print('='*60 + " 参数演示示例" + '='*60)
demo_response = demo_model.invoke("请用三句话解释为什么天空是蓝色的。")
print_model_result(demo_response)


"""
调用:必须调用聊天模型才能生成输出。有三种主要的调用方法，每种适用于不同的用例。
"""


"""
Invoke
调用模型最直接的方法是使用 invoke\(\) 传递单条消息或消息列表。
"""

print('=' * 60 + " Invoke：单条消息 " + '=' * 60)
response = model.invoke("为什么鹦鹉有彩色的羽毛？")
print_model_result(response)



print('=' * 60 + " Invoke：dict 对话消息列表 " + '=' * 60)
conversation = [
    {"role": "system", "content": "你是一个将英语翻译成法语的有用助手。"},
    {"role": "user", "content": "翻译：I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "翻译：I love building applications."}
]
response = model.invoke(conversation)
print_model_result(response)

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

print('=' * 60 + " Invoke：Message 对象列表 " + '=' * 60)
conversation = [
    SystemMessage(content="你是一个将英语翻译成法语的有用助手。"),
    HumanMessage(content="翻译：I love programming."),
    AIMessage(content="J'adore la programmation."),
    HumanMessage(content="翻译：I love building applications.")
]
response = model.invoke(conversation)
print_model_result(response)

# 注意：
# 如果返回类型是字符串，通常说明你调用的是传统 LLM（文本补全）而不是聊天模型。
# LangChain 聊天模型通常以 Chat 开头，例如 ChatOpenAI。

"""
Stream
大多数模型支持在生成过程中流式返回内容，可显著改善长回复场景下的体验。
"""
import asyncio

# 预期输出：会看到一段被“|”分隔的逐步文本，例如“因|为|羽|毛...|”。
# 原因：stream() 每次返回的是增量 chunk，chunk.text 是当前时间片新增的内容，不是一次性完整答案。
print("=" * 60 + " Stream：基础文本流 " + "=" * 60)
for chunk in model.stream("为什么鹦鹉有彩色的羽毛？"):
    print(getattr(chunk, "text", ""), end="|", flush=True)
print()


# 预期输出：可能混合出现“推理：...”“工具调用块：...”“纯文本...”，不同模型显示的块类型可能不同。
# 原因：content_blocks 把流式内容按语义切块（reasoning/tool_call_chunk/text），代码按 type 分支分别打印。
print("=" * 60 + " Stream：content_blocks 解析 " + "=" * 60)
for chunk in model.stream("天空是什么颜色的？"):
    for block in getattr(chunk, "content_blocks", []) or []:
        block_type = block.get("type")
        if block_type == "reasoning":
            reasoning = block.get("reasoning")
            if reasoning:
                print(f"推理：{reasoning}")
        elif block_type == "tool_call_chunk":
            print(f"工具调用块：{block}")
        elif block_type == "text":
            print(block.get("text", ""))


# 预期输出：会看到文本从短到长不断“累积”，最后两行是完整聚合结果和完整 content_blocks。
# 原因：AIMessageChunk 支持相加，full = full + chunk 会把历史片段与新片段合并成更完整的消息对象。
print("=" * 60 + " Stream：块聚合为完整消息 " + "=" * 60)
full = None
for chunk in model.stream("天空是什么颜色的？"):
    full = chunk if full is None else full + chunk
    print(getattr(full, "text", ""))

print("最终聚合文本：", getattr(full, "text", ""))
print("最终 content_blocks：", getattr(full, "content_blocks", []))


# 预期输出：通常先打印 start 输入，再连续打印多个 stream token，最后打印 end 的完整消息。
# 原因：astream_events() 暴露模型生命周期事件，开始/流式中间态/结束会以不同 event 名称分阶段触发。
print("=" * 60 + " astream_events：语义事件流 " + "=" * 60)
async def demo_astream_events() -> None:
    async for event in model.astream_events("你好"):
        event_type = event.get("event")
        data = event.get("data", {})
        if event_type == "on_chat_model_start":
            print(f"输入：{data.get('input')}")
        elif event_type == "on_chat_model_stream":
            chunk = data.get("chunk")
            print(f"Token: {getattr(chunk, 'text', '')}")
        elif event_type == "on_chat_model_end":
            output = data.get("output")
            print(f"完整消息：{getattr(output, 'text', '')}")

asyncio.run(demo_astream_events())


"""
Batch
对独立请求进行批处理，可在客户端并行调用模型，常用于问答、分类、摘要等高吞吐场景。
"""

print("=" * 60 + " Batch：基础并行调用 " + "=" * 60)
batch_inputs = [
    "为什么鹦鹉有彩色的羽毛？",
    "飞机是如何飞行的？",
    "什么是量子计算？",
]
batch_responses = model.batch(batch_inputs)
for i, resp in enumerate(batch_responses):
    print_model_result(resp, title=f"Batch 响应 #{i}")

print("=" * 60 + " Batch：按完成顺序返回（可能乱序） " + "=" * 60)
for idx, resp in model.batch_as_completed(batch_inputs):
    print_model_result(resp, title=f"batch_as_completed 响应(原索引={idx})")

print("=" * 60 + " Batch：限制最大并发数 " + "=" * 60)
limited_responses = model.batch(
    batch_inputs,
    config={"max_concurrency": 2},  # 最多 2 个并行请求
)
for i, resp in enumerate(limited_responses):
    print_model_result(resp, title=f"并发限制响应 #{i}")

"""
示例：用 batch 一次性生成“长段文档说明”
说明：
1. 目标：批量生成多段可直接用于项目文档的长文本说明，减少串行调用耗时。
2. 做法：把多个“文档写作任务”放入同一个输入列表，通过 batch 并行执行。
3. 输出：每个输入对应一条完整结果，顺序与输入一致；可直接保存到 Markdown 文件。
4. 建议：
   - 如果是长文档，建议设置较高 max_tokens，避免内容截断。
   - 如果输入很多，建议通过 config.max_concurrency 控制并发，平衡速度与限流风险。
   - 若对稳定性要求高，保留较高 max_retries，并记录 response_metadata 便于排障。
5. 与 batch_as_completed 的区别：
   - batch：结果按输入顺序返回，便于直接落盘。
   - batch_as_completed：谁先完成先返回，适合实时消费与进度展示。
"""

print("=" * 60 + " Batch：批量生成长文档说明 " + "=" * 60)
doc_prompts = [
    (
        "请写一份面向初学者的长文档，主题是“什么是 RAG（检索增强生成）”。"
        "要求：分章节（背景、核心流程、优缺点、落地建议、常见误区），"
        "使用中文，结构清晰，内容详细，约 800~1200 字。"
    ),
    (
        "请写一份面向工程团队的长文档，主题是“LLM 应用中的提示词工程实践”。"
        "要求：包含设计原则、模板示例、评估方法、线上监控与迭代策略，"
        "使用中文，结构清晰，内容详细，约 800~1200 字。"
    ),
]

doc_responses = model.batch(
    doc_prompts,
    config={"max_concurrency": 2},
)

for i, resp in enumerate(doc_responses):
    print("\n" + "─" * 90)
    print(f"文档说明 #{i + 1}")
    print("─" * 90)
    print(getattr(resp, "content", ""))
