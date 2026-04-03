"""
LangChain 实现了 Document 抽象，旨在表示文本单元和相关元数据。它有三个属性：

page_content：表示内容的字符串

metadata：包含任意元数据的字典

id：（可选）文档的字符串标识符

metadata 属性可以捕获文档来源、与其他文档的关系以及其他信息。请注意，单个 Document 对象通常代表较大文档的一个块。

我们可以在需要时生成示例文档：
"""
import dotenv
from dotenv import load_dotenv
from langchain_core.documents import Document

documents = [
    Document(
        page_content="狗是很好的伴侣，以其忠诚和友善著称。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="猫是独立的宠物，通常喜欢自己的空间。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
]
"""
然而，LangChain 生态系统实现了 文档加载器，与数百种常见源集成。
这使得将这些源的数据轻松集成到您的 AI 应用程序中变得容易。
"""
"""
加载文档
让我们将 PDF 加载到 Document 对象序列中。
这是一个示例 PDF —— 耐克公司 2023 年的 10-K 文件。
我们可以查阅 LangChain 文档了解 可用的 PDF 文档加载器。
"""
from langchain_community.document_loaders import PyPDFLoader

file_path = "./nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
"""
PyPDFLoader 每个 PDF 页面加载一个 Document 对象。对于每个对象，我们可以轻松访问：
页面的字符串内容
包含文件名和页码的元数据
"""
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)


"""
分割
为了信息检索和下游问答目的，页面可能表示得过于粗略。我们的最终目标是检索回答输入查询的 Document 对象，
进一步分割我们的PDF将有助于确保相关文档部分的含义不会被周围文本"冲淡"。
我们可以使用文本分割器来实现这个目的。这里我们将使用一个简单的文本分割器，它基于字符进行分区。
我们将文档分割成 1000 个字符的块，块之间有 200 个字符的重叠。重叠有助于防止将陈述与其相关的重要上下文分离。
我们使用RecursiveCharacterTextSplitter，它将递归地使用常见分隔符（如换行符）分割文档，直到每个块达到适当的大小。
这是通用文本用例的推荐文本分割器。
我们设置add_start_index=True，以便每个分割后的Document在初始Document中的起始字符索引作为元数据属性"start_index"保留。
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(len(all_splits))

"""
2. 嵌入
向量搜索是一种存储和搜索非结构化数据（如非结构化文本）的常见方法。其思想是存储与文本关联的数字向量。
给定一个查询，我们可以将其 嵌入 为相同维度的向量，并使用向量相似性度量（如余弦相似性）来识别相关文本。
LangChain 支持来自 数十个提供商 的嵌入。这些模型指定了如何将文本转换为数字向量。让我们选择一个模型：
"""
import os
load_dotenv()
MODELSCOPE_API_KEY = os.getenv("ModelScope_API_KEY")
MODELSCOPE_BASE_URL = os.getenv("ModelScope_BASE_URL")

from langchain_openai import OpenAIEmbeddings

# 使用 ModelScope 的 OpenAI 兼容 API
embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-8B",
    api_key="ms-287d22f9-a746-4e96-b041-e40ff62483b5",
    base_url="https://api-inference.modelscope.cn/v1",
)
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])