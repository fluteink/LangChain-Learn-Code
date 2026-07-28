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
# MODELSCOPE_API_KEY = os.getenv("ModelScope_API_KEY")
# MODELSCOPE_BASE_URL = os.getenv("ModelScope_BASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["ALL_PROXY"] = "socks5://127.0.0.1:7897"
# print(f"ModelScope API Key: {MODELSCOPE_API_KEY}\nModelScope Base URL: {MODELSCOPE_BASE_URL}")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

"""
3. 向量存储
LangChain VectorStore 对象包含将文本和 Document 对象添加到存储中的方法，并使用各种相似性度量对它们进行查询。
它们通常使用 嵌入 模型初始化，这些模型决定了如何将文本数据转换为数字向量。
LangChain 包含一套与不同向量存储技术集成的 集成。一些向量存储由提供商托管（例如各种云提供商），需要特定凭据才能使用；
一些（如 Postgres）在单独的基础设施中运行，可以在本地运行或通过第三方运行；其他可以在内存中运行以处理轻量级工作负载。让我们选择一个向量存储：
"""


from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

"""
请注意，大多数向量存储实现将允许您连接到现有的向量存储——例如，通过提供客户端、索引名称或其他信息。
请参阅特定 集成 的文档以获取更多详情。
一旦我们实例化了一个包含文档的 VectorStore，我们就可以查询它。VectorStore 包括以下查询方法：
·通过字符串查询和向量查询
·返回或不返回相似性分数
·通过相似性搜索和 最大边际相关性（以平衡查询的相似性与检索结果的多样性）
·这些方法的输出通常包含 Document 对象列表。
"""
"""
用法
嵌入通常将文本表示为"稠密"向量，使得含义相近的文本在几何上接近。这让我们只需传入问题即可检索相关信息，
而无需了解文档中使用的任何特定关键术语。
基于与字符串查询的相似性返回文档：
"""
print("\n查询：How many distribution centers does Nike have in the US?\n")
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(results[0])
"""
# 异步查询：
results = await vector_store.asimilarity_search("When was Nike incorporated?")

print(results[0])
"""

# 返回分数：
# 请注意，不同提供商实现不同的分数；这里的分数
# 是一个与相似性成反比的距离度量。
print("\n查询：耐克 2023 年的收入是多少？\n")
results = vector_store.similarity_search_with_score("耐克 2023 年的收入是多少？")
doc, score = results[0]
print(f"分数：{score}\n")
print(doc)


# 基于与嵌入查询的相似性返回文档：
print("\n查询：耐克 2023 年的利润率受到什么影响？\n")
embedding = embeddings.embed_query("耐克 2023 年的利润率受到什么影响？")

results = vector_store.similarity_search_by_vector(embedding)
print(results[0])

"""
4. 检索器
LangChain VectorStore 对象不继承 Runnable。LangChain 检索器 是 Runnable，
因此它们实现了一组标准方法（例如同步和异步的 invoke 和 batch 操作）。
虽然我们可以从向量存储构建检索器，但检索器也可以与非向量存储的数据源交互（例如外部 API）。

我们可以自己创建一个简单的版本，而无需继承 Retriever。如果我们选择希望使用什么方法来检索文档，
我们可以轻松创建一个可运行对象。下面我们将围绕 similarity_search 方法构建一个：
"""
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)
print("\n批量查询\n")

batch_results = retriever.batch(
    [
        "耐克在美国有多少个分销中心？",
        "耐克是什么时候成立的？",
    ],
)
print(batch_results)

batch_results_2 = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
print(batch_results_2)
"""
向量存储实现了 as_retriever 方法，该方法将生成一个检索器，具体是 VectorStoreRetriever。
这些检索器包含特定的 search_type 和 search_kwargs 属性，用于标识要调用的底层向量存储方法，
以及如何参数化它们。例如，我们可以使用以下方式复制上面的内容：
"""
print("\nas_retriever演示\n")
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

as_retriever_results = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
print(as_retriever_results)
"""
VectorStoreRetriever 支持 "similarity"（默认）、"mmr"（最大边际相关性，如上所述）和 "similarity_score_threshold" 搜索类型。
我们可以使用后者通过相似性分数对检索器输出的文档进行阈值处理。
检索器可以轻松集成到更复杂的应用程序中，例如 检索增强生成 (RAG) 应用程序，该应用程序将给定问题与检索到的上下文结合到 LLM 的提示中。
要了解有关构建此类应用程序的更多信息，请查看 RAG 教程 教程。
"""