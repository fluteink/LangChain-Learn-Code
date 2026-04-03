# 使用 LangChain 构建语义搜索引擎

## 概述

本教程将帮助您熟悉 LangChain 的 [文档加载器](/oss/python/integrations/document_loaders)、[嵌入](/oss/python/integrations/embeddings) 和 [向量存储](/oss/python/integrations/vectorstores) 抽象。这些抽象旨在支持从（向量）数据库和其他源检索数据，以便与 LLM 工作流集成。对于需要检索数据作为模型推理一部分的应用程序（如检索增强生成或 [RAG](/oss/python/langchain/retrieval)），它们非常重要。

在这里，我们将基于 PDF 文档构建一个搜索引擎。这将允许我们检索与输入查询相似的 PDF 段落。本指南还在搜索引擎的基础上包含了一个最小化的 RAG 实现。

### 核心概念

本指南专注于文本数据检索。我们将涵盖以下概念：

* [文档和文档加载器](/oss/python/integrations/document_loaders)
* [文本分割器](/oss/python/integrations/splitters)
* [嵌入](/oss/python/integrations/embeddings)
* [向量存储](/oss/python/integrations/vectorstores) 和 [检索器](/oss/python/integrations/retrievers)

## 环境设置

### 安装

本教程需要 `langchain-community` 和 `pypdf` 包：

```bash
pip install langchain-community pypdf
```

如需更多详情，请参阅我们的 [安装指南](/oss/python/langchain/install)。

### LangSmith

使用 LangChain 构建的许多应用程序将包含多个步骤和多次 LLM 调用。
随着这些应用程序变得越来越复杂，能够检查链或代理内部的实际运行情况变得至关重要。
最好的方法是使用 [LangSmith](https://smith.langchain.com)。

在上述链接注册后，请确保设置环境变量以开始记录跟踪：

```shell
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

或者，如果在笔记本中，可以使用以下方式设置：

```python
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

## 1. 文档和文档加载器

LangChain 实现了 [Document](https://reference.langchain.com/python/langchain-core/documents/base/Document) 抽象，旨在表示文本单元和相关元数据。它有三个属性：

* `page_content`：表示内容的字符串
* `metadata`：包含任意元数据的字典
* `id`：（可选）文档的字符串标识符

`metadata` 属性可以捕获文档来源、与其他文档的关系以及其他信息。请注意，单个 [`Document`](https://reference.langchain.com/python/langchain-core/documents/base/Document) 对象通常代表较大文档的一个块。

我们可以在需要时生成示例文档：

```python
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
```

然而，LangChain 生态系统实现了 [文档加载器](/oss/python/integrations/document_loaders)，与数百种常见源集成。这使得将这些源的数据轻松集成到您的 AI 应用程序中变得容易。

### 加载文档

让我们将 PDF 加载到 [`Document`](https://reference.langchain.com/python/langchain-core/documents/base/Document) 对象序列中。[这是一个示例 PDF](https://github.com/langchain-ai/langchain/blob/v0.3/docs/docs/example_data/nke-10k-2023.pdf) —— 耐克公司 2023 年的 10-K 文件。我们可以查阅 LangChain 文档了解 [可用的 PDF 文档加载器](/oss/python/integrations/document_loaders/#pdfs)。

```python
from langchain_community.document_loaders import PyPDFLoader

file_path = "../example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
```

```text
107
```

`PyPDFLoader` 每个 PDF 页面加载一个 [`Document`](https://reference.langchain.com/python/langchain-core/documents/base/Document) 对象。对于每个对象，我们可以轻松访问：

* 页面的字符串内容
* 包含文件名和页码的元数据

```python
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)
```

```text
目录
美国
证券交易委员会
华盛顿特区 20549
10-K 表格
（标记一）
☑ 根据 1934 年证券交易法第 13 或 15(D) 条提交的年度报告
FO

{'source': '../example_data/nke-10k-2023.pdf', 'page': 0}
```

### 分割

为了信息检索和下游问答目的，页面可能表示得过于粗略。我们的最终目标是检索回答输入查询的 [`Document`](https://reference.langchain.com/python/langchain-core/documents/base/Document) 对象，进一步分割我们的 PDF 将有助于确保相关文档部分的含义不会被周围文本"冲淡"。

我们可以使用 [文本分割器](/oss/python/integrations/splitters) 来实现这个目的。这里我们将使用一个简单的文本分割器，它基于字符进行分区。我们将文档分割成 1000 个字符的块，块之间有 200 个字符的重叠。重叠有助于防止将陈述与其相关的重要上下文分离。我们使用 `RecursiveCharacterTextSplitter`，它将递归地使用常见分隔符（如换行符）分割文档，直到每个块达到适当的大小。这是通用文本用例的推荐文本分割器。

我们设置 `add_start_index=True`，以便每个分割后的 Document 在初始 Document 中的起始字符索引作为元数据属性"start_index"保留。

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))
```

```text
514
```

## 2. 嵌入

向量搜索是一种存储和搜索非结构化数据（如非结构化文本）的常见方法。其思想是存储与文本关联的数字向量。给定一个查询，我们可以将其 [嵌入](/oss/python/integrations/embeddings) 为相同维度的向量，并使用向量相似性度量（如余弦相似性）来识别相关文本。

LangChain 支持来自 [数十个提供商](/oss/python/integrations/embeddings/) 的嵌入。这些模型指定了如何将文本转换为数字向量。让我们选择一个模型：

<Tabs>
  <Tab title="OpenAI">
    ```shell
    pip install -U "langchain-openai"
    ```

    ```python
    import getpass
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("输入 OpenAI 的 API 密钥：")

    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    ```
  </Tab>

  <Tab title="Azure">
    ```shell
    pip install -U "langchain-openai"
    ```

    ```python
    import getpass
    import os

    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("输入 Azure 的 API 密钥：")

    from langchain_openai import AzureOpenAIEmbeddings

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    ```
  </Tab>

  <Tab title="Google Gemini">
    ```shell
    pip install -qU langchain-google-genai
    ```

    ```python
    import getpass
    import os

    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("输入 Google Gemini 的 API 密钥：")

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    ```
  </Tab>

  <Tab title="Google Vertex">
    ```shell
    pip install -qU langchain-google-vertexai
    ```

    ```python
    from langchain_google_vertexai import VertexAIEmbeddings

    embeddings = VertexAIEmbeddings(model="text-embedding-005")
    ```
  </Tab>

  <Tab title="AWS">
    ```shell
    pip install -qU langchain-aws
    ```

    ```python
    from langchain_aws import BedrockEmbeddings

    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    ```
  </Tab>

  <Tab title="HuggingFace">
    ```shell
    pip install -qU langchain-huggingface
    ```

    ```python
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    ```
  </Tab>

  <Tab title="Ollama">
    ```shell
    pip install -qU langchain-ollama
    ```

    ```python
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model="llama3")
    ```
  </Tab>

  <Tab title="Cohere">
    ```shell
    pip install -qU langchain-cohere
    ```

    ```python
    import getpass
    import os

    if not os.environ.get("COHERE_API_KEY"):
        os.environ["COHERE_API_KEY"] = getpass.getpass("输入 Cohere 的 API 密钥：")

    from langchain_cohere import CohereEmbeddings

    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    ```
  </Tab>

  <Tab title="MistralAI">
    ```shell
    pip install -qU langchain-mistralai
    ```

    ```python
    import getpass
    import os

    if not os.environ.get("MISTRALAI_API_KEY"):
        os.environ["MISTRALAI_API_KEY"] = getpass.getpass("输入 MistralAI 的 API 密钥：")

    from langchain_mistralai import MistralAIEmbeddings

    embeddings = MistralAIEmbeddings(model="mistral-embed")
    ```
  </Tab>

  <Tab title="Nomic">
    ```shell
    pip install -qU langchain-nomic
    ```

    ```python
    import getpass
    import os

    if not os.environ.get("NOMIC_API_KEY"):
        os.environ["NOMIC_API_KEY"] = getpass.getpass("输入 Nomic 的 API 密钥：")

    from langchain_nomic import NomicEmbeddings

    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    ```
  </Tab>

  <Tab title="NVIDIA">
    ```shell
    pip install -qU langchain-nvidia-ai-endpoints
    ```

    ```python
    import getpass
    import os

    if not os.environ.get("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = getpass.getpass("输入 NVIDIA 的 API 密钥：")

    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")
    ```
  </Tab>

  <Tab title="Voyage AI">
    ```shell
    pip install -qU langchain-voyageai
    ```

    ```python
    import getpass
    import os

    if not os.environ.get("VOYAGE_API_KEY"):
        os.environ["VOYAGE_API_KEY"] = getpass.getpass("输入 Voyage AI 的 API 密钥：")

    from langchain-voyageai import VoyageAIEmbeddings

    embeddings = VoyageAIEmbeddings(model="voyage-3")
    ```
  </Tab>

  <Tab title="IBM watsonx">
    ```shell
    pip install -qU langchain-ibm
    ```

    ```python
    import getpass
    import os

    if not os.environ.get("WATSONX_APIKEY"):
        os.environ["WATSONX_APIKEY"] = getpass.getpass("输入 IBM watsonx 的 API 密钥：")

    from langchain_ibm import WatsonxEmbeddings

    embeddings = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="<WATSONX 项目 ID>",
    )
    ```
  </Tab>

  <Tab title="Fake">
    ```shell
    pip install -qU langchain-core
    ```

    ```python
    from langchain_core.embeddings import DeterministicFakeEmbedding

    embeddings = DeterministicFakeEmbedding(size=4096)
    ```
  </Tab>

  <Tab title="Isaacus">
    ```shell
    pip install -qU langchain-isaacus
    ```

    ```python
    import getpass
    import os

    if not os.environ.get("ISAACUS_API_KEY"):
        os.environ["ISAACUS_API_KEY"] = getpass.getpass("输入 Isaacus 的 API 密钥：")

    from langchain_isaacus import IsaacusEmbeddings

    embeddings = IsaacusEmbeddings(model="kanon-2-embedder")
    ```
  </Tab>
</Tabs>

```python
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"生成长度为 {len(vector_1)} 的向量\n")
print(vector_1[:10])
```

```text
生成长度为 1536 的向量

[-0.008586574345827103, -0.03341241180896759, -0.008936782367527485, -0.0036674530711025, 0.010564599186182022, 0.009598285891115665, -0.028587326407432556, -0.015824200585484505, 0.0030416189692914486, -0.012899317778646946]
```

有了生成文本嵌入的模型，我们接下来可以将它们存储在支持高效相似性搜索的特殊数据结构中。

## 3. 向量存储

LangChain [VectorStore](https://reference.langchain.com/python/langchain-core/vectorstores/base/VectorStore) 对象包含将文本和 [`Document`](https://reference.langchain.com/python/langchain-core/documents/base/Document) 对象添加到存储中的方法，并使用各种相似性度量对它们进行查询。它们通常使用 [嵌入](/oss/python/integrations/embeddings) 模型初始化，这些模型决定了如何将文本数据转换为数字向量。

LangChain 包含一套与不同向量存储技术集成的 [集成](/oss/python/integrations/vectorstores)。一些向量存储由提供商托管（例如各种云提供商），需要特定凭据才能使用；一些（如 [Postgres](/oss/python/integrations/vectorstores/pgvector)）在单独的基础设施中运行，可以在本地运行或通过第三方运行；其他可以在内存中运行以处理轻量级工作负载。让我们选择一个向量存储：

<Tabs>
  <Tab title="内存">
    ```shell
    pip install -U "langchain-core"
    ```

    ```python
    from langchain_core.vectorstores import InMemoryVectorStore

    vector_store = InMemoryVectorStore(embeddings)
    ```
  </Tab>

  <Tab title="Amazon OpenSearch">
    ```shell
    pip install -qU  boto3
    ```

    ```python
    from opensearchpy import RequestsHttpConnection

    service = "es"  # 必须将服务设置为'es'
    region = "us-east-2"
    credentials = boto3.Session(
        aws_access_key_id="xxxxxx", aws_secret_access_key="xxxxx"
    ).get_credentials()
    awsauth = AWS4Auth("xxxxx", "xxxxxx", region, service, session_token=credentials.token)

    vector_store = OpenSearchVectorSearch.from_documents(
        docs,
        embeddings,
        opensearch_url="host url",
        http_auth=awsauth,
        timeout=300,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        index_name="test-index",
    )
    ```
  </Tab>

  <Tab title="AstraDB">
    ```shell
    pip install -U "langchain-astradb"
    ```

    ```python
    from langchain_astradb import AstraDBVectorStore

    vector_store = AstraDBVectorStore(
        embedding=embeddings,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name="astra_vector_langchain",
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_NAMESPACE,
    )
    ```
  </Tab>

  <Tab title="Chroma">
    ```shell
    pip install -qU langchain-chroma
    ```

    ```python
    from langchain_chroma import Chroma

    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # 本地保存数据的位置，如果不需要可以移除
    )
    ```
  </Tab>

  <Tab title="FAISS">
    ```shell
    pip install -qU langchain-community faiss-cpu
    ```

    ```python
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS

    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    ```
  </Tab>

  <Tab title="Milvus">
    ```shell
    pip install -qU langchain-milvus
    ```

    ```python
    from langchain_milvus import Milvus

    URI = "./milvus_example.db"

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )
    ```
  </Tab>

  <Tab title="MongoDB">
    ```shell
    pip install -qU langchain-mongodb
    ```

    ```python
    from langchain_mongodb import MongoDBAtlasVectorSearch

    vector_store = MongoDBAtlasVectorSearch(
        embedding=embeddings,
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        relevance_score_fn="cosine",
    )
    ```
  </Tab>

  <Tab title="PGVector">
    ```shell
    pip install -qU langchain-postgres
    ```

    ```python
    from langchain_postgres import PGVector

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="my_docs",
        connection="postgresql+psycopg://...",
    )
    ```
  </Tab>

  <Tab title="PGVectorStore">
    ```shell
    pip install -qU langchain-postgres
    ```

    ```python
    from langchain_postgres import PGEngine, PGVectorStore

    pg_engine = PGEngine.from_connection_string(
        url="postgresql+psycopg://..."
    )

    vector_store = PGVectorStore.create_sync(
        engine=pg_engine,
        table_name='test_table',
        embedding_service=embeddings
    )
    ```
  </Tab>

  <Tab title="Pinecone">
    ```shell
    pip install -qU langchain-pinecone
    ```

    ```python
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    pc = Pinecone(api_key=...)
    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(embedding=embeddings, index=index)
    ```
  </Tab>

  <Tab title="Qdrant">
    ```shell
    pip install -qU langchain-qdrant
    ```

    ```python
    from qdrant_client.models import Distance, VectorParams
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")

    vector_size = len(embeddings.embed_query("sample text"))

    if not client.collection_exists("test"):
        client.create_collection(
            collection_name="test",
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test",
        embedding=embeddings,
    )
    ```
  </Tab>
</Tabs>

实例化向量存储后，我们现在可以索引文档。

```python
ids = vector_store.add_documents(documents=all_splits)
```

请注意，大多数向量存储实现将允许您连接到现有的向量存储——例如，通过提供客户端、索引名称或其他信息。请参阅特定 [集成](/oss/python/integrations/vectorstores) 的文档以获取更多详情。

一旦我们实例化了一个包含文档的 [`VectorStore`](https://reference.langchain.com/python/langchain-core/vectorstores/base/VectorStore)，我们就可以查询它。[VectorStore](https://reference.langchain.com/python/langchain-core/vectorstores/base/VectorStore) 包括以下查询方法：

* 通过字符串查询和向量查询
* 返回或不返回相似性分数
* 通过相似性搜索和 [最大边际相关性](https://reference.langchain.com/python/langchain-core/vectorstores/base/VectorStore/max_marginal_relevance_search)（以平衡查询的相似性与检索结果的多样性）

这些方法的输出通常包含 [Document](https://reference.langchain.com/python/langchain-core/documents/base/Document) 对象列表。

**用法**

嵌入通常将文本表示为"稠密"向量，使得含义相近的文本在几何上接近。这让我们只需传入问题即可检索相关信息，而无需了解文档中使用的任何特定关键术语。

基于与字符串查询的相似性返回文档：

```python
results = vector_store.similarity_search(
    "耐克在美国有多少个分销中心？"
)

print(results[0])
```

```text
page_content='直销业务通过以下数量的零售店在美国销售产品：
美国零售店数量
耐克品牌工厂店 213
耐克品牌直营店（包括仅限员工店）74
匡威店（包括工厂店）82
总计 369
在美国，耐克有八个重要的分销中心。更多信息请参阅项目 2. 物业。
2023 年 10-K 表格 2' metadata={'page': 4, 'source': '../example_data/nke-10k-2023.pdf', 'start_index': 3125}
```

异步查询：

```python
results = await vector_store.asimilarity_search("耐克是什么时候成立的？")

print(results[0])
```

```text
page_content='目录
第一部分
项目 1. 业务
概述
耐克公司于 1967 年根据俄勒冈州法律注册成立。如本年 10-K 表格（本"年度报告"）所用，除非上下文另有说明，否则"我们"、"我们的"、
"耐克"和"公司"等术语指代耐克公司及其前身、子公司和关联公司。
我们的主要业务活动是运动鞋、服装、设备、配件和服务的设计、开发和全球营销销售。耐克是
世界上最大的运动鞋和服装销售商。我们通过耐克直销业务销售产品，包括耐克拥有的零售店
和通过我们的数字平台销售（也称为"耐克品牌数字"），向零售客户和独立分销商、被许可人和销售的混合体销售' metadata={'page': 3, 'source': '../example_data/nke-10k-2023.pdf', 'start_index': 0}
```

返回分数：

```python
# 请注意，不同提供商实现不同的分数；这里的分数
# 是一个与相似性成反比的距离度量。

results = vector_store.similarity_search_with_score("耐克 2023 年的收入是多少？")
doc, score = results[0]
print(f"分数：{score}\n")
print(doc)
```

```text
分数：0.23699893057346344

page_content='目录
2023 财年耐克品牌收入亮点
以下表格展示了按报告经营部门、分销渠道和主要产品线分类的耐克品牌收入：
2023 财年与 2022 财年比较
•耐克公司 2023 财年收入为 512 亿美元，与 2022 财年相比，按报告基础和货币中性基础分别增长 10% 和 16%。
增长是由于北美、欧洲、中东和非洲（"EMEA"）、APLA 和大中华区的收入增加，分别贡献了约 7、6、
2 和 1 个百分点的耐克公司收入。
•耐克品牌收入占耐克公司收入的 90% 以上，按报告基础和货币中性基础分别增长 10% 和 16%。这
一增长主要是由于男士、乔丹品牌、女士和儿童系列的收入增加，按批发
等价基础分别增长 17%、35%、11% 和 10%。' metadata={'page': 35, 'source': '../example_data/nke-10k-2023.pdf', 'start_index': 0}
```

基于与嵌入查询的相似性返回文档：

```python
embedding = embeddings.embed_query("耐克 2023 年的利润率受到什么影响？")

results = vector_store.similarity_search_by_vector(embedding)
print(results[0])
```

```text
page_content='目录
毛利率
2023 财年与 2022 财年比较
2023 财年，我们的合并毛利润增长了 4%，达到 222.92 亿美元，而 2022 财年为 214.79 亿美元。2023 财年的毛利率
下降了 250 个基点，从 2022 财年的 46.0% 降至 43.5%，原因如下：
*批发等价基础
2023 财年毛利率下降主要是由于：
•耐克品牌产品成本增加，按批发等价基础计算，主要原因是投入成本增加以及入境货运和物流成本上升以及
产品组合变化；
•耐克直销业务利润率下降，原因是本期促销活动增加以清理库存，而前期由于可用库存供应减少，促销活动较少；
•外币汇率的不利变化，包括对冲；以及
•按批发等价基础计算的折扣店利润率下降。
这部分被以下因素所抵消：' metadata={'page': 36, 'source': '../example_data/nke-10k-2023.pdf', 'start_index': 0}
```

了解更多：

* [API 参考](https://reference.langchain.com/python/langchain-core/vectorstores/base/VectorStore)
* [特定集成文档](/oss/python/integrations/vectorstores)

## 4. 检索器

LangChain [`VectorStore`](https://reference.langchain.com/python/langchain-core/vectorstores/base/VectorStore) 对象不继承 [Runnable](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable)。LangChain [检索器](https://reference.langchain.com/python/langchain-core/retrievers/BaseRetriever) 是 Runnable，因此它们实现了一组标准方法（例如同步和异步的 `invoke` 和 `batch` 操作）。虽然我们可以从向量存储构建检索器，但检索器也可以与非向量存储的数据源交互（例如外部 API）。

我们可以自己创建一个简单的版本，而无需继承 `Retriever`。如果我们选择希望使用什么方法来检索文档，我们可以轻松创建一个可运行对象。下面我们将围绕 `similarity_search` 方法构建一个：

```python
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


retriever.batch(
    [
        "耐克在美国有多少个分销中心？",
        "耐克是什么时候成立的？",
    ],
)
```

```text
[[Document(metadata={'page': 4, 'source': '../example_data/nke-10k-2023.pdf', 'start_index': 3125}, page_content='直销业务通过以下数量的零售店在美国销售产品：\n美国零售店数量\n耐克品牌工厂店 213 \n耐克品牌直营店（包括仅限员工店）74 \n匡威店（包括工厂店）82 \n总计 369 \n在美国，耐克有八个重要的分销中心。更多信息请参阅项目 2. 物业。\n2023 年 10-K 表格 2')],
 [Document(metadata={'page': 3, 'source': '../example_data/nke-10k-2023.pdf', 'start_index': 0}, page_content='目录\n第一部分\n项目 1. 业务\n概述\n耐克公司于 1967 年根据俄勒冈州法律注册成立。如本年 10-K 表格（本"年度报告"）所用，除非上下文另有说明，否则"我们"、"我们的"、\n"耐克"和"公司"等术语指代耐克公司及其前身、子公司和关联公司。\n我们的主要业务活动是运动鞋、服装、设备、配件和服务的设计、开发和全球营销销售。耐克是\n世界上最大的运动鞋和服装销售商。我们通过耐克直销业务销售产品，包括耐克拥有的零售店\n和通过我们的数字平台销售（也称为"耐克品牌数字"），向零售客户和独立分销商、被许可人和销售的混合体销售')]]
```

向量存储实现了 `as_retriever` 方法，该方法将生成一个检索器，具体是 [`VectorStoreRetriever`](https://reference.langchain.com/python/langchain-core/vectorstores/base/VectorStoreRetriever)。这些检索器包含特定的 `search_type` 和 `search_kwargs` 属性，用于标识要调用的底层向量存储方法，以及如何参数化它们。例如，我们可以使用以下方式复制上面的内容：

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

retriever.batch(
    [
        "耐克在美国有多少个分销中心？",
        "耐克是什么时候成立的？",
    ],
)
```

```text
[[Document(metadata={'page': 4, 'source': '../example_data/nke-10k-2023.pdf', 'start_index': 3125}, page_content='直销业务通过以下数量的零售店在美国销售产品：\n美国零售店数量\n耐克品牌工厂店 213 \n耐克品牌直营店（包括仅限员工店）74 \n匡威店（包括工厂店）82 \n总计 369 \n在美国，耐克有八个重要的分销中心。更多信息请参阅项目 2. 物业。\n2023 年 10-K 表格 2')],
 [Document(metadata={'page': 3, 'source': '../example_data/nke-10k-2023.pdf', 'start_index': 0}, page_content='目录\n第一部分\n项目 1. 业务\n概述\n耐克公司于 1967 年根据俄勒冈州法律注册成立。如本年 10-K 表格（本"年度报告"）所用，除非上下文另有说明，否则"我们"、"我们的"、\n"耐克"和"公司"等术语指代耐克公司及其前身、子公司和关联公司。\n我们的主要业务活动是运动鞋、服装、设备、配件和服务的设计、开发和全球营销销售。耐克是\n世界上最大的运动鞋和服装销售商。我们通过耐克直销业务销售产品，包括耐克拥有的零售店\n和通过我们的数字平台销售（也称为"耐克品牌数字"），向零售客户和独立分销商、被许可人和销售的混合体销售')]]
```

`VectorStoreRetriever` 支持 `"similarity"`（默认）、`"mmr"`（最大边际相关性，如上所述）和 `"similarity_score_threshold"` 搜索类型。我们可以使用后者通过相似性分数对检索器输出的文档进行阈值处理。

检索器可以轻松集成到更复杂的应用程序中，例如 [检索增强生成 (RAG)](/oss/python/langchain/retrieval) 应用程序，该应用程序将给定问题与检索到的上下文结合到 LLM 的提示中。要了解有关构建此类应用程序的更多信息，请查看 [RAG 教程](/oss/python/langchain/rag) 教程。

## 下一步

您现在已经了解了如何基于 PDF 文档构建语义搜索引擎。

有关文档加载器的更多信息：

* [概述](/oss/python/langchain/retrieval)
* [可用集成](/oss/python/integrations/document_loaders/)

有关嵌入的更多信息：

* [概述](/oss/python/langchain/retrieval)
* [可用集成](/oss/python/integrations/embeddings/)

有关向量存储的更多信息：

* [概述](/oss/python/langchain/retrieval)
* [可用集成](/oss/python/integrations/vectorstores/)

有关 RAG 的更多信息，请参阅：

* [构建检索增强生成 (RAG) 应用](/oss/python/langchain/rag/)

***

<div className="source-links">
  <Callout icon="edit">
    [在 GitHub 上编辑此页面](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/knowledge-base.mdx) 或 [提交问题](https://github.com/langchain-ai/docs/issues/new/choose)。
  </Callout>

  <Callout icon="terminal-2">
    [连接这些文档](/use-these-docs) 到 Claude、VSCode 等，通过 MCP 获取实时答案。
  </Callout>
</div>
