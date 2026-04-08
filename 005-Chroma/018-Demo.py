import os

import chromadb
from chromadb import Documents, EmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
ModelScope_Token = os.getenv("ModelScope_API_KEY")
# 创建自定义的嵌入函数类，使用 Qwen3-Embedding-8B 模型
class QwenEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key=ModelScope_Token,  # ModelScope Token
        )
    
    def __call__(self, input: Documents) -> list[list[float]]:
        response = self.client.embeddings.create(
            model='Qwen/Qwen3-Embedding-8B',  # ModelScope Model-Id, required
            input=input,
            encoding_format="float"
        )
        # 按原始顺序排序（API 可能会重新排序）
        sorted_embeddings = sorted(response.data, key=lambda x: x.index)
        return [embedding.embedding for embedding in sorted_embeddings]

# 使用自定义嵌入函数创建 Chroma 客户端
embed_fn = QwenEmbeddingFunction()
chroma_client = chromadb.Client()

# 创建集合并指定嵌入函数
collection = chroma_client.create_collection(
    name="my_collection",
    embedding_function=embed_fn
)

# 添加文档（会自动使用 Qwen3-Embedding-8B 进行嵌入）
collection.add(
    documents=["This is a document about engineer", "This is a document about steak"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}],
    ids=["id1", "id2"]
)

# 查询（查询文本也会使用相同的嵌入模型）
results = collection.query(
    query_texts=["Which food is the best?"],
    n_results=2
)

print(results)