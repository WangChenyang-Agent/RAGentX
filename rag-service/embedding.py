import ollama
from typing import List
from langchain_core.embeddings import Embeddings

class EmbeddingService(Embeddings):
    def __init__(self):
        self.model = "qwen3-embedding:4b"

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

    def embed_query(self, text: str) -> List[float]:
        """生成单个文本的嵌入向量"""
        try:
            response = ollama.embeddings(
                model=self.model,
                prompt=text
            )
            return response["embedding"]
        except Exception as e:
            print(f"Embedding error: {e}")
            return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本的嵌入向量"""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            if embedding:
                embeddings.append(embedding)
        return embeddings

    def embed(self, text):
        """生成文本的嵌入向量"""
        return self.embed_query(text)

    def embed_batch(self, texts):
        """批量生成文本的嵌入向量"""
        return self.embed_documents(texts)
