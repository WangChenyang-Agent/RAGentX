import ollama

class EmbeddingService:
    def __init__(self):
        self.model = "qwen3-embedding:4b"
    
    def embed(self, text):
        """生成文本的嵌入向量"""
        try:
            response = ollama.embeddings(
                model=self.model,
                prompt=text
            )
            return response["embedding"]
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def embed_batch(self, texts):
        """批量生成文本的嵌入向量"""
        embeddings = []
        for text in texts:
            embedding = self.embed(text)
            if embedding:
                embeddings.append(embedding)
        return embeddings
