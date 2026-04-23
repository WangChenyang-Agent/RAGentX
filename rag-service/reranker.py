import ollama

class Reranker:
    def __init__(self):
        self.model = "bona/bge-reranker-v2-m3"
    
    def rerank(self, query, documents, top_n=3):
        """对检索结果进行重排序"""
        try:
            # 构建重排序请求
            prompt = f"Query: {query}\n\n"
            for i, doc in enumerate(documents):
                prompt += f"Document {i+1}: {doc['document']}\n"
            
            # 调用reranker模型
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1}
            )
            
            # 解析结果（简化实现）
            # 实际项目中应该根据模型输出进行更复杂的解析
            reranked = documents[:top_n]
            return reranked
        except Exception as e:
            print(f"Rerank error: {e}")
            return documents[:top_n]
