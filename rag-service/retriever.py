import faiss
import numpy as np
import os

class Retriever:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.index = self.load_index()
        self.documents = self.load_documents()
    
    def load_index(self):
        """加载FAISS索引"""
        index_path = "../data/index.faiss"
        if os.path.exists(index_path):
            return faiss.read_index(index_path)
        else:
            # 创建空索引
            dimension = 1536  # qwen3 embedding维度
            return faiss.IndexFlatL2(dimension)
    
    def load_documents(self):
        """加载文档"""
        documents = []
        doc_dir = "../data/docs"
        
        if os.path.exists(doc_dir):
            for filename in os.listdir(doc_dir):
                if filename.endswith((".txt", ".md")):
                    try:
                        with open(os.path.join(doc_dir, filename), "r", encoding="utf-8") as f:
                            content = f.read()
                            documents.append(content)
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
        
        return documents if documents else ["Document 1", "Document 2", "Document 3"]
    
    def retrieve(self, query, k=5):
        """检索相关文档"""
        embedding = self.embedding_service.embed(query)
        if embedding is None:
            return []
        
        # 转换为numpy数组
        embedding_np = np.array([embedding], dtype=np.float32)
        
        # 检索
        distances, indices = self.index.search(embedding_np, k)
        
        # 获取文档
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "distance": distances[0][i]
                })
        
        return results
