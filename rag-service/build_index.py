# build_index.py
import os
import faiss
import numpy as np
from embedding import EmbeddingService

def build_index():
    embedding_service = EmbeddingService()
    documents = []
    doc_dir = "../data/docs"
    
    # 读取文档
    for filename in os.listdir(doc_dir):
        if filename.endswith((".txt", ".md")):
            with open(os.path.join(doc_dir, filename), "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(content)
    
    # 生成嵌入
    embeddings = embedding_service.embed_batch(documents)
    
    # 创建FAISS索引
    dimension = len(embeddings[0]) if embeddings else 1536
    index = faiss.IndexFlatL2(dimension)
    
    # 添加向量
    if embeddings:
        index.add(np.array(embeddings, dtype=np.float32))
    
    # 保存索引
    faiss.write_index(index, "../data/index.faiss")
    print(f"Index built with {len(documents)} documents")

if __name__ == "__main__":
    build_index()