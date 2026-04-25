import ollama
import re

class Reranker:
    def __init__(self):
        self.model = "bona/bge-reranker-v2-m3"
    
    def rerank(self, query, documents, top_n=5):
        """对检索结果进行重排序"""
        try:
            # 限制处理的文档数量
            documents = documents[:10]  # 只处理前10个文档
            
            # 构建重排序请求
            prompt = f"Query: {query}\n\n"
            prompt += "Please rank the following documents in order of relevance to the query.\n"
            prompt += "Consider both semantic relevance and context matching.\n"
            prompt += "Return only the document numbers separated by commas, with the most relevant first.\n\n"
            
            for i, doc in enumerate(documents):
                # 限制每个文档的长度
                if isinstance(doc, dict):
                    # 兼容旧格式（document）和新格式（content）
                    if "content" in doc:
                        doc_content = doc['content'][:600]  # 限制为600字符
                    elif "document" in doc:
                        doc_content = doc['document'][:600]  # 限制为600字符
                    else:
                        doc_content = str(doc)[:600]
                else:
                    doc_content = str(doc)[:600]
                prompt += f"Document {i+1}: {doc_content}\n\n"
            
            # 调用reranker模型
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1, "max_tokens": 100}
            )
            
            # 解析结果
            response_text = response["response"]
            # 提取数字
            ranks = re.findall(r'\d+', response_text)
            
            if ranks:
                # 转换为索引
                ranked_indices = [int(r) - 1 for r in ranks if 0 <= int(r) - 1 < len(documents)]
                # 去重并保持顺序
                seen = set()
                ranked_indices = [i for i in ranked_indices if not (i in seen or seen.add(i))]
                
                # 构建重排序后的结果
                reranked = []
                for i in ranked_indices[:top_n]:
                    if i < len(documents):
                        reranked.append(documents[i])
                
                # 如果重排序结果不足，使用原始排序补充
                if len(reranked) < top_n:
                    for doc in documents:
                        if doc not in reranked:
                            reranked.append(doc)
                            if len(reranked) >= top_n:
                                break
                
                return reranked[:top_n]
            else:
                # 如果解析失败，使用原始排序
                return documents[:top_n]
        except Exception as e:
            print(f"Rerank error: {e}")
            # 发生错误时，返回原始排序的前top_n个文档
            return documents[:top_n]
