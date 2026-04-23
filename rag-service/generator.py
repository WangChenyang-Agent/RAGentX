import ollama

class Generator:
    def __init__(self):
        self.model = "deepseek-r1:1.5b"
    
    def generate(self, query, context):
        """生成回答"""
        try:
            # 构建prompt
            prompt = f"You are a technical interview assistant. Answer the following question based on the provided context.\n\n"
            prompt += f"Question: {query}\n\n"
            prompt += "Context:\n"
            for doc in context:
                prompt += f"- {doc['document']}\n"
            
            # 调用Ollama本地模型
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.7}
            )
            
            # 解析结果
            answer = response["response"]
            return answer
        except Exception as e:
            print(f"Generate error: {e}")
            # 降级处理
            return f"Based on the context, the answer to your question '{query}' is..."
