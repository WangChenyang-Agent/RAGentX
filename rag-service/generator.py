import os
import requests
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    def generate(self, query, context):
        """生成回答"""
        try:
            # 构建prompt
            prompt = f"You are a technical interview assistant. Answer the following question based on the provided context.\n\n"
            prompt += f"Question: {query}\n\n"
            prompt += "Context:\n"
            for doc in context:
                prompt += f"- {doc['document']}\n"
            
            # 调用OpenAI API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a technical interview assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
            
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            # 解析结果
            answer = response.json()["choices"][0]["message"]["content"]
            return answer
        except Exception as e:
            print(f"Generate error: {e}")
            # 降级处理
            return f"Based on the context, the answer to your question '{query}' is..."
