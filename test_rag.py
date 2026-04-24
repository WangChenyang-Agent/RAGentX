import requests
import json

# 测试RAG系统
def test_rag():
    url = "http://localhost:8000/api/ask"
    payload = {
        "query": "Go语言中的锁有哪些"
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("=== 回答 ===")
        print(result['answer'])
        print("\n=== 引用来源 ===")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source}")
    else:
        print(f"错误: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_rag()