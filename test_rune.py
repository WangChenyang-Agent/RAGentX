import requests
import json

# 测试RAG系统关于rune类型的回答
def test_rune():
    url = "http://localhost:8000/api/ask"
    payload = {
        "query": "什么是rune类型"
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
    test_rune()