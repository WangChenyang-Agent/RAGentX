import requests
import json

# 测试默认参数问题
def test_default_parameters():
    url = "http://localhost:8000/api/ask"
    payload = {
        "query": "Go 支持默认参数或可选参数吗？"
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("=== 测试：Go 支持默认参数或可选参数吗？ ===")
        print("回答:")
        print(result['answer'])
        print("\n引用来源:")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source[:150]}...")
    else:
        print(f"错误: {response.status_code}")
        print(response.text)

# 测试锁问题（确保仍然正常工作）
def test_locks():
    url = "http://localhost:8000/api/ask"
    payload = {
        "query": "Go语言中的锁有哪些"
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("\n=== 测试：Go语言中的锁有哪些 ===")
        print("回答:")
        print(result['answer'])
        print("\n引用来源:")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source[:150]}...")
    else:
        print(f"错误: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_default_parameters()
    test_locks()