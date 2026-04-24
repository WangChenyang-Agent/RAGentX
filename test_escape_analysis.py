import requests

# 测试逃逸分析相关问题
def test_escape_analysis():
    url = "http://localhost:8000/api/ask"
    payload = {
        "query": "函数返回局部变量的指针是否安全？",
        "top_k": 2
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        print("=== 测试：函数返回局部变量的指针是否安全？ ===")
        print("回答:")
        print(result['answer'])
        print("\n引用来源:")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source[:150]}...")
    else:
        print(f"错误: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_escape_analysis()