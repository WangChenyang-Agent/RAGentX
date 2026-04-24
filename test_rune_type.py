import requests

# 测试什么是rune类型
def test_rune_type():
    url = "http://localhost:8000/api/ask"
    payload = {
        "query": "什么是rune类型？",
        "top_k": 1
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        print("=== 测试：什么是rune类型？ ===")
        print("回答:")
        print(result['answer'])
        print("\n引用来源:")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source[:150]}...")
    else:
        print(f"错误: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_rune_type()