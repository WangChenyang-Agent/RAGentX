import requests

# 测试如何判断map中是否包含某个key
def test_map_key_check():
    url = "http://localhost:8000/api/ask"
    payload = {
        "query": "如何判断map中是否包含某个key？",
        "top_k": 1
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        print("=== 测试：如何判断map中是否包含某个key？ ===")
        print("回答:")
        print(result['answer'])
        print("\n引用来源:")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source[:150]}...")
    else:
        print(f"错误: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_map_key_check()