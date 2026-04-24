import requests

# 测试Go GC原理问题
def test_gc_principle():
    url = "http://localhost:8000/api/ask"
    payload = {
        "query": "简述 Go GC 原理",
        "top_k": 2
    }

    print(f"发送请求到: {url}")
    print(f"请求体: {payload}")
    response = requests.post(url, json=payload)
    print(f"响应状态码: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("=== 测试：简述 Go GC 原理 ===")
        print("回答:")
        print(result['answer'])
        print("\n引用来源:")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source[:150]}...")
    else:
        print(f"错误: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_gc_principle()