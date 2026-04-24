import requests
import time

# 测试健康检查
print("Testing health check...")
r = requests.get('http://localhost:8000/api/health')
print(f"Health check status: {r.status_code}")
print(f"Health check response: {r.json()}")

# 测试查询
print("\nTesting query...")
start_time = time.time()
r = requests.post('http://localhost:8000/api/ask', json={'query': '什么是rune类型？', 'top_k': 3})
end_time = time.time()
print(f"Query status: {r.status_code}")
print(f"Query time: {end_time - start_time:.2f} seconds")
print(f"Answer: {r.json()['answer']}")
