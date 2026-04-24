import requests
import time

# 测试健康检查
print("Testing health check...")
r = requests.get('http://localhost:8000/api/health')
print(f"Health check status: {r.status_code}")
print(f"Health check response: {r.json()}")

# 测试文档处理
print("\nTesting document processing...")
start_time = time.time()
r = requests.post('http://localhost:8000/api/process-documents')
end_time = time.time()
print(f"Document processing status: {r.status_code}")
print(f"Document processing time: {end_time - start_time:.2f} seconds")
print(f"Document processing response: {r.json()}")

# 测试查询
print("\nTesting query...")
start_time = time.time()
r = requests.post('http://localhost:8000/api/ask', json={'query': 'Go语言的局部变量分配在栈上还是堆上？', 'top_k': 3})
end_time = time.time()
print(f"Query status: {r.status_code}")
print(f"Query time: {end_time - start_time:.2f} seconds")
print(f"Answer: {r.json()['answer']}")

# 测试另一个查询
print("\nTesting another query...")
start_time = time.time()
r = requests.post('http://localhost:8000/api/ask', json={'query': '如何高效地拼接字符串？', 'top_k': 3})
end_time = time.time()
print(f"Query status: {r.status_code}")
print(f"Query time: {end_time - start_time:.2f} seconds")
print(f"Answer: {r.json()['answer']}")
