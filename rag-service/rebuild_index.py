from unified_rag_processor import UnifiedRAGProcessor

# 初始化处理器
processor = UnifiedRAGProcessor()

# 重新构建索引
print("Building index...")
processor.process_folder()
print("Index built successfully")

# 测试检索
print("\nTesting retrieval for '什么是协程':")
results = processor.retrieve("什么是协程")
print(f"Number of results: {len(results)}")

for i, result in enumerate(results):
    content = result.get("content", result.get("document", str(result)))
    print(f"Result {i+1}: {content[:300]}...")