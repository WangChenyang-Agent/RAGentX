from unified_rag_processor import UnifiedRAGProcessor

# 初始化处理器
processor = UnifiedRAGProcessor()

# 重新构建索引
print("Building index...")
processor.process_folder()
print("Index built successfully")

# 检查块的数量
print(f"Number of chunks: {len(processor.chunks)}")

# 检查前20个块，看看是否包含"Q5什么是协程"或"协程"相关内容
print("\nFirst 20 chunks:")
for i, chunk in enumerate(processor.chunks[:20]):
    content = chunk.page_content
    has_coroutine = "协程" in content
    has_q5 = "Q5" in content
    print(f"Chunk {i+1}: has_coroutine={has_coroutine}, has_q5={has_q5}")
    if has_coroutine or has_q5:
        print(f"  Content: {content[:200]}...")

# 测试检索
print("\nTesting retrieval for '什么是协程':")
results = processor.retrieve("什么是协程")
print(f"Number of results: {len(results)}")

for i, result in enumerate(results):
    content = result.get("content", result.get("document", str(result)))
    print(f"Result {i+1}: {content[:200]}...")