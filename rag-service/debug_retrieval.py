from unified_rag_processor import UnifiedRAGProcessor

# 初始化处理器
processor = UnifiedRAGProcessor()

# 测试检索
print("Testing retrieval for '什么是协程':")
results = processor.retrieve("什么是协程")
print(f"Number of results: {len(results)}")

for i, result in enumerate(results):
    content = result.get("content", result.get("document", str(result)))
    print(f"Result {i+1}:")
    print(f"  Content: {content.strip()[:300]}...")
    print(f"  Metadata: {result.get('metadata', {})}")
    print()

# 检查所有块
print("\nChecking all chunks:")
counter = 0
for i, chunk in enumerate(processor.chunks):
    if "什么是协程" in chunk.page_content:
        counter += 1
        print(f"Chunk {i+1} contains '什么是协程':")
        print(f"  Content: {chunk.page_content.strip()[:300]}...")
        print()

print(f"Found {counter} chunks containing '什么是协程'")