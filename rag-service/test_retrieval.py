from unified_rag_processor import UnifiedRAGProcessor

# 初始化处理器
processor = UnifiedRAGProcessor()

# 测试检索
print('Testing retrieval for "什么是协程":')
results = processor.retrieve("什么是协程")
print(f'Number of results: {len(results)}')

# 输出前3个结果
for i, result in enumerate(results[:3]):
    content = result.get("content", result.get("document", str(result)))
    print(f"\nResult {i+1}:")
    print(f"  Content: {content.strip()[:500]}...")