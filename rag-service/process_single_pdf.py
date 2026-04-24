from unified_rag_processor import UnifiedRAGProcessor
import os

# 初始化处理器
processor = UnifiedRAGProcessor()

# 只处理Go语言笔试面试题汇总1-基础语法.pdf
pdf_file = 'Go语言笔试面试题汇总1-基础语法.pdf'
pdf_path = os.path.join(processor.docs_dir, pdf_file)

print(f"Processing: {pdf_file}")
print('='*60)

# 处理文档
result = processor.process_document(pdf_path)
print(f"\nProcessing result: {result}")

# 检查生成的chunks
print(f"\nNumber of chunks: {len(processor.chunks)}")
for i, chunk in enumerate(processor.chunks):
    content = chunk.page_content
    metadata = chunk.metadata
    question = metadata.get('question', '')
    print(f"\nChunk {i+1}:")
    print(f"  Question: {question}")
    print(f"  Content: {content[:100]}...")

# 测试检索
print(f"\n{'='*60}")
print("Testing retrieval for '什么是协程':")
results = processor.retrieve("什么是协程")
print(f"Number of results: {len(results)}")

for i, result in enumerate(results):
    content = result.get("content", result.get("document", str(result)))
    print(f"\nResult {i+1}:")
    print(f"  Content: {content.strip()[:300]}...")