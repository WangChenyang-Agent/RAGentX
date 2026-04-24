from unified_rag_processor import UnifiedRAGProcessor

# 初始化处理器
processor = UnifiedRAGProcessor()

# 读取测试文件
with open('data/formatted/Go语言笔试面试题汇总1-基础语法.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 测试预处理
print("Testing _preprocess_qa_text...")
processed = processor._preprocess_qa_text(content)
print(f"Processed text length: {len(processed)}")
print(f"First 1000 chars: {processed[:1000]}...")

# 检查是否包含"什么是协程"
if "什么是协程" in processed:
    print("Found '什么是协程' in processed text")
else:
    print("NOT found '什么是协程' in processed text")

# 测试创建块
print("\nTesting _create_semantic_chunks...")
chunks = processor._create_semantic_chunks(content, "test.md")
print(f"Created {len(chunks)} chunks")

# 检查块内容
for i, chunk in enumerate(chunks):
    if "协程" in chunk.page_content:
        print(f"\nChunk {i+1} contains '协程':")
        print(f"Content: {chunk.page_content[:300]}...")
        print(f"Metadata: {chunk.metadata}")