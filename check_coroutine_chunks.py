import json

# 读取chunks.json文件
with open('data/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print(f"Total chunks: {len(chunks)}")

# 查找包含协程定义的块
found = False
for i, chunk in enumerate(chunks):
    content = chunk.get('content', '')
    metadata = chunk.get('metadata', {})
    question = metadata.get('question', '')
    
    # 检查是否包含协程定义
    if '什么是协程' in content or ('协程' in content and 'Goroutine' in content and '泄露' not in content):
        print(f"\nFound potential coroutine definition in chunk {i+1}:")
        print(f"Question: {question}")
        print(f"Content: {content[:300]}...")
        found = True
        break

if not found:
    print("\nNo coroutine definition found in chunks.")
    # 检查所有包含协程的块
    print("\nAll chunks containing '协程':")
    for i, chunk in enumerate(chunks):
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})
        question = metadata.get('question', '')
        if '协程' in content:
            print(f"\nChunk {i+1}:")
            print(f"Question: {question}")
            print(f"Content: {content[:100]}...")