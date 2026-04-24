import json

# 读取chunks.json文件
with open('data/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print(f"Total chunks: {len(chunks)}")

# 检查是否包含"什么是协程"
counter = 0
for i, chunk in enumerate(chunks):
    content = chunk.get('content', '')
    if "什么是协程" in content:
        counter += 1
        print(f"\nFound in chunk {i+1}:")
        print(f"Content: {content[:300]}...")

print(f"\nTotal chunks containing '什么是协程': {counter}")

# 检查是否包含"协程"但不是"协程泄露"
coroutine_counter = 0
for i, chunk in enumerate(chunks):
    content = chunk.get('content', '')
    if "协程" in content and "协程泄露" not in content:
        coroutine_counter += 1
        print(f"\nFound '协程' (not '协程泄露') in chunk {i+1}:")
        print(f"Content: {content[:300]}...")

print(f"\nTotal chunks containing '协程' (not '协程泄露'): {coroutine_counter}")