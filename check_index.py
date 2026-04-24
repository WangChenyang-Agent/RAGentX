import os
import json

# 检查向量索引和chunks
chunks_path = os.path.join('data', 'chunks.json')

if os.path.exists(chunks_path):
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f'Total chunks: {len(chunks)}')
    print('\nFirst 10 chunks:')
    for i, chunk in enumerate(chunks[:10]):
        content = chunk.get('content', '').strip()[:100]
        print(f'{i+1}. {content}...')
    
    # 搜索包含"默认参数"的chunk
    print('\nChunks containing "默认参数":')
    found = False
    for i, chunk in enumerate(chunks):
        content = chunk.get('content', '')
        if '默认参数' in content or '可选参数' in content:
            print(f'Found in chunk {i+1}:')
            print(content[:200] + '...')
            found = True
    
    if not found:
        print('No chunks found with "默认参数" or "可选参数"')
else:
    print('Chunks file not found')
    print('Current directory:', os.getcwd())
    print('Files in data directory:', os.listdir('data') if os.path.exists('data') else 'data directory not found')