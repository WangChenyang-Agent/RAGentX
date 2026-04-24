import os
import json

# 搜索包含默认参数相关内容的chunk
chunks_path = os.path.join('data', 'chunks.json')

if os.path.exists(chunks_path):
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f'Total chunks: {len(chunks)}')
    
    # 搜索包含相关关键词的chunk
    keywords = ['默认参数', '可选参数', 'Q9']
    print('\nSearching for chunks with keywords:', keywords)
    
    found_chunks = []
    for i, chunk in enumerate(chunks):
        content = chunk.get('content', '')
        for keyword in keywords:
            if keyword in content:
                found_chunks.append((i+1, content))
                break
    
    if found_chunks:
        print(f'Found {len(found_chunks)} chunks:')
        for chunk_idx, content in found_chunks:
            print(f'\nChunk {chunk_idx}:')
            print(content[:300] + '...')
    else:
        print('No chunks found with the keywords')
else:
    print('Chunks file not found')