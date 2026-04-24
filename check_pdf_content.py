import PyPDF2
import re

# 打开PDF文件
pdf_path = 'data/docs/Go语言笔试面试题汇总1-基础语法.pdf'

with open(pdf_path, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    text = ''
    
    # 提取所有页面的文本
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + '\n'

# 搜索"协程"相关内容
print('Searching for 协程 in PDF:')
matches = re.finditer(r'协程', text)

found = False
for match in matches:
    # 提取匹配位置前后的文本
    start = max(0, match.start() - 100)
    end = min(len(text), match.end() + 200)
    
    print(f'\nMatch found at position {match.start()}:')
    print(text[start:end])
    found = True
    break

if not found:
    print('No matches found for 协程')

# 搜索Q5相关内容
print('\nSearching for Q5 in PDF:')
q5_matches = re.finditer(r'Q5', text)

for match in q5_matches:
    start = max(0, match.start() - 50)
    end = min(len(text), match.end() + 300)
    
    print(f'\nQ5 found at position {match.start()}:')
    print(text[start:end])
    break