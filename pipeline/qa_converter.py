import json

# 读取原始文件
with open('data/formatted/go面试资料整理(24页-带答案).json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取内容
sections = data['content']['sections']

# 合并所有文本
all_texts = []
for section in sections:
    if 'content' in section:
        all_texts.extend(section['content'])

# 定义分类
categories = {
    'go语言基础': [],
    'go语言进阶': [],
    'mysql': [],
    'kafka': [],
    'redis': []
}

# 当前分类
current_category = None

# 问题和答案缓冲区
current_question = None
current_answer = []

# 处理文本，按照分类和QA结构组织
for text in all_texts:
    text = text.strip()
    if not text:
        continue
    
    # 检查是否是分类标题
    if text in categories.keys():
        # 保存当前缓冲区的内容
        if current_question and current_answer:
            if current_category:
                answer_text = '\n'.join(current_answer)
                categories[current_category].append({
                    'type': 'qa',
                    'question': current_question,
                    'answer': answer_text
                })
        # 重置缓冲区
        current_question = None
        current_answer = []
        current_category = text
    elif text.endswith('？') or text.endswith('?'):
        # 保存当前缓冲区的内容
        if current_question and current_answer:
            if current_category:
                answer_text = '\n'.join(current_answer)
                categories[current_category].append({
                    'type': 'qa',
                    'question': current_question,
                    'answer': answer_text
                })
        # 开始新的问题
        current_question = text
        current_answer = []
    else:
        # 累积答案内容
        if current_question:
            current_answer.append(text)

# 保存最后一个问题
if current_question and current_answer and current_category:
    answer_text = '\n'.join(current_answer)
    categories[current_category].append({
        'type': 'qa',
        'question': current_question,
        'answer': answer_text
    })

# 构建新的JSON结构
new_sections = []
q_count = 1

for category, qa_pairs in categories.items():
    # 添加分类标题
    new_sections.append({
        'type': 'section',
        'title': category,
        'content': []
    })
    
    # 添加QA条目
    for qa in qa_pairs:
        new_sections.append({
            'type': 'qa',
            'question': f"Q{q_count}: {qa['question']}",
            'answer': qa['answer']
        })
        q_count += 1

# 构建最终数据
new_data = {
    "source": data["source"],
    "processed_at": data["processed_at"],
    "content": {
        "sections": new_sections
    }
}

# 保存转换后的文件
with open('data/formatted/go面试资料整理(24页-带答案)-qa.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

# 打印统计信息
total_qa = sum(len(pairs) for pairs in categories.values())
print(f"转换完成！")
print(f"总QA数量: {total_qa}")
for category, pairs in categories.items():
    print(f"{category}: {len(pairs)}个问题")
