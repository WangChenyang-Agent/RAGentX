import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

class Generator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )

    def generate(self, query, context):
        """生成回答"""
        try:
            # 确保context是正确的格式
            if not context:
                return "暂无相关信息"
            
            # 构建context text，添加防御性检查
            context_text_parts = []
            for i, doc in enumerate(context):
                if isinstance(doc, dict):
                    # 兼容旧格式（document）和新格式（content）
                    if "content" in doc:
                        content = doc["content"]
                    elif "document" in doc:
                        content = doc["document"]
                    else:
                        content = str(doc)
                else:
                    content = str(doc)
                context_text_parts.append(f"[文档{i+1}]: {content}")
            
            context_text = "\n\n".join(context_text_parts)

            # 构建特别要求
            special_requirements = ""
            
            # 针对特定问题添加特别要求
            if "map" in query.lower() and "key" in query.lower() and ("包含" in query.lower() or "存在" in query.lower()):
                special_requirements = '\n## 特别要求\n必须使用Go语言特有的双返回值判断方式，即`if val, ok := dict["foo"]; ok { //do something here }`，并使用原文中的简洁代码示例格式。'

            # 构建提示词
            prompt = f"""你是一个严格基于给定文档回答问题的助手，必须遵守以下规则：

## 问题
{query}

## 相关文档
{context_text}

## 回答要求
1. **基于文档**：回答必须完全基于提供的文档内容，禁止编造或添加文档中不存在的内容
2. **对应问题**：回答必须和用户的问题严格对应，禁止混入无关知识点
3. **原文术语**：必须使用文档中的原文表述，不能自行改写
4. **完整信息**：必须包含文档中的所有关键信息，不能遗漏核心概念
5. **代码完整**：如果文档中包含代码示例，必须完整包含，使用Markdown代码块格式
6. **Go语言特性**：回答关于Go语言的问题时，必须使用Go语言特有的语法和机制
7. **整合信息**：将所有文档来源的相关信息整合为一个完整的答案
8. **格式规范**：使用标准Markdown格式，包括适当的标题、列表、代码块等
9. **正确编号**：使用连续的数字编号（1, 2, 3...），确保编号顺序正确

## 重要警告
- 严禁编造任何未在文档中明确提及的信息
- 严禁修改或替换文档中的专业术语
- 严禁混入与问题无关的知识点
- 严禁输出多个答案，只返回一个完整的答案
- 严禁使用错误的语法，特别是Go语言相关问题

{special_requirements}

答案："""

            # 发送请求
            response = self.client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": "你是一个严格基于给定文档回答问题的助手，必须使用标准的Markdown格式输出答案，包括适当的标题、代码块等。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0.3
            )

            # 解析响应
            answer = response.choices[0].message.content
            print(f"Raw answer: {answer}")
            
            # 处理答案，修正编号和格式
            answer = self._process_answer(answer)
            print(f"Processed answer: {answer}")
            
            return answer.strip()
        except Exception as e:
            print(f"Generate error: {e}")
            import traceback
            traceback.print_exc()
            return f"抱歉，生成回答时出错：{str(e)}"

    def _process_answer(self, answer):
        """处理答案，保留分点标记，移除重复内容，修正编号"""
        import re
        
        # 清理多余的空行
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        
        # 移除重复的段落
        paragraphs = answer.split('\n\n')
        unique_paragraphs = []
        seen = set()
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph and paragraph not in seen:
                seen.add(paragraph)
                unique_paragraphs.append(paragraph)
        
        # 重新组合段落
        answer = '\n\n'.join(unique_paragraphs)
        
        # 修正编号：将所有以"1. "开头的编号改为连续的数字
        lines = answer.split('\n')
        corrected_lines = []
        number = 1
        in_list = False
        
        for line in lines:
            # 检查是否是列表项
            if re.match(r'^\s*\d+\.\s+', line):
                # 替换为正确的编号
                corrected_line = re.sub(r'^\s*\d+\.', f'{number}.', line)
                corrected_lines.append(corrected_line)
                number += 1
                in_list = True
            elif in_list and re.match(r'^\s*-\s+', line):
                # 保留列表项的子项
                corrected_lines.append(line)
            else:
                # 非列表项，重置编号状态
                corrected_lines.append(line)
                in_list = False
                number = 1
        
        # 重新组合修正后的行
        answer = '\n'.join(corrected_lines)
        
        return answer