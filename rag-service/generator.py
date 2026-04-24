import requests
import json

class Generator:
    def __init__(self):
        self.api_key = "sk-9d66c6a185ef4c3bb76d67d8485e7a17"
        self.api_url = "https://api.deepseek.com/v1/chat/completions"

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

            # 构建特别要求和格式要求
            special_requirements = ""
            format_requirement = "请使用标准的Markdown格式输出，包括适当的标题、代码块等。"
            
            # 针对特定问题添加特别要求
            if "map" in query.lower() and "key" in query.lower() and ("包含" in query.lower() or "存在" in query.lower()):
                special_requirements = '\n## 特别要求\n必须使用Go语言特有的双返回值判断方式，即`if val, ok := dict["foo"]; ok { //do something here }`，并使用原文中的简洁代码示例格式。'

            prompt = f"""你是一个严格基于给定文档回答问题的助手，必须遵守以下规则：

## 问题
{query}

## 相关文档
{context_text}

## 回答要求
1. **完全基于文档**：回答必须完全基于提供的文档内容，**禁止编造、篡改术语，禁止添加文档中不存在的内容**
2. **严格对应问题**：回答必须和用户的问题严格对应，**禁止混入无关知识点**
3. **使用原文术语**：必须使用文档中的原文表述，不能自行改写
4. **保留核心术语**：必须完整保留原文中的所有关键专业术语，不能简化或替换
5. **包含完整代码**：如果文档中包含代码示例，**必须完整包含代码示例**，不能省略或简化，并且使用Markdown代码块格式
6. **使用Go语言特有的方式**：回答关于Go语言的问题时，必须使用Go语言特有的语法和机制
7. **整合所有信息**：将所有文档来源的相关信息整合为**一个**完整的答案，**绝对不允许输出多个答案**
8. **回答完整**：必须包含文档中的所有关键信息，不能遗漏核心概念
9. **使用标准Markdown格式**：使用适当的标题、列表、代码块等Markdown元素，保持结构清晰

## 重要警告
- **严禁编造**：任何未在文档中明确提及的信息都视为无效
- **严禁篡改**：不能修改或替换文档中的专业术语
- **严禁串台**：不能混入与问题无关的知识点
- **严禁多答案**：无论有多少个文档来源，只返回一个答案，绝对不允许输出多个答案
- **严禁使用错误的语法**：回答关于Go语言的问题时，必须使用正确的Go语言语法，不能使用其他语言的语法

{special_requirements}

## 回答格式
{format_requirement}

答案："""

            # 构建请求体
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个严格基于给定文档回答问题的助手，必须使用标准的Markdown格式输出答案，包括适当的标题、代码块等。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 1000  # 增加最大token数，确保能包含完整代码
            }

            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # 检查请求是否成功

            # 解析响应
            response_data = response.json()
            answer = response_data["choices"][0]["message"]["content"]
            print(f"Raw answer: {answer}")
            # 处理答案，移除编号列表和重复内容
            answer = self._process_answer(answer)
            print(f"Processed answer: {answer}")
            return answer.strip()
        except Exception as e:
            print(f"Generate error: {e}")
            import traceback
            traceback.print_exc()
            return f"抱歉，生成回答时出错：{str(e)}"

    def _process_answer(self, answer):
        """处理答案，保留分点标记，移除重复内容"""
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
        
        return answer