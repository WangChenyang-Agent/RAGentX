import ollama

class Generator:
    def __init__(self):
        self.model = "deepseek-r1:1.5b"

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

            prompt = f"""你是一个严格的技术问答助手。

## 问题
{query}

## 相关文档
{context_text}

## 回答要求（必须严格遵守）
1. **只回答问题本身涉及的知识点**，不要扩展到其他无关内容
2. **简洁明了**，只列出关键特点、原理或答案，不需要冗长的背景介绍
3. **严格基于提供的文档回答**，不要编造或推测文档外的信息
4. **如果文档中没有相关信息，直接回答"暂无相关信息"**
5. **格式清晰**，使用要点或编号列表

## 回答格式
直接给出答案，不要说"根据文档"、"在本文中"等引导语。

答案："""

            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.0, "max_tokens": 500}
            )

            answer = response["response"]
            return answer.strip()
        except Exception as e:
            print(f"Generate error: {e}")
            import traceback
            traceback.print_exc()
            return f"抱歉，生成回答时出错：{str(e)}"