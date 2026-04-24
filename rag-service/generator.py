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

            # 提取问题关键词，用于严格过滤
            query_keywords = query.lower()
            is_lock_query = "锁" in query or "lock" in query or "mutex" in query

            # 构建特别要求和格式要求
            special_requirements = ""
            format_requirement = "直接给出答案，不要使用任何引导语，只列出文档中明确提到的要点。"
            
            if is_lock_query:
                special_requirements = """## 特别要求（针对本问题）
- **只回答"锁"相关的内容**：sync.Mutex、sync.RWMutex等同步原语
- **不回答其他内容**：channel、map、goroutine等不是"锁"，不要提及
- **答案简洁**：文档中提到几个锁，就回答几个，不要扩展
"""
                format_requirement = "直接给出答案，不要使用任何引导语，只列出文档中明确提到的锁类型。"

            prompt = f"""你是一个严格的技术问答助手，**必须完全基于提供的文档内容回答问题**。

## 问题
{query}

## 相关文档
{context_text}

## 回答要求（必须严格遵守）
1. **严格基于文档**：只使用文档中明确提到的信息，**不得使用任何文档外的知识**
2. **逐字核对**：确保每个答案点都能在文档中找到**精确对应**的内容
3. **拒绝幻觉**：如果文档中没有相关信息，**直接回答"暂无相关信息"**，不要猜测
4. **简洁精准**：只列出文档中明确提到的要点，**不要添加任何解释或扩展**
5. **格式规范**：使用编号列表，每个要点直接对应文档中的内容

## 重要警告
- **严禁编造**：任何未在文档中明确提及的信息都视为无效
- **严禁扩展**：只回答问题本身，不要添加背景、原理或其他无关内容
- **严禁臆测**：不要对文档内容进行任何推理或演绎

{special_requirements}

## 回答格式
{format_requirement}

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