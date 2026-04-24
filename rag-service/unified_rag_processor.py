import os
import json
import asyncio
import re
from typing import List, Dict, Optional, Any, Callable
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from embedding import EmbeddingService
from generator import Generator

try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
except ImportError:
    print("Marker not available, will use fallback method")
    MARKER_AVAILABLE = False

# 强制禁用Marker，使用PyPDF2（避免下载大模型和权限问题）
FORCE_DISABLE_MARKER = True

try:
    from raganything.modalprocessors import ImageModalProcessor, TableModalProcessor
    RAG_ANYTHING_AVAILABLE = True
except ImportError:
    print("RAG-Anything multimodal processors not available")
    RAG_ANYTHING_AVAILABLE = False

project_root = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(project_root, ".marker_cache")
os.makedirs(cache_dir, exist_ok=True)

os.environ["XDG_CACHE_HOME"] = cache_dir
os.environ["MARKER_CACHE_HOME"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["TORCH_HOME"] = cache_dir


class UnifiedRAGProcessor:
    """统一的RAG处理器：LangChain + Marker + 部分RAG-Anything"""

    def __init__(
        self,
        docs_dir: str = None,
        output_dir: str = None,
        index_dir: str = None,
        chunk_size: int = 1000,  # 增大切块大小
        chunk_overlap: int = 200  # 增大重叠部分
    ):
        if docs_dir is None:
            docs_dir = os.path.join(project_root, "..", "data", "docs")
        if output_dir is None:
            output_dir = os.path.join(project_root, "..", "data", "formatted")
        if index_dir is None:
            index_dir = os.path.join(project_root, "..", "data")

        self.docs_dir = docs_dir
        self.output_dir = output_dir
        self.index_dir = index_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        os.makedirs(self.output_dir, exist_ok=True)

        self.embedding_service = EmbeddingService()
        self.generator = Generator()

        self.vectorstore = None
        self.chunks = []
        self.markdown_files = []

        self.marker_models = None
        self.marker_converter = None
        self._marker_initialized = False

        # 不再这里初始化Marker，改为延迟加载
        # self._initialize_marker()
        self._load_existing_index()

    def _ensure_marker_initialized(self):
        """延迟初始化Marker模型"""
        if self._marker_initialized:
            return

        if FORCE_DISABLE_MARKER:
            print("Marker is disabled by configuration, using PyPDF2 fallback")
            self._marker_initialized = True
            return

        if not MARKER_AVAILABLE:
            print("Marker package not installed")
            self._marker_initialized = True
            return

        if self.marker_models is None:
            print("Initializing Marker models (lazy)...")
            try:
                self.marker_models = create_model_dict()
                self.marker_converter = PdfConverter(artifact_dict=self.marker_models)
                print("Marker models initialized successfully!")
            except Exception as e:
                print(f"Failed to initialize Marker models: {e}")
                print("Will use fallback PDF processing")
                self.marker_models = None
                self.marker_converter = None

        self._marker_initialized = True

    def _convert_pdf_to_markdown_fallback(self, pdf_path: str) -> str:
        """使用PyPDF2的备用转换方法"""
        print("Using fallback PDF text extraction...")

        try:
            import PyPDF2
            import re

            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"

            # 1. 清理文本
            text = text.replace('\r', '')
            text = re.sub(r'[\u00a0\u200b\u200c\u200d\u2060\ufeff]', ' ', text)
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)  # 清理连续的空行
            
            print(f"Extracted text length: {len(text)}")
            print(f"First 500 chars: {text[:500]}...")
            
            # 2. 结构化Q&A
            result = []
            
            # 查找所有Q1, Q2, Q3...的位置
            q_pattern = re.compile(r'Q\d+')
            matches = list(q_pattern.finditer(text))
            
            print(f"Found {len(matches)} Q patterns")
            
            if matches:
                # 处理每个Q&A块
                for i, match in enumerate(matches):
                    q_start = match.start()
                    q_text = match.group()
                    
                    # 找到下一个Q的位置
                    if i < len(matches) - 1:
                        next_q_start = matches[i + 1].start()
                    else:
                        next_q_start = len(text)
                    
                    # 提取Q&A块
                    qa_block = text[q_start:next_q_start].strip()
                    
                    # 分离问题和答案
                    # 模式1: Q1: 问题内容\n答案内容
                    # 模式2: Q1\n问题内容\n答案内容
                    # 模式3: Q1 问题内容\n答案内容
                    q_match = re.search(r'Q\d+[:\s]*(.+?)(?:\n\s*答案\s*\n|\n\s*)(.+)$', qa_block, re.DOTALL)
                    if q_match:
                        question = q_match.group(1).strip()
                        answer = q_match.group(2).strip()
                    else:
                        # 简单分割
                        parts = qa_block.split('\n', 1)
                        if len(parts) == 2:
                            question = parts[0].replace(q_text, '').strip()
                            # 提取答案，跳过可能的"答案"行
                            answer_parts = parts[1].split('\n', 1)
                            if len(answer_parts) == 2 and answer_parts[0].strip() == '答案':
                                answer = answer_parts[1].strip()
                            else:
                                answer = parts[1].strip()
                        else:
                            question = qa_block.replace(q_text, '').strip()
                            answer = ""
                    
                    if question:
                        # 提取标签（根据问题内容）
                        tags = self._extract_tags(question)
                        
                        # 格式化输出
                        result.append("【问题】" + question)
                        result.append("\n【答案】")
                        result.append(answer)
                        if tags:
                            result.append("\n【标签】")
                            result.append(' / '.join(tags))
                        result.append('\n')
                        print(f"Added structured Q&A: {question[:50]}...")
            else:
                # 没有找到Q格式，尝试其他模式
                # 查找"问题"、"问："等标记
                other_patterns = [
                    r'问题[：:](.+?)(?:答案[：:]|\n)(.+?)(?=问题|$)',
                    r'问[：:](.+?)(?:答[：:]|\n)(.+?)(?=问|$)'
                ]
                
                for pattern in other_patterns:
                    matches = re.finditer(pattern, text, re.DOTALL)
                    for match in matches:
                        question = match.group(1).strip()
                        answer = match.group(2).strip()
                        tags = self._extract_tags(question)
                        
                        result.append("【问题】" + question)
                        result.append("\n【答案】")
                        result.append(answer)
                        if tags:
                            result.append("\n【标签】")
                            result.append(' / '.join(tags))
                        result.append('\n')
                
                if not result:
                    # 仍然没有找到，返回原文本
                    result.append(text.strip())
                    print("No Q&A patterns found, returning original text")
            
            formatted_text = '\n'.join(result)
            print(f"Formatted text length: {len(formatted_text)}")
            print(f"First 500 chars of formatted text: {formatted_text[:500]}...")
            
            return formatted_text

        except Exception as e:
            print(f"Fallback extraction error: {e}")
            # 即使出错，也尝试返回清理后的文本
            try:
                return text.strip()
            except:
                return ""

    def _extract_tags(self, question: str) -> List[str]:
        """从问题中提取标签"""
        tags = []
        
        # 技术领域标签
        tech_tags = {
            'Golang': ['go', 'golang'],
            'Redis': ['redis'],
            'MySQL': ['mysql'],
            'MongoDB': ['mongodb'],
            'Linux': ['linux'],
            '网络': ['网络', 'tcp', 'ip', 'http', 'https'],
            '并发': ['并发', 'goroutine', 'channel', '锁', 'mutex'],
            '分布式': ['分布式', '微服务', '集群'],
            '算法': ['算法', '数据结构'],
            '操作系统': ['操作系统', 'os']
        }
        
        # 问题类型标签
        type_tags = {
            '基础': ['什么是', '如何', '怎样', '原理', '定义'],
            '进阶': ['底层', '源码', '实现', '原理'],
            '面经': ['面试', '面经', '问题', '答案']
        }
        
        # 提取技术标签
        question_lower = question.lower()
        for tag, keywords in tech_tags.items():
            if any(keyword in question_lower for keyword in keywords):
                tags.append(tag)
        
        # 提取问题类型标签
        for tag, keywords in type_tags.items():
            if any(keyword in question for keyword in keywords):
                tags.append(tag)
        
        # 去重
        return list(set(tags))
    
    def _extract_topic(self, text: str) -> str:
        """从文本中提取主题"""
        import re
        
        # 常见技术主题
        topics = {
            'Golang': ['go', 'golang', 'goroutine', 'channel', 'mutex'],
            'Redis': ['redis', '缓存', 'key-value'],
            'MySQL': ['mysql', '数据库', 'sql'],
            'MongoDB': ['mongodb', 'nosql'],
            'Linux': ['linux', '操作系统', 'shell'],
            '网络': ['网络', 'tcp', 'ip', 'http', 'https'],
            '并发': ['并发', '多线程', '同步'],
            '分布式': ['分布式', '微服务', '集群'],
            '算法': ['算法', '数据结构'],
            '操作系统': ['操作系统', 'os', '进程', '线程']
        }
        
        # 从文本中提取主题
        text_lower = text.lower()
        for topic, keywords in topics.items():
            if any(keyword in text_lower for keyword in keywords):
                return topic
        
        # 提取问题中的主要技术词汇
        tech_keywords = re.findall(r'[A-Za-z]+[0-9]*', text)
        if tech_keywords:
            return tech_keywords[0]
        
        return "通用技术"

    def _clean_text(self, text: str) -> str:
        """清理提取的文本，保留换行符"""
        # 只替换连续的空格，保留换行符
        text = re.sub(r'[ \t]+', ' ', text)
        # 清理中文前后的空格
        text = re.sub(r' (?=[\u4e00-\u9fa5])', '', text)
        text = re.sub(r'(?<=[\u4e00-\u9fa5]) ', '', text)
        # 清理连续的空行
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _enhance_markdown_structure(self, text: str) -> str:
        """增强Markdown结构，专门针对Q&A格式优化"""
        if not text:
            return text

        # 1. 清理和预处理
        text = text.replace('\r', '')  # 移除回车符
        text = re.sub(r'[\u00a0\u200b\u200c\u200d\u2060\ufeff]', ' ', text)  # 移除零宽字符
        
        # 2. 先处理Q&A格式
        text = self._process_qa_format(text)
        
        # 3. 处理特殊内容（如rune类型）
        text = re.sub(r'([^\n])(rune\s+绫诲瀷|rune\s+type)', r'\1\n\n### \2', text)
        
        # 4. 增强段落分隔
        # 确保段落之间有足够的空行
        enhanced_text = re.sub(r'\n([^\n])', r'\n\n\1', text)
        # 清理多余的空行
        enhanced_text = re.sub(r'\n{3,}', '\n\n', enhanced_text)
        # 确保文件开头没有空行
        enhanced_text = enhanced_text.lstrip()
        
        return enhanced_text
    
    def _process_qa_format(self, text: str) -> str:
        """处理Q&A格式，将连续文本分割成结构化Markdown"""
        if not text:
            return text
        
        # 1. 清理文本
        text = text.replace('\r', '')
        text = re.sub(r'[\u00a0\u200b\u200c\u200d\u2060\ufeff]', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 2. 直接分割Q&A
        # 使用正则表达式分割Q1, Q2, Q3...
        parts = re.split(r'(Q\d+)', text)
        
        if len(parts) < 3:
            # 没有找到Q格式，尝试其他格式
            return text
        
        # 3. 构建结构化Markdown
        structured_text = []
        
        # 添加标题部分（如果有）
        if parts[0].strip():
            structured_text.append(parts[0].strip())
            structured_text.append('')
        
        # 处理每个Q&A块
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                q_text = parts[i]
                content = parts[i + 1]
                structured_text.append(f"### {q_text}{content}")
                structured_text.append('')
        
        result = '\n'.join(structured_text)
        return result

    def _convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """将PDF转换为Markdown"""
        print(f"Converting PDF to Markdown: {pdf_path}")

        self._ensure_marker_initialized()

        if self.marker_converter is None:
            print("Marker not available, using fallback method...")
            return self._convert_pdf_to_markdown_fallback(pdf_path)

        try:
            rendered = self.marker_converter(pdf_path)
            markdown_text = text_from_rendered(rendered)
            
            # 增强Markdown结构化
            markdown_text = self._enhance_markdown_structure(markdown_text)
            return markdown_text

        except Exception as e:
            print(f"Marker conversion failed: {e}")
            print("Falling back to basic text extraction...")
            # 备用方法已经处理了结构化
            return self._convert_pdf_to_markdown_fallback(pdf_path)

    def _save_markdown(self, markdown_text: str, output_filename: str) -> str:
        """保存Markdown文件"""
        output_path = os.path.join(self.output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)

        return output_path

    def _format_document_to_json(self, markdown_text: str, source_filename: str) -> Dict[str, Any]:
        """将文档格式化为JSON键值对形式"""
        import re
        import time
        
        json_structure = {
            "source": source_filename,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "content": {
                "sections": []
            }
        }
        
        # 处理Q&A格式
        qa_pattern = re.compile(r'Q\d+|【问题】', re.IGNORECASE)
        if qa_pattern.search(markdown_text):
            # 分割Q&A块
            parts = re.split(r'(Q\d+|【问题】)', markdown_text)
            
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    question = parts[i].strip()
                    answer = parts[i + 1].strip()
                    
                    # 清理问题和答案
                    if question.startswith('Q'):
                        question = question
                    elif question == '【问题】':
                        # 提取问题内容
                        q_match = re.search(r'(.+?)[\n\r]', answer)
                        if q_match:
                            question = q_match.group(1).strip()
                            answer = answer[q_match.end():].strip()
                    
                    if answer:
                        json_structure["content"]["sections"].append({
                            "type": "qa",
                            "question": question,
                            "answer": answer
                        })
        else:
            # 处理普通文档格式，按标题分割
            lines = markdown_text.split('\n')
            current_section = {
                "type": "section",
                "title": "",
                "content": []
            }
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 检查是否是标题
                if line.startswith('#'):
                    # 保存当前section
                    if current_section["title"] or current_section["content"]:
                        json_structure["content"]["sections"].append(current_section)
                    
                    # 提取标题和级别
                    title_level = len(line) - len(line.lstrip('#'))
                    title_text = line.lstrip('#').strip()
                    
                    current_section = {
                        "type": "section",
                        "title": title_text,
                        "level": title_level,
                        "content": []
                    }
                else:
                    current_section["content"].append(line)
            
            # 保存最后一个section
            if current_section["title"] or current_section["content"]:
                json_structure["content"]["sections"].append(current_section)
        
        return json_structure

    def _save_json(self, json_data: Dict[str, Any], output_filename: str) -> str:
        """保存JSON文件"""
        output_path = os.path.join(self.output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        return output_path

    def _create_semantic_chunks(self, markdown_text: str, source: str = "") -> List[Document]:
        """创建语义化的文本块，针对技术文档和Q&A格式优化"""
        import re
        
        # 1. 预处理文本
        processed_text = self._preprocess_qa_text(markdown_text)
        
        # 2. 按Q&A块分割（优先保证Q&A完整性）
        # 查找所有【问题】标记的位置
        q_pattern = re.compile(r'\n?【问题】')
        q_matches = list(q_pattern.finditer(processed_text))  # 查找所有【问题】标记
        
        chunks = []
        
        if q_matches:
            print("Splitting by Q&A blocks...")
            print(f"Found {len(q_matches)} Q&A blocks")
            # 处理每个Q&A块
            for i, match in enumerate(q_matches):
                q_start = match.start()
                
                # 找到下一个Q的位置
                if i < len(q_matches) - 1:
                    next_q_start = q_matches[i + 1].start()
                else:
                    next_q_start = len(processed_text)
                
                # 提取完整的Q&A块
                qa_block = processed_text[q_start:next_q_start].strip()
                
                # 确保块包含完整的Q&A结构
                if "【问题】" in qa_block:
                    # 提取问题、答案和标签
                    question_match = re.search(r'【问题】(.+?)(?=【答案】|$)', qa_block, re.DOTALL)
                    answer_match = re.search(r'【答案】(.+?)(?=【标签】|$)', qa_block, re.DOTALL)
                    tags_match = re.search(r'【标签】(.+)', qa_block, re.DOTALL)
                    
                    question = question_match.group(1).strip() if question_match else ""
                    answer = answer_match.group(1).strip() if answer_match else ""
                    tags = tags_match.group(1).strip() if tags_match else ""
                    
                    # 提取标签列表
                    tag_list = [tag.strip() for tag in tags.split('/')] if tags else []
                    
                    # 提取主题
                    topic = self._extract_topic(question)
                    
                    # 构建主chunk的metadata
                    main_metadata = {
                        "source": source,
                        "type": "qa",
                        "topic": topic,
                        "tags": tag_list,
                        "question": question,
                        "chunk_level": "main"
                    }
                    
                    # 添加主QA块（完整的Q&A）
                    chunks.append(Document(page_content=qa_block, metadata=main_metadata))
                    
                    # 3. 对answer进行细粒度切片（子chunk）
                    if answer:
                        # 构建子chunk的metadata
                        sub_metadata = {
                            "source": source,
                            "type": "qa_detail",
                            "topic": topic,
                            "tags": tag_list,
                            "question": question,
                            "chunk_level": "sub"
                        }
                        
                        # 按语义分割answer
                        answer_chunks = self._split_answer_semantically(answer)
                        
                        # 添加子chunk
                        for j, sub_chunk in enumerate(answer_chunks):
                            if self._is_valuable_chunk(sub_chunk):
                                chunks.append(Document(page_content=sub_chunk, metadata=sub_metadata))
        else:
            # 没有Q&A格式，使用传统分割
            print("No Q&A blocks found, using semantic splitting...")
            
            # 提取主题
            topic = self._extract_topic(processed_text[:500])  # 从文本开头提取主题
            
            # 构建metadata
            metadata = {
                "source": source,
                "type": "general",
                "topic": topic,
                "tags": [],
                "chunk_level": "main"
            }
            
            # 使用语义分割，确保完整性
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # 适中的分块大小，符合用户建议
                chunk_overlap=100,  # 合适的重叠比例，避免信息断裂
                separators=[
                    "\n\n",      # 按段落分割
                    "\n",        # 按行分割
                    "。",        # 按中文句号分割
                    ". ",        # 按英文句号分割
                    "；",        # 按中文分号分割
                    ";",         # 按英文分号分割
                    "，",        # 按中文逗号分割
                    ", ",        # 按英文逗号分割
                    ""           # 最后按字符分割
                ]
            )
            base_doc = Document(page_content=processed_text, metadata=metadata)
            chunks = text_splitter.split_documents([base_doc])

        # 过滤并后处理块
        filtered_chunks = []
        for chunk in chunks:
            content = chunk.page_content.strip()
            # 保留包含关键词的块，即使较短
            if self._is_valuable_chunk(content):
                filtered_chunks.append(chunk)

        print(f"Created {len(filtered_chunks)} semantic chunks")
        return filtered_chunks
    
    def _split_answer_semantically(self, answer: str) -> List[str]:
        """对答案进行语义分割，创建细粒度的子chunk"""
        import re
        
        # 1. 预处理答案文本
        answer = answer.strip()
        
        # 2. 按语义分割
        # 分割点：段落、列表、代码块
        
        # 提取所有代码块
        code_blocks = re.findall(r'```[\s\S]*?```', answer)
        
        # 替换代码块为占位符
        non_code_text = re.sub(r'```[\s\S]*?```', '[[CODE_BLOCK]]', answer)
        
        # 按段落和列表分割
        sub_chunks = []
        
        # 处理段落
        paragraphs = re.split(r'\n\n{2,}', non_code_text)
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 处理列表
            list_items = re.split(r'\n\s*[•●○▸▹▶]\s*', para)
            if len(list_items) > 1:
                # 包含列表，每个列表项作为一个chunk
                for item in list_items:
                    item = item.strip()
                    if item:
                        sub_chunks.append(item)
            else:
                # 普通段落
                sub_chunks.append(para)
        
        # 3. 替换回代码块
        final_chunks = []
        code_index = 0
        
        for chunk in sub_chunks:
            if "[[CODE_BLOCK]]" in chunk:
                # 替换回代码块
                if code_index < len(code_blocks):
                    chunk = chunk.replace("[[CODE_BLOCK]]", code_blocks[code_index])
                    code_index += 1
            
            # 控制块大小
            if len(chunk) > 800:
                # 太大，进一步分割
                smaller_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(smaller_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _split_large_chunk(self, text: str) -> List[str]:
        """分割大块文本"""
        import re
        
        # 按句子分割
        sentences = re.split(r'[。！？.!?]\s*', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) > 500:
                # 块大小适中，添加到chunks
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += sentence + "。"
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _preprocess_qa_text(self, text: str) -> str:
        """预处理Q&A文本，添加分隔符 - 针对面试资料格式优化"""
        import re
        
        # 1. 清理文本
        text = text.replace('\r', '')
        text = re.sub(r'[\u00a0\u200b\u200c\u200d\u2060\ufeff]', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 2. 识别并标准化Q&A格式
        
        # 模式1: Q1: 问题内容 或 Q1 问题内容
        # 注意：确保不会破坏代码块，使用更精确的匹配
        text = re.sub(r'(Q\d+)([：:]?)(\s*)(.+?)(?=Q\d+|$)', r'【问题】\4\n【答案】\n', text, flags=re.DOTALL)
        
        # 模式2: 问题：内容
        text = re.sub(r'问题[：:](\s*)([^\n]+)\n', r'【问题】\2\n【答案】\n', text)
        
        # 模式3: 问：内容
        text = re.sub(r'问[：:](\s*)([^\n]+)\n', r'【问题】\2\n【答案】\n', text)
        
        # 3. 处理答案标记
        text = re.sub(r'答案[：:](\s*)', r'【答案】\n', text)
        text = re.sub(r'答[：:](\s*)', r'【答案】\n', text)
        
        # 4. 处理标签
        # 查找可能的标签格式
        text = re.sub(r'标签[：:](\s*)(.+)', r'【标签】\2', text)
        text = re.sub(r'分类[：:](\s*)(.+)', r'【标签】\2', text)
        
        # 5. 识别并处理缩进的代码块
        # 暂时注释掉这部分逻辑，以提高性能
        # lines = text.split('\n')
        # processed_lines = []
        # in_code_block = False
        # 
        # for line in lines:
        #     # 检查是否是代码行（以空格或制表符开头，且包含代码特征）
        #     stripped_line = line.strip()
        #     is_code_line = False
        #     
        #     # 代码特征：包含关键字、符号或函数定义
        #     code_patterns = [
        #         r'^func\s+',  # 函数定义
        #         r'^var\s+',   # 变量定义
        #         r'^type\s+',  # 类型定义
        #         r'^import\s+', # 导入语句
        #         r'^for\s+',   # for循环
        #         r'^if\s+',    # if语句
        #         r'^else',     # else语句
        #         r'^return\s+', # return语句
        #         r'^case\s+',  # case语句
        #         r'^default:',  # default语句
        #         r'[{};]\s*$',  # 包含大括号或分号
        #         r'\s*=\s*',    # 赋值语句
        #         r'\s*\+\+|\s*--', # 自增自减
        #         r'\s*\+|-|\*|/|%|\^|&|\||<<|>>', # 运算符
        #         r'\s*\(|\)',   # 括号
        #         r'\s*\[|\]',   # 方括号
        #         r'\s*\.|->',   # 点或箭头操作符
        #     ]
        #     
        #     # 检查是否是代码行
        #     if stripped_line and (line.startswith(' ') or line.startswith('\t')):
        #         for pattern in code_patterns:
        #             if re.search(pattern, stripped_line):
        #                 is_code_line = True
        #                 break
        #     
        #     # 处理代码块
        #     if is_code_line and not in_code_block:
        #         # 开始代码块
        #         processed_lines.append('```go')
        #         processed_lines.append(line)
        #         in_code_block = True
        #     elif is_code_line and in_code_block:
        #         # 代码块内的行
        #         processed_lines.append(line)
        #     elif not is_code_line and in_code_block:
        #         # 结束代码块
        #         processed_lines.append('```')
        #         processed_lines.append(line)
        #         in_code_block = False
        #     else:
        #         # 普通行
        #         processed_lines.append(line)
        # 
        # # 确保代码块结束
        # if in_code_block:
        #     processed_lines.append('```')
        # 
        # text = '\n'.join(processed_lines)
        
        # 6. 确保每个Q&A块之间有清晰的分隔
        text = re.sub(r'【问题】', r'\n【问题】', text)
        
        # 7. 清理多余的空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _is_valuable_chunk(self, content: str) -> bool:
        """判断块是否有价值"""
        if not content:
            return False
        
        # 包含重要关键词的块，即使较短也保留
        important_keywords = [
            '锁', 'Mutex', 'RWMutex', '互斥锁', '读写锁',
            'channel', 'goroutine', '协程',
            '答案', 'Q\d+', '问题'
        ]
        
        for keyword in important_keywords:
            if re.search(keyword, content, re.IGNORECASE):
                return True
        
        # 其他块需要一定长度
        return len(content) > 50

    def _build_vector_index(self, chunks: List[Document]):
        """构建向量索引"""
        if not chunks:
            return

        print(f"Building vector index for {len(chunks)} chunks...")

        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(texts)

        if not embeddings:
            print("Failed to generate embeddings")
            return

        # 直接使用EmbeddingService实例作为embedding参数
        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=self.embedding_service,
            metadatas=[chunk.metadata for chunk in chunks]
        )

        index_path = os.path.join(self.index_dir, "faiss_index")
        self.vectorstore.save_local(index_path)

        chunks_data = [
            {"content": chunk.page_content, "metadata": chunk.metadata}
            for chunk in chunks
        ]
        with open(os.path.join(self.index_dir, "chunks.json"), 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False)

        print(f"Vector index saved to: {index_path}")

    def _load_existing_index(self):
        """加载已存在的向量索引"""
        index_path = os.path.join(self.index_dir, "faiss_index")
        chunks_path = os.path.join(self.index_dir, "chunks.json")

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            print(f"Loading existing index from: {index_path}")
            try:
                # 直接使用EmbeddingService实例作为embeddings参数
                self.vectorstore = FAISS.load_local(
                    index_path,
                    embeddings=self.embedding_service,
                    allow_dangerous_deserialization=True
                )

                with open(chunks_path, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                    self.chunks = [Document(page_content=c['content'], metadata=c.get('metadata', {})) for c in chunks_data]

                print(f"Loaded index with {len(self.chunks)} chunks")
            except Exception as e:
                print(f"Failed to load existing index: {e}")

    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """处理单个PDF文档"""
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(pdf_path)}")
        print('='*60)

        markdown_text = self._convert_pdf_to_markdown(pdf_path)

        if not markdown_text:
            return {"success": False, "error": "Failed to extract text"}

        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        markdown_filename = f"{filename}.md"
        self._save_markdown(markdown_text, markdown_filename)
        self.markdown_files.append(markdown_filename)

        # 保存JSON格式
        json_data = self._format_document_to_json(markdown_text, os.path.basename(pdf_path))
        json_filename = f"{filename}.json"
        json_path = self._save_json(json_data, json_filename)
        print(f"JSON file saved: {json_path}")

        # 调试：检查JSON数据中的sections
        sections = json_data.get('content', {}).get('sections', [])
        print(f"Found {len(sections)} sections in JSON")
        for i, section in enumerate(sections):
            if section.get('type') == 'qa':
                question = section.get('question', '')
                print(f"  QA {i+1}: {question[:50]}...")

        chunks = self._create_semantic_chunks(markdown_text, os.path.basename(pdf_path))
        self.chunks.extend(chunks)

        return {
            "success": True,
            "pdf_path": pdf_path,
            "markdown_file": markdown_filename,
            "json_file": json_filename,
            "chunks_count": len(chunks),
            "markdown_length": len(markdown_text)
        }

    def process_markdown(self, md_path: str) -> Dict[str, Any]:
        """处理单个Markdown文档"""
        print(f"\n{'='*60}")
        print(f"Processing Markdown: {os.path.basename(md_path)}")
        print('='*60)

        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                markdown_text = f.read()

            if not markdown_text:
                return {"success": False, "error": "Empty Markdown file"}

            filename = os.path.splitext(os.path.basename(md_path))[0]
            markdown_filename = f"{filename}.md"
            self.markdown_files.append(markdown_filename)

            # 保存JSON格式
            json_data = self._format_document_to_json(markdown_text, os.path.basename(md_path))
            json_filename = f"{filename}.json"
            json_path = self._save_json(json_data, json_filename)
            print(f"JSON file saved: {json_path}")

            chunks = self._create_semantic_chunks(markdown_text, os.path.basename(md_path))
            self.chunks.extend(chunks)

            return {
                "success": True,
                "md_path": md_path,
                "markdown_file": markdown_filename,
                "json_file": json_filename,
                "chunks_count": len(chunks),
                "markdown_length": len(markdown_text)
            }

        except Exception as e:
            print(f"Error processing markdown: {e}")
            return {"success": False, "error": str(e)}

    def process_folder(self) -> List[Dict[str, Any]]:
        """处理文档文件夹"""
        results = []

        # 获取所有PDF和Markdown文件
        pdf_files = [f for f in os.listdir(self.docs_dir) if f.endswith('.pdf')]
        md_files = [f for f in os.listdir(self.docs_dir) if f.endswith('.md')]

        print(f"\nFound {len(pdf_files)} PDF files and {len(md_files)} Markdown files to process")

        # 处理PDF文件
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing PDF: {pdf_file}")

            pdf_path = os.path.join(self.docs_dir, pdf_file)
            result = self.process_document(pdf_path)
            results.append(result)

        # 处理Markdown文件
        for i, md_file in enumerate(md_files, 1):
            print(f"\n[{i}/{len(md_files)}] Processing Markdown: {md_file}")

            md_path = os.path.join(self.docs_dir, md_file)
            result = self.process_markdown(md_path)
            results.append(result)

        if self.chunks:
            self._build_vector_index(self.chunks)

        return results

    def _process_content(self, content):
        """处理内容，确保字符编码正确"""
        try:
            # 尝试将内容编码为 UTF-8，然后解码
            return content.encode('utf-8', errors='replace').decode('utf-8')
        except Exception:
            # 如果失败，返回原始内容
            return content
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,  # 增加Top-K值，确保召回足够多的相关文档
        mode: str = "hybrid",
        similarity_threshold: float = 0.2  # 进一步降低阈值以获取更多信息
    ) -> List[Dict[str, Any]]:
        """检索相关块，带相似度阈值过滤和关键词增强"""
        if not self.vectorstore:
            print("Vector index not built. Call process_folder() first.")
            return []

        print(f"\nRetrieving for query: {query}")
        
        # 关键词匹配增强 - 针对特定查询优化
        query_lower = query.lower()
        keyword_boost = {}
        
        # 定义关键词到相关内容的映射
        keyword_mappings = {
            '锁': ['锁', 'Mutex', 'RWMutex', '互斥锁', '读写锁', 'sync.Mutex', 'sync.RWMutex'],
            'mutex': ['Mutex', '互斥锁', '锁', 'sync.Mutex'],
            'channel': ['channel', '通道', 'chan ', '发送阻塞', '接收阻塞'],
            'goroutine': ['goroutine', '协程', 'Goroutine'],
            '协程': ['协程', 'goroutine', 'Goroutine'],
            'map': ['map', '哈希', 'hashmap', 'key', '键', '包含', '存在'],
            '调度': ['调度', 'GPM', '调度器', 'goroutine调度'],
            'rune': ['rune', '字符', 'unicode', 'codepoint', 'int32', '字符编码'],
            'gc': ['gc', '垃圾回收', '垃圾收集', '标记清除', '三色标记法', '写屏障', 'stw'],
            '死锁': ['死锁', 'deadlock', 'dead lock'],
            '协程泄露': ['协程泄露', 'goroutine leak'],
            '无限循环': ['无限循环', 'infinite loop', '循环', '重试'],
            '逃逸分析': ['逃逸分析', 'escape analysis', '局部变量', '指针'],
            'csp': ['csp', '并发模型', 'communicating sequential processes'],
            'gpm': ['gpm', '调度模型', 'goroutine', 'processor', 'machine'],
        }
        
        # 提取查询中的关键词
        matched_keywords = []
        for keyword, related_terms in keyword_mappings.items():
            # 检查查询中是否包含任何相关术语
            for term in related_terms:
                if term in query_lower:
                    matched_keywords.extend(related_terms)
                    break
        
        matched_keywords = list(set(matched_keywords))  # 去重
        print(f"Matched keywords: {matched_keywords}")

        if mode == "vector" or mode == "hybrid":
            # 使用向量检索，获取更多结果
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=top_k * 10)  # 增加到10倍以获取更多结果
            
            # 打印前10个结果的分数
            print(f"Top 10 vector search results scores:")
            for i, (doc, score) in enumerate(docs_and_scores[:10]):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                content = self._process_content(content)  # 处理字符编码
                print(f"  {i+1}. Score: {score:.4f}, Content: {content[:50]}...")
            
            # 合并并去重
            seen_content = set()
            all_docs = []
            
            for doc, score in docs_and_scores:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                content = self._process_content(content)  # 处理字符编码
                content_hash = hash(content[:200])  # 增加哈希长度，减少碰撞
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    # 检查是否包含关键词
                    has_keyword = any(kw in content for kw in matched_keywords) if matched_keywords else False
                    # 调试：打印关键词匹配情况
                    if '什么是协程' in content or 'Goroutine 是' in content:
                        print(f"Debug: Checking keywords for coroutine definition:")
                        print(f"Debug: Content: {content[:100]}...")
                        print(f"Debug: Matched keywords: {matched_keywords}")
                        print(f"Debug: Has keyword: {has_keyword}")
                        for kw in matched_keywords:
                            print(f"Debug: Keyword '{kw}' in content: {kw in content}")
                    all_docs.append((doc, score, has_keyword))
                    # 调试：打印文档内容和得分
                    if '什么是协程' in content or 'Goroutine 是' in content:
                        print(f"Debug: Found coroutine definition - Score: {score}, Has keyword: {has_keyword}")
                        print(f"Debug: Content: {content[:100]}...")
            
            # 优先返回包含关键词的文档
            keyword_docs = []
            other_docs = []
            
            for doc, score, has_keyword in all_docs:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                
                # 调试：打印文档处理情况
                if '什么是协程' in content or 'Goroutine 是' in content:
                    print(f"Debug: Processing coroutine definition doc:")
                    print(f"Debug: Content: {content[:100]}...")
                    print(f"Debug: Has keyword: {has_keyword}")
                    print(f"Debug: Query contains definition terms: {any(term in query for term in ['什么是', '定义', '概念', '含义'])}")
                    print(f"Debug: Content contains '泄露' or '问题': {'泄露' in content or '问题' in content}")
                    print(f"Debug: Content contains definition terms: {any(def_term in content for def_term in ['是指', '定义', '概念', '含义', '解释', '是 ', '是与', '可以被认为', '可以被认为是', '可以认为', '是与其他', '是 与其他']) or '什么是协程' in content or 'Goroutine 是' in content}")
                
                # 智能排序：根据内容相关性和查询意图
                if has_keyword:
                    # 对于定义类查询，优先返回包含定义的内容
                    if any(term in query for term in ['什么是', '定义', '概念', '含义']):
                        # 优先匹配包含核心概念但不包含负面/问题场景的内容
                        # 检查内容中是否包含负面词汇，而不是标题中的"问题"关键词
                        content_lower = content.lower()
                        if '泄露' not in content_lower and not ( '问题' in content_lower and '场景' in content_lower):
                            # 检查是否包含定义相关词汇
                            if any(def_term in content for def_term in ['是指', '定义', '概念', '含义', '解释', '是 ', '是与', '可以被认为', '可以被认为是', '可以认为', '是与其他', '是 与其他']) or '什么是协程' in content or 'Goroutine 是' in content:
                                # 大幅降低得分，使其排前面（向量相似度得分越低越相似）
                                adjusted_score = score * 0.1  # 进一步降低得分，确保排前面
                                keyword_docs.append((doc, adjusted_score))
                                # 调试：打印调整后的得分
                                if '什么是协程' in content or 'Goroutine 是' in content:
                                    print(f"Debug: Coroutine definition adjusted score: {adjusted_score}")
                            else:
                                # 适度降低得分
                                adjusted_score = score * 0.7
                                keyword_docs.append((doc, adjusted_score))
                        else:
                            keyword_docs.append((doc, score))
                    else:
                        keyword_docs.append((doc, score))
                elif score >= similarity_threshold:
                    other_docs.append((doc, score))
            
            # 特殊处理：如果没有关键词匹配，但有与查询相关的低分文档，优先保留这些文档
            if not keyword_docs:
                # 从all_docs中筛选出可能与查询相关的文档
                query_terms = query.lower().split()
                print(f"Query terms: {query_terms}")
                relevant_docs = []
                irrelevant_docs = []
                
                for doc, score, hk in all_docs:
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    content_lower = content.lower()
                    # 检查文档内容是否包含查询中的关键词
                    has_relevant_terms = any(term in content_lower for term in query_terms if len(term) > 1)
                    print(f"Checking doc: score={score:.4f}, has_relevant_terms={has_relevant_terms}")
                    print(f"Doc content: {content[:50]}...")
                    if has_relevant_terms:
                        relevant_docs.append((doc, score))
                    else:
                        irrelevant_docs.append((doc, score))
                
                print(f"Relevant docs found: {len(relevant_docs)}")
                print(f"Irrelevant docs found: {len(irrelevant_docs)}")
                
                # 如果有相关文档，优先使用相关文档
                if relevant_docs:
                    # 按分数排序，取前top_k个
                    relevant_docs.sort(key=lambda x: x[1])
                    other_docs = relevant_docs[:top_k]
                    print(f"Using relevant docs: {len(other_docs)}")
                elif not other_docs and all_docs:
                    # 如果没有相关文档且other_docs为空，按分数排序取前top_k个
                    sorted_docs = sorted(all_docs, key=lambda x: x[1])
                    other_docs = [(d, s) for d, s, hk in sorted_docs[:top_k]]
                    print(f"Using sorted docs: {len(other_docs)}")
            
            # 打印过滤后的结果
            print(f"Filtered results - keyword_docs: {len(keyword_docs)}, other_docs: {len(other_docs)}")
            for i, (doc, score) in enumerate(other_docs[:5]):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                print(f"  Other doc {i+1}. Score: {score:.4f}, Content: {content[:50]}...")
            
            # 合并：关键词文档优先，然后按相似度排序
            # 对keyword_docs按得分排序（得分越低越相似）
            keyword_docs.sort(key=lambda x: x[1])
            # 对other_docs按得分排序
            other_docs.sort(key=lambda x: x[1])
            
            # 调试：打印keyword_docs的前5个结果
            print(f"Debug: Top 5 keyword_docs:")
            for i, (doc, score) in enumerate(keyword_docs[:5]):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                print(f"  Debug: {i+1}. Score: {score:.4f}, Content: {content[:50]}...")
            
            # 合并并取前top_k个
            combined_docs = keyword_docs + other_docs
            docs = [doc for doc, _ in combined_docs[:top_k]]
            
            print(f"Vector search: {len(keyword_docs)} keyword-matched, {len(other_docs)} other docs")
            
        elif mode == "keyword":
            # 简单的关键词匹配检索
            query_keywords = [k for k in matched_keywords if len(k) > 1]  # 使用已提取的关键词
            if not query_keywords:
                query_keywords = [query]  # 如果没有关键词，使用整个查询
            
            scored_chunks = []
            for chunk in self.chunks:
                content = chunk.page_content
                score = 0
                for kw in query_keywords:
                    if kw in content:
                        score += content.count(kw) * len(kw)  # 权重：出现次数 * 关键词长度
                if score > 0:
                    scored_chunks.append((chunk, score))
            
            # 按分数排序
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            docs = [chunk for chunk, _ in scored_chunks[:top_k]]

        results = []
        for doc in docs:
            if hasattr(doc, 'page_content'):
                doc_content = doc.page_content
            elif hasattr(doc, 'content'):
                doc_content = doc.content
            else:
                doc_content = str(doc)
            
            results.append({
                "content": doc_content,
                "metadata": getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else {}
            })

        print(f"Returning {len(results)} results")
        return results

    def query(
        self,
        question: str,
        top_k: int = 3,
        retrieval_mode: str = "hybrid",
        similarity_threshold: float = 0.3  # 降低阈值以获取更多信息
    ) -> Dict[str, Any]:
        """RAG查询"""
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print('='*60)

        retrieved_docs = self.retrieve(question, top_k, retrieval_mode, similarity_threshold)

        print(f"Retrieved {len(retrieved_docs)} docs")
        if retrieved_docs:
            print(f"First doc type: {type(retrieved_docs[0])}, keys: {list(retrieved_docs[0].keys()) if isinstance(retrieved_docs[0], dict) else 'not dict'}")

        if not retrieved_docs:
            return {
                "question": question,
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": []
            }

        # 构建context
        try:
            context = []
            for doc in retrieved_docs:
                if hasattr(doc, 'page_content'):
                    content = self._process_content(doc.page_content)  # 处理字符编码
                    context.append({"content": content})
                elif isinstance(doc, dict) and "content" in doc:
                    content = self._process_content(doc["content"])  # 处理字符编码
                    context.append({"content": content})
                else:
                    content = self._process_content(str(doc))  # 处理字符编码
                    context.append({"content": content})
            print(f"Context built: {len(context)} items")
        except Exception as e:
            print(f"Context build error: {e}, doc: {retrieved_docs[0] if retrieved_docs else 'empty'}")
            context = []
            for doc in retrieved_docs:
                try:
                    if hasattr(doc, 'page_content'):
                        content = self._process_content(doc.page_content)  # 处理字符编码
                        context.append({"content": content})
                    elif isinstance(doc, dict) and "content" in doc:
                        content = self._process_content(doc["content"])  # 处理字符编码
                        context.append({"content": content})
                    else:
                        content = self._process_content(str(doc))  # 处理字符编码
                        context.append({"content": content})
                except Exception:
                    # 如果处理失败，跳过该文档
                    pass

        answer = self.generator.generate(question, context)

        # 提取sources为字符串数组
        sources = []
        for doc in retrieved_docs:
            try:
                if hasattr(doc, 'page_content'):
                    content = self._process_content(doc.page_content)  # 处理字符编码
                    sources.append(content)
                elif isinstance(doc, dict) and "content" in doc:
                    content = self._process_content(doc["content"])  # 处理字符编码
                    sources.append(content)
                else:
                    content = self._process_content(str(doc))  # 处理字符编码
                    sources.append(content)
            except Exception:
                # 如果处理失败，尝试使用原始字符串
                try:
                    sources.append(str(doc))
                except Exception:
                    # 如果仍然失败，跳过该文档
                    pass

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieval_mode": retrieval_mode
        }

    def query_with_multimodal(
        self,
        question: str,
        multimodal_content: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """支持多模态内容的RAG查询（RAG-Anything部分功能）"""
        if multimodal_content and RAG_ANYTHING_AVAILABLE:
            print("Processing multimodal content...")

            multimodal_context = []
            for content in multimodal_content:
                content_type = content.get("type", "unknown")
                if content_type == "table":
                    table_data = content.get("table_data", "")
                    multimodal_context.append(f"[Table]\n{table_data}")
                elif content_type == "image":
                    img_caption = content.get("image_caption", "")
                    multimodal_context.append(f"[Image: {img_caption}]")

            if multimodal_context:
                question = f"{question}\n\nAdditional context:\n" + "\n".join(multimodal_context)

        return self.query(question)

    async def aquery(
        self,
        question: str,
        top_k: int = 1,  # 减少召回数量，避免噪声
        mode: str = "hybrid"
    ) -> Dict[str, Any]:
        """异步RAG查询"""
        print(f"aquery called with question: {question}, mode: {mode}")
        # 使用更低的相似度阈值，确保能找到包含正确答案的文档
        return self.query(question, top_k, mode, similarity_threshold=0.3)  # 降低相似度阈值

    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计"""
        return {
            "total_chunks": len(self.chunks),
            "total_markdown_files": len(self.markdown_files),
            "markdown_files": self.markdown_files,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }


async def main():
    """测试主函数"""
    print("=" * 60)
    print("Unified RAG Processor: LangChain + Marker + RAG-Anything")
    print("=" * 60)

    processor = UnifiedRAGProcessor()

    print("\nProcessing documents...")
    results = processor.process_folder()

    print(f"\n{'='*60}")
    print("Processing Complete!")
    print('='*60)

    stats = processor.get_statistics()
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total markdown files: {stats['total_markdown_files']}")

    print(f"\n{'='*60}")
    print("Testing Query...")
    print('='*60)

    result = processor.query("Go语言的GPM调度器是什么？")

    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {len(result['sources'])} documents")


if __name__ == "__main__":
    asyncio.run(main())