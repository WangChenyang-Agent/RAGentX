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
        chunk_size: int = 500,
        chunk_overlap: int = 100
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
            
            print(f"Extracted text length: {len(text)}")
            print(f"First 500 chars: {text[:500]}...")
            
            # 2. 直接分割Q&A
            # 手动分割Q1, Q2, Q3...
            result = []
            
            # 查找所有Q1, Q2, Q3...的位置
            q_pattern = re.compile(r'Q\d+')
            matches = list(q_pattern.finditer(text))
            
            print(f"Found {len(matches)} Q patterns")
            
            if matches:
                # 添加标题部分
                if matches[0].start() > 0:
                    title = text[:matches[0].start()].strip()
                    result.append(title)
                    result.append('')
                    print(f"Added title: {title[:100]}...")
                
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
                    result.append(f"### {qa_block}")
                    result.append('')
                    print(f"Added Q&A block: {q_text}...")
            else:
                # 没有找到Q格式，返回原文本
                result.append(text.strip())
                print("No Q patterns found, returning original text")
            
            formatted_text = '\n'.join(result)
            print(f"Formatted text length: {len(formatted_text)}")
            print(f"First 500 chars of formatted text: {formatted_text[:500]}...")
            
            return formatted_text

        except Exception as e:
            print(f"Fallback extraction error: {e}")
            return ""

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

    def _create_semantic_chunks(self, markdown_text: str) -> List[Document]:
        """使用LangChain创建语义化分块，针对Q&A文档优化"""
        # 首先尝试识别Q&A模式并添加分隔符
        processed_text = self._preprocess_qa_text(markdown_text)
        
        headers_to_split_on = [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
            ("####", "H4"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )

        md_chunks = markdown_splitter.split_text(processed_text)

        if len(md_chunks) < 2:
            print("No headers found, using Q&A-aware text splitting...")
            # 使用更小的chunk_size来确保单个Q&A不被拆分
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # 适当增大以确保完整Q&A
                chunk_overlap=100,
                separators=[
                    "\n【问题】",  # 优先按问题标记分割（新增）
                    "\n\n",      # 按段落分割
                    "\n",        # 按行分割
                    "。",        # 按中文句号分割
                    ". ",        # 按英文句号分割
                    "；",        # 按中文分号分割
                    ";",         # 按英文分号分割
                    ""           # 最后按字符分割
                ]
            )
            md_chunks = text_splitter.split_documents([Document(page_content=processed_text)])

        # 过滤并后处理块
        filtered_chunks = []
        for chunk in md_chunks:
            content = chunk.page_content.strip()
            # 保留包含关键词的块，即使较短
            if self._is_valuable_chunk(content):
                filtered_chunks.append(chunk)

        print(f"Created {len(filtered_chunks)} semantic chunks")
        return filtered_chunks
    
    def _preprocess_qa_text(self, text: str) -> str:
        """预处理Q&A文本，添加分隔符 - 针对面试资料格式优化"""
        # 在常见问题前添加清晰的分隔
        qa_patterns = [
            r'(go中有哪些锁[？?])',  # 锁相关问题
            r'(CSP并发模型[？?])',
            r'(GPM模型[？?])',
            r'(channel底层的数据结构[？?])',
            r'(goroutine的调度时机[？?])',
            r'(分布式系统[？?])',
            r'(微服务架构[？?])',
            r'(Redis[？?])',
            r'(MySQL[？?])',
            r'(MongoDB[？?])',
            r'(Linux[？?])',
            r'(网络[？?])',
        ]
        
        for pattern in qa_patterns:
            text = re.sub(pattern, r'\n【问题】\1\n', text)
        
        # 在答案标记前添加分隔
        text = re.sub(r'(答案)', r'\n【答案】\1', text)
        
        # 在sync.Mutex/RWMutex前添加换行
        text = re.sub(r'(sync\.Mutex)', r'\n\1', text)
        text = re.sub(r'(sync\.RWMutex)', r'\n\1', text)
        
        # 在数字列表前添加换行
        text = re.sub(r'(\d+、)', r'\n\1', text)
        
        # 在Q编号前添加换行
        text = re.sub(r'\s(Q\d+)', r'\n\1', text)
        
        return text
    
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

        self._load_existing_index()
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

        chunks = self._create_semantic_chunks(markdown_text)
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

            chunks = self._create_semantic_chunks(markdown_text)
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

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        similarity_threshold: float = 0.5  # 降低阈值以获取更多信息
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
            '锁': ['锁', 'Mutex', 'RWMutex', '互斥锁', '读写锁'],
            'mutex': ['Mutex', '互斥锁', '锁'],
            'channel': ['channel', '通道', 'chan '],
            'goroutine': ['goroutine', '协程', 'Goroutine'],
            'map': ['map', '哈希', 'hashmap'],
            '调度': ['调度', 'GPM', '调度器'],
        }
        
        # 提取查询中的关键词
        matched_keywords = []
        for keyword, related_terms in keyword_mappings.items():
            if keyword in query_lower:
                matched_keywords.extend(related_terms)
        
        matched_keywords = list(set(matched_keywords))  # 去重
        print(f"Matched keywords: {matched_keywords}")

        if mode == "vector" or mode == "hybrid":
            # 使用向量检索，获取更多结果
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=top_k * 10)  # 增加到10倍以获取更多结果
            
            # 打印前10个结果的分数
            print(f"Top 10 vector search results scores:")
            for i, (doc, score) in enumerate(docs_and_scores[:10]):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                print(f"  {i+1}. Score: {score:.4f}, Content: {content[:50]}...")
            
            # 合并并去重
            seen_content = set()
            all_docs = []
            
            for doc, score in docs_and_scores:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                content_hash = hash(content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    # 检查是否包含关键词
                    has_keyword = any(kw in content for kw in matched_keywords) if matched_keywords else False
                    all_docs.append((doc, score, has_keyword))
            
            # 优先返回包含关键词的文档
            keyword_docs = [(d, s) for d, s, hk in all_docs if hk]
            other_docs = [(d, s) for d, s, hk in all_docs if not hk and s >= similarity_threshold]
            
            # 打印过滤后的结果
            print(f"Filtered results - keyword_docs: {len(keyword_docs)}, other_docs: {len(other_docs)}")
            for i, (doc, score) in enumerate(other_docs[:5]):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                print(f"  Other doc {i+1}. Score: {score:.4f}, Content: {content[:50]}...")
            
            # 合并：关键词文档优先，然后按相似度排序
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
        similarity_threshold: float = 0.7
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
                    context.append({"content": doc.page_content})
                elif isinstance(doc, dict) and "content" in doc:
                    context.append({"content": doc["content"]})
                else:
                    context.append({"content": str(doc)})
            print(f"Context built: {len(context)} items")
        except Exception as e:
            print(f"Context build error: {e}, doc: {retrieved_docs[0] if retrieved_docs else 'empty'}")
            context = [{"content": str(doc)} for doc in retrieved_docs]

        answer = self.generator.generate(question, context)

        # 提取sources为字符串数组
        sources = []
        for doc in retrieved_docs:
            if hasattr(doc, 'page_content'):
                sources.append(doc.page_content)
            elif isinstance(doc, dict) and "content" in doc:
                sources.append(doc["content"])
            else:
                sources.append(str(doc))

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
        top_k: int = 5,
        mode: str = "hybrid"
    ) -> Dict[str, Any]:
        """异步RAG查询"""
        print(f"aquery called with question: {question}, mode: {mode}")
        # 使用更低的相似度阈值，确保能找到包含正确答案的文档
        return self.query(question, top_k, mode, similarity_threshold=0.3)

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