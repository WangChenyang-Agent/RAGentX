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
from reranker import Reranker

try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
except ImportError:
    print("Marker not available, will use fallback method")
    MARKER_AVAILABLE = False

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
        self.reranker = Reranker()

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

            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"

            return self._clean_text(text)

        except Exception as e:
            print(f"Fallback extraction error: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """清理提取的文本"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r' (?=[\u4e00-\u9fa5])', '', text)
        text = re.sub(r'(?<=[\u4e00-\u9fa5]) ', '', text)
        text = re.sub(r'\n\n+', '\n\n', text)

        return text.strip()

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
            return markdown_text

        except Exception as e:
            print(f"Marker conversion failed: {e}")
            print("Falling back to basic text extraction...")
            return self._convert_pdf_to_markdown_fallback(pdf_path)

    def _save_markdown(self, markdown_text: str, output_filename: str) -> str:
        """保存Markdown文件"""
        output_path = os.path.join(self.output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)

        return output_path

    def _create_semantic_chunks(self, markdown_text: str) -> List[Document]:
        """使用LangChain创建语义化分块"""
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

        md_chunks = markdown_splitter.split_text(markdown_text)

        if len(md_chunks) < 2:
            print("No headers found, using recursive text splitting...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            md_chunks = text_splitter.split_documents([Document(page_content=markdown_text)])

        return md_chunks

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

        chunks = self._create_semantic_chunks(markdown_text)
        self.chunks.extend(chunks)

        return {
            "success": True,
            "pdf_path": pdf_path,
            "markdown_file": markdown_filename,
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

            chunks = self._create_semantic_chunks(markdown_text)
            self.chunks.extend(chunks)

            return {
                "success": True,
                "md_path": md_path,
                "markdown_file": markdown_filename,
                "chunks_count": len(chunks),
                "markdown_length": len(markdown_text)
            }
        except Exception as e:
            print(f"Error processing Markdown: {e}")
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
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """检索相关块，带相似度阈值过滤"""
        if not self.vectorstore:
            print("Vector index not built. Call process_folder() first.")
            return []

        print(f"\nRetrieving for query: {query}")

        if mode == "vector":
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=top_k * 3)
            filtered_docs = [
                (doc, score) for doc, score in docs_and_scores
                if score >= similarity_threshold
            ]
            docs = [doc for doc, _ in filtered_docs[:top_k]]
        elif mode == "keyword":
            retriever = BM25Retriever.from_texts(
                texts=[chunk.page_content for chunk in self.chunks],
                metadatas=[chunk.metadata for chunk in self.chunks]
            )
            retriever.k = top_k * 3
            docs = retriever.get_relevant_documents(query)
            docs = docs[:top_k]
        else:
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=top_k * 3)
            filtered_docs = [
                (doc, score) for doc, score in docs_and_scores
                if score >= similarity_threshold
            ]
            docs = [doc for doc, _ in filtered_docs[:top_k]]

        results = []
        for doc in docs:
            print(f"Doc type: {type(doc)}, dir: {[attr for attr in dir(doc) if not attr.startswith('_')]}")
            if hasattr(doc, 'page_content'):
                doc_content = doc.page_content
                print(f"Has page_content: {len(doc_content) if doc_content else 0} chars")
            elif hasattr(doc, 'content'):
                doc_content = doc.content
                print(f"Has content: {len(doc_content) if doc_content else 0} chars")
            else:
                doc_content = str(doc)
                print(f"No content found, using str: {doc_content[:50]}...")
            
            results.append({
                "content": doc_content,
                "metadata": getattr(doc, 'metadata', {}) if hasattr(doc, 'metadata') else {}
            })

        print(f"Final results: {len(results)} docs, first doc keys: {list(results[0].keys()) if results else []}")
        return results

    def query(
        self,
        question: str,
        top_k: int = 5,
        retrieval_mode: str = "hybrid",
        use_reranker: bool = False,
        similarity_threshold: float = 0.5
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

        if use_reranker and len(retrieved_docs) > 1:
            print("Reranking results...")
            try:
                reranked = self.reranker.rerank(question, retrieved_docs)
                print(f"Reranked docs: {len(reranked)}")
            except Exception as e:
                print(f"Rerank failed, using original: {e}")
                reranked = retrieved_docs
        else:
            reranked = retrieved_docs

        print(f"Reranked docs: {len(reranked)}")
        if reranked:
            print(f"First reranked doc type: {type(reranked[0])}, keys: {list(reranked[0].keys()) if isinstance(reranked[0], dict) else 'not dict'}")

        # 构建context
        try:
            context = []
            for doc in reranked:
                if hasattr(doc, 'page_content'):
                    context.append({"content": doc.page_content})
                elif isinstance(doc, dict) and "content" in doc:
                    context.append({"content": doc["content"]})
                else:
                    context.append({"content": str(doc)})
            print(f"Context built: {len(context)} items")
        except Exception as e:
            print(f"Context build error: {e}, doc: {reranked[0] if reranked else 'empty'}")
            context = [{"content": str(doc)} for doc in reranked]

        answer = self.generator.generate(question, context)

        # 提取sources为字符串数组
        sources = []
        for doc in reranked:
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
        return self.query(question, top_k, mode, use_reranker=False)

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