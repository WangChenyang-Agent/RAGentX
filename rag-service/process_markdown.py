import os
import json
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from embedding import EmbeddingService

project_root = os.path.dirname(os.path.abspath(__file__))
docs_dir = os.path.join(project_root, "..", "data", "docs")
index_dir = os.path.join(project_root, "..", "data")

class MarkdownProcessor:
    """专门处理Markdown文件的处理器"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.docs_dir = docs_dir
        self.index_dir = index_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embedding_service = EmbeddingService()
        self.vectorstore = None
        self.chunks = []
        self.markdown_files = []

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

        print(f"Vector index saved to: {index_path}")

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

        # 只获取Markdown文件
        md_files = [f for f in os.listdir(self.docs_dir) if f.endswith('.md')]

        print(f"\nFound {len(md_files)} Markdown files to process")

        # 处理Markdown文件
        for i, md_file in enumerate(md_files, 1):
            print(f"\n[{i}/{len(md_files)}] Processing Markdown: {md_file}")

            md_path = os.path.join(self.docs_dir, md_file)
            result = self.process_markdown(md_path)
            results.append(result)

        if self.chunks:
            self._build_vector_index(self.chunks)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计"""
        return {
            "total_chunks": len(self.chunks),
            "total_markdown_files": len(self.markdown_files),
            "markdown_files": self.markdown_files,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }

    def test_query(self, query: str):
        """测试查询"""
        if not self.vectorstore:
            print("Vector index not built. Call process_folder() first.")
            return

        print(f"\n{'='*60}")
        print(f"Testing query: {query}")
        print('='*60)

        # 执行相似度搜索
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=5)

        print(f"Found {len(docs_and_scores)} relevant chunks")

        for i, (doc, score) in enumerate(docs_and_scores, 1):
            print(f"\n[{i}] Score: {score:.4f}")
            print(f"Content: {doc.page_content[:200]}...")

if __name__ == "__main__":
    print("=" * 60)
    print("Markdown Processor: LangChain + Embedding")
    print("=" * 60)

    processor = MarkdownProcessor()

    print("\nProcessing Markdown files...")
    results = processor.process_folder()

    print(f"\n{'='*60}")
    print("Processing Complete!")
    print('='*60)

    stats = processor.get_statistics()
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total markdown files: {stats['total_markdown_files']}")
    print(f"Processed files: {stats['markdown_files']}")

    print(f"\n{'='*60}")
    print("Testing Queries...")
    print('='*60)

    # 测试几个查询
    test_queries = [
        "Go语言的特点",
        "Go的并发特性",
        "Go的GPM调度器"
    ]

    for query in test_queries:
        processor.test_query(query)
