import os
import json
import asyncio
from typing import List, Dict, Optional, Any
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

project_root = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(project_root, ".marker_cache")
os.makedirs(cache_dir, exist_ok=True)

os.environ["XDG_CACHE_HOME"] = cache_dir
os.environ["MARKER_CACHE_HOME"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["TORCH_HOME"] = cache_dir

try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
except ImportError:
    print("Marker not available, will use fallback method")
    MARKER_AVAILABLE = False

class MarkerPDFProcessor:
    """基于Marker的PDF转Markdown处理器"""

    def __init__(
        self,
        output_dir: str = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        if output_dir is None:
            output_dir = os.path.join(project_root, "..", "data", "formatted_marker")

        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        os.makedirs(self.output_dir, exist_ok=True)

        self.models = None
        self.converter = None

    def _initialize_marker(self):
        """初始化Marker模型"""
        if not MARKER_AVAILABLE:
            print("Marker package not installed")
            return

        if self.models is None:
            print("Initializing Marker models...")
            print("This may take a while on first run (models will be downloaded)...")

            try:
                self.models = create_model_dict()
                self.converter = PdfConverter(artifact_dict=self.models)
                print("Marker models initialized successfully!")
            except Exception as e:
                print(f"Error initializing Marker models: {e}")
                print("\nWill use fallback PDF processing method...")
                self.models = None
                self.converter = None

    def convert_pdf_to_markdown_fallback(self, pdf_path: str) -> str:
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
        import re

        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r' (?=[\u4e00-\u9fa5])', '', text)
        text = re.sub(r'(?<=[\u4e00-\u9fa5]) ', '', text)
        text = re.sub(r'\n\n+', '\n\n', text)

        return text.strip()

    def convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """将PDF转换为Markdown"""
        self._initialize_marker()

        print(f"Converting PDF to Markdown: {pdf_path}")

        if self.converter is None:
            print("Marker not available, using fallback method...")
            return self.convert_pdf_to_markdown_fallback(pdf_path)

        try:
            rendered = self.converter(pdf_path)
            markdown_text = text_from_rendered(rendered)

            return markdown_text

        except Exception as e:
            print(f"Error converting PDF with Marker: {e}")
            print("Falling back to basic text extraction...")
            return self.convert_pdf_to_markdown_fallback(pdf_path)

    def save_markdown(self, markdown_text: str, output_filename: str) -> str:
        """保存Markdown文件"""
        output_path = os.path.join(self.output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)

        print(f"Markdown saved to: {output_path}")
        return output_path

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """处理单个PDF文件"""
        print(f"\n{'='*60}")
        print(f"Processing PDF with Marker: {os.path.basename(pdf_path)}")
        print('='*60)

        markdown_text = self.convert_pdf_to_markdown(pdf_path)

        if not markdown_text:
            return {
                "success": False,
                "pdf_path": pdf_path,
                "markdown": "",
                "chunks": []
            }

        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        markdown_filename = f"{filename}.md"
        self.save_markdown(markdown_text, markdown_filename)

        chunks = self.create_chunks(markdown_text)

        return {
            "success": True,
            "pdf_path": pdf_path,
            "markdown": markdown_text,
            "markdown_file": markdown_filename,
            "chunks": chunks,
            "chunk_count": len(chunks)
        }

    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """处理文件夹中的所有PDF"""
        results = []

        pdf_files = [
            f for f in os.listdir(folder_path)
            if f.endswith('.pdf')
        ]

        print(f"\nFound {len(pdf_files)} PDF files to process")

        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file}")

            pdf_path = os.path.join(folder_path, pdf_file)
            result = self.process_pdf(pdf_path)

            results.append(result)

        return results

    def create_chunks(self, markdown_text: str) -> List[Dict[str, Any]]:
        """从Markdown创建语义化分块"""
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
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            text_chunks = text_splitter.split_text(markdown_text)

            chunks = []
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "content": chunk,
                    "chunk_id": i,
                    "metadata": {"source": "text_split"}
                })
        else:
            chunks = []
            for i, chunk in enumerate(md_chunks):
                chunks.append({
                    "content": chunk.page_content,
                    "chunk_id": i,
                    "metadata": chunk.metadata
                })

        return chunks

    def get_chunk_preview(self, chunks: List[Dict], num_chunks: int = 5) -> List[str]:
        """获取chunks预览"""
        previews = []

        for i, chunk in enumerate(chunks[:num_chunks]):
            content = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            previews.append(f"Chunk {i+1}:\n{content}\n")

        return previews


async def test_marker_conversion():
    """测试Marker PDF转换"""
    print("=" * 60)
    print("Testing Marker PDF to Markdown Conversion")
    print("=" * 60)

    processor = MarkerPDFProcessor()

    test_pdf = os.path.join(project_root, "..", "data", "docs", "Go语言GPM调度器.pdf")

    if not os.path.exists(test_pdf):
        print(f"Test PDF not found: {test_pdf}")
        print("Please ensure you have a PDF file to test")
        return

    result = processor.process_pdf(test_pdf)

    if result['success']:
        print(f"\n{'='*60}")
        print("Conversion Results")
        print('='*60)
        print(f"PDF: {result['pdf_path']}")
        print(f"Markdown file: {result['markdown_file']}")
        print(f"Total chunks: {result['chunk_count']}")
        print(f"Markdown length: {len(result['markdown'])} characters")

        print(f"\n{'='*60}")
        print("Chunk Previews")
        print('='*60)
        previews = processor.get_chunk_preview(result['chunks'], num_chunks=3)
        for preview in previews:
            print(preview)
    else:
        print("Conversion failed!")


def main():
    """主函数"""
    asyncio.run(test_marker_conversion())


if __name__ == "__main__":
    main()