import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from rag_service.unified_rag_processor import UnifiedRAGProcessor

# 测试JSON格式化功能
def test_json_formatting():
    print("Testing JSON formatting functionality...")
    
    # 创建处理器实例
    processor = UnifiedRAGProcessor()
    
    # 处理所有PDF文档
    docs_dir = os.path.join(os.path.dirname(__file__), "data", "docs")
    
    if not os.path.exists(docs_dir):
        print(f"Docs directory not found: {docs_dir}")
        return
    
    print(f"Processing documents from: {docs_dir}")
    
    # 处理每个PDF文件
    for filename in os.listdir(docs_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(docs_dir, filename)
            result = processor.process_document(pdf_path)
            
            if result['success']:
                print(f"\n✓ Processed: {filename}")
                print(f"  Markdown: {result['markdown_file']}")
                print(f"  JSON: {result['json_file']}")
                print(f"  Chunks: {result['chunks_count']}")
            else:
                print(f"\n✗ Failed to process: {filename}")
                print(f"  Error: {result['error']}")
    
    # 检查生成的JSON文件
    output_dir = os.path.join(os.path.dirname(__file__), "data", "formatted")
    if os.path.exists(output_dir):
        print(f"\nGenerated JSON files in: {output_dir}")
        json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        if json_files:
            print("Generated JSON files:")
            for json_file in json_files:
                print(f"  - {json_file}")
                # 显示文件大小
                file_path = os.path.join(output_dir, json_file)
                file_size = os.path.getsize(file_path) / 1024
                print(f"    Size: {file_size:.2f} KB")
        else:
            print("No JSON files generated")
    
    print("\nJSON formatting test completed!")

if __name__ == "__main__":
    test_json_formatting()