import os
import json

# 直接运行在rag-service目录中
from unified_rag_processor import UnifiedRAGProcessor

# 测试JSON格式化功能
def test_json_formatting():
    print("Testing JSON formatting functionality...")
    
    # 创建处理器实例
    processor = UnifiedRAGProcessor()
    
    # 处理所有PDF文档
    docs_dir = os.path.join("..", "data", "docs")
    
    if not os.path.exists(docs_dir):
        print(f"Docs directory not found: {docs_dir}")
        return
    
    print(f"Processing documents from: {docs_dir}")
    
    # 处理每个PDF文件
    for filename in os.listdir(docs_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(docs_dir, filename)
            print(f"\nProcessing: {filename}")
            result = processor.process_document(pdf_path)
            
            if result['success']:
                print(f"Success!")
                print(f"  Markdown: {result['markdown_file']}")
                print(f"  JSON: {result['json_file']}")
                print(f"  Chunks: {result['chunks_count']}")
            else:
                print(f"Failed: {result['error']}")
    
    # 检查生成的JSON文件
    output_dir = os.path.join("..", "data", "formatted")
    if os.path.exists(output_dir):
        print(f"\nGenerated files in: {output_dir}")
        all_files = os.listdir(output_dir)
        md_files = [f for f in all_files if f.endswith('.md')]
        json_files = [f for f in all_files if f.endswith('.json')]
        
        print(f"Markdown files: {len(md_files)}")
        print(f"JSON files: {len(json_files)}")
        
        if json_files:
            print("\nJSON files generated:")
            for json_file in json_files[:3]:  # 只显示前3个
                print(f"  - {json_file}")
                # 显示JSON文件的前几行
                json_path = os.path.join(output_dir, json_file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                    print(f"    Sections: {len(content['content']['sections'])}")
                    if content['content']['sections']:
                        first_section = content['content']['sections'][0]
                        print(f"    First section type: {first_section.get('type')}")
                except Exception as e:
                    print(f"    Error reading JSON: {e}")
    
    print("\nJSON formatting test completed!")

if __name__ == "__main__":
    test_json_formatting()