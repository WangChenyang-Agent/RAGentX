import os
import sys

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing UnifiedRAGProcessor initialization...")

# 测试导入
try:
    from unified_rag_processor import UnifiedRAGProcessor
    print("OK: UnifiedRAGProcessor imported successfully")
except Exception as e:
    print(f"ERROR: Failed to import UnifiedRAGProcessor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试初始化
try:
    print("Initializing UnifiedRAGProcessor...")
    processor = UnifiedRAGProcessor()
    print("OK: UnifiedRAGProcessor initialized successfully")
    
    # 检查向量索引
    if processor.vectorstore:
        print(f"OK: Vector index loaded with {len(processor.chunks)} chunks")
    else:
        print("WARNING: No vector index found")
    
    # 测试查询
    print("Testing query...")
    result = processor.query("Go语言的特点")
    print("OK: Query completed")
    print(f"Answer: {result['answer'][:100]}...")
    print(f"Sources: {len(result['sources'])} documents")
    
    print("\nAll tests passed!")
    
except Exception as e:
    print(f"ERROR: Failed to initialize or use UnifiedRAGProcessor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
