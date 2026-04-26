# -*- coding: utf-8 -*-
import sys
import io
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(__file__)
rag_service_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(rag_service_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, rag_service_dir)
print(f"Added to Python path: {project_root}")
print(f"Added to Python path: {rag_service_dir}")

# 设置默认编码为UTF-8，避免Windows下的GBK编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn
import os

from core.embedding import EmbeddingService
from core.reranker import Reranker
from core.generator import Generator
from cache.redis_cache import clear_all_cache, is_redis_available

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理"""
    global unified_rag_processor
    
    # 启动时
    print("Starting RAGentX Service...")
    print(f"UNIFIED_RAG_AVAILABLE: {UNIFIED_RAG_AVAILABLE}")

    if UNIFIED_RAG_AVAILABLE:
        print("Initializing Unified RAG Processor (LangChain + Marker)...")
        try:
            # 禁用Marker初始化，避免模型下载阻塞
            print("Creating UnifiedRAGProcessor instance...")
            unified_rag_processor = UnifiedRAGProcessor()
            print("Unified RAG Processor instance created!")
            print("Skipping document processing during startup...")
            # 暂时禁用启动时的文档处理，加快启动速度
            # results = unified_rag_processor.process_folder()
            # print(f"Document processing completed! Created {len(unified_rag_processor.chunks)} chunks")
            if unified_rag_processor.vectorstore:
                print(f"Vector index loaded successfully with {len(unified_rag_processor.chunks)} chunks")
            else:
                print("No existing vector index found. Please call /api/process-documents to build index.")
            print("Unified RAG Processor initialized!")
        except Exception as e:
            print(f"Failed to initialize Unified RAG Processor: {e}")
            import traceback
            traceback.print_exc()
            unified_rag_processor = None
    else:
        print("Unified RAG not available, skipping initialization")
    
    print("RAGentX Service startup completed!")
    
    yield
    
    # 关闭时
    print("Shutting down RAGentX Service...")
    if unified_rag_processor:
        print("Cleaning up Unified RAG Processor...")
        # 这里可以添加清理代码
    print("RAGentX Service shutdown completed!")


# 计算前端目录路径
current_dir = os.path.dirname(__file__)
rag_service_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(rag_service_dir)
FRONTEND_DIR = os.path.join(project_root, "frontend")
print(f"Frontend directory: {FRONTEND_DIR}")
print(f"Frontend directory exists: {os.path.exists(FRONTEND_DIR)}")

unified_rag_processor = None

try:
    from core.unified_rag_processor import UnifiedRAGProcessor
    UNIFIED_RAG_AVAILABLE = True
except ImportError as e:
    print(f"Unified RAG not available: {e}")
    UNIFIED_RAG_AVAILABLE = False

app = FastAPI(title="RAGentX Service", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str
    use_unified_rag: bool = True
    top_k: int = 3
    retrieval_mode: str = "hybrid"


class AskResponse(BaseModel):
    answer: str
    sources: list
    time: str
    mode: str = "unified_rag"


class MultimodalQueryRequest(BaseModel):
    query: str
    multimodal_content: list = []
    mode: str = "hybrid"


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        # 使用安全的方式打印日志，避免编码错误
        try:
            print(f"接收到请求: {request.query}")
        except Exception:
            print("接收到请求: [编码错误]")
        if not UNIFIED_RAG_AVAILABLE or not unified_rag_processor:
            raise HTTPException(status_code=503, detail="Unified RAG not available")

        result = await unified_rag_processor.aquery(
            request.query,
            top_k=request.top_k,
            mode=request.retrieval_mode
        )

        def process_content(content):
            """处理内容，确保字符编码正确"""
            try:
                return content.encode('utf-8', errors='replace').decode('utf-8')
            except Exception:
                return content
        
        sources = []
        if result.get("sources"):
            for source in result["sources"]:
                if isinstance(source, dict):
                    content = source.get("content", "")
                    content = process_content(content)
                    sources.append(content[:200] + "..." if len(content) > 200 else content)
                else:
                    content = process_content(str(source))
                    sources.append(content[:200])

        # 处理answer的字符编码
        answer = result.get("answer", "")
        answer = process_content(answer)
        
        # 使用安全的方式打印日志，避免编码错误
        try:
            print(f"返回回答: {answer}")
        except Exception:
            print("返回回答: [编码错误]")
        return AskResponse(
            answer=answer,
            sources=sources,
            time="2026-04-23T14:00:00Z",
            mode="unified_rag"
        )
    except Exception as e:
        print(f"处理请求时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_compat(request: AskRequest):
    """兼容旧的请求路径"""
    return await ask(request)


@app.post("/api/multimodal-query")
async def multimodal_query(request: MultimodalQueryRequest):
    """多模态查询接口"""
    if not UNIFIED_RAG_AVAILABLE or not unified_rag_processor:
        raise HTTPException(status_code=503, detail="Unified RAG not available")

    try:
        result = unified_rag_processor.query_with_multimodal(
            request.query,
            request.multimodal_content
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process-documents")
async def process_documents():
    """处理所有文档"""
    if not UNIFIED_RAG_AVAILABLE or not unified_rag_processor:
        raise HTTPException(status_code=503, detail="Unified RAG not available")

    try:
        print("Starting document processing...")
        results = unified_rag_processor.process_folder()
        print("Document processing completed successfully")

        return {
            "status": "success",
            "results": results,
            "statistics": unified_rag_processor.get_statistics()
        }
    except Exception as e:
        print(f"Error processing documents: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/statistics")
async def get_statistics():
    """获取处理统计"""
    if not UNIFIED_RAG_AVAILABLE or not unified_rag_processor:
        raise HTTPException(status_code=503, detail="Unified RAG not available")

    try:
        return unified_rag_processor.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "service": "RAGentX Service",
        "unified_rag": "available" if UNIFIED_RAG_AVAILABLE else "unavailable",
        "redis": "available" if is_redis_available() else "unavailable"
    }


@app.post("/api/cache/clear")
async def clear_cache():
    """清除所有 RAG 查询缓存"""
    count = clear_all_cache()
    return {
        "status": "success",
        "message": f"Cleared {count} cache keys"
    }


@app.get("/health")
async def health_check_compat():
    """兼容旧的健康检查路径"""
    return {
        "status": "ok",
        "service": "RAGentX Service",
        "unified_rag": "available" if UNIFIED_RAG_AVAILABLE else "unavailable"
    }


@app.get("/")
async def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    print(f"Root path requested")
    print(f"Index path: {index_path}")
    print(f"Index path exists: {os.path.exists(index_path)}")
    if os.path.exists(index_path):
        print(f"Returning FileResponse for: {index_path}")
        return FileResponse(index_path)
    print("Frontend not found, returning HTML response")
    return HTMLResponse("<h1>RAGentX Service</h1><p>Frontend not found.</p>")


@app.get("/frontend/{file_path:path}")
async def serve_frontend(file_path: str):
    full_path = os.path.join(FRONTEND_DIR, file_path)
    if os.path.exists(full_path):
        return FileResponse(full_path)
    raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    print("Starting RAGentX Service...")
    print("=" * 60)
    print("RAGentX Service - Unified RAG with LangChain + Marker")
    print("=" * 60)
    print("Features:")
    print("  - LangChain for text splitting and retrieval")
    print("  - Marker/Fallback for PDF to Markdown conversion")
    print("  - Hybrid retrieval (vector + keyword)")
    print("  - Multimodal support (disabled by default)")
    print("=" * 60)
    print("Starting server on http://0.0.0.0:8000...")
    
    # 使用uvicorn直接启动
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
