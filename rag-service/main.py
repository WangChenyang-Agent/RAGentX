from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn
import os

from embedding import EmbeddingService
from reranker import Reranker
from generator import Generator

app = FastAPI(title="RAGentX Service")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

unified_rag_processor = None

try:
    from unified_rag_processor import UnifiedRAGProcessor
    UNIFIED_RAG_AVAILABLE = True
except ImportError as e:
    print(f"Unified RAG not available: {e}")
    UNIFIED_RAG_AVAILABLE = False


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


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global unified_rag_processor

    print("Starting RAGentX Service...")
    print(f"UNIFIED_RAG_AVAILABLE: {UNIFIED_RAG_AVAILABLE}")

    if UNIFIED_RAG_AVAILABLE:
        print("Initializing Unified RAG Processor (LangChain + Marker)...")
        try:
            # 禁用Marker初始化，避免模型下载阻塞
            print("Creating UnifiedRAGProcessor instance...")
            unified_rag_processor = UnifiedRAGProcessor()
            print("Unified RAG Processor instance created!")
            print("Loading existing index...")
            # 手动加载索引，以便查看详细信息
            if unified_rag_processor.vectorstore:
                print(f"Vector index loaded successfully with {len(unified_rag_processor.chunks)} chunks")
            else:
                print("No existing vector index found")
            print("Unified RAG Processor initialized!")
        except Exception as e:
            print(f"Failed to initialize Unified RAG Processor: {e}")
            import traceback
            traceback.print_exc()
            unified_rag_processor = None
    else:
        print("Unified RAG not available, skipping initialization")
    
    print("RAGentX Service startup completed!")


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        if not UNIFIED_RAG_AVAILABLE or not unified_rag_processor:
            raise HTTPException(status_code=503, detail="Unified RAG not available")

        result = await unified_rag_processor.aquery(
            request.query,
            top_k=request.top_k,
            mode=request.retrieval_mode
        )

        sources = []
        if result.get("sources"):
            for source in result["sources"]:
                if isinstance(source, dict):
                    content = source.get("content", "")
                    sources.append(content[:200] + "..." if len(content) > 200 else content)
                else:
                    sources.append(str(source)[:200])

        return AskResponse(
            answer=result.get("answer", ""),
            sources=sources,
            time="2026-04-23T14:00:00Z",
            mode="unified_rag"
        )
    except Exception as e:
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
        results = unified_rag_processor.process_folder()

        return {
            "status": "success",
            "results": results,
            "statistics": unified_rag_processor.get_statistics()
        }
    except Exception as e:
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
        "unified_rag": "available" if UNIFIED_RAG_AVAILABLE else "unavailable"
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
    if os.path.exists(index_path):
        return FileResponse(index_path)
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
    
    # 直接使用uvicorn命令启动，而不是通过Python API
    import subprocess
    subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
