from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from embedding import EmbeddingService
from retriever import Retriever
from reranker import Reranker
from generator import Generator
from router import AdaptiveRouter
from cache import CacheService

app = FastAPI(title="RAGentX Service")

# 初始化服务
embedding_service = EmbeddingService()
retriever = Retriever(embedding_service)
reranker = Reranker()
generator = Generator()
cache_service = CacheService()
adaptive_router = AdaptiveRouter(retriever, reranker, generator, cache_service)

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    time: str

@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        # 检查缓存（暂时禁用）
        # cached = cache_service.get(request.query)
        # if cached:
        #     return AskResponse(**cached)
        
        # 使用Adaptive RAG处理
        answer, sources = adaptive_router.route(request.query)
        
        # 构建响应
        response = AskResponse(
            answer=answer,
            sources=sources,
            time="2026-04-23T14:00:00Z"
        )
        
        # 存入缓存（暂时禁用）
        # cache_service.set(request.query, response.model_dump())
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "RAGentX RAG Service"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
