package handler

import (
	"RAGentX/backend-go/middleware"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

type Handler struct{}

type AskRequest struct {
	Query string `json:"query" binding:"required"`
}

type AskResponse struct {
	Answer  string   `json:"answer"`
	Sources []string `json:"sources"`
	Time    string   `json:"time"`
}

func NewHandler() *Handler {
	return &Handler{}
}

func (h *Handler) Ask(c *gin.Context) {
	var req AskRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	// 检查Redis缓存
	if cached, found := middleware.GetFromCache(req.Query); found {
		c.JSON(http.StatusOK, cached)
		return
	}

	// 调用RAG服务
	answer, sources := h.callRagService(req.Query)

	// 构建响应
	resp := AskResponse{
		Answer:  answer,
		Sources: sources,
		Time:    time.Now().Format(time.RFC3339),
	}

	// 存入Redis缓存
	middleware.SetToCache(req.Query, middleware.AskResponse{
		Answer:  resp.Answer,
		Sources: resp.Sources,
		Time:    resp.Time,
	})

	c.JSON(http.StatusOK, resp)
}

func (h *Handler) callRagService(query string) (string, []string) {
	// 这里调用Python RAG服务
	// 实际项目中应该使用HTTP请求
	return "This is a sample answer for: " + query, []string{"source1", "source2"}
}

func (h *Handler) HealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":  "ok",
		"service": "RAGentX API Gateway",
		"time":    time.Now().Format(time.RFC3339),
	})
}
