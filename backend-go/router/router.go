package router

import (
	"RAGentX/backend-go/handler"
	"github.com/gin-gonic/gin"
)

func SetupRoutes(r *gin.Engine, h *handler.Handler) {
	// API路由组
	api := r.Group("/api")
	{
		// 问答接口
		api.POST("/ask", h.Ask)
		
		// 健康检查
		api.GET("/health", h.HealthCheck)
	}
}