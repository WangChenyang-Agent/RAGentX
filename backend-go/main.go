package main

import (
	"RAGentX/backend-go/handler"
	"RAGentX/backend-go/middleware"
	"RAGentX/backend-go/router"
	"fmt"
	"github.com/gin-gonic/gin"
)

func main() {
	// 初始化Redis
	middleware.InitRedis()

	// 创建Gin引擎
	r := gin.Default()

	// 注册路由
	router.SetupRoutes(r, handler.NewHandler())

	// 启动服务
	port := 8080
	fmt.Printf("Server running on port %d\n", port)
	r.Run(fmt.Sprintf(":%d", port))
}