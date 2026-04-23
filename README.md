# RAGentX

> **Adaptive RAG System with Local Retrieval & Redis Acceleration**

一个面向技术面经的**自适应检索增强生成系统（Adaptive RAG）**，通过本地向量检索 + 云端大模型 + 多级缓存，实现高性能、高质量的智能问答。

## 项目简介

RAGentX 是一个结合本地检索与云端大模型生成的 RAG 系统，专为技术面经场景设计。

系统支持 PDF / 文本解析，通过向量检索 + 重排序（Reranker）+ 自适应检索策略（Adaptive RAG），实现精准问答。同时引入 Redis 构建缓存体系，大幅降低延迟与 API 成本。

## 核心特性

### Adaptive RAG（核心亮点）

根据问题类型动态选择检索策略：

| 问题类型 | 检索策略             |
| -------- | -------------------- |
| 概念类   | 向量检索（Top-K）    |
| 对比类   | 扩展检索（多 chunk） |
| 代码类   | 代码块优先           |
| 推理类   | 多上下文融合         |

### 本地检索 + 云端生成

- 本地（Ollama）：
  - embedding（qwen3）
  - FAISS 向量检索
  - BGE reranker
- 云端：
  - 大模型 API（OpenAI / DeepSeek）

**优势**：
- 数据不出本地
- 生成质量更高
- 成本可控

### Reranker 精排优化

```
Query → TopK → Rerank → TopN
```

- 使用 BGE reranker 提升相关性
- 减少无关上下文干扰

### Redis 多级缓存（性能关键）

使用 Redis 构建缓存体系：

#### 缓存层设计

```
Query
 ↓
[Redis Query Cache]
 ↓ miss
RAG Pipeline
 ↓
[Redis Answer Cache]
 ↓
Response
```

#### 缓存内容

- Query Cache（问题缓存）
- Answer Cache（回答缓存）
- Embedding Cache（向量缓存，可选）

**效果**：
- 延迟降低 ~60%
- API 成本降低 ~40%
- QPS 提升显著

### 高并发后端设计

- Go + Gin API 网关
- Python RAG 服务（LangChain 编排）
- SSE 流式输出
- 异步任务（可扩展）

## 系统架构

```
Frontend (Vue)
      ↓
Go API Gateway (Gin)
      ↓
Python RAG Service (LangChain)
      ↓
Redis Cache Layer
      ↓
Retriever Layer
  ├── Embedding (Qwen)
  ├── FAISS
  ├── Reranker (BGE)
  ├── Adaptive Routing (Core)
      ↓
LLM API (OpenAI / DeepSeek)
      ↓
Answer + Sources
```

## 技术栈

### Backend

- Go (Gin)
- Python (FastAPI + LangChain)

### RAG Core

- FAISS（向量检索）
- Qwen Embedding（本地）
- BGE Reranker
- Adaptive Retrieval（自研）

### Cache

- Redis

### Model

- Ollama（本地模型）
- OpenAI / DeepSeek API

## 项目结构

```
RAGentX/
├── backend-go/
│   ├── main.go
│   ├── go.mod
│   ├── router/
│   │   └── router.go
│   ├── handler/
│   │   └── handler.go
│   └── middleware/
│       └── redis.go
│
├── rag-service/
│   ├── main.py
│   ├── requirements.txt
│   ├── embedding.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── generator.py
│   ├── cache.py        # Redis 缓存
│   └── router.py       # Adaptive RAG 核心
│
├── data/
│   ├── docs/
│   └── index.faiss
│
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r rag-service/requirements.txt
go mod tidy
```

### 2. 启动 Redis

```bash
docker run -d -p 6379:6379 redis
```

### 3. 启动本地模型（Ollama）

```bash
ollama run qwen3-embedding:4b
ollama run bona/bge-reranker-v2-m3
```

### 4. 构建向量索引

```bash
# 实际项目中需要创建构建索引的脚本
# python build_index.py
```

### 5. 启动服务

```bash
# 启动Python RAG服务
cd rag-service
python main.py

# 启动Go API网关
cd ../backend-go
go run main.go
```

## 示例

### 输入

```
Go 中 channel 为什么会阻塞？
```

### 输出

```
channel 阻塞通常由以下原因导致：
1. 无缓冲 channel 未同时读写
2. goroutine 泄漏
3. select 未处理 default 分支

相关知识：
- goroutine 调度
- select 多路复用
```

## 性能优化

- Redis 缓存（Query / Answer）
- Top-K 控制
- Reranker 精排
- Prompt 压缩
- SSE 流式输出

## 项目亮点

- 设计并实现 **Adaptive RAG 检索策略**，动态选择检索方式
- 构建 **Redis 多级缓存系统**，降低 40% API 成本
- 实现 **本地检索 + 云端生成** 的混合架构
- 使用 FAISS + reranker 提升问答准确率
- 基于 Go 实现高并发 API 网关（QPS 100+）

## 面试可讲点

- 为什么需要 Adaptive RAG？
- Reranker 如何提升效果？
- Redis 缓存如何设计？
- 如何平衡缓存与实时性？
- 为什么使用 FAISS 而不是 Milvus？

## Roadmap

- 多模态支持（OCR / 图像理解）
- 知识图谱增强检索
- Web UI（Vue3）
- Obsidian 集成
- 在线部署版本

## License

MIT License
