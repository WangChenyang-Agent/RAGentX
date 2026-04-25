# RAGentX

> **Adaptive RAG System for Technical Interview Q&A**

一个面向技术面经的**自适应检索增强生成系统（Adaptive RAG）**，通过本地向量检索 + 大模型生成，实现高性能、高质量的智能问答。

## 项目简介

RAGentX 是一个结合本地检索与大模型生成的 RAG 系统，专为技术面经场景设计。

系统支持 PDF / 文本解析，通过向量检索 + 自适应检索策略，实现精准问答。

## 核心特性

### 本地检索 + 模型生成

- 本地：
  - embedding（Qwen3）
  - FAISS 向量检索
  - BGE reranker
  - Redis 缓存
- 模型：
  - DeepSeek API（deepseek-v4-pro）

**优势**：
- 数据不出本地（检索部分）
- 生成质量更高
- 响应速度快
- 缓存加速，系统吞吐量高

### 自适应检索策略

根据问题类型动态选择检索策略，提高检索准确性和效率。

### Reranker 精排优化

```
Query → TopK → Rerank → TopN
```

- 使用 BGE reranker 提升相关性
- 减少无关上下文干扰

### 智能分块与排序

- 基于语义的文本分块，支持QA级和细粒度子chunk混合索引
- 智能排序，优先返回定义类内容，提升检索准确性

### Markdown 渲染支持

- 前端支持 Markdown 格式渲染，包括代码块和标题样式
- 保留原始文档的格式结构

## 系统架构

```
Frontend
      ↓
FastAPI Service
      ↓
Unified RAG Processor
      ↓
Retriever Layer
  ├── Embedding (Qwen)
  ├── FAISS Vector Store
  ├── Reranker (BGE)
  └── Redis Cache
      ↓
Generator (DeepSeek API)
      ↓
Answer + Sources
```

## 技术栈

### Backend

- Python (FastAPI)
- LangChain

### RAG Core

- FAISS（向量检索）
- Qwen Embedding（本地）
- BGE Reranker
- Adaptive Retrieval

### Model

- DeepSeek API（deepseek-v4-pro）

## 项目结构

```
RAGentX/
├── data/
│   ├── docs/           # 文档目录
│   ├── formatted/      # 处理后的文档
│   └── faiss_index/    # 向量索引
├── frontend/           # 前端
│   └── index.html
├── rag-service/        # RAG 服务
│   ├── main.py         # FastAPI 入口
│   ├── requirements.txt
│   ├── embedding.py    # 向量嵌入
│   ├── generator.py    # 回答生成
│   ├── reranker.py     # 重排序
│   ├── redis_cache.py  # Redis 缓存
│   ├── unified_rag_processor.py  # 核心处理器
│   ├── marker_pdf_processor.py   # PDF 处理
│   └── process_markdown.py       # Markdown 处理
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
cd rag-service
pip install -r requirements.txt
```

### 2. 启动 Redis 容器

```bash
docker run -d -p 6379:6379 --name redis redis
```

### 3. 配置 DeepSeek API

在 `rag-service/.env` 文件中添加 DeepSeek API 密钥：

```env
# DeepSeek API配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### 4. 启动服务

```bash
# 启动 RAG 服务
cd rag-service
python main.py
```

服务将在 http://0.0.0.0:8000 上运行

## 使用方法

### 1. 处理文档

服务启动后，会自动加载现有的向量索引。如果需要处理新的文档，可以调用API：

```bash
# 处理所有文档
curl -X POST http://localhost:8000/api/process-documents
```

### 2. 发送查询

使用POST请求发送查询：

```bash
# 发送查询
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是协程", "use_unified_rag": true, "top_k": 3, "retrieval_mode": "hybrid"}'
```

### 3. 访问前端

打开浏览器访问 http://localhost:8000，使用前端界面进行交互。

### 4. 缓存管理

```bash
# 清除所有缓存
curl -X POST http://localhost:8000/api/cache/clear

# 查看缓存状态
curl http://localhost:8000/api/health
```

## 示例

### 输入

```
什么是协程
```

### 输出

```
协程（Goroutine）是与其他函数或方法同时运行的函数或方法。Goroutines 可以被认为是轻量级的线程。与线程相比，创建 Goroutine 的开销很小。Go 应用程序同时运行数千个 Goroutine 是非常常见的做法。
```

### 输入

```
简述 Go GC 原理
```

### 输出

```
Go GC 使用标记清除算法，分为标记和清除两个阶段，采用三色标记法（黑、灰、白），通过写屏障技术处理并发标记时的引用变化，整个过程分为标记准备、标记、标记结束、清理四个阶段，其中标记准备和标记结束阶段会暂停程序，并发标记阶段可以与程序并行执行。
```

## 性能优化

- Top-K 控制（减少召回数量，避免噪声）
- Reranker 精排（提升相关性）
- Prompt 优化（减少幻觉和串台）
- 答案后处理（确保格式正确）
- 智能分块（提升检索准确性）
- 自适应排序（优先返回定义类内容）
- Redis 缓存（减少重复查询延迟，提升系统吞吐量）

## 项目亮点

- 设计并实现 **Adaptive RAG 检索策略**，动态选择检索方式
- 实现 **本地检索 + 模型生成** 的混合架构
- 使用 FAISS + reranker 提升问答准确率
- 基于 FastAPI 实现高性能 API 服务
- 支持 PDF 文档解析和处理
- 智能分块与排序，提升检索准确性
- Markdown 渲染支持，保留原始文档格式
- 集成 Redis 缓存，提升系统响应速度和吞吐量

## License

MIT License