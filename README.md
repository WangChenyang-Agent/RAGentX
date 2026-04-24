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
- 模型：
  - 本地大模型（Ollama）

**优势**：
- 数据不出本地
- 生成质量更高
- 成本可控

### 自适应检索策略

根据问题类型动态选择检索策略，提高检索准确性和效率。

### Reranker 精排优化

```
Query → TopK → Rerank → TopN
```

- 使用 BGE reranker 提升相关性
- 减少无关上下文干扰

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
      ↓
Generator (Ollama)
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

- Ollama（本地模型）

## 项目结构

```
RAGentX/
├── data/
│   ├── docs/           # 文档目录
│   └── chunks.json     # 分块数据
├── frontend/           # 前端
│   └── index.html
├── rag-service/        # RAG 服务
│   ├── main.py         # FastAPI 入口
│   ├── requirements.txt
│   ├── embedding.py    # 向量嵌入
│   ├── generator.py    # 回答生成
│   ├── reranker.py     # 重排序
│   ├── unified_rag_processor.py  # 核心处理器
│   └── marker_pdf_processor.py   # PDF 处理
├── README.md
└── test_*.py           # 测试脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r rag-service/requirements.txt
```

### 2. 启动本地模型（Ollama）

```bash
ollama run deepseek-r1:1.5b
```

### 3. 启动服务

```bash
# 启动 RAG 服务
cd rag-service
python main.py
```

### 4. 测试

```bash
# 测试 Go GC 原理
python test_gc_principle.py

# 测试函数返回局部变量指针
python test_escape_analysis.py

# 测试 map key 判断
python test_map_key_check.py

# 测试 rune 类型
python test_rune_type.py
```

## 示例

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

## 项目亮点

- 设计并实现 **Adaptive RAG 检索策略**，动态选择检索方式
- 实现 **本地检索 + 模型生成** 的混合架构
- 使用 FAISS + reranker 提升问答准确率
- 基于 FastAPI 实现高性能 API 服务
- 支持 PDF 文档解析和处理

## 面试可讲点

- 为什么需要 Adaptive RAG？
- Reranker 如何提升效果？
- 如何平衡检索准确性和效率？
- 为什么使用 FAISS 进行向量检索？
- 如何处理模型幻觉和串台问题？

## License

MIT License