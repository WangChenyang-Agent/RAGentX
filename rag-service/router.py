import re

class AdaptiveRouter:
    def __init__(self, retriever, reranker, generator, cache_service):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.cache_service = cache_service
    
    def classify_query(self, query):
        """分类问题类型"""
        query = query.lower()
        
        # 概念类问题
        if any(keyword in query for keyword in ["什么是", "定义", "概念", "原理", "如何", "怎样"]):
            return "concept"
        
        # 对比类问题
        if any(keyword in query for keyword in ["对比", "区别", "比较", "vs", "和"]):
            return "compare"
        
        # 代码类问题
        if any(keyword in query for keyword in ["代码", "实现", "编程", "write", "code"]):
            return "code"
        
        # 推理类问题
        if any(keyword in query for keyword in ["为什么", "原因", "原理", "机制", "如何工作"]):
            return "reasoning"
        
        # 默认类型
        return "concept"
    
    def route(self, query):
        """根据问题类型选择检索策略"""
        query_type = self.classify_query(query)
        
        if query_type == "concept":
            return self.handle_concept_query(query)
        elif query_type == "compare":
            return self.handle_compare_query(query)
        elif query_type == "code":
            return self.handle_code_query(query)
        elif query_type == "reasoning":
            return self.handle_reasoning_query(query)
        else:
            return self.handle_concept_query(query)
    
    def handle_concept_query(self, query):
        """处理概念类问题"""
        # 向量检索（Top-K）
        results = self.retriever.retrieve(query, k=5)
        reranked = self.reranker.rerank(query, results, top_n=3)
        context = [r["document"] for r in reranked]
        answer = self.generator.generate(query, reranked)
        return answer, context
    
    def handle_compare_query(self, query):
        """处理对比类问题"""
        # 扩展检索（多chunk）
        results = self.retriever.retrieve(query, k=8)
        reranked = self.reranker.rerank(query, results, top_n=5)
        context = [r["document"] for r in reranked]
        answer = self.generator.generate(query, reranked)
        return answer, context
    
    def handle_code_query(self, query):
        """处理代码类问题"""
        # 代码块优先
        results = self.retriever.retrieve(query, k=6)
        # 过滤出代码相关的文档
        code_related = [r for r in results if "code" in r["document"].lower()]
        if not code_related:
            code_related = results
        reranked = self.reranker.rerank(query, code_related, top_n=3)
        context = [r["document"] for r in reranked]
        answer = self.generator.generate(query, reranked)
        return answer, context
    
    def handle_reasoning_query(self, query):
        """处理推理类问题"""
        # 多上下文融合
        results = self.retriever.retrieve(query, k=7)
        reranked = self.reranker.rerank(query, results, top_n=4)
        context = [r["document"] for r in reranked]
        answer = self.generator.generate(query, reranked)
        return answer, context
