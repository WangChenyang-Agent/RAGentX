#!/usr/bin/env python3
"""
测试 RAG 知识库功能的脚本

使用方法：
1. 确保 RAG 服务正在运行（http://localhost:8000）
2. 运行：python test.py
3. 输入问题，按回车获取答案
4. 输入 'exit' 退出
"""

import requests
import json

def ask_question(question):
    """向 RAG 服务发送问题并获取答案"""
    url = "http://localhost:8000/api/ask"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "query": question
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # 检查响应状态
        result = response.json()
        return result
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

def display_answer(result):
    """显示回答和来源"""
    if result:
        print("\n=== 回答 ===")
        print(result.get("answer", "无回答"))
        print("\n=== 来源 ===")
        sources = result.get("sources", [])
        for i, source in enumerate(sources, 1):
            print(f"{i}. {source[:100]}..." if len(source) > 100 else f"{i}. {source}")
        print("\n" + "-" * 50 + "\n")
    else:
        print("无法获取回答\n")

def main():
    """主函数"""
    print("RAG 知识库测试工具")
    print("================")
    print("输入问题，按回车获取答案")
    print("输入 'exit' 退出\n")
    
    while True:
        question = input("问题: ")
        if question.lower() == 'exit':
            print("退出测试...")
            break
        if not question.strip():
            continue
        
        print("正在获取答案...")
        result = ask_question(question)
        display_answer(result)

if __name__ == "__main__":
    main()