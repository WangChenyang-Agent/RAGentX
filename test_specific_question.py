# -*- coding: utf-8 -*-
import sys
import io
import os

# 设置环境变量强制使用UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 设置默认编码为UTF-8，避免Windows下的GBK编码问题
if sys.platform == 'win32':
    # Windows系统特殊处理
    import locale
    # 尝试设置控制台代码页为UTF-8
    os.system('chcp 65001 >nul 2>&1')
    
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import requests

def safe_print(content):
    """安全打印，处理编码问题"""
    try:
        print(content)
    except Exception:
        try:
            print(content.encode('utf-8', errors='replace').decode('utf-8'))
        except Exception:
            print("[编码错误，无法显示内容]")

# 测试之前失败的问题
def test_failing_question():
    url = "http://localhost:8000/api/ask"
    
    # 之前失败的问题
    question = "非接口的任意类型 T() 都能够调用 *T 的方法吗？反过来呢？"
    
    print(f"\n=== 测试：{question} ===")
    payload = {
        "query": question,
        "top_k": 3
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print("回答:")
            print(result['answer'])
            safe_print("\n引用来源:")
            for i, source in enumerate(result['sources'], 1):
                safe_print(f"{i}. {source[:150]}...")
        else:
            print(f"错误: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"异常: {str(e)}")

if __name__ == "__main__":
    test_failing_question()