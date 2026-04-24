import requests

# 测试不同类型的问题
def test_questions():
    url = "http://localhost:8000/api/ask"
    
    # 测试问题列表
    questions = [
        "Go语言的局部变量分配在栈上还是堆上?",
        "草莓好吃吗",
        "非接口的任意类型 T() 都能够调用 *T 的方法吗？反过来呢？",
        "什么是rune类型？",
        "简述 Go GC 原理"
    ]
    
    for question in questions:
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
                print("\n引用来源:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source[:150]}...")
            else:
                print(f"错误: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"异常: {str(e)}")

if __name__ == "__main__":
    test_questions()