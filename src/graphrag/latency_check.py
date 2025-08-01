import time
import requests

def test_knowledge_base(query="你好"):
    url = "http://localhost:8100/search/local"   # GraphRAG API 地址
    params = {"query": query}
    t0 = time.time()
    response = requests.get(url, params=params)
    t1 = time.time()
    if response.status_code == 200:
        print(f"[知识库查询耗时] {t1 - t0:.4f} 秒")
    else:
        print(f"[知识库查询失败] 状态码: {response.status_code}, 耗时: {t1 - t0:.4f} 秒")

def test_model_inference(prompt="你好", model_name="/root/autodl-tmp/unsloth/merged_model"):
    url = "http://localhost:8200/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": False
    }
    t0 = time.time()
    response = requests.post(url, headers=headers, json=payload)
    t1 = time.time()
    if response.status_code == 200:
        result = response.json()
        text = result["choices"][0]["message"]["content"]
        print(f"[模型推理耗时] {t1 - t0:.4f} 秒")
        print("模型输出:", text)
    else:
        print(f"[模型推理失败] 状态码: {response.status_code}, 响应内容: {response.text}")

if __name__ == "__main__":
    test_knowledge_base()
    test_model_inference()
