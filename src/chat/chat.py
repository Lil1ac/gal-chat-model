import requests
import json
import time

MAX_HISTORY_LEN = 10
MAX_KB_LEN = 1000

def knowledge_search(query: str):
    try:
        t0 = time.time()
        response = requests.get("http://localhost:8100/search/local", params={"query": query})
        response.raise_for_status()
        t1 = time.time()
        data = response.json()
        print(f"[知识库查询耗时] {t1 - t0:.4f} 秒")  # 增加耗时打印
        return data.get("response", ""), data.get("context_data", {})
    except Exception as e:
        print(f"[知识库查询失败] {e}")
        return "", {}

def trim_kb_text(text, max_len=MAX_KB_LEN):
    if not text:
        return ""
    text = str(text)
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text

def generate_response_stream(messages, max_tokens=1024, temperature=0.7):
    url = "http://localhost:8200/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "/root/autodl-tmp/unsloth/merged_model",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "stream": True
    }
    try:
        t0 = time.time()
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            response.raise_for_status()
            full_text = ""
            for line in response.iter_lines():
                if line:
                    if line.startswith(b"data: "):
                        data_str = line[len(b"data: "):].decode("utf-8")
                        if data_str == "[DONE]":
                            break
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"].get("content", "")
                        print(delta, end="", flush=True)
                        full_text += delta
            print()
        t1 = time.time()
        print(f"[模型推理耗时] {t1 - t0:.4f} 秒")  # 增加耗时打印
        return full_text.strip()
    except Exception as e:
        print(f"[模型调用失败] {e}")
        return ""

def trim_history(history, max_len=MAX_HISTORY_LEN):
    system = history[0]
    conv = history[1:]
    if len(conv) > max_len:
        conv = conv[-max_len:]
    return [system] + conv

import re
def clean_kb_response(text):
    if not text:
        return ""
    # 去掉 think 标签和 Data 标签
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'\[Data:.*?\]', '', text)
    return text.strip()

    
def rewrite_query_via_model(history, user_input):
    context = "\n".join([f"{msg['role']}：{msg['content']}" for msg in history if msg['role'] != 'system'])

    prompt = (
        "你是一个智能搜索助手。对话是直哉（用户）与本间心铃（角色）的交流。\n"
        "根据对话历史和直哉当前的问题，生成一句**精准、简洁**的知识库查询语句。\n"
        "生成规则：\n"
        "1. 如果问题与具体角色或对象（如“蓝是谁”）相关，直接围绕该对象生成查询。\n"
        "2. 如果问题与心铃本人或心铃相关的关系、经历、想法有关，则重点突出心铃。\n"
        "3. 保留亲密语气和对话关系背景，但不要无条件将所有查询都改成“心铃”。\n\n"
        f"对话历史：\n{context}\n"
        f"直哉的问题：{user_input}\n"
        "只输出查询语句，不要附加其他内容："
    )

    query = generate_response_stream([{"role": "user", "content": prompt}], max_tokens=30, temperature=0)
    query = re.sub(r'<.*?>', '', query).strip()
    return query







def main():
    print("角色扮演对话开始，输入 exit 退出")

    system_prompt = (
        "你是本间心铃，游戏《樱之刻》中的女主角，圣卢安学院的学生。\n"
        "你有一头黑色双马尾和棕色瞳孔，外貌与母亲中村丽华相似，举止得体，眼神锐利但温柔，"
        "性格乖巧且深思熟虑。\n"
        "你擅长绘画，是一位天才画家，能够敏锐地辨识真伪，善于观察细节并体察他人心情。\n"
        "你正在与直哉对话。\n"
        "体现她的性格特点和情感深度。\n"
    )

    conversation_history = [{"role": "system", "content": system_prompt}]

    while True:
        user_input = input("你说: ").strip()
        if user_input.lower() == "exit":
            print("对话结束")
            break
        
        # 先用模型根据对话历史和当前输入生成更精准的查询句
        search_query = rewrite_query_via_model(conversation_history, user_input)
        print(f"[重写后的查询句]: {search_query}")
        
        # 用生成的查询句去知识库查找
        kb_response, _ = knowledge_search(search_query)
        kb_response = clean_kb_response(kb_response)
        kb_response = trim_kb_text(kb_response)
        
        user_content = (
            f"你现在是本间心铃。"
            f"直哉刚刚对你说：『{user_input}』。\n"
            f"结合以下知识库线索，用心铃自己的话回答，不要直接照搬原文中的第一人称表达。（可参考，可忽略）：{kb_response}\n"
            f"请你用心铃的口吻直接回应直哉，"
            f"专注当前对话情绪，不要总结关系或分析立场，不要旁白式表达。"
        )
        
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history = trim_history(conversation_history, max_len=MAX_HISTORY_LEN)
        
        messages = [conversation_history[0]] + conversation_history[1:] + [{"role": "user", "content": user_content}]
        
        import pprint
        pprint.pprint(messages)
        
        print("心铃: ", end="", flush=True)
        answer = generate_response_stream(messages)
        print()
        
        conversation_history.append({"role": "assistant", "content": answer})
        conversation_history = trim_history(conversation_history, max_len=MAX_HISTORY_LEN)


if __name__ == "__main__":
    main()
