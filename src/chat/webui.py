import gradio as gr
import requests
import json
import re
import time

MAX_HISTORY_LEN = 10
MAX_KB_LEN = 1000

def knowledge_search(query: str):
    try:
        response = requests.get("http://localhost:8100/search/local", params={"query": query})
        response.raise_for_status()
        data = response.json()
        return data.get("response", ""), data.get("context_data", {})
    except Exception as e:
        return "", {}

def clean_kb_response(text):
    if not text:
        return ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'\[Data:.*?\]', '', text)
    return text.strip()

def generate_response_stream(messages, max_tokens=300, temperature=0.7):
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
        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line and line.startswith(b"data: "):
                    data_str = line[len(b"data: "):].decode("utf-8")
                    if data_str == "[DONE]":
                        break
                    data = json.loads(data_str)
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
    except Exception as e:
        yield f"[模型调用失败] {e}"

def rewrite_query_via_model(history, user_input):
    context = "\n".join([f"{msg[0]}：{msg[1]}" for msg in history[-MAX_HISTORY_LEN:]])
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
    query = ""
    for delta in generate_response_stream([{"role": "user", "content": prompt}], max_tokens=30, temperature=0):
        query += delta
    query = re.sub(r'<.*?>', '', query).strip()
    return query

def chatbot_interface(user_input, history):
    system_prompt = {
        "role": "system",
        "content": (
            "你是本间心铃，游戏《樱之刻》中的女主角，圣卢安学院的学生。\n"
            "你有一头黑色双马尾和棕色瞳孔，外貌与母亲中村丽华相似，举止得体，眼神锐利但温柔，"
            "性格乖巧且深思熟虑。\n"
            "你擅长绘画，是一位天才画家，能够敏锐地辨识真伪，善于观察细节并体察他人心情。\n"
            "你正在与直哉对话。\n"
            "体现她的性格特点和情感深度。\n"
            )
    }

    # 先用对话历史和当前输入生成精准查询
    search_query = rewrite_query_via_model(history, user_input)

    # 查询知识库
    kb_response, _ = knowledge_search(search_query)
    kb_response = clean_kb_response(kb_response)
    if len(kb_response) > MAX_KB_LEN:
        kb_response = kb_response[:MAX_KB_LEN] + "..."

    user_content = (
            f"你现在是本间心铃。"
            f"直哉刚刚对你说：『{user_input}』。\n"
            f"结合以下知识库线索，用心铃自己的话回答，不要直接照搬原文中的第一人称表达。（可参考，可忽略）：{kb_response}\n"
            f"请你用心铃的口吻直接回应直哉，"
            f"专注当前对话情绪，不要总结关系或分析立场，不要旁白式表达。"
    )

    messages = [system_prompt]
    # 限制历史长度
    for u, a in history[-MAX_HISTORY_LEN:]:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_content})

    partial_answer = ""
    for delta in generate_response_stream(messages, max_tokens=300):
        partial_answer += delta
        yield history + [(user_input, partial_answer)]
    history.append((user_input, partial_answer))

def clear_history():
    return [], []

with gr.Blocks(css="""
#chatbot {height: 600px}
.send-btn {background-color: #4CAF50; color: white;}
""") as demo:
    gr.Markdown("## 本间心铃 - 对话系统")
    chatbot = gr.Chatbot(elem_id="chatbot", label="对话")
    with gr.Row():
        with gr.Column(scale=8):
            msg = gr.Textbox(show_label=False, placeholder="输入你的话...")
        with gr.Column(scale=2, min_width=80):
            send = gr.Button("发送", elem_classes="send-btn")
    clear = gr.Button("清空对话")

    state = gr.State([])

    send.click(fn=chatbot_interface,
               inputs=[msg, state],
               outputs=[chatbot],
               show_progress=False)
    msg.submit(fn=chatbot_interface,
               inputs=[msg, state],
               outputs=[chatbot],
               show_progress=False)
    clear.click(clear_history, None, [chatbot, state], queue=False)

demo.launch(server_name="0.0.0.0", server_port=7860)
