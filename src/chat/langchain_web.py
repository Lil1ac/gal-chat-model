import gradio as gr
import re
import requests
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage  # 从这里导入
from langchain.schema import BaseMessage, HumanMessage, AIMessage


class GraphRAGRetriever(BaseRetriever):
    endpoint: str = "http://localhost:8100"

    def _get_relevant_documents(self, query: str) -> List[Document]:
        print(f"[检索器] 请求知识库接口，查询语句：{query}")
        try:
            resp = requests.get(f"{self.endpoint}/search/local", params={"query": query}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("context_text", "") or data.get("response", "")
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            text = re.sub(r"\[Data:.*?\]", "", text).strip()
            print(f"[检索器] 返回知识库文本：{text}")
            return [Document(page_content=text)]
        except Exception as e:
            print(f"[检索器] 调用失败，异常：{e}")
            return [Document(page_content=f"[GraphRAG调用失败] {e}")]


from langchain_openai import ChatOpenAI
from typing import Any, Dict, List


class Qwen3ChatOpenAI(ChatOpenAI):
    def _call(self, messages: List[Dict[str, Any]], stop: List[str] = None, **kwargs) -> str:
        # 这里把 enable_thinking=False 传给底层接口
        if "params" not in kwargs:
            kwargs["params"] = {}
        kwargs["params"]["enable_thinking"] = False

        # 调用父类方法，传递参数
        return super()._call(messages, stop=stop, **kwargs)


base_llm = Qwen3ChatOpenAI(
    model="/root/autodl-tmp/unsloth/merged_model_no_think",
    temperature=0.7,
    streaming=True,
    openai_api_base="http://localhost:8200/v1",
    api_key="dummy"
)

base_llm_qwen = Qwen3ChatOpenAI(
    model="/root/autodl-tmp/Qwen/Qwen3-8B",
    temperature=0.7,
    streaming=True,
    openai_api_base="http://localhost:8000/v1",
    api_key="dummy"
)

# 使用base_llm创建转换chain
from langchain.chains import LLMChain

rewrite_prompt = PromptTemplate(
    input_variables=["history", "question"],
    template=(
        "你是一个智能搜索助手，帮助用户生成精准、简洁的知识库查询。\n"
        "根据对话历史和直哉当前的问题，生成一句**精准、简洁**的知识库查询语句。"
        "生成规则："
        "1. 如果问题与具体角色或对象相关，直接围绕该对象生成查询。"
        "2. 如果问题与心铃本人或心铃相关的关系、经历、想法有关，则重点突出心铃。"
        "3. 保留亲密语气和对话关系背景，但不要无条件将所有查询都改成“心铃”。"
        "对话历史：{history}\n"
        "用户问题：{question}\n"
        "请只输出最终的查询语句，不要附加其他内容。"
    )
)
rewrite_chain = LLMChain(llm=base_llm_qwen, prompt=rewrite_prompt)
retriever = GraphRAGRetriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

system_message = SystemMessage(content=(
    "你是本间心铃，游戏《樱之刻》中的女主角，圣卢安学院的学生。\n"
    "你有一头黑色双马尾和棕色瞳孔，举止得体，眼神锐利但温柔，"
    "性格乖巧且深思熟虑。\n"
    "你擅长绘画，是一位天才画家，能够敏锐地辨识真伪，善于观察细节并体察他人心情。\n"
    "你正在与直哉对话。\n"
))

# **重点修改这里，必须包含 context，且声明 input_variables**
qa_prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="chat_history"),
    ("human",
     "{question}\n"
     "结合以下知识库线索，用你自己的话回答。"
     "：{context}\n"
     "（知识库仅作为参考，可忽略）请你用第一人称的口吻直接回应直哉。")
])
qa_prompt.input_variables = ["question", "context", "chat_history"]  # 这里必须加，否则报错
stuff_chain = LLMChain(llm=base_llm, prompt=qa_prompt)  # 你自己的QA prompt链

# 新增一个把结构化关系转换成自然语言的Prompt和chain
summary_prompt = PromptTemplate(
    input_variables=["raw_knowledge"],
    template=(
        "请将以下结构化的知识点转换成连贯的自然语言描述，"
        "使其更适合用作对话系统的知识库内容：\n\n"
        "{raw_knowledge}\n\n"
        "请用简洁流畅的语言表达，不要逐条罗列。"
    )
)

summary_chain = LLMChain(llm=base_llm_qwen, prompt=summary_prompt)

simple_chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")  # 直接传用户输入
])
simple_chat_prompt.input_variables = ["question", "chat_history"]


class GradioStreamCallback(BaseCallbackHandler):
    def __init__(self):
        self.tokens = ""
        self.on_update = None

    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens += token
        if self.on_update:
            self.on_update(self.tokens)

    def get_text(self):
        return self.tokens


def format_history_for_gradio(history: List[Tuple[str, str]]):
    """
    格式化聊天记录，给用户和助手的消息分别添加class，前端根据class区分左右气泡
    """
    msgs = []
    for user_text, bot_text in history:
        msgs.append({"role": "user", "content": user_text, "class_name": "chatbot-user"})
        styled_bot_text = process_cot_tags(add_cot_tags(bot_text))
        msgs.append({"role": "assistant", "content": styled_bot_text, "class_name": "chatbot-assistant"})
    return msgs


# ======= 修改这里，把括号替换成 <cot> 标签 =======
def add_cot_tags(text: str) -> str:
    def replacer(match):
        inner = match.group(1)
        return f"<cot>{inner}</cot>"

    # 支持 ( ... ) 和 （ ... ）
    return re.sub(r'[（(](.*?)[）)]', replacer, text, flags=re.DOTALL)


# ======= 处理 <cot> 标签，转成html带缩进的思考块 =======
def process_cot_tags(text: str) -> str:
    level = 0

    def replace_cot(match):
        nonlocal level
        inner_text = match.group(1)
        current_level = level
        level += 1
        return f'<div class="thought-process cot-level-{current_level}"><i>💭 {inner_text}</i></div>'

    return re.sub(r'<cot>(.*?)</cot>', replace_cot, text, flags=re.DOTALL)


def remove_inner_brackets(text: str) -> str:
    """去掉括号及括号内容，用于TTS"""
    return re.sub(r"[（(].*?[）)]", "", text)


import requests


def synthesize_speech(text: str, filename: str = "output.wav"):
    local_url = "http://127.0.0.1:9880/tts"
    print(f"[音频] 请求GPT-SOVITS，生成语音：{text}")
    data = {
        "text": text,
        "text_lang": "zh",
        "ref_audio_path": "/root/autodl-tmp/fem_mis_00051.ogg",
        "prompt_text": "初対面で色々とはぐらかされると，あまりいい気分ではありません",
        "prompt_lang": "ja",
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(local_url, json=data, headers=headers)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"✅ TTS 合成成功: {filename}")
        return filename
    else:
        print(f"❌ TTS 失败: {response.text}")
        return None


def chatbot_interface(user_input: str, history: List[Tuple[str, str]], kb_flag: bool):
    print("======================================")
    print("[用户输入]", user_input)
    print(f"[历史记录] 共{len(history)}条，最近5条：{history[-5:] if history else '无历史'}")

    history.append((user_input, ""))
    # 第一次 yield（清空输入框、暂时没有音频）
    yield format_history_for_gradio(history), history, "", None

    callback = GradioStreamCallback()

    def on_token(token: str, **kwargs):
        callback.tokens += token
        # history 里存的是原始文本（无 HTML）
        history[-1] = (user_input, callback.tokens)

    callback.on_llm_new_token = on_token

    import threading
    import time

    done_flag = threading.Event()

    def invoke_chain():
        try:
            # ===== 使用memory而不是手动format =====
            memory.chat_memory.add_user_message(user_input)

            if kb_flag:
                # 用知识库检索并拼接上下文
                context = "\n".join([f"{u}：{a}" for u, a in history[-10:]])
                rewrite_result = rewrite_chain.invoke({"history": context, "question": user_input})
                search_query = rewrite_result["text"]

                # 调用检索器拿结构化知识
                docs = retriever._get_relevant_documents(search_query)

                # 这里将结构化知识的文本先拼接起来，传给摘要chain转换成自然语言
                raw_knowledge = "\n".join(doc.page_content for doc in docs)
                natural_language_knowledge = summary_chain.invoke({"raw_knowledge": raw_knowledge})
                natural_language_knowledge = natural_language_knowledge["text"]
                natural_language_knowledge = re.sub(r"<think>.*?</think>", "", natural_language_knowledge,
                                                    flags=re.DOTALL)
                print(f"[转换后的自然语言知识]: {natural_language_knowledge}")

                qa_input = {
                    "question": user_input,
                    "context": natural_language_knowledge,
                    **memory.load_memory_variables({})
                }
                stuff_chain.invoke(qa_input, config={"callbacks": [callback]})

            else:
                inputs = {"question": user_input, **memory.load_memory_variables({})}
                messages = simple_chat_prompt.format_messages(**inputs)
                base_llm.invoke(messages, config={"callbacks": [callback]})

            # 把模型回答写入memory
            memory.chat_memory.add_ai_message(callback.tokens)
        finally:
            done_flag.set()

    thread = threading.Thread(target=invoke_chain)
    thread.start()

    last_len = 0
    while not done_flag.is_set() or last_len < len(callback.tokens):
        if len(callback.tokens) > last_len:
            last_len = len(callback.tokens)
            # 仅用于显示的 HTML
            styled_text = process_cot_tags(add_cot_tags(callback.tokens))
            yield format_history_for_gradio(history[:-1] + [(user_input, styled_text)]), history, "", None
        time.sleep(0.1)

    thread.join()

    print("[模型回答]:", callback.tokens)

    history[-1] = (user_input, callback.tokens)
    # 调用TTS
    # === 去除心理活动内容再生成音频 ===
    tts_text = remove_inner_brackets(callback.tokens)
    tts_file = synthesize_speech(tts_text, filename="output.wav")
    # 最终返回4个值（包含语音路径）
    styled_text = process_cot_tags(add_cot_tags(callback.tokens))
    yield format_history_for_gradio(history[:-1] + [(user_input, styled_text)]), history, "", tts_file
    print("======================================\n")


css_style = """
/* 页面基础样式，左右水平居中 */
body {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    background: linear-gradient(135deg, #e3f2fd, #fce4ec);
}

/* 主容器，竖直布局，宽度固定，高度90vh */
#main-container {
    flex-direction: column;
    width: 880px;
    height: 90vh;
    padding: 24px;
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    border: 1px solid rgba(255,255,255,0.4);
}

/* 聊天区域，撑满剩余空间并允许滚动 */
#chatbot {
    flex: 1 1 auto;
    overflow-y: auto;
    background: rgba(255,255,255,0.25);
    backdrop-filter: blur(8px);
    border-radius: 16px;
    padding: 16px;
    box-shadow: inset 0 0 12px rgba(0,0,0,0.08);
    margin-bottom: 12px;
}

/* 输入区域容器，固定高度，不收缩 */
.input-container {
    position: relative;
    width: 100%;
    flex-shrink: 0;
}

/* 输入框，100%宽度，右侧留空间给发送按钮 */
.input-container textarea {
    width: 100%;
    font-size: 16px;
    padding: 12px 48px 12px 12px;
    background: rgba(255,255,255,0.35);
    transition: background 0.3s ease;
    resize: none;
    box-sizing: border-box;
    border-radius: 8px;
    border: 1px solid #ccc;
}

.input-container textarea:focus {
    background: rgba(255,255,255,0.6);
    outline: none;
    border-color: #10a37f;
}

/* 发送按钮，绝对定位在输入框右下角 */
.send-btn {
    position: absolute !important;
    bottom: 10px;
    right: 10px;
    width: 36px !important;
    height: 36px !important;
    border-radius: 50% !important;
    background: #10a37f !important;
    color: white !important;
    font-size: 16px !important;
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: pointer;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    border: none;
    z-index: 2;
}

/* 清除和切换知识库按钮容器，横向排列 */
#btn-row {
    margin-top: 10px;
    display: flex;
    gap: 12px;
}

/* 清除按钮和切换知识库按钮样式 */
.clear-btn {
    padding: 4px 10px;
    font-size: 13px;
    border-radius: 8px;
    background: #f5f5f5;
    color: #333;
    cursor: pointer;
    transition: background 0.2s ease;
    border: none;
}
.clear-btn:hover {
    background: #e0e0e0 !important;
}

/* 用户聊天气泡（右侧） */
.chatbot-user {
    background: rgba(186,104,200,0.9);
    color: white;
    padding: 12px 18px;
    border-radius: 22px 22px 0 22px;
    max-width: 70%;
    margin-left: auto;  /* 靠右 */
    margin-bottom: 14px;
    word-wrap: break-word;
    font-size: 16px;
    line-height: 1.4;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}

/* 助手聊天气泡（左侧） */
.chatbot-assistant {
    background: rgba(255,255,255,0.85);
    color: #333;
    padding: 12px 18px;
    border-radius: 22px 22px 22px 0;
    max-width: 70%;
    margin-right: auto; /* 靠左 */
    margin-bottom: 14px;
    word-wrap: break-word;
    font-size: 16px;
    line-height: 1.4;
    box-shadow: 0 4px 10px rgba(0,0,0,0.12);
}


/* 思考块 */
.thought-process {
    font-style: italic;
    color: #7b7b7b;
    margin-left: 16px;
    margin-top: 6px;
    margin-bottom: 6px;
    border-left: 3px solid #ba68c8;
    padding-left: 8px;
    background-color: rgba(255,255,255,0.4);
    border-radius: 4px;
    font-size: 14px;
}
"""


def clear_all():
    memory.clear()  # 清空 langchain 内存
    return [], [], "", None


def toggle_kb_fn(current_state: bool):
    new_state = not current_state
    label = "知识库: 开启" if new_state else "知识库: 关闭"
    return new_state, label


with gr.Blocks(css=css_style) as demo:
    with gr.Column(elem_id="main-container"):
        gr.Markdown("### 🌸 本间心铃 - 智能对话系统")

        chatbot = gr.Chatbot(elem_id="chatbot", label="对话", type="messages")

        # 输入框 + 发送按钮（悬浮在输入框内部）
        with gr.Row():
            with gr.Column(elem_classes="input-container"):
                msg = gr.Textbox(show_label=False, placeholder="和心铃说点什么吧…", lines=1)
                send = gr.Button("➤", elem_classes="send-btn")
        # 按钮区：清空 + 知识库切换
        with gr.Row():
            clear = gr.Button("清空对话", elem_classes="clear-btn")
            toggle_kb = gr.Button("知识库: 开启", elem_classes="clear-btn")
        with gr.Row():
            audio_player = gr.Audio(label="语音播放", type="filepath", autoplay=True)

        # 状态
        state = gr.State([])  # 聊天历史
        kb_state = gr.State(True)  # 知识库状态

        send.click(fn=chatbot_interface, inputs=[msg, state, kb_state],
                   outputs=[chatbot, state, msg, audio_player])
        msg.submit(fn=chatbot_interface, inputs=[msg, state, kb_state],
                   outputs=[chatbot, state, msg, audio_player])

        clear.click(clear_all, None, [chatbot, state, msg, audio_player], queue=False)
        toggle_kb.click(fn=toggle_kb_fn, inputs=[kb_state], outputs=[kb_state, toggle_kb])

demo.launch(server_name="0.0.0.0", server_port=7860)




