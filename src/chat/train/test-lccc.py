from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from transformers import TextStreamer

# ---------- 模型路径 ----------
base_model_path = "/root/autodl-tmp/Qwen/Qwen3-8B"
lora_model_path = "../../model/lora_model_no_think"

# ---------- 1. 加载分词器 ----------
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# ---------- 2. 加载基础模型 ----------
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ---------- 3. 加载 LoRA 适配器 ----------
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()

# ---------- 4. 输出捕获 ----------
class CaptureStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=False, **kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **kwargs)
        self.generated_text = ""

    def on_finalized_text(self, text: str, stream_end: bool = False):
        # 先替换
        clean_text = text.replace("<|im_end|>", "")
        self.generated_text += clean_text
        super().on_finalized_text(clean_text, stream_end=stream_end)

    def get_output(self):
        return self.generated_text.strip()

# ---------- 5. 角色设定 + 历史对话 ----------
character_prompt = """
你是一个典型的日系galgame女主角，扮演游戏《樱之刻》中角色本间心铃。
你的性格是温柔、善解人意，有时会害羞，喜欢和玩家互动。
无论用户说什么，你都要用角色身份回应，并保持可爱风格。
【本间心铃角色背景参考】
- 本名：本间心铃，游戏《樱之刻》及衍生作品女主角，圣卢安学院学生。
- 外貌：黑色双马尾，棕色瞳孔，外貌与母亲中村丽华相似，性格乖巧且深思熟虑，举止得体，眼神深邃锐利。
- 亲属：父亲本间礼次郎，哥哥本间心佐夫，母亲中村丽华。
- 特点：能够辨识真伪，擅长画画，是天才画家，性格理智细腻。
- 人际关系：与草薙直哉有深厚感情，最终携手共度人生；师从圭，经历过成长与磨砺。
"""
history = [
    {"role": "system", "content": character_prompt.strip()},
]

def ask(user_message):
    history.append({"role": "user", "content": user_message})
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    streamer = CaptureStreamer(tokenizer, skip_prompt=True)

    with torch.no_grad():
        _ = model.generate(
            **tokenizer(text, return_tensors="pt").to("cuda"),
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
        )

    output = streamer.get_output()
    history.append({"role": "assistant", "content": output})
    return output

# ---------- 6. CLI 连续对话 ----------
if __name__ == "__main__":
    print("===== Galgame 对话启动 =====")
    print("你正在和心铃对话，输入 exit() 退出。")
    while True:
        user_input = input("你：")
        if user_input.strip().lower() == "exit()":
            print("对话结束。")
            break
        answer = ask(user_input)
