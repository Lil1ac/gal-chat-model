import json
import glob
import os

input_pattern = r"D:/GalPkgs/text/sakura/output/*.json"
output_dir = r"D:/GalPkgs/text/sakura/train_data"
os.makedirs(output_dir, exist_ok=True)

system_prompt = (
    "你是游戏《樱之刻》中的角色“本间心铃”，"
    "她性格冷峻、理性，语气温柔却有力，善于思考。"
)

def get_recent_non_xinling_texts(data, current_index, max_sentences=2):
    texts = []
    idx = current_index - 1
    while idx >= 0 and len(texts) < max_sentences:
        speaker = data[idx].get('speaker')
        text = data[idx].get('text')
        if speaker != "心铃" and text and text.strip():
            texts.insert(0, text.replace('\n', ' ').strip())  # 头插保持顺序
        idx -= 1
    return " ".join(texts)

def extract_dialog_pairs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dialogs = []
    for i, entry in enumerate(data):
        speaker = entry.get('speaker')
        text = entry.get('text')
        if not text or not text.strip():
            continue
        text = text.replace('\n', ' ').strip()

        if speaker == "心铃":
            user_context = get_recent_non_xinling_texts(data, i)
            if user_context:
                dialogs.append((user_context, text))

    return dialogs

all_dialogs = []
file_list = glob.glob(input_pattern)
print(f"找到 {len(file_list)} 个文件")

for filename in file_list:
    dialogs = extract_dialog_pairs(filename)
    print(f"{os.path.basename(filename)} 提取了 {len(dialogs)} 对对话")
    all_dialogs.extend(dialogs)

output_file = os.path.join(output_dir, "train_data.jsonl")
with open(output_file, "w", encoding="utf-8") as f:
    for user_text, assistant_text in all_dialogs:
        sample = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text}
            ]
        }
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"微调数据已写入 {output_file}，共 {len(all_dialogs)} 条样本")
