import os
import json
import random

input_folder = "qwen_max_lastest_dialogs_second"
train_ratio = 0.9  # 90% 训练, 10% 测试
random.seed(42)

all_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
random.shuffle(all_files)

split_idx = int(len(all_files) * train_ratio)
train_files = all_files[:split_idx]
test_files = all_files[split_idx:]

def convert_to_chat_format(file_list, output_path):
    with open(output_path, "w", encoding="utf-8") as out:
        for file_name in file_list:
            with open(os.path.join(input_folder, file_name), "r", encoding="utf-8") as f:
                data = json.load(f)
                history = data.get("conversation_history", [])
                system_prompt = data.get("system prompt", "")

                # 确保历史记录为偶数长度
                if len(history) % 2 == 1:
                    history = history[:-1]

                messages = [{"role": "system", "content": system_prompt}]
                for i, text in enumerate(history):
                    role = "user" if i % 2 == 0 else "assistant"
                    messages.append({"role": role, "content": text})

                out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

# 生成 train.jsonl 和 test.jsonl
convert_to_chat_format(train_files, "../data/train.jsonl")
convert_to_chat_format(test_files, "../data/test.jsonl")

print(f"训练集数量: {len(train_files)}, 测试集数量: {len(test_files)}")
