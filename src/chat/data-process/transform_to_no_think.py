import json
import os

# ========= 输入输出文件 =========
files = [
    ("train.jsonl", "train_no_think.jsonl"),
    ("test.jsonl", "test_no_think.jsonl"),
]

for input_file, output_file in files:
    if not os.path.exists(input_file):
        print(f"文件 {input_file} 不存在，跳过")
        continue

    with open(input_file, "r", encoding="utf-8") as fr, \
         open(output_file, "w", encoding="utf-8") as fw:
        for line in fr:
            data = json.loads(line)
            if "messages" not in data:
                continue

            # 遍历消息并处理
            for m in data["messages"]:
                if m["role"] == "user":
                    # 在用户消息末尾添加 /no_think（如果没有的话）
                    if not m["content"].strip().endswith("/no_think"):
                        m["content"] = m["content"].rstrip() + " /no_think"

            fw.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"转换完成: {input_file} -> {output_file}")
