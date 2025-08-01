import json

filename = "D:/GalPkgs/text/sakura/lccc_style_clean.jsonl"  # 你的文件名


with open(filename, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if idx > 10:
            break
        data = json.loads(line)
        dialog = data.get("dialog", [])
        print(f"----- 对话条目 {idx} 开始 -----\n")
        for i, sentence in enumerate(dialog, 1):
            print(f"{i}: {sentence}\n")
        print(f"----- 对话条目 {idx} 结束 -----\n\n")

