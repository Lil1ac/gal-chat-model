import os
import json

INPUT_FOLDER = r"D:\GalPkgs\text\sakura\merged_output"
OUTPUT_FOLDER = r"D:\GalPkgs\text\sakura\extracted_dialogs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def convert_dialog_to_chat_piece(data):
    chat_lines = []
    for item in data:
        speaker = item.get("speaker")
        text = item.get("text", "").strip()
        if not text:
            continue
        if speaker is None:
            chat_lines.append(f"({text})")  # 旁白括号包裹
        else:
            chat_lines.append(f"{speaker}：{text}")
    return "\n".join(chat_lines)

def main():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".json")]
    for filename in files:
        filepath = os.path.join(INPUT_FOLDER, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            chat_piece = convert_dialog_to_chat_piece(data)
            if chat_piece.strip():
                out_path = os.path.join(OUTPUT_FOLDER, filename.replace(".json", ".txt"))
                with open(out_path, "w", encoding="utf-8") as fw:
                    fw.write(chat_piece.replace("\n", "\\n"))
                print(f"{filename} 提取成功，保存到 {out_path}")
            else:
                print(f"{filename} 内容为空，跳过")
        except Exception as e:
            print(f"读取{filename}失败: {e}")

if __name__ == "__main__":
    main()
