import os
import json
import re

INPUT_DIR = r"D:\GalPkgs\text\sakura\extracted_dialogs"
OUTPUT_JSON = r"D:\GalPkgs\text\sakura\dialog_pieces.json"

def clean_text(text):
    # 去除括号，换行符替换为空格，连续空格替换成一个空格
    text = re.sub(r"[（）()]", "", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_into_pieces(full_text, max_len=300):
    # 简单按句号分割，再按长度拆分，防止太长
    sentences = re.split(r"(。|！|？|!|\?)", full_text)
    pieces = []
    current_piece = ""
    for i in range(0, len(sentences), 2):
        sent = sentences[i]
        punct = sentences[i+1] if i+1 < len(sentences) else ""
        combined = sent + punct
        if len(current_piece) + len(combined) > max_len:
            if current_piece:
                pieces.append(current_piece.strip())
            current_piece = combined
        else:
            current_piece += combined
    if current_piece:
        pieces.append(current_piece.strip())
    return pieces

def main():
    all_pieces = []
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(INPUT_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        clean_content = clean_text(content)
        pieces = split_into_pieces(clean_content)
        for p in pieces:
            # 这里包装成单条对话的格式，作为对话片段
            all_pieces.append([p])  # 一条对话只有一段文本作为起点，可以改成多段对话形式
    # 保存所有对话片段json
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_pieces, f, ensure_ascii=False, indent=2)
    print(f"共提取{len(all_pieces)}条对话片段，保存到 {OUTPUT_JSON}")

if __name__ == "__main__":
    main()


