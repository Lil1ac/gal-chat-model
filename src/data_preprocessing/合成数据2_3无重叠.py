import os
import json
import hashlib

INPUT_DIR = r"D:\GalPkgs\text\sakura\merged_output"
OUTPUT_JSON = r"D:\GalPkgs\text\sakura\dialog_pieces_xinling2.json"
WINDOW_SIZE = 5  # 上下文窗口大小，可调节

def hash_text_block(block):
    """对文本块生成一个哈希，用于去重"""
    return hashlib.md5("".join(block).encode('utf-8')).hexdigest()

def extract_kokoro_contexts(data, window=WINDOW_SIZE):
    results = []
    seen_hashes = set()
    i = 0

    while i < len(data):
        item = data[i]
        speaker = item.get("speaker") or ""
        if "心铃" in speaker:

            start = max(0, i - window)
            end = min(len(data), i + window + 1)
            context_slice = data[start:end]

            text_piece = []
            for entry in context_slice:
                spk = entry.get("speaker")
                txt = entry.get("text", "").strip()
                if not spk:
                    if not (txt.startswith("（") and txt.endswith("）")):
                        txt = f"（{txt}）"
                    text_piece.append(txt)
                else:
                    text_piece.append(f"{spk}：{txt}")

            h = hash_text_block(text_piece)
            if h not in seen_hashes:
                seen_hashes.add(h)
                results.append(["\n".join(text_piece)])

            i = end  # 👉 跳到当前窗口之后，避免重叠
        else:
            i += 1

    return results


def main():
    all_pieces = []

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(INPUT_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"{filename} 读取失败: {e}")
                continue
        pieces = extract_kokoro_contexts(data)
        print(f"{filename} 提取到 {len(pieces)} 条心铃上下文")
        all_pieces.extend(pieces)

    print(f"最终合并前共有 {len(all_pieces)} 条片段")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_pieces, f, ensure_ascii=False, indent=2)

    print(f"共提取 {len(all_pieces)} 条心铃上下文，保存到 {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
