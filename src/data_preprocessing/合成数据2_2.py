import os
import json

INPUT_DIR = r"D:\GalPkgs\text\sakura\merged_output"
OUTPUT_JSON = r"D:\GalPkgs\text\sakura\dialog_pieces_xinling.json"
WINDOW_SIZE = 5  # 上下文窗口大小，可调节

def extract_kokoro_contexts(data, window=WINDOW_SIZE):
    results = []
    kokoro_indices = [i for i, item in enumerate(data) if item.get("speaker") and ("心铃" in item.get("speaker"))]

    for idx in kokoro_indices:
        start = max(0, idx - window)
        end = min(len(data), idx + window + 1)
        context_slice = data[start:end]

        text_piece = []
        for item in context_slice:
            speaker = item.get("speaker")
            text = item.get("text", "").strip()

            if not speaker:  # 旁白
                # 如果文本本身已经有括号包围，就不加了
                if not (text.startswith("（") and text.endswith("）")) and not (text.startswith("(") and text.endswith(")")):
                    text = f"（{text}）"
                text_piece.append(text)
            else:
                text_piece.append(f"{speaker}：{text}")

        results.append(["\n".join(text_piece)])

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

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_pieces, f, ensure_ascii=False, indent=2)
    print(f"共提取 {len(all_pieces)} 条心铃上下文，保存到 {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
