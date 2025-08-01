import re
import os
import json


def extract_blocks_from_ast(ast_text):  # 提取所有block及其对话内容
    pattern = re.compile(r'(block_\d+)\s*=\s*{', re.MULTILINE)
    matches = list(pattern.finditer(ast_text))
    print(f"[调试] 找到 {len(matches)} 个 block 开头")

    blocks = {}
    for i, match in enumerate(matches):
        block_name = match.group(1)
        start = match.end()
        depth = 1
        pos = start
        while depth > 0 and pos < len(ast_text):
            if ast_text[pos] == '{':
                depth += 1
            elif ast_text[pos] == '}':
                depth -= 1
            pos += 1
        block_content = ast_text[start:pos - 1].strip()
        print(f"[调试] 提取到 {block_name} 内容长度: {len(block_content)} 字符")
        blocks[block_name] = block_content
    return blocks


def extract_text_ja_from_block(block_content):  # 提取 ja = { ... } 中的对话
    # 先提取 text = { ... } 块
    text_pos = block_content.find("text")
    if text_pos == -1:
        print("[调试] 当前 block 无 text 字段")
        return []

    text_block, end_pos = extract_balanced_block(block_content, text_pos)
    if text_block is None:
        print("[调试] 无法正确提取 text 块")
        return []

    # 提取 ja = { ... } 块
    ja_match = re.search(r'ja\s*=\s*{', text_block)
    if not ja_match:
        print("[调试] text 中无 ja 语言内容")
        return []

    ja_start = ja_match.end() - 1  # '{' 位置
    ja_block, ja_end_pos = extract_balanced_block(text_block, ja_start)
    if ja_block is None:
        print("[调试] 无法正确提取 ja 块")
        return []

    # ja_block 是多条 {} 包裹的列表，形如 { {...}, {...}, ... }
    # 手动拆分顶层 {} 内容
    entries = []
    depth = 0
    entry_start = None
    for i, c in enumerate(ja_block):
        if c == '{':
            if depth == 0:
                entry_start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and entry_start is not None:
                entries.append(ja_block[entry_start + 1:i].strip())  # 不含外层{}
                entry_start = None

    print(f"[调试] ja 中找到 {len(entries)} 个条目")

    results = []
    for idx, entry in enumerate(entries):
        # print(entries) # ja内部块
        print(f"[调试] 处理第 {idx + 1} 个条目，长度: {len(entry)}")
        # Step 1: 提取说话人 name
        name_match = re.search(r'name\s*=\s*{(.*?)}', entry, re.DOTALL)
        if name_match:
            name_raw = name_match.group(1)
            name_str_match = re.search(r'"(.*?)"', name_raw)
            speaker = name_str_match.group(1) if name_str_match else "未知"
            print(f"[调试] 说话人: {speaker}")
            # Step 2: 去掉整个 name = {...} 字段
            entry = re.sub(r'name\s*=\s*{.*?},?', '', entry, flags=re.DOTALL)
        else:
            speaker = None
            print("[调试] 无说话人字段")

        # Step 3: 提取剩下部分所有字符串，拼成一句话
        texts = re.findall(r'"(.*?)"', entry)
        lines = []
        # print(texts) # 所有对话
        for t in texts:
            # 跳过标签内容，例如 rt2
            if t.startswith("rt"):
                # print(f"[调试] 跳过标签: {t}")
                continue
            # 去掉日文括号和空白
            line = t.strip("「」").strip()
            if line:
                lines.append(line)

        if lines:
            full_text = " ".join(lines)
            print(f"[调试] 提取台词: {full_text}")
            results.append({"speaker": speaker, "text": full_text})

    return results


def save_dialogues(dialogues, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=2)
    print(f"[输出] 保存到 {output_path}，共 {len(dialogues)} 条")


def extract_balanced_block(text, start_pos=0, open_sym='{', close_sym='}'):  # 提取 { ... }块
    pos = text.find(open_sym, start_pos)
    if pos == -1:
        return None, None
    depth = 0
    for i in range(pos, len(text)):
        if text[i] == open_sym:
            depth += 1
        elif text[i] == close_sym:
            depth -= 1
            if depth == 0:
                return text[pos + 1:i], i + 1
    return None, None


if __name__ == "__main__":
    input_dir = r"D:\GalPkgs\text\sakura\script"
    output_dir = r"D:\GalPkgs\text\sakura\output"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".ast"):
            input_path = os.path.join(input_dir, filename)
            print(f"[处理] 开始处理文件: {filename}")
            with open(input_path, encoding="utf-8") as f:
                content = f.read()

            blocks = extract_blocks_from_ast(content)

            all_dialogues = []
            for block_name, block_content in blocks.items():
                print(f"[调试] 开始处理 {block_name}")
                dialogues = extract_text_ja_from_block(block_content)
                if dialogues:
                    all_dialogues.extend(dialogues)
                else:
                    print(f"[调试] {block_name} 无有效对话")

            print(f"[调试] 总共提取到 {len(all_dialogues)} 条对话")
            for d in all_dialogues:
                print(f"{d['speaker'] or '旁白'}: {d['text']}")

            output_path = os.path.join(output_dir, filename.replace(".ast", ".json"))
            # # 保存结果到文本文件
            # with open(output_path, "w", encoding="utf-8") as f:
            #     for d in all_dialogues:
            #         speaker = d['speaker'] if d['speaker'] else "旁白"
            #         line = f"{speaker}: {d['text']}\n"
            #         f.write(line)
            #
            # print(f"[调试] 对话已保存到: {output_path}")

            # 保存为 JSON 格式
            save_dialogues(all_dialogues, output_path)
