import os
import re
import json

input_dir = r'D:\GalPkgs\text\sakura\merged_output'
output_path = r'D:\GalPkgs\text\sakura\graphrag_output\xinling.txt'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

filename_pattern = re.compile(r'^(\d+)_(?:(mis|mak|kei)?)(\d+)\.json$', re.IGNORECASE)
special_filename = 'gend.json'

route_name_map = {
    None: "普通",
    "mis": "心铃",
    "mak": "真琴",
    "kei": "圭",
    "gend": "终章"
}

def parse_filename(filename):
    if filename.lower() == special_filename:
        return "终章", "终章", None
    m = filename_pattern.match(filename)
    if not m:
        return None, None, None
    chapter = int(m.group(1))
    route_key = m.group(2)
    scene = int(m.group(3))
    route_name = route_name_map.get(route_key, route_key if route_key else "普通")
    return chapter, route_name, scene

def json_to_text_entries(json_data):
    entries = []
    for entry in json_data:
        speaker = entry.get('speaker')
        if not speaker or speaker == "null":
            speaker = "旁白"
        text = entry.get('text') or ""
        entries.append((speaker, text))
    return entries

def main():
    all_lines = []
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith('.json'):
            continue
        chapter, route_name, scene = parse_filename(filename)
        if chapter is None:
            print(f"跳过无法解析的文件名: {filename}")
            continue

        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"文件读取或解析失败 {filename}: {e}")
            continue

        entries = json_to_text_entries(data)
        for speaker, text in entries:
            if speaker != "心铃":  # 只要心铃的台词
                continue
            header = f"[章节: {chapter}] [线路: {route_name}] [场景: {scene if scene is not None else '-'}] [角色: {speaker}]"
            all_lines.append(header)
            all_lines.append(text)
            all_lines.append("")

    if not all_lines:
        print("没有找到心铃对话。")
        return

    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(all_lines))

    print(f"已生成总文件: {output_path}")

if __name__ == "__main__":
    main()
