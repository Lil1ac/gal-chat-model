import os
import re
import json

input_dir = r'D:\GalPkgs\text\sakura\merged_output'
output_dir = r'D:\GalPkgs\text\sakura\graphrag_output_xinling'  # 改成目录
os.makedirs(output_dir, exist_ok=True)

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
        lines = []
        for speaker, text in entries:

            if speaker != "心铃":  # 只要心铃
                continue

            header = f"[章节: {chapter}] [线路: {route_name}] [场景: {scene if scene is not None else '-'}] [角色: {speaker}]"
            lines.append(header)
            lines.append(text)
            lines.append("")

        out_filename = os.path.splitext(filename)[0] + '.txt'
        out_path = os.path.join(output_dir, out_filename)
        with open(out_path, 'w', encoding='utf-8') as fout:
            fout.write('\n'.join(lines))

        print(f"已生成: {out_path}")

if __name__ == "__main__":
    main()
