import os
import re
import json
from collections import defaultdict
import shutil


def get_merge_key(filename):
    base = filename.rsplit('.', 1)[0]

    # 规则3：含gend的，统一合并为gend.json
    if 'gend' in base:
        return 'gend'

    # 规则1：含 mak/mis/kei 的，匹配数字_字母+数字部分作为key，比如 03_mak01
    if any(x in base for x in ['mak', 'mis', 'kei']):
        m = re.match(r'(\d+_(?:mak|mis|kei)\d+)', base)
        if m:
            return m.group(1)

    # 规则2：形如 01_04_1 结尾的数字_数字_数字，合并为数字_数字
    m = re.match(r'(\d+_\d+)_\d+$', base)
    if m:
        return m.group(1)

    # 不合并，返回None
    return None


def load_json_list(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 如果是列表，直接返回
    if isinstance(data, list):
        return data
    # 如果是dict且有"dialog"字段，包成列表返回
    if isinstance(data, dict) and "dialog" in data:
        return [data]
    return [data]


def main():
    input_dir = r'D:\GalPkgs\text\sakura\output'  # 你的输入目录
    output_dir = os.path.join(os.path.dirname(input_dir), 'merged_output')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    merge_groups = defaultdict(list)
    singles = []

    for fname in os.listdir(input_dir):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(input_dir, fname)
        key = get_merge_key(fname)
        if key:
            merge_groups[key].append(fpath)
        else:
            singles.append(fpath)

    print(f"共找到 {len(merge_groups)} 组需合并文件，{len(singles)} 个单独文件")

    # 合并写入
    for key, files in merge_groups.items():
        merged_data = []
        for f in files:
            data = load_json_list(f)
            merged_data.extend(data)
        out_file = os.path.join(output_dir, f"{key}.json")
        with open(out_file, 'w', encoding='utf-8') as fw:
            json.dump(merged_data, fw, ensure_ascii=False, indent=2)
        print(f"合并 {len(files)} 个文件 -> {out_file}")

    # 单独文件复制
    for f in singles:
        fname = os.path.basename(f)
        dst = os.path.join(output_dir, fname)
        shutil.copyfile(f, dst)
        print(f"复制单文件 {fname} -> merged_output")


if __name__ == '__main__':
    main()
