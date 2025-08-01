import os
import json
import re

def parse_filename(filename):
    """
    解析文件名，提取章节、场景、线路信息
    支持：
    - 01_01.json => chapter=1, scene=1, line=None
    - 03_mak01.json => chapter=3, scene=None, line='mak01'
    - gend.json => chapter='gend', scene=None, line=None
    """
    name = os.path.splitext(filename)[0]
    if name == 'gend':
        return {'chapter': 'gend', 'scene': None, 'line': None}
    # 先匹配章节_场景，如 01_01, 03_05
    m = re.match(r'(\d{2})_(\d{2})$', name)
    if m:
        return {'chapter': int(m.group(1)), 'scene': int(m.group(2)), 'line': None}
    # 匹配章节_线路，如 03_mak01, 04_kei04
    m2 = re.match(r'(\d{2})_([a-z]+[0-9]+)$', name)
    if m2:
        return {'chapter': int(m2.group(1)), 'scene': None, 'line': m2.group(2)}
    # 匹配章节_线路无数字，如 03_mak
    m3 = re.match(r'(\d{2})_([a-z]+)$', name)
    if m3:
        return {'chapter': int(m3.group(1)), 'scene': None, 'line': m3.group(2)}
    # 其他情况直接返回
    return {'chapter': None, 'scene': None, 'line': None}

def merge_consecutive_dialogues(dialogues):
    """
    合并连续相同speaker的对话为一条，方便后续使用
    """
    merged = []
    prev_speaker = None
    prev_text = []
    for d in dialogues:
        speaker = d.get('speaker')
        text = d.get('text', '').strip()
        if not text:
            continue
        if speaker == prev_speaker:
            prev_text.append(text)
        else:
            if prev_speaker is not None:
                merged.append({'speaker': prev_speaker, 'text': ' '.join(prev_text)})
            prev_speaker = speaker
            prev_text = [text]
    # 最后一个也添加
    if prev_speaker is not None:
        merged.append({'speaker': prev_speaker, 'text': ' '.join(prev_text)})
    return merged

def process_all_files(root_dir):
    """
    处理目录下所有JSON文件，返回统一格式数据列表
    """
    all_data = []
    for fname in os.listdir(root_dir):
        if not fname.endswith('.json'):
            continue
        meta = parse_filename(fname)
        full_path = os.path.join(root_dir, fname)
        with open(full_path, 'r', encoding='utf-8') as f:
            try:
                dialogues = json.load(f)
            except Exception as e:
                print(f"文件{fname}解析失败: {e}")
                continue
        merged_dialogues = merge_consecutive_dialogues(dialogues)
        # 遍历每条对话，组装数据
        for idx, d in enumerate(merged_dialogues):
            entry = {
                'chapter': meta['chapter'],
                'scene': meta['scene'],
                'line': meta['line'],
                'dialogue_index': idx,
                'speaker': d['speaker'],
                'text': d['text']
            }
            all_data.append(entry)
    return all_data

if __name__ == '__main__':
    root_dir = r'D:\GalPkgs\text\sakura\merged_output'  # 你的对话json目录
    print("开始处理目录：", root_dir)
    all_dialogue_data = process_all_files(root_dir)
    print(f"共处理对话段落：{len(all_dialogue_data)} 条")
    # 输出到一个文件
    output_path = os.path.join(root_dir, 'merged_dialogues_structured.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_dialogue_data, f, ensure_ascii=False, indent=2)
    print(f"结构化数据已保存到：{output_path}")
