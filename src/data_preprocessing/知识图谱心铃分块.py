# import os
#
# input_dir = r'D:\GalPkgs\text\sakura\chunk'  # 你之前转换的带标签文本文件夹
# output_dir = r'D:\GalPkgs\text\sakura\graphrag_chunks'
# os.makedirs(output_dir, exist_ok=True)
#
# def load_and_group_by_role(file_path):
#     """
#     读取单个文本文件，按章节线路场景角色分组连续对话。
#     假设格式类似：
#     [章节: 3] [线路: 普通] [场景: 8] [角色: 心铃]
#     对话内容
#     """
#     with open(file_path, encoding='utf-8') as f:
#         lines = f.read().splitlines()
#
#     groups = []
#     current_group = None
#     buffer = []
#
#     for line in lines:
#         if line.startswith('[章节:'):
#             # 遇到新标签行，先把之前积累的对话保存
#             if current_group and buffer:
#                 groups.append((current_group, "\n".join(buffer)))
#                 buffer = []
#
#             # 解析标签
#             # 举例：[章节: 3] [线路: 普通] [场景: 8] [角色: 心铃]
#             parts = {}
#             for part in line.strip().split(']'):
#                 part = part.strip()
#                 if not part:
#                     continue
#                 key_val = part.strip('[').split(':')
#                 if len(key_val) == 2:
#                     key = key_val[0].strip()
#                     val = key_val[1].strip()
#                     parts[key] = val
#
#             chapter = parts.get('章节', '0')
#             route = parts.get('线路', '未知')
#             scene = parts.get('场景', '0')
#             role = parts.get('角色', '未知')
#
#             current_group = (chapter, route, scene, role)
#         else:
#             # 普通对话内容
#             if current_group:
#                 buffer.append(line.strip())
#
#     # 最后一个group
#     if current_group and buffer:
#         groups.append((current_group, "\n".join(buffer)))
#
#     return groups
#
# def main():
#     all_groups = []
#     for filename in sorted(os.listdir(input_dir)):
#         if not filename.endswith('.txt'):
#             continue
#         file_path = os.path.join(input_dir, filename)
#         groups = load_and_group_by_role(file_path)
#         all_groups.extend(groups)
#
#     # 按每个分块生成文件
#     for i, (group_key, content) in enumerate(all_groups):
#         chapter, route, scene, role = group_key
#         safe_route = route.replace(' ', '_')
#         safe_role = role.replace(' ', '_')
#         filename = f"chapter{chapter}_route{safe_route}_scene{scene}_role{safe_role}_{i}.txt"
#         filepath = os.path.join(output_dir, filename)
#
#         with open(filepath, 'w', encoding='utf-8') as f:
#             header = f"[章节: {chapter}] [线路: {route}] [场景: {scene}] [角色: {role}]\n\n"
#             f.write(header)
#             f.write(content)
#
#         print(f"写入文件: {filepath}")
#
# if __name__ == "__main__":
#     main()
import os

input_dir = r'D:\GalPkgs\text\sakura\chunk'  # 你之前转换的带标签文本文件夹
output_file = r'D:\GalPkgs\text\sakura\all_texts_merged.txt'  # 合并后的单文件路径

def load_and_group_by_role(file_path):
    with open(file_path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    groups = []
    current_group = None
    buffer = []

    for line in lines:
        if line.startswith('[章节:'):
            if current_group and buffer:
                groups.append((current_group, "\n".join(buffer)))
                buffer = []

            parts = {}
            for part in line.strip().split(']'):
                part = part.strip()
                if not part:
                    continue
                key_val = part.strip('[').split(':')
                if len(key_val) == 2:
                    key = key_val[0].strip()
                    val = key_val[1].strip()
                    parts[key] = val

            chapter = parts.get('章节', '0')
            route = parts.get('线路', '未知')
            scene = parts.get('场景', '0')
            role = parts.get('角色', '未知')

            current_group = (chapter, route, scene, role)
        else:
            if current_group:
                buffer.append(line.strip())

    if current_group and buffer:
        groups.append((current_group, "\n".join(buffer)))

    return groups

def main():
    all_groups = []
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith('.txt'):
            continue
        file_path = os.path.join(input_dir, filename)
        groups = load_and_group_by_role(file_path)
        all_groups.extend(groups)

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (group_key, content) in enumerate(all_groups):
            chapter, route, scene, role = group_key
            header = f"[章节: {chapter}] [线路: {route}] [场景: {scene}] [角色: {role}]\n\n"
            f.write(header)
            f.write(content)
            f.write('\n\n' + ('='*40) + '\n\n')  # 用分隔线区分不同块

    print(f"所有内容已合并写入到: {output_file}")

if __name__ == "__main__":
    main()
