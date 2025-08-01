def check_braces_match_verbose(text):
    stack = []
    for i, ch in enumerate(text):
        if ch == '{':
            stack.append(i)
        elif ch == '}':
            if not stack:
                # 多余右括号，打印周围字符帮助定位
                start = max(0, i-20)
                end = min(len(text), i+20)
                print(f"错误：在位置{i}处发现多余的右括号 '}}'")
                print(f"位置上下文：{text[start:end]!r}")
                return False
            stack.pop()
    if stack:
        i = stack[-1]
        start = max(0, i-20)
        end = min(len(text), i+20)
        print(f"错误：在位置{i}处发现多余的左括号 '{{'")
        print(f"位置上下文：{text[start:end]!r}")
        return False
    return True

if __name__ == "__main__":
    filename = "D:/GalPkgs/text/sakura/prompt/prompt.txt"  # 把这里替换成你的文件路径，比如 "/mnt/data/test.txt"
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"读取文件时出错: {e}")
        exit(1)

# 替换调用
if check_braces_match_verbose(content):
    print("括号匹配正确！")
else:
    print("括号匹配错误！")
# import re
#
# input_file = "D:/GalPkgs/text/sakura/prompt/prompt.txt"    # 你的原始文件路径
# output_file = "D:/GalPkgs/text/sakura/prompt/prompt2.txt"  # 修正后保存的文件路径
#
# with open(input_file, "r", encoding="utf-8") as f:
#     content = f.read()
#
# # 用正则替换：
# # 匹配每条记录最后一个非换行的 '}'，替换为 ')'
# # 这里假设每条记录是类似于
# # ("entity"{tuple_delimiter}xxx{tuple_delimiter}xxx{tuple_delimiter}xxx}
# # {record_delimiter}
# # 这样的格式
#
# def fix_brackets(text):
#     # 用正则匹配每条记录中最后一个 '}' 并替换为 ')'
#     # 注意只替换每条记录末尾的那个 }，不影响中间的 }
#     pattern = re.compile(r'\}(?=\s*\{record_delimiter\})')
#     fixed_text = pattern.sub(')', text)
#     return fixed_text
#
# fixed_content = fix_brackets(content)
#
# with open(output_file, "w", encoding="utf-8") as f:
#     f.write(fixed_content)
#
# print("括号替换完成，已保存到", output_file)

