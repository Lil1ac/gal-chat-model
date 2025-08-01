import os
import asyncio
import platform
import json
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

# ======== 配置 ========
API_KEY = os.getenv("aliQwen-api")


INPUT_DIR = r"D:\GalPkgs\text\sakura\graphrag_output"
OUTPUT_DIR = r"D:\GalPkgs\text\sakura\summaries"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONCURRENCY = 5

def build_prompt(scene_text):
    import re
    m = re.match(r"(\[章节:.*?\])\s*(\[线路:.*?\])\s*(\[场景:.*?\])", scene_text)
    prefix = ""
    if m:
        prefix = " ".join(m.groups())

    prompt = f"""你是一名专业的Galgame剧情内容分析助手。请基于以下游戏剧情文本，生成一段剧情摘要。

请在摘要最开始写出本段内容的章节、线路、场景信息，格式为：
{prefix}

然后接着写剧情摘要，要求语言简洁、连贯，重点提炼关键事件、角色关系和情感变化，字数控制在500-1000字之间。

请只输出章节信息和摘要内容，不要添加其他多余说明。

剧情文本：
{scene_text}
"""
    return prompt




# 初始化OpenAI异步客户端
client = AsyncOpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

async def call_bailian_api(scene_text, filename, semaphore):
    prompt = build_prompt(scene_text)
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.5,
                top_p=0.9,
            )
            summary = response.choices[0].message.content.strip()
            print(f"{filename} 摘要生成完成")
            return summary
        except Exception as e:
            print(f"{filename} 摘要生成失败: {e}")
            return None

async def process_file(filepath, semaphore):
    filename = os.path.basename(filepath)
    output_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(output_path):
        print(f"{filename} 已存在摘要，跳过")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    summary = await call_bailian_api(text, filename, semaphore)

    if summary:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)

async def main():
    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    print(f"共找到 {len(files)} 个场景文件待处理")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [process_file(f, semaphore) for f in files]

    await tqdm_asyncio.gather(*tasks)

if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
