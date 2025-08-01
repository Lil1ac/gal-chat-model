import os
import asyncio
import platform
import json
from tqdm import tqdm
from openai import AsyncOpenAI

# 替换成你的apikey
API_KEY = os.getenv("aliQwen-api")

# 目录和文件路径
INPUT_JSON = r"D:\GalPkgs\text\sakura\dialog_pieces_xinling.json"
RESULTS_FOLDER = r"D:\GalPkgs\text\sakura\generated_dialogs"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 角色扮演Prompt模板，符合你之前提供的设计
PROMPT_TEMPLATE = """请根据以下对话片段，扮演游戏《樱之刻》中角色本间心铃，生成一段真实、自然且符合角色设定的对话。

1. 先明确本段对话的“对话目标”（goal），简明扼要地描述对话的主题和目的。

2. 以第一人称视角，撰写“system prompt”字段，详细介绍本间心铃的身份、性格、当前心情和所处环境（时间、天气、地点等），并说明她参与本次对话的原因。请参考以下角色背景撰写，内容自然流畅，能合理引出对话开头：

【本间心铃角色背景参考】
- 本名：本间心铃，游戏《樱之刻》及衍生作品女主角，圣卢安学院学生。
- 外貌：黑色双马尾，棕色瞳孔，外貌与母亲中村丽华相似，性格乖巧且深思熟虑，举止得体，眼神深邃锐利。
- 亲属：父亲本间礼次郎，哥哥本间心佐夫，母亲中村丽华。
- 特点：能够辨识真伪，擅长画画，是天才画家，性格理智细腻。
- 人际关系：与草薙直哉有深厚感情，最终携手共度人生；师从圭，经历过成长与磨砺。
- 心情与环境：请结合当前对话环境（时间、天气、地点）自然描述参与对话的原因和心境。

3. 在“conversation_history”字段中，围绕对话目标，创造6至12轮对话（用户和本间心铃轮流发言）。  
   - 语言符合日常口语，贴合角色个性。  
   - 遇到无法回答的问题，可以表达疑惑并主动澄清。  
   - 对话连贯自然，避免突兀或中断。  
   - 每轮不超过100字。

请严格按照如下JSON格式输出，不要带任何额外解释：

{
  "goal": "对话目标，简洁明确",
  "system prompt": "第一人称详细描述角色身份、性格、当前环境及对话缘由，语言自然流畅，不出现“你”字",
  "conversation_history": [
    "用户的第一句发言（无角色前缀）",
    "本间心铃的第一句回复",
    "用户的第二句发言",
    "本间心铃的第二句回复",
    ...
  ]
}

给定对话片段：

${chat_piece}}
"""

# OpenAI客户端配置
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

CONCURRENCY = 5
BATCH_SIZE = 50
semaphore = asyncio.Semaphore(CONCURRENCY)

def format_chat_piece(dialog):
    # 因为dialog是列表（多段），这里用换行拼接，也可以自定义格式
    return "\n".join(dialog)

async def call_model(dialog_piece, index, semaphore):
    async with semaphore:
        try:
            print(f"开始生成第{index}条对话...")
            prompt = PROMPT_TEMPLATE.replace("${chat_piece}", format_chat_piece(dialog_piece))
            response = await client.chat.completions.create(
                model="qwen-max-latest",
                messages=[
                    {"role": "system", "content": "你是一位擅长编写真实，日常情景对话的专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                top_p=0.9,
            )
            print(f"第{index}条对话生成完成")
            return index, response.choices[0].message.content
        except Exception as e:
            print(f"第{index}条对话生成失败：{e}")
            return index, None


from tqdm.asyncio import tqdm_asyncio


async def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        dialogs = json.load(f)

    print(f"总共要处理的对话条数：{len(dialogs)}")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [call_model(dialogs[i], i, semaphore) for i in range(len(dialogs))]

    results = await asyncio.gather(*tasks)

    for idx, res in enumerate(results):
        print(f"第{idx}条处理完成")
        if res:
            out_path = os.path.join(RESULTS_FOLDER, f"dialog_{idx}.json")
            try:
                json_obj = json.loads(res)
                json_obj["original_dialog_piece"] = dialogs[idx]
                with open(out_path, "w", encoding="utf-8") as fw:
                    json.dump(json_obj, fw, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"第{idx}条结果解析失败: {e}")
        else:
            print(f"第{idx}条没有返回结果")
# async def main():
#     with open(INPUT_JSON, "r", encoding="utf-8") as f:
#         dialogs = json.load(f)
#
#     semaphore = asyncio.Semaphore(CONCURRENCY)
#     # 只调用第0条，测试
#     tasks = [call_model(dialogs[0], 0, semaphore)]
#
#     results = await asyncio.gather(*tasks)
#
#     for idx, res in enumerate(results):
#         print(f"第{idx}条处理完成")
#         if res and res[1]:  # res 是 (index, 内容)
#             out_path = os.path.join(RESULTS_FOLDER, f"dialog_{res[0]}.json")
#             try:
#                 json_obj = json.loads(res[1])
#                 json_obj["original_dialog_piece"] = dialogs[res[0]]
#                 with open(out_path, "w", encoding="utf-8") as fw:
#                     json.dump(json_obj, fw, ensure_ascii=False, indent=2)
#             except Exception as e:
#                 print(f"第{res[0]}条结果解析失败: {e}")
#         else:
#             print(f"第{res[0]}条没有返回结果")


if __name__ == "__main__":
    import platform
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

