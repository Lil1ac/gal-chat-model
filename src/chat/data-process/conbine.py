import os
import asyncio
import platform
import json
import re
from tqdm import tqdm
from datasets import load_from_disk
from openai import AsyncOpenAI

# 配置OpenAI客户端
client = AsyncOpenAI(
    api_key= os.getenv("aliQwen-api"),  # 替换为你自己的api key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

prompt = """
你的任务是根据给定的对话片段，创造一个独特且鲜活的角色，并生成一段生动、深入的galgame风格对话。

请确保每次生成的角色在姓名、年龄、性别、职业和性格等方面都具有较大差异，避免与之前生成的角色职业或身份雷同。  
为此，可以从不同的社会阶层、行业或生活背景中挑选职业，并尝试创造不常见的、有趣的角色设定。

首先，设定一个明确且富有挑战性的对话目标，作为"goal"字段的值。该目标应紧密契合给定的对话片段内容和语境，体现情感张力或剧情冲突。

给定对话片段为:
${chat_piece}

请从第一人称视角，详细塑造一个复杂且立体的角色，包括：
1. 基本信息：姓名（须独特且富有个性）、年龄、性别、职业等；
2. 性格特征：具体且丰富的性格描写，体现独特的语言风格和行为习惯，避免空洞标签；
3. 个人经历：具深度的背景故事，揭示塑造性格的关键事件；
4. 当前心情与环境：细致描述角色此刻的情绪状态及所处的物理和社会环境（光线、声音、气味、天气等细节）；
5. 对话缘由：说明角色参与本对话的原因及背景。

确保角色塑造自然引出后续对话，场景连贯且真实可信。

在"conversation_history"字段中，重写用户与角色之间的对话，要求：
- 生成6至12轮对话，每轮不超过100词；
- 在括号内加入动作、表情和语气变化等细节描写；
- 补充环境氛围细节，增强沉浸感；
- 适当穿插角色内心独白，展现人物内心复杂性；
- 保持对话自然流畅，符合日常交流习惯；
- 明显体现角色独特性格及情感波动；
- 遇到无法回答的问题时，表现合理困惑或求解态度；
- 对话发展合理完整，避免突兀或无结尾。

请严格按照以下JSON格式输出结果，不要做任何解释：

{
  "goal": "对话目标",
  "system prompt": "模型第一人称独白，详细介绍自己、用户、当前对话上下文和物理环境",
  "conversation_history": [
    "用户的第一次发言（避免出现`用户:`，稍作润色使对话更顺畅）",
    "你的第一次回复（避免出现`角色:`，内容丰富且连贯）",
    "用户的第二次发言",
    "你的第二次回复",
    ...
  ]
}
"""



# 创建结果保存文件夹
RESULTS_FOLDER = "qwen_max_lastest_dialogs_second"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 加载数据集
dataset = load_from_disk("./lccc-clean-train")  # 请替换成你的数据路径
dialogs = dataset["dialog"][:3000]  # 你想处理的对话数量

# 并发和批次设置
CONCURRENCY = 5
BATCH_SIZE = 100
semaphore = asyncio.Semaphore(CONCURRENCY)


def get_processed_indices():
    processed = set()
    for fname in os.listdir(RESULTS_FOLDER):
        match = re.search(r'_(\d+)\.json$', fname)
        if match:
            idx = int(match.group(1))
            processed.add(idx)
    return processed


def map_fn(dialog):
    # 偶数为用户，奇数为角色
    res = []
    for i, msg in enumerate(dialog):
        speaker = "用户" if i % 2 == 0 else "角色"
        res.append(f"{speaker}：{msg.strip()}")
    return "\n".join(res)


async def process_dialog(dialog):
    async with semaphore:
        try:
            dialog_text = prompt.replace("${chat_piece}", map_fn(dialog))
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "你是一位擅长编写真实，日常情景对话的专家。"},
                    {"role": "user", "content": dialog_text},
                ],
                model="qwen-max",
                temperature=0.7,
                top_p=0.9,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error processing dialog: {e}")
            return None


def safe_filename(s, max_len=15):
    # 只保留中文、英文、数字、下划线，空格替换成下划线
    s = s.replace(" ", "_")
    s = "".join(c for c in s if c.isalnum() or c == "_" or ('\u4e00' <= c <= '\u9fff'))
    return s[:max_len]

async def process_and_save(dialog, index):
    if index in processed_indices:
        print(f"跳过已处理对话 index={index}")
        return
    result = await process_dialog(dialog)
    if result:
        try:
            result_json = json.loads(result)
            goal = result_json.get('goal', f'unknown_goal_{index}')
            result_json["ori_chat"] = dialog
            goal_clean = safe_filename(goal)
            if not goal_clean:
                goal_clean = f"dialog_{index}"
            filename = f"{goal_clean}_{index}.json"
            with open(os.path.join(RESULTS_FOLDER, filename), "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)
            print(f"已保存对话 index={index} 文件名={filename}")
        except json.JSONDecodeError:
            print(f"JSON解析错误 index={index}")
        except Exception as e:
            print(f"保存结果异常 index={index}: {e}")



async def process_batch(batch, start_index):
    tasks = [process_and_save(dialog, i) for i, dialog in enumerate(batch, start=start_index)]
    await asyncio.gather(*tasks)


async def main():
    global processed_indices
    processed_indices = get_processed_indices()
    print(f"已处理对话数量：{len(processed_indices)}，准备处理剩余 {len(dialogs) - len(processed_indices)} 条")

    with tqdm(total=len(dialogs), desc="Processing dialogs") as pbar:
        for i in range(0, len(dialogs), BATCH_SIZE):
            batch = dialogs[i:i + BATCH_SIZE]
            # 统计这一批次跳过数量
            batch_indices = set(range(i, i + len(batch)))
            to_process_indices = batch_indices - processed_indices
            if not to_process_indices:
                pbar.update(len(batch))
                continue
            await process_batch(batch, i)
            pbar.update(len(batch))


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
