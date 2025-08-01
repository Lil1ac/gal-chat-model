import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from swanlab.integration.transformers import SwanLabCallback
from trl import SFTTrainer, SFTConfig

# ========== 1. 加载 Qwen 模型 ==========
model_name = "/root/autodl-tmp/Qwen/Qwen3-8B"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    full_finetuning=False,
)

# ========== 2. LoRA 配置 ==========
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ========== 3. 加载自定义数据 ==========
train_dataset = load_dataset("json", data_files="train_no_think.jsonl", split="train")
test_dataset = load_dataset("json", data_files="test_no_think.jsonl", split="train")

# 标准化为 sharegpt 格式
train_dataset = standardize_sharegpt(train_dataset)
test_dataset = standardize_sharegpt(test_dataset)

# 应用 chat_template 生成纯文本
train_text = tokenizer.apply_chat_template(train_dataset["messages"], tokenize=False)
test_text = tokenizer.apply_chat_template(test_dataset["messages"], tokenize=False)

# 转换成 HF Dataset
train_hf = Dataset.from_pandas(pd.DataFrame({"text": train_text})).shuffle(seed=3407)
test_hf = Dataset.from_pandas(pd.DataFrame({"text": test_text})).shuffle(seed=3407)

# ========== 4. SwanLab 回调 ==========
swanlab_callback = SwanLabCallback(
    project="Qwen3-8B-finetune",
    experiment_name="Galgame-dialog-no-think",
    description="使用自制 galgame 对话数据微调 Qwen3-8B",
    config={
        "model": model_name,
        "train_data_number": len(train_hf),
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }
)

# ========== 5. SFT Trainer ==========
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset_text_field="text",
    train_dataset=train_hf,
    eval_dataset=test_hf,
    callbacks=[swanlab_callback],
    args=SFTConfig(
        output_dir="../../model/lora_model_no_think",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        save_steps=100,
    ),
)

# ========== 6. 显示 GPU 信息 ==========
gpu_stats = torch.cuda.get_device_properties(0)
print(f"GPU = {gpu_stats.name}, Max memory = {round(gpu_stats.total_memory/1024/1024/1024,3)} GB.")

# ========== 7. 训练 ==========
trainer_stats = trainer.train()  # 如果需要断点恢复加 resume_from_checkpoint

# ========== 8. 保存模型 ==========
model.save_pretrained("lora_model_no_think")
tokenizer.save_pretrained("lora_model_no_think")

print(f"训练完成: 样本数 {len(train_hf)}")
