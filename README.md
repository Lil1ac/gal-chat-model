# gal-chat

gal-chat 是一个基于大语言模型的对话代理系统，专注于对话生成、知识图谱构建与检索增强生成（RAG）。

## 项目概述

LilacAgent 旨在为 AI 研究人员、对话系统开发者和内容创作者提供一个功能齐全的对话代理平台。该项目整合了多种先进技术，包括大语言模型微调、知识图谱构建和检索增强生成（RAG），以提供高质量的对话体验。

## 项目流程

本项目的工作流程主要包括以下几个阶段：

### 1. 数据准备与预处理
- 使用 [src/data_preprocessing](src/data_preprocessing) 目录下的脚本进行原始数据处理
- 提取对话内容、合并相同场景文本
- 合成新的对话数据用于模型训练

### 2. 模型训练与微调
- 基于 LCCC 数据集合成对话数据
- 使用 LoRA 技术对 Qwen3-8B 模型进行微调
- 训练后的模型保存在 [src/model/lora_model_no_think](src/model/lora_model_no_think) 目录下

### 3. 知识库构建
- 构造章节、线路、场景、角色、对话的结构化文本
- 使用 GraphRAG 技术构建知识库
- 知识库相关文件存储在 [src/graphrag/summary2](src/graphrag/summary2) 目录下

### 4. 系统部署与服务启动
- 使用 vLLM 部署 LoRA 微调后的 Qwen3 模型
- 使用 FastChat 部署 Qwen3-embedding-0.6b 模型
- 启动 GraphRAG API 服务（基于 FastAPI）
- 启动 Web UI 界面（基于 Gradio）

### 5. 对话处理流程
- 用户通过 Web UI 提出问题
- 系统通过 API 将问题传入 GraphRAG 进行 embedding
- 检索相关知识库内容
- 将问题和检索到的知识库内容一起传给模型生成回答

## 目录结构

```
.
├── graphrag-api/                 # GraphRAG API 服务
│   ├── services/                 # 核心服务模块
│   ├── static/                   # 静态资源
│   ├── artifacts/                # 构件文件
│   └── ...                       # 其他配置和脚本文件
├── src/
│   ├── chat/                     # 对话系统模块
│   │   ├── data-process/         # 数据处理脚本
│   │   └── train/                # 训练脚本
│   ├── data_preprocessing/       # 数据预处理工具
│   ├── model/                    # 模型文件和合并脚本
│   │   └── lora_model_no_think/  # LoRA 微调模型文件
│   ├── graphrag/                 # GraphRAG 相关模块
│   │   └── summary2/             # 摘要数据
│   ├── data/                     # 数据文件
│   └── utils/                    # 工具模块
└── ...
```

## 主要功能模块

### 1. 对话系统 (Chat System)
位于 [src/chat](src/chat) 目录下，包含以下组件：
- [chat.py](src/chat/chat.py) - 核心对话功能
- [langchain_web.py](src/chat/langchain_web.py) - LangChain 集成的 Web 接口
- [webui.py](src/chat/webui.py) - Web 用户界面（基于Gradio）
- [train/](src/chat/train) - 模型训练和测试脚本
- [data-process/](src/chat/data-process) - 数据处理脚本

### 2. 数据预处理 (Data Preprocessing)
位于 [src/data_preprocessing](src/data_preprocessing) 目录下，包含多种数据处理脚本：
- 对话提取和场景合并工具
- 知识图谱构建脚本
- 数据合成和转换工具

### 3. 模型 (Model)
位于 [src/model](src/model) 目录下：
- [lora_model_no_think/](src/model/lora_model_no_think) - LoRA 微调模型文件
- [merge-model.py](src/model/merge-model.py) - 模型合并脚本

### 4. GraphRAG API
位于 [graphrag-api](graphrag-api) 目录下，提供基于 FastAPI 的 GraphRAG 查询服务：
- 全局和本地结构化搜索
- RESTful API 接口
- 可视化查询界面

### 5. 工具 (Utils)
位于 [src/utils](src/utils) 目录下，包含各种辅助工具和配置文件。

## 安装和使用

### 环境要求
- Python 3.10+
- transformers >=4.36.0
- PyTorch >=2.0.0
- sentence-transformers >=2.2.2
- Faiss CPU >=1.7.4
- bitsandbytes >=0.41.0 (用于模型量化)
- accelerate >=0.25.0

### 安装步骤
```bash
# 克隆项目
git clone <项目地址>
cd LilacAgent

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 启动服务

#### 启动本地原生模型
vllm启动微调前的Qwen3-8B

#### 启动本地微调后的模型
vllm启动LoRA微调后的Qwen3-8B-lora

#### 启动Embedding模型
FastChat启动Qwen3-embedding-0.6B模型

#### 启动 GraphRAG API 服务
使用Fastapi封装的GraphRAG API

#### 启动对话系统
```bash
cd src/chat
python langchain_web.py
```

## 技术特点

1. **对话生成** - 使用微调模型（如 Qwen）进行对话响应生成
2. **知识图谱构建** - 从文本中提取结构化知识并构建图谱
3. **数据预处理** - 提供丰富的脚本用于提取对话、合并场景、合成数据等
4. **检索增强生成（RAG）** - 结合 Faiss 索引与向量检索提升生成质量
5. **模型训练与微调** - 提供训练脚本与 LoRA 模型支持

## 开发和部署

- 支持本地部署与推理加速（如 bitsandbytes 量化）
- 支持 LangChain 集成的 Web 接口
- 多种模型配置与合并策略支持
- 可部署为 Web 服务（如 FastAPI + Uvicorn）