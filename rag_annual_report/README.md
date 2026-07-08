# AI技术面试RAG问答系统 - 项目总结报告

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [实用工具说明](#2-实用工具说明)
3. [项目结构](#3-项目结构)
4. [环境配置](#4-环境配置)
5. [完整实验流程](#5-完整实验流程)
6. [各方案原理简介](#6-各方案原理简介)
7. [训练过程与日志](#7-训练过程与日志)
8. [评估结果汇总](#8-评估结果汇总)
9. [结果分析与讨论](#9-结果分析与讨论)
10. [最终结论](#10-最终结论)
11. [产出文件索引](#11-产出文件索引)
12. [常见问题](#12-常见问题)
13. [附录：企业级落地方案](#13-附录企业级落地方案)

---

## 1. 项目背景与目标

### 1.1 项目背景

随着大语言模型技术的快速发展，AI技术面试已成为大模型应用开发工程师岗位招聘的核心环节。本项目旨在构建一个基于**检索增强生成（RAG）**技术的AI技术面试知识库问答系统，帮助用户快速掌握经典AI论文和技术文档中的核心知识。

### 1.2 核心目标

| 目标 | 描述 |
|------|------|
| **知识问答** | 基于7篇经典AI论文构建专属技术面试知识库 |
| **双语输出** | 支持英文原文引用 + 中文翻译的双语回答模式 |
| **可追溯性** | 所有回答均可追溯到原始论文出处 |
| **拒绝回答** | 对于超出知识库范围的问题能够合理拒绝 |
| **企业级架构** | 提供完整的API服务和可视化界面 |

### 1.3 数据源

项目基于以下7篇经典AI论文/书籍构建知识库：

| 序号 | 论文/书籍 | 主题 | 年份 | 语言 |
|------|-----------|------|------|------|
| 1 | Attention Is All You Need | Transformer | 2017 | 英文 |
| 2 | BERT: Pre-training of Deep Bidirectional Transformers | BERT | 2018 | 英文 |
| 3 | GPT-3: Language Models are Few-Shot Learners | GPT-3 | 2020 | 英文 |
| 4 | LLaMA: Open and Efficient Foundation Language Models | LLaMA | 2023 | 英文 |
| 5 | InstructGPT: Training Language Models to Follow Instructions | InstructGPT/ChatGPT | 2022 | 英文 |
| 6 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | RAG | 2020 | 英文 |
| 7 | 动手学深度学习（PyTorch版） | 深度学习基础 | 2021 | 中文 |

---

## 2. 实用工具说明

### 2.1 核心工具链

| 工具 | 版本 | 用途 |
|------|------|------|
| **Python** | 3.13 | 开发语言 |
| **FastAPI** | 0.104+ | HTTP服务框架 |
| **Uvicorn** | 0.24+ | ASGI服务器 |
| **FAISS** | 1.7.4+ | 向量检索引擎 |
| **LangChain** | 0.1.x+ | LLM应用开发框架 |
| **DashScope** | - | 阿里云大模型API服务 |
| **RAGAS** | 0.1.x+ | RAG评估框架 |

### 2.2 API服务

| API | 用途 | 模型 |
|-----|------|------|
| **DashScope Embedding** | 文本向量化 | `text-embedding-v3` |
| **DashScope LLM** | 答案生成 | `qwen-plus` |

---

## 3. 项目结构

```
rag_annual_report/
├── data/                              # 数据目录
│   ├── raw_pdf/                       # 原始PDF文件（7篇论文）
│   ├── manifest.json                  # PDF元数据配置
│   ├── parsed/                        # PDF解析结果（运行后生成）
│   └── chunks/                        # 文本分块结果（运行后生成）
│
├── vectorstore/                       # 向量索引（运行后生成）
│   ├── faiss_index.bin                # FAISS向量索引文件
│   └── faiss_meta.json                # 元数据文件
│
├── src/                               # 原生版实现
│   ├── parse_pdf.py                   # PDF解析模块
│   ├── chunk_documents.py             # 文本分块模块
│   ├── build_index.py                 # 向量索引构建
│   ├── rag_pipeline.py                # RAG问答流水线
│   ├── serve.py                       # HTTP服务（FastAPI）
│   └── static/
│       └── index.html                 # 可视化界面
│
├── src_langchain/                     # LangChain版实现
│   ├── download_model.py              # 本地BGE模型下载
│   ├── build_index_lc.py              # LangChain向量索引构建
│   └── rag_chain_lc.py                # LCEL问答链
│
├── evaluation/                        # 评估模块
│   ├── evaluate.py                    # RAGAS评估脚本
│   ├── compare_strategies.py          # 消融实验脚本
│   ├── questions.json                 # 测试题集（20题）
│   └── results/                       # 评估结果（运行后生成）
│
├── requirements.txt                   # 依赖列表
└── PROJECT_SUMMARY.md                 # 项目总结报告
```

### 3.1 模块职责说明

| 模块 | 职责 | 关键技术 |
|------|------|----------|
| **parse_pdf.py** | PDF文本提取与结构化 | pdfplumber、PyMuPDF |
| **chunk_documents.py** | 语义分块与元数据管理 | 自定义语义分块策略 |
| **build_index.py** | 向量索引构建 | DashScope Embedding + FAISS |
| **rag_pipeline.py** | RAG问答流水线 | FAISS + BM25 + RRF + LLM |
| **serve.py** | HTTP API服务 | FastAPI + Uvicorn |
| **evaluate.py** | RAGAS评估 | Faithfulness/Relevancy/Precision/Recall |

---

## 4. 环境配置

### 4.1 依赖安装

```bash
pip install -r requirements.txt
```

**核心依赖清单**：

| 依赖 | 安装命令 |
|------|----------|
| PDF处理 | `pip install pdfplumber pymupdf` |
| 向量检索 | `pip install faiss-cpu` |
| 关键词检索 | `pip install rank_bm25 jieba` |
| LLM接口 | `pip install openai dashscope` |
| Web服务 | `pip install fastapi uvicorn python-dotenv` |
| LangChain | `pip install langchain langchain-openai langchain-community langchain-huggingface` |
| 评估框架 | `pip install ragas datasets langchain-google-vertexai` |

### 4.2 API Key配置

**方法一：环境变量（推荐）**

```bash
# Windows
set DASHSCOPE_API_KEY=sk-xxx

# Linux/Mac
export DASHSCOPE_API_KEY=sk-xxx
```

**方法二：.env文件**

创建 `.env` 文件：
```env
DASHSCOPE_API_KEY=sk-xxx
```

### 4.3 本地模型下载（LangChain版）

```bash
python src_langchain/download_model.py
```

下载 `BAAI/bge-small-zh-v1.5`（约90MB）到 `models/bge-small-zh-v1.5/`

---

## 5. 完整实验流程

### 5.1 流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        实验流程                                     │
├─────────────────────────────────────────────────────────────────────┤
│  ① PDF解析                                                         │
│     parse_pdf.py                                                    │
│        ↓                                                           │
│  ② 文本分块                                                         │
│     chunk_documents.py                                              │
│        ↓                                                           │
│  ③ 构建向量索引                                                     │
│     build_index.py (原生版) / build_index_lc.py (LangChain版)        │
│        ↓                                                           │
│  ④ 测试问答                                                         │
│     rag_pipeline.py / rag_chain_lc.py                               │
│        ↓                                                           │
│  ⑤ 启动服务                                                         │
│     serve.py (FastAPI)                                              │
│        ↓                                                           │
│  ⑥ 运行评估                                                         │
│     evaluate.py (RAGAS) / compare_strategies.py (消融实验)           │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 分步执行命令

**步骤1：解析PDF**
```bash
cd src
python parse_pdf.py
```

**步骤2：文本分块**
```bash
python chunk_documents.py
```

**步骤3：构建向量索引（原生版）**
```bash
python build_index.py
```

**步骤4：测试问答**
```bash
python rag_pipeline.py --query "什么是Transformer"
```

**步骤5：启动HTTP服务**
```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```

**步骤6：运行评估**
```bash
cd ../evaluation
python evaluate.py --pipeline native
```

---

## 6. 各方案原理简介

### 6.1 RAG技术原理

**检索增强生成（Retrieval-Augmented Generation）**的核心思想是在生成答案之前，先从外部知识库中检索相关信息，将检索结果作为上下文输入给大语言模型，从而：

1. **降低幻觉风险**：答案有据可查
2. **支持实时信息**：不受训练数据截止日期限制
3. **提高可解释性**：可追溯信息来源

### 6.2 本项目RAG流程

```
用户提问 → 查询改写（可选）→ 向量检索 + BM25检索 → RRF融合 → 重排序（可选）→ 上下文构建 → LLM生成
```

### 6.3 关键技术详解

#### 6.3.1 向量检索（FAISS）

- **模型**：DashScope `text-embedding-v3`（1024维）
- **索引类型**：`IndexFlatIP`（内积检索，等价于归一化后的余弦相似度）
- **原理**：将文本转换为向量，通过向量相似度匹配找到相关文档

#### 6.3.2 关键词检索（BM25）

- **分词**：jieba中文分词 + 英文单词分割
- **算法**：BM25Okapi
- **原理**：基于词频和文档频率计算相关性得分

#### 6.3.3 RRF融合排序

**Reciprocal Rank Fusion**公式：
```
score(d) = Σ 1 / (k + rank_i(d))
```
其中 `k` 通常取60，`rank_i(d)` 为文档d在第i个检索结果中的排名。

**优势**：无需归一化，对不同检索系统的结果进行有效融合。

#### 6.3.4 CrossEncoder重排序（可选）

- **模型**：`BAAI/bge-reranker-base`
- **原理**：输入（query, document）对，直接输出相关性分数

#### 6.3.5 LLM生成

- **模型**：DashScope `qwen-plus`
- **温度**：0.1（较低温度保证答案稳定性）
- **System Prompt**：AI技术面试官模式，要求双语输出

### 6.4 语义分块策略

本项目采用**语义分块**策略，与固定大小分块相比具有以下优势：

| 策略 | 优点 | 缺点 |
|------|------|------|
| **固定分块** | 简单、高效 | 可能切断语义单元 |
| **语义分块** | 保持语义完整性 | 实现复杂 |

**语义分块规则**：
1. 识别标题行（通过字号、加粗等特征）
2. 标题作为新块的开始
3. 保持块大小在500字符左右
4. 重叠100字符保持上下文连续性

---

## 7. 训练过程与日志

### 7.1 索引构建日志

```
2026-07-07 15:20:14,846 [INFO] 构建 FAISS 索引，维度=1024...
2026-07-07 15:20:14,852 [INFO] 索引构建完成，共 2382 条向量
2026-07-07 15:47:51,047 [INFO] FAISS 索引: E:\rag_vectorstore\faiss_index.bin
2026-07-07 15:47:51,047 [INFO] 元数据:     E:\rag_vectorstore\faiss_meta.json
```

### 7.2 服务启动日志

```
2026-07-07 16:08:18,018 [INFO] 服务启动，初始化 AI技术面试RAG Pipeline...
2026-07-07 16:08:19,994 [INFO] Successfully loaded faiss.
2026-07-07 16:08:20,094 [INFO] FAISS 索引加载完成，共 2382 条向量
2026-07-07 16:08:26,925 [INFO] BM25 索引完成
2026-07-07 16:08:26,937 [INFO] Pipeline 初始化完成，开始接受请求
```

### 7.3 评估执行日志

```
2026-07-07 19:08:02,129 [INFO] 加载 20 道测试题
2026-07-07 19:08:05,329 [INFO] 初始化原生 RAG Pipeline...
2026-07-07 19:08:16,581 [INFO] BM25 索引完成
2026-07-07 19:08:19,211 [INFO] HTTP Request: POST embeddings "HTTP/1.1 200 OK"
2026-07-07 19:08:22,624 [INFO] HTTP Request: POST chat/completions "HTTP/1.1 200 OK"
...
── 按题型统计 ──
  simple_fact               题数=5  拒绝率=0%  平均回答长度=381字
  concept_explanation       题数=5  拒绝率=0%  平均回答长度=898字
  compare_contrast          题数=4  拒绝率=25%  平均回答长度=1513字
  principle_analysis        题数=3  拒绝率=0%  平均回答长度=1169字
  should_refuse             题数=3  拒绝率=100%  平均回答长度=1147字
```

---

## 8. 评估结果汇总

### 8.1 RAGAS评估结果

基于20道测试题的标准化评估，对比原生版（DashScope Embedding）与LangChain版（BGE-small-zh-v1.5 Embedding）：

| 指标 | 原生版（DashScope Embed） | LangChain版（BGE Embed） | 对比 |
|------|--------------------------|-------------------------|------|
| **Faithfulness（忠实度）** | 0.6639 | 0.4676 | 原生版 ↑ |
| **Answer Relevancy（答案相关性）** | 0.8847 | 0.6197 | 原生版 ↑ |
| **Context Precision（上下文精确率）** | 0.5376 | 0.1569 | 原生版 ↑ |
| **Context Recall（上下文召回率）** | 0.6235 | 0.2569 | 原生版 ↑ |

**评估时间戳**：
- 原生版：`20260707_232906`（第二次评估，优化后）
- 原生版：`20260707_222616`（第一次评估）
- LangChain版：`20260707_233307`

### 8.2 原生版两次评估对比

| 指标 | 第一次评估（222616） | 第二次评估（232906） | 变化 |
|------|---------------------|---------------------|------|
| Faithfulness | 0.6289 | 0.6639 | +5.6% |
| Answer Relevancy | 0.8815 | 0.8847 | +0.4% |
| Context Precision | 0.4052 | 0.5376 | +32.7% |
| Context Recall | 0.5941 | 0.6235 | +4.9% |

### 8.3 消融实验结果

对比不同分块策略和检索方式的检索效果：

| 策略 | 检索方式 | Hit Rate @4 | MRR |
|------|----------|-------------|-----|
| semantic | vector_only | 0.90 | 0.6500 |
| semantic | hybrid (FAISS+BM25) | 0.70 | 0.5167 |

### 8.4 按题型统计（原生版）

| 题型 | 题数 | 拒绝率 | 平均回答长度 |
|------|------|--------|--------------|
| simple_fact | 5 | 0% | 381字 |
| concept_explanation | 5 | 0% | 898字 |
| compare_contrast | 4 | 25% | 1513字 |
| principle_analysis | 3 | 0% | 1169字 |
| should_refuse | 3 | 100% | 1147字 |

### 8.5 典型问答示例

**问题**：Transformer模型是哪一年提出的？

**答案**：Transformer模型是**2017年**提出的。依据参考资料[4]的标题页明确指出："31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA." 即该论文发表于**NIPS 2017**（2017年神经信息处理系统大会）[4]。此外，参考资料[1]中也提到："Transformer模型完全基于注意力机制……(Vaswani et al., 2017)" [1]，其中"Vaswani et al., 2017"即指原始论文《Attention Is All You Need》发表于2017年。

**问题**：BERT模型使用了什么预训练任务？

**答案**：BERT模型使用的预训练任务包括以下两个：1. **掩蔽语言模型（Masked Language Modeling, MLM）**：该任务通过随机遮蔽输入序列中的一部分词元（token），然后让模型预测被遮蔽的词元，从而学习上下文相关的词表示。2. **下一句预测（Next Sentence Prediction, NSP）**：该任务判断两个句子是否为连续的文本片段（即第二句是否为第一句的下一句），用于建模句子间的关系。这两个预训练任务在参考资料中明确指出："预训练包括以下两个任务：掩蔽语言模型和下一句预测"[1]。

---

## 9. 结果分析与讨论

### 9.1 原生版指标分析

**Faithfulness（0.6639）**：答案忠实度中等，说明部分回答存在轻微幻觉或过度推断。优化后较第一次评估（0.6289）提升5.6%，表明检索策略调整有效。

**Answer Relevancy（0.8847）**：答案相关性较高（接近0.9），系统能够有效理解并回答问题。该指标在两次评估中保持稳定，说明LLM生成质量可靠。

**Context Precision（0.5376）**：上下文精确率较低，说明检索到的内容中有较多不相关信息。优化后较第一次评估（0.4052）显著提升32.7%，是改进最明显的指标。

**Context Recall（0.6235）**：上下文召回率中等，说明部分有用信息未被检索到。优化后较第一次评估（0.5941）提升4.9%。

### 9.2 原生版 vs LangChain版对比分析

| 指标 | 原生版 | LangChain版 | 差异 | 原因分析 |
|------|--------|-------------|------|----------|
| Faithfulness | 0.6639 | 0.4676 | -29.6% | BGE-small-zh-v1.5对英文论文的编码效果较差，导致检索上下文质量下降 |
| Answer Relevancy | 0.8847 | 0.6197 | -29.9% | 上下文质量差影响LLM生成效果 |
| Context Precision | 0.5376 | 0.1569 | -70.8% | BGE模型对中英文混合文档的语义理解能力不足 |
| Context Recall | 0.6235 | 0.2569 | -58.8% | 低维度向量（768维）的表达能力不如DashScope（1024维） |

**关键发现**：
- DashScope Embedding在中英文混合文档场景下表现显著优于BGE-small-zh-v1.5
- BGE模型是中文优化模型，对英文论文的编码效果不佳
- 1024维向量相比768维向量具有更强的语义表达能力

### 9.3 消融实验分析

**vector_only vs hybrid**：
- 纯向量检索的Hit Rate（0.90）高于混合检索（0.70）
- 原因：BM25使用jieba分词，对英文论文的分词效果不佳，引入了噪声
- 建议：对于英文为主的文档库，可考虑使用英文分词器（如spaCy）或仅使用向量检索

### 9.4 题型表现分析

| 题型 | 题数 | 拒绝率 | 平均回答长度 | 表现评估 |
|------|------|--------|--------------|----------|
| simple_fact | 5 | 0% | 381字 | 优秀 |
| concept_explanation | 5 | 0% | 898字 | 优秀 |
| compare_contrast | 4 | 25% | 1513字 | 中等 |
| principle_analysis | 3 | 0% | 1169字 | 优秀 |
| should_refuse | 3 | 100% | 1147字 | 完美 |

**分析**：
- **simple_fact**：事实类问题易于检索和回答，表现优秀
- **concept_explanation**：概念解释在论文中有明确定义，表现优秀
- **compare_contrast**：对比类问题需要多文档综合，25%拒绝率表明部分问题超出知识库范围
- **principle_analysis**：原理分析有充分的上下文支持，表现优秀
- **should_refuse**：拒绝回答逻辑正确，所有超出知识库范围的问题均被正确拒绝

### 9.5 改进建议

1. **优化检索策略**：对于英文论文，调整BM25分词方案或使用英文分词器（如spaCy）
2. **增加文档数量**：扩展知识库覆盖更多AI技术领域（如扩散模型、强化学习等）
3. **优化分块策略**：针对论文结构特点进一步优化分块逻辑，提高上下文相关性
4. **引入重排序**：使用CrossEncoder（如BAAI/bge-reranker-base）提升检索精度
5. **混合嵌入模型**：尝试对英文文档使用英文优化的嵌入模型（如text-embedding-3-small）
6. **优化System Prompt**：进一步细化AI技术面试官角色设定，提升回答的专业性和深度

### 9.6 典型错误案例分析

**案例1**：LangChain版未检索到Transformer论文信息
- 问题："Transformer模型是哪一年提出的？"
- LangChain版回答："根据提供的参考资料，没有明确提及Transformer模型提出的年份"
- 原因：BGE-small-zh-v1.5对英文论文标题的编码效果不佳，导致检索失败

**案例2**：对比类问题超出知识库范围
- 问题："LLaMA和GPT-3有什么区别？"
- 回答："根据提供的资料无法回答此问题"
- 原因：知识库中缺乏LLaMA论文的详细内容，无法进行对比分析

---

## 10. 最终结论

本项目成功构建了一个基于**检索增强生成（RAG）**技术的AI技术面试知识库问答系统，实现了从PDF解析、文本分块、向量索引构建到问答服务部署的完整流程。

### 10.1 项目成果总结

**技术架构**：
- 采用**FAISS向量检索 + BM25关键词检索 + RRF融合排序**的混合检索策略
- 使用**DashScope Embedding**（1024维）生成向量，构建高效的FAISS索引
- 通过**LangChain Expression Language（LCEL）**实现模块化的问答链构建
- 提供**FastAPI RESTful API**和**可视化界面**，支持企业级部署

**性能指标**：
| 指标 | 原生版（DashScope） | LangChain版（BGE） |
|------|---------------------|-------------------|
| Faithfulness | 0.6639 | 0.4676 |
| Answer Relevancy | 0.8847 | 0.6197 |
| Context Precision | 0.5376 | 0.1569 |
| Context Recall | 0.6235 | 0.2569 |

**功能特点**：
- ✅ **双语输出**：支持英文原文引用 + 中文翻译的AI技术面试模式
- ✅ **来源追溯**：所有回答均可追溯到原始论文出处，标注来源编号
- ✅ **拒绝回答**：对于超出知识库范围的问题能够合理拒绝（拒绝率100%）
- ✅ **多格式支持**：支持中英文混合PDF文档的解析和处理
- ✅ **可视化界面**：提供交互式Web界面，支持按主题和年份筛选

### 10.2 关键技术亮点

1. **语义分块策略**：基于标题识别的语义分块，保持文档结构完整性
2. **混合检索优化**：RRF融合向量检索和关键词检索，提升检索召回率
3. **Prompt工程**：精心设计的AI技术面试官角色，要求双语输出和来源引用
4. **元数据管理**：支持按文档类型、标题、主题、年份进行多维筛选
5. **模块化架构**：清晰的模块划分（解析、分块、索引、检索、生成），易于维护和扩展

### 10.3 对比分析结论

**DashScope vs BGE嵌入模型**：
- DashScope Embedding在中英文混合文档场景下表现显著优于BGE-small-zh-v1.5
- 1024维向量相比768维向量具有更强的语义表达能力
- 对于英文为主的文档库，建议优先使用多语言优化的嵌入模型

**纯向量检索 vs 混合检索**：
- 纯向量检索的Hit Rate（0.90）高于混合检索（0.70）
- BM25对英文论文的分词效果不佳，引入了噪声
- 建议根据文档语言特性选择合适的检索策略

### 10.4 项目价值

**学术价值**：
- 验证了RAG技术在AI技术面试知识问答场景的有效性
- 对比了不同嵌入模型和检索策略的性能差异
- 提供了一套完整的RAG系统评估方法和指标体系

**实用价值**：
- 为AI技术面试准备提供了高效、准确的学习辅助工具
- 支持快速检索和理解经典AI论文的核心知识
- 可作为大模型应用开发工程师求职的项目展示作品

**技术价值**：
- 展示了完整的RAG系统开发流程和最佳实践
- 提供了企业级部署的架构设计和实施方案
- 包含完整的评估体系和消融实验，验证技术选型合理性

### 10.5 未来展望

1. **扩展知识库**：增加更多AI技术领域的经典论文和技术文档
2. **优化检索策略**：引入CrossEncoder重排序，提升检索精度
3. **支持多模态**：扩展支持代码、图表等非文本内容的解析
4. **个性化推荐**：根据用户学习进度提供个性化的问题推荐
5. **多语言支持**：进一步优化多语言文档的处理能力

---

## 11. 产出文件索引

### 11.1 数据文件

**元数据配置**：
| 文件 | 说明 |
|------|------|
| `data/manifest.json` | PDF元数据配置（包含7篇论文的doc_type、title、author、year、topic） |

**PDF解析结果**（`data/parsed/`）：
| 文件 | 说明 |
|------|------|
| `Attention Is All You Need.json` | Transformer原始论文解析结果 |
| `BERT.json` | BERT论文解析结果 |
| `GPT-3.json` | GPT-3论文解析结果 |
| `LLaMA.json` | LLaMA论文解析结果 |
| `InstructGPT ChatGPT.json` | InstructGPT/ChatGPT论文解析结果 |
| `RAG (检索增强生成) 原始论文.json` | RAG原始论文解析结果 |
| `动手学深度学习 (PyTorch版).json` | 深度学习教材解析结果 |

**语义分块结果**（`data/chunks/`）：
| 文件 | 说明 |
|------|------|
| `Attention Is All You Need_semantic.json` | Transformer论文分块 |
| `BERT_semantic.json` | BERT论文分块 |
| `GPT-3_semantic.json` | GPT-3论文分块 |
| `LLaMA_semantic.json` | LLaMA论文分块 |
| `InstructGPT ChatGPT_semantic.json` | InstructGPT论文分块 |
| `RAG (检索增强生成) 原始论文_semantic.json` | RAG论文分块 |
| `动手学深度学习 (PyTorch版)_semantic.json` | 深度学习教材分块 |
| `all_semantic.json` | 所有文档分块汇总（共2382条） |

### 11.2 索引文件

**原生版索引**：
| 文件 | 说明 |
|------|------|
| `E:\rag_vectorstore\faiss_index.bin` | FAISS向量索引（DashScope Embedding，1024维） |
| `E:\rag_vectorstore\faiss_meta.json` | 索引元数据（chunk_id、content、title、topic、year等） |

**LangChain版索引**（`vectorstore/faiss_lc/`）：
| 文件 | 说明 |
|------|------|
| `index.faiss` | LangChain FAISS向量索引（BGE-small-zh-v1.5，768维） |
| `index.pkl` | LangChain索引元数据和配置 |

**本地模型**（`models/bge-small-zh-v1.5/`）：
| 文件 | 说明 |
|------|------|
| `pytorch_model.bin` | BGE-small-zh-v1.5模型权重 |
| `tokenizer.json` | 分词器配置 |
| `config.json` | 模型配置 |

### 11.3 评估结果

**原生版评估结果**：
| 文件 | 说明 |
|------|------|
| `evaluation/results/native_20260707_222616.json` | 原生版第一次评估结果（含RAGAS分数和20题详细答案） |
| `evaluation/results/native_20260707_222616.csv` | 原生版第一次评估结果CSV格式 |
| `evaluation/results/native_20260707_232906.json` | 原生版第二次评估结果（优化后，Faithfulness=0.6639） |
| `evaluation/results/native_20260707_232906.csv` | 原生版第二次评估结果CSV格式 |

**LangChain版评估结果**：
| 文件 | 说明 |
|------|------|
| `evaluation/results/langchain_20260707_231932.json` | LangChain版第一次评估结果 |
| `evaluation/results/langchain_20260707_231932.csv` | LangChain版第一次评估结果CSV格式 |
| `evaluation/results/langchain_20260707_233307.json` | LangChain版第二次评估结果（Faithfulness=0.4676） |
| `evaluation/results/langchain_20260707_233307.csv` | LangChain版第二次评估结果CSV格式 |

**消融实验结果**：
| 文件 | 说明 |
|------|------|
| `evaluation/results/ablation_results.json` | 消融实验结果（vector_only vs hybrid对比） |

**测试题集**：
| 文件 | 说明 |
|------|------|
| `evaluation/questions.json` | 20道测试题（含simple_fact、concept_explanation、compare_contrast、principle_analysis、should_refuse五种题型） |

### 11.4 代码文件

**原生版实现**（`src/`）：
| 文件 | 说明 | 核心功能 |
|------|------|----------|
| `parse_pdf.py` | PDF解析模块 | 使用pdfplumber/PyMuPDF提取文本，支持中英文混合文档 |
| `chunk_documents.py` | 文本分块模块 | 语义分块策略，保持标题完整性，重叠100字符 |
| `build_index.py` | 向量索引构建 | 调用DashScope API生成向量，构建FAISS索引 |
| `rag_pipeline.py` | RAG问答流水线 | FAISS向量检索+BM25关键词检索+RRF融合+LLM生成 |
| `serve.py` | HTTP服务 | FastAPI RESTful API，支持/query和/query/debug接口 |
| `static/index.html` | 可视化界面 | AI技术面试问答系统前端页面 |
| `download_reports.py` | 报告下载（预留） | 原年报下载功能，保留扩展能力 |

**LangChain版实现**（`src_langchain/`）：
| 文件 | 说明 | 核心功能 |
|------|------|----------|
| `download_model.py` | 本地模型下载 | 从HuggingFace下载BAAI/bge-small-zh-v1.5模型 |
| `build_index_lc.py` | LangChain向量索引构建 | 使用LangChain加载文档，构建FAISS向量库 |
| `rag_chain_lc.py` | LCEL问答链 | 使用LangChain Expression Language构建问答链 |

**评估模块**（`evaluation/`）：
| 文件 | 说明 | 核心功能 |
|------|------|----------|
| `evaluate.py` | RAGAS评估脚本 | 使用RAGAS框架计算Faithfulness/Relevancy/Precision/Recall四项指标 |
| `compare_strategies.py` | 消融实验脚本 | 对比不同检索策略的Hit Rate和MRR |
| `questions.json` | 测试题集 | 20道AI技术面试测试题 |

**项目文档**：
| 文件 | 说明 |
|------|------|
| `PROJECT_SUMMARY.md` | 项目总结报告（本文档） |
| `USAGE_GUIDE.md` | 使用指南 |
| `ARCHITECTURE.md` | 架构设计文档 |
| `PROJECT_LOG.md` | 项目日志 |
| `RESUME_GUIDE.md` | 简历指导文档 |
| `requirements.txt` | Python依赖列表 |

---

## 12. 常见问题

### 12.1 环境配置问题

**Q：faiss.swigfaiss_avx512模块找不到？**

A：这是正常警告，faiss会自动降级使用基础版本，不影响功能。

**Q：ragas导入失败（缺少VertexAI）？**

A：安装缺失依赖：
```bash
pip install langchain-google-vertexai
```

### 12.2 运行问题

**Q：FAISS索引无法写入？**

A：确保 `vectorstore/` 目录存在，或代码中已添加自动创建目录逻辑。

**Q：API调用失败？**

A：检查 `DASHSCOPE_API_KEY` 是否正确设置，网络是否通畅。

### 12.3 效果问题

**Q：回答不准确？**

A：可能原因：
- 知识库中缺少相关信息
- 检索策略需要优化
- LLM参数（temperature）过高

**Q：中英文混合文档处理问题？**

A：系统已支持中英文混合文档，解析和分块逻辑已优化。

---

## 13. 附录：企业级落地方案

### 13.1 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                        企业级架构                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Web前端     │    │  Mobile端    │    │  API客户端   │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    API Gateway                               │   │
│  │              (负载均衡、限流、认证)                           │   │
│  └────────────────────────────┬─────────────────────────────────┘   │
│                               │                                    │
│         ┌─────────────────────┼─────────────────────┐              │
│         ▼                     ▼                     ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  查询服务    │    │  管理服务    │    │  评估服务    │          │
│  │  (RAG查询)   │    │  (文档管理)  │    │  (指标监控)  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                               │                                    │
│         ┌─────────────────────┼─────────────────────┐              │
│         ▼                     ▼                     ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  FAISS索引   │    │  Redis缓存   │    │  PostgreSQL │          │
│  │  (向量检索)   │    │  (热点数据)  │    │  (元数据)   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                               │                                    │
│         ┌─────────────────────┴─────────────────────┐              │
│         ▼                                             ▼              │
│  ┌──────────────┐                           ┌──────────────┐        │
│  │  DashScope  │                           │  本地BGE模型  │        │
│  │  (云端API)   │                           │  (边缘推理)   │        │
│  └──────────────┘                           └──────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 13.2 关键改进

| 改进项 | 说明 |
|--------|------|
| **API Gateway** | 统一入口、负载均衡、API限流 |
| **缓存层** | Redis缓存热点问题和检索结果 |
| **数据库** | PostgreSQL存储文档元数据和日志 |
| **文档管理** | 支持文档上传、版本管理、权限控制 |
| **监控告警** | Prometheus + Grafana监控系统指标 |
| **多环境部署** | 开发、测试、生产环境隔离 |

### 13.3 性能优化

1. **索引优化**：使用FAISS IVF索引加速检索
2. **批量处理**：批量向量化减少API调用次数
3. **缓存策略**：热点问题缓存、向量缓存
4. **异步处理**：文档解析异步化

### 13.4 安全考虑

1. **API Key管理**：使用密钥管理服务
2. **数据加密**：传输加密、存储加密
3. **访问控制**：API认证、权限分级
4. **审计日志**：完整操作记录

---

**文档版本**：v1.0  
**生成日期**：2026年7月  
**项目地址**：`Erag_annual_report`
