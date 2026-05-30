# 基于 BERT 与大模型的中文文本分类对比实验

本项目在 **TNEWS 15 类中文新闻标题** 任务上，完成 BERT 池化消融实验，并对比 **BERT 微调**、**LLM 零样本**、**LLM LoRA 指令微调** 三种文本分类范式。实验在本地 **CPU 快速验证模式** 下完成，适合课程实验与轻量化部署场景。

---

## 一、实验目的

1. 完成 BERT 文本分类 **消融实验**，对比 `cls` / `mean` / `max` 三种池化方式；
2. 对比 **传统微调 BERT**、**LLM 零样本分类**、**LLM 指令微调（LoRA）** 三种方法的性能；
3. 分析不同池化策略与训练范式对分类效果的影响，给出可落地的方案选择建议。

---

## 二、实验环境与平台

| 项目 | 说明 |
|------|------|
| 运行平台 | 本地 CPU（快速验证模式） |
| 框架 | PyTorch、Transformers、PEFT（LoRA） |
| BERT 模型 | `bert-base-chinese`（约 102.3M 参数） |
| LLM 模型 | `Qwen2-0.5B-Instruct`（约 494M 参数） |
| 任务 | 15 类中文文本分类（TNEWS） |
| 评估指标 | 准确率（Accuracy）、宏平均 F1（macro F1） |
| BERT 训练策略 | `--fast`：冻结 BERT 主干，仅训练分类头 |
| LLM 训练策略 | LoRA 微调，可训练参数约 **0.1093%** |

---

## 三、项目结构

```
text_classification项目/
├── data/                          # 数据集（train / val / test / label_map.json）
├── pretrain_models/
│   ├── bert-base-chinese/         # BERT 预训练权重
│   └── Qwen2-0.5B-Instruct/       # Qwen2 预训练权重（含 model.safetensors）
├── src/                           # BERT 相关脚本
│   ├── train.py                   # BERT 训练（支持 --fast / --pool）
│   ├── evaluate.py                # BERT 评估（支持 --all_pools）
│   ├── predict.py                 # 单条 / 批量推理
│   ├── model.py                   # BertClassifier（三种池化）
│   ├── dataset.py                 # 数据加载
│   └── compare_pools.py           # 池化对比（等价 evaluate --all_pools）
├── src_llm/                       # 大模型相关脚本
│   ├── classify_llm.py            # LLM 零样本分类
│   ├── train_sft.py               # LLM LoRA 指令微调
│   ├── evaluate_sft.py            # SFT 模型评估
│   └── qwen_tokenizer.py          # Qwen2 tokenizer 兼容加载
├── outputs/                       # 训练日志、checkpoint、评估结果
│   ├── checkpoints/               # best_cls.pt / best_mean.pt / best_max.pt
│   ├── sft_adapter/               # LoRA adapter
│   ├── train_log_*.json
│   ├── llm_zero_shot_results.json
│   └── llm_sft_results.json
├── requirements.txt
├── USAGE_GUIDE.md                 # 详细使用说明
└── ARCHITECTURE.md                # 架构说明
```

---

## 四、环境准备

### 4.1 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖：`torch`、`transformers`、`peft`（LoRA）、`scikit-learn`、`tqdm` 等。

### 4.2 预训练模型

将模型放入 `pretrain_models/`：

```
pretrain_models/
├── bert-base-chinese/
│   ├── config.json
│   └── vocab.txt（及权重文件）
└── Qwen2-0.5B-Instruct/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json（可选，脚本可自动生成）
    └── model.safetensors
```

### 4.3 数据

若 `data/` 为空，可在 `src/` 下运行：

```bash
cd src
python download_data.py
```

---

## 五、实验方案

### 方案 1：BERT 消融实验（三种池化）

固定 BERT 主干（`--fast` 模式下冻结），仅改变池化方式：

| 池化 | 说明 |
|------|------|
| `cls` | 取 `[CLS]` token 向量 |
| `mean` | 对有效 token 做平均池化 |
| `max` | 对有效 token 做最大池化 |

### 方案 2：LLM 零样本分类

无训练，直接使用 `Qwen2-0.5B-Instruct` 按指令格式输出类别名。

### 方案 3：LLM 指令微调（LoRA）

将分类任务转为 chat 格式 SFT，用 LoRA 仅微调约 0.1% 参数。

---

## 六、完整运行命令

### 6.1 BERT 训练（消融实验）

```bash
cd src

python train.py --pool cls  --epochs 1 --fast
python train.py --pool mean --epochs 1 --fast
python train.py --pool max  --epochs 1 --fast
```

### 6.2 BERT 模型评估

```bash
python evaluate.py --all_pools --fast
```

### 6.3 BERT 推理

```bash
# 单条
python predict.py --pool mean --text "今天股市大幅下跌"

# 批量（500 条快速试跑）
python predict.py --pool mean --input_file ../data/val.json --fast
```

### 6.4 LLM 零样本分类

```bash
cd ../src_llm

python classify_llm.py --demo
python classify_llm.py --text "今天股市大幅下跌" --fast
```

### 6.5 LLM LoRA 指令微调

```bash
python train_sft.py --demo    # 冒烟（最快）
python train_sft.py --fast    # CPU 试跑（约 90s）
```

### 6.6 SFT 模型评估

```bash
python evaluate_sft.py --demo
python evaluate_sft.py --fast
```

---

## 七、实验结果

> 以下数据来自本项目 **CPU + `--fast` 快速模式** 的实际运行日志（验证集各 2000 条；LLM 评估为 `--demo` 5 条采样）。

### 表 1：BERT 不同池化方式消融实验

| 池化方式 | 准确率 (Acc) | 宏平均 F1 | 模型大小 | 单 epoch 耗时 | 排名 |
|----------|-------------|-----------|----------|---------------|------|
| **Mean 池化** | **0.5415** | **0.5302** | 102.3M | ~46 min | **1（最优）** |
| CLS 池化 | 0.5215 | 0.5108 | 102.3M | ~54 min | 2 |
| Max 池化 | 0.5170 | 0.4828 | 102.3M | ~45 min | 3 |

### 表 2：三种分类方法整体对比

| 训练方法 | 准确率 (Acc) | 训练成本 | 推理速度 | 资源占用 | 适用场景 |
|----------|-------------|----------|----------|----------|----------|
| **BERT 微调（Mean 池化）** | **0.5415** | 极低 | 极快 | 低 | 小样本、轻量化部署 |
| LLM 零样本 | 0.4000 | 无 | 慢 | 中 | 无标注数据、快速验证 |
| LLM LoRA 指令微调 | 0.4000 | 低 | 中 | 中 | 小样本指令适配 |

**补充指标：**

| 指标 | 数值 |
|------|------|
| BERT 批量推理准确率（500 条 val） | 0.5020 |
| LLM 零样本无法解析率（5 条 demo） | 40.0% |
| LLM LoRA 可训练参数占比 | 0.1093% |
| LLM LoRA 快速训练 val_loss | 1.4694 |

---

## 八、结果分析

### 8.1 BERT 消融实验

- **Mean 池化最优**：平均池化聚合全句 token 语义，特征更稳定，在 15 分类任务上 Acc / F1 均最高。
- **CLS 池化次之**：仅依赖 `[CLS]` 向量，丢失部分序列细节，效果略低于 mean。
- **Max 池化最差**：只保留局部最强响应，易受噪声干扰，macro F1 明显偏低（0.4828）。
- 三种方式 **模型大小、训练流程一致**，差异 purely 来自池化策略。

### 8.2 三种范式对比

- **BERT 微调**：在 CPU 快速模式下仍显著优于 LLM 方案，适合作为生产首选。
- **LLM 零样本**：零训练成本，但 0.5B 小模型理解有限，且常输出非标准类名（如「股市下跌」而非「财经」），导致解析失败。
- **LLM LoRA**：解决了输出无法解析问题（unparseable 0%），但受模型规模与小样本训练限制，demo 准确率未超过 BERT。

---

## 九、实验结论

1. **BERT 池化消融**：**Mean（平均）池化** 为本实验最优选择，优于 CLS 与 Max。
2. **方法选型建议**：
   - 追求 **速度、低成本、高精度** → **BERT 微调（Mean 池化）**
   - **无训练数据**、仅需快速验证 → **LLM 零样本**
   - 需要 **指令化输出**、轻量适配 → **LLM LoRA 指令微调**
3. 在小样本、CPU 轻量场景下，**专用微调的 BERT 明显优于小参数量大模型**。

---

## 十、输出文件说明

| 类型 | 路径 |
|------|------|
| BERT 最优 checkpoint | `outputs/checkpoints/best_cls.pt` / `best_mean.pt` / `best_max.pt` |
| BERT 训练日志 | `outputs/train_log_cls.json` / `train_log_mean.json` / `train_log_max.json` |
| LLM LoRA adapter | `outputs/sft_adapter/` |
| LLM 零样本结果 | `outputs/llm_zero_shot_results.json` |
| LLM SFT 结果 | `outputs/llm_sft_results.json` |
| LLM 训练日志 | `outputs/train_log_sft.json` |

---

## 十一、常见问题

**Q：`classify_llm.py` 报 `chat_template is not set`？**  
A：运行时会自动使用内置 Qwen2 模板，或生成 `tokenizer_config.json`；也可执行 `python download_tokenizer_config.py`。

**Q：CPU 训练太慢？**  
A：BERT 使用 `train.py --fast`；LLM 使用 `train_sft.py --demo` 或 `--fast`。

**Q：如何复现完整精度？**  
A：去掉 `--fast`，增大 `epochs`、`max_length`，使用 GPU 全量 fine-tune BERT 或增大 LLM 训练数据量。

---

## 十二、许可证与致谢

- 数据集：[CLUE / TNEWS](https://github.com/CLUEbenchmark/CLUE)  
- 预训练模型：[bert-base-chinese](https://huggingface.co/bert-base-chinese)、[Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
