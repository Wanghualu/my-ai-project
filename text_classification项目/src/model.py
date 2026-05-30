"""
BERT 文本分类模型：BertModel + 自定义分类头

教学重点：
  1. 为什么不用 BertForSequenceClassification？
     — 手写分类头让结构一目了然，且方便替换不同的向量提取策略
  2. 三种向量提取策略对比：
     — cls     : 取 [CLS] 位置输出，BERT 原始论文的分类方式
     — mean    : 对所有真实 token（attention_mask=1）求均值，实践中常更鲁棒
     — max     : 对所有真实 token 取 element-wise max，保留最显著特征
  3. Dropout 在 fine-tuning 中的作用：防止过拟合，通常 0.1~0.3

使用方式：
  from model import BertClassifier

  # CLS 策略（默认）
  model = BertClassifier(bert_path, num_labels=15, pool="cls")

  # 均值池化
  model = BertClassifier(bert_path, num_labels=15, pool="mean")

  # 最大值池化
  model = BertClassifier(bert_path, num_labels=15, pool="max")
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers import BertModel
import transformers


POOL_OPTIONS = ("cls", "mean", "max")


class BertClassifier(nn.Module):
    """
    结构：BertModel → 池化策略 → Dropout → Linear → logits

    参数：
      bert_path  : 预训练权重路径（本地文件夹或 HuggingFace 模型名）
      num_labels : 分类类别数
      pool       : 向量提取策略，可选 'cls' / 'mean' / 'max'
      dropout    : 分类头前的 dropout 比例
    """

    def __init__(
        self,
        bert_path: str,
        num_labels: int,
        pool: str = "cls",
        dropout: float = 0.1,
    ):
        super().__init__()
        assert pool in POOL_OPTIONS, f"pool 必须是 {POOL_OPTIONS} 之一，收到: {pool}"

        self.pool = pool
        self._bert_frozen = False
        # transformers 5.x 加载时会打印 LOAD REPORT（含 UNEXPECTED keys），
        # bert-base-chinese 权重含预训练头（cls.*）但 BertModel 不需要，属正常现象，降低日志级别屏蔽
        _prev_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path)
        transformers.logging.set_verbosity(_prev_verbosity)
        hidden_size = self.bert.config.hidden_size  # bert-base = 768

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def freeze_bert(self) -> "BertClassifier":
        """
        冻结 BERT 参数；forward 时对 BERT 使用 torch.no_grad()。
        仅训练分类头时，CPU 上可明显加速（避免为 BERT 构建无用计算图）。
        """
        for param in self.bert.parameters():
            param.requires_grad = False
        self._bert_frozen = True
        self.bert.eval()
        return self

    def train(self, mode: bool = True):
        """冻结 BERT 时保持其 eval 模式，避免 Dropout 干扰表征。"""
        super().train(mode)
        if self._bert_frozen:
            self.bert.eval()
        return self

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """BERT 编码，返回 last_hidden_state [B, L, H]。"""
        ctx = torch.no_grad() if self._bert_frozen else nullcontext()
        with ctx:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
        return outputs.last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
        token_type_ids: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        """
        返回 logits: [B, num_labels]，未经 softmax（交叉熵 loss 内部做）
        """
        last_hidden = self._encode(input_ids, attention_mask, token_type_ids)

        vec = self._pool(last_hidden, attention_mask)  # [B, H]
        vec = self.dropout(vec)
        logits = self.classifier(vec)                  # [B, num_labels]
        return logits

    def _pool(
        self,
        last_hidden: torch.Tensor,    # [B, L, H]
        attention_mask: torch.Tensor, # [B, L]
    ) -> torch.Tensor:                # [B, H]
        if self.pool == "cls":
            # [CLS] 在位置 0，直接切片
            return last_hidden[:, 0, :]

        # 构造 mask：[B, L, 1]，0 位置（padding）对应 token 不参与计算
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]

        if self.pool == "mean":
            # 有效 token 的均值，避免 padding 影响均值
            sum_hidden = (last_hidden * mask).sum(dim=1)      # [B, H]
            count      = mask.sum(dim=1).clamp(min=1e-9)      # [B, 1]
            return sum_hidden / count

        if self.pool == "max":
            # 把 padding 位置设为 -inf，再取 max 就不会被选中
            masked = last_hidden + (1 - mask) * (-1e9)
            return masked.max(dim=1).values                   # [B, H]

        raise ValueError(f"未知池化策略: {self.pool}")


def build_model(
    bert_path: str,
    num_labels: int,
    pool: str = "cls",
    dropout: float = 0.1,
) -> BertClassifier:
    """工厂函数，统一构建入口，便于 train.py 调用。"""
    model = BertClassifier(
        bert_path, num_labels=num_labels, pool=pool, dropout=dropout
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_bert   = sum(p.numel() for p in model.bert.parameters()) / 1e6
    n_head   = sum(p.numel() for p in model.classifier.parameters()) / 1e3
    print(f"模型参数量: {n_params:.1f}M  "
          f"(BERT: {n_bert:.1f}M, 分类头: {n_head:.1f}K)")
    print(f"池化策略: {pool}")
    return model
