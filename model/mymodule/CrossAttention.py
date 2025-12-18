import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable

from model.mymodule.CRAttention import CRA
import clip


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h   # d_k每个注意力头处理的通道数
        self.h = h  # 注意力头的数量
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        # self.alpha = nn.Parameter(torch.tensor(0.7))

    def forward(self, query, key1, key2, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        print(query.size())

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key1, key2 = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key1, key2))]
        value = value.repeat(self.h, 1, 1).transpose(0, 1).contiguous().unsqueeze(-1)
        # 2) Apply attention on all the projected vectors in batch.
        d_k = query.size(-1)
        scores1 = torch.matmul(query, key1.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores2 = torch.matmul(query, key2.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        # scores = torch.nn.functional.relu(scores1 - scores2)  # 负数自动归零
        scores = scores1 * 0.35 + scores2 * 0.65  # 全部计算相似度和前景计算相似度相加得到更具有前景信息的相似度
        # scores = scores2
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value)
        # x, self.attn = attention(query, key, value, mask=mask,
        #                          dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        return torch.mean(x, -3)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]#format with class
            texts = clip.tokenize(texts).to(device) # tokenize
            class_embeddings = model.encode_text(texts) # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights.t()
if __name__ == "__main__":
    query = torch.randn(12, 576, 512)
    key1 = torch.randn(12, 5*576, 512)
    key2 = torch.randn(12, 5*576, 512)
    value = torch.randn(12, 5*576)
    m = MultiHeadedAttention(h=8, d_model=512, dropout=0.5)
    output = m(query, key1, key2, value)
    print('input_size:', value.size())
    print('output_size:', output.size())
