import math

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_length: int, embed_dim: int):
        super().__init__()
        self.max_length = int(max_length)
        self.embedding = nn.Parameter(torch.zeros(1, self.max_length, embed_dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        if length > self.max_length:
            raise ValueError(f"Sequence length {length} exceeds max_length={self.max_length}.")
        return x + self.embedding[:, :length, :]


class AttentivePooling(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.score = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(x), dim=1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled, weights


class FeedForwardModule(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.pointwise_in = nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=embed_dim,
        )
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2)
        y = self.pointwise_in(y)
        y = self.glu(y)
        y = self.depthwise(y)
        y = self.batch_norm(y)
        y = self.activation(y)
        y = self.pointwise_out(y)
        y = self.dropout(y)
        return y.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, conv_kernel_size: int, dropout: float):
        super().__init__()
        self.ffn1 = FeedForwardModule(embed_dim, ffn_dim, dropout)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConvModule(embed_dim, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(embed_dim, ffn_dim, dropout)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        y = self.self_attn_norm(x)
        y, _ = self.self_attn(y, y, y, need_weights=False)
        x = x + self.self_attn_dropout(y)
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


def conv2d_token_count(size: int, kernel: int, stride: int) -> int:
    return math.floor((size - kernel) / stride) + 1
