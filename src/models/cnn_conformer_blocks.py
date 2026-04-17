import math

import torch
import torch.nn as nn

from src.models.transformer_blocks import ConvModule, FeedForwardModule, apply_sequence_mask


class RelativePositionMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, max_relative_position: int = 128):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}.")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.max_relative_position = int(max_relative_position)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        table_size = 2 * self.max_relative_position + 1
        self.relative_bias = nn.Embedding(table_size, self.num_heads)

    def _relative_position_index(self, length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(length, device=device)
        relative = positions[:, None] - positions[None, :]
        relative = relative.clamp(-self.max_relative_position, self.max_relative_position)
        return relative + self.max_relative_position

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        rel_index = self._relative_position_index(seq_len, x.device)
        rel_bias = self.relative_bias(rel_index).permute(2, 0, 1)
        attn_scores = attn_scores + rel_bias.unsqueeze(0)

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                torch.finfo(attn_scores.dtype).min,
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            return attn_output, attn_weights.mean(dim=1)
        return attn_output, None


class CNNConformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        conv_kernel_size: int,
        dropout: float,
        attention_type: str = "relative",
        max_relative_position: int = 128,
    ):
        super().__init__()
        self.attention_type = str(attention_type)
        self.ffn1 = FeedForwardModule(embed_dim, ffn_dim, dropout)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        if self.attention_type == "relative":
            self.self_attn = RelativePositionMultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                max_relative_position=max_relative_position,
            )
        elif self.attention_type == "absolute":
            self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        else:
            raise ValueError(f"Unsupported cnn_conformer attention_type: {self.attention_type}")

        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConvModule(embed_dim, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(embed_dim, ffn_dim, dropout)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        y = self.self_attn_norm(x)
        if self.attention_type == "relative":
            y, _ = self.self_attn(y, key_padding_mask=key_padding_mask, need_weights=False)
        else:
            y, _ = self.self_attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.self_attn_dropout(y)
        x = x + self.conv_module(x, key_padding_mask=key_padding_mask)
        x = x + 0.5 * self.ffn2(x)
        x = apply_sequence_mask(x, key_padding_mask)
        x = self.final_norm(x)
        return apply_sequence_mask(x, key_padding_mask)
