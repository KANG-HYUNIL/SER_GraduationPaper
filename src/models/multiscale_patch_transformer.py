import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.transformer_blocks import AttentivePooling, LearnedPositionalEncoding, conv2d_token_count
from src.utils.registry import register_model


class PatchBranch(nn.Module):
    def __init__(
        self,
        input_height: int,
        input_width: int,
        embed_dim: int,
        patch_size: list[int],
        patch_stride: list[int],
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=tuple(patch_size),
            stride=tuple(patch_stride),
            bias=False,
        )
        token_h = conv2d_token_count(input_height, patch_size[0], patch_stride[0])
        token_w = conv2d_token_count(input_width, patch_size[1], patch_stride[1])
        if token_h <= 0 or token_w <= 0:
            raise ValueError("Patch configuration produces no tokens. Check patch_size and patch_stride.")
        self.position = LearnedPositionalEncoding(token_h * token_w, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = self.position(x)
        x = self.encoder(x)
        return self.norm(x)


@register_model("multiscale_patch_transformer")
class MultiScalePatchTransformerSER(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        num_classes = 8
        embed_dim = int(cfg.model.get("embed_dim", 192))
        num_heads = int(cfg.model.get("num_heads", 4))
        num_layers = int(cfg.model.get("num_layers", 3))
        ffn_dim = int(cfg.model.get("ffn_dim", embed_dim * 4))
        dropout = float(cfg.model.get("dropout", 0.2))
        fine_patch = list(cfg.model.get("fine_patch_size", [8, 8]))
        fine_stride = list(cfg.model.get("fine_patch_stride", [8, 8]))
        coarse_patch = list(cfg.model.get("coarse_patch_size", [16, 16]))
        coarse_stride = list(cfg.model.get("coarse_patch_stride", [16, 16]))
        pooling = str(cfg.model.get("pooling", "attention"))

        input_height = int(cfg.data.resize_height)
        input_width = int(cfg.data.resize_width)
        self.fine_branch = PatchBranch(
            input_height,
            input_width,
            embed_dim,
            fine_patch,
            fine_stride,
            num_layers,
            num_heads,
            ffn_dim,
            dropout,
        )
        self.coarse_branch = PatchBranch(
            input_height,
            input_width,
            embed_dim,
            coarse_patch,
            coarse_stride,
            num_layers,
            num_heads,
            ffn_dim,
            dropout,
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.pooling_type = pooling
        self.fine_pool = AttentivePooling(embed_dim) if pooling == "attention" else None
        self.coarse_pool = AttentivePooling(embed_dim) if pooling == "attention" else None
        self.output_norm = nn.LayerNorm(embed_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def _pool(self, tokens: torch.Tensor, pooler: AttentivePooling | None) -> torch.Tensor:
        if self.pooling_type == "mean":
            return tokens.mean(dim=1)
        pooled, _ = pooler(tokens)
        return pooled

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        fine = self.fine_branch(x)
        coarse = self.coarse_branch(x)
        cross, _ = self.cross_attention(fine, coarse, coarse, need_weights=False)
        gate = self.fusion_gate(torch.cat([fine, cross], dim=-1))
        fused_fine = gate * fine + (1.0 - gate) * cross

        fine_embedding = self._pool(fused_fine, self.fine_pool)
        coarse_embedding = self._pool(coarse, self.coarse_pool)
        return self.output_norm(torch.cat([fine_embedding, coarse_embedding], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.get_embedding(x)
        return self.classifier(self.dropout(embedding))
