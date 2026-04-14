import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.transformer_blocks import AttentivePooling, LearnedPositionalEncoding, conv2d_token_count
from src.utils.registry import register_model


@register_model("pure_transformer")
class PureTransformerSER(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        num_classes = 8
        embed_dim = int(cfg.model.get("embed_dim", 192))
        num_heads = int(cfg.model.get("num_heads", 4))
        num_layers = int(cfg.model.get("num_layers", 4))
        ffn_dim = int(cfg.model.get("ffn_dim", embed_dim * 4))
        dropout = float(cfg.model.get("dropout", 0.2))
        patch_size = list(cfg.model.get("patch_size", [16, 16]))
        patch_stride = list(cfg.model.get("patch_stride", patch_size))
        pooling = str(cfg.model.get("pooling", "attention"))

        self.pooling_type = pooling
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=tuple(patch_size),
            stride=tuple(patch_stride),
            bias=False,
        )

        token_h = conv2d_token_count(int(cfg.data.resize_height), patch_size[0], patch_stride[0])
        token_w = conv2d_token_count(int(cfg.data.resize_width), patch_size[1], patch_stride[1])
        if token_h <= 0 or token_w <= 0:
            raise ValueError("Patch configuration produces no tokens. Check patch_size and patch_stride.")
        self.num_tokens = token_h * token_w

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if pooling == "cls" else None
        pos_tokens = self.num_tokens + (1 if self.cls_token is not None else 0)
        self.position = LearnedPositionalEncoding(pos_tokens, embed_dim)
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
        self.attentive_pool = AttentivePooling(embed_dim) if pooling == "attention" else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = self.position(x)
        x = self.encoder(x)
        return self.norm(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._encode_tokens(x)
        if self.pooling_type == "cls":
            pooled = tokens[:, 0]
        elif self.pooling_type == "mean":
            pooled = tokens.mean(dim=1)
        else:
            pooled, _ = self.attentive_pool(tokens)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.get_embedding(x)
        return self.classifier(self.dropout(embedding))
