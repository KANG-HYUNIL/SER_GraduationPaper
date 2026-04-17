import math

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.transformer_blocks import AttentivePooling, conv2d_token_count
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
        self.patch_size = tuple(patch_size)
        self.patch_stride = tuple(patch_stride)
        self.patch_embed = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
            bias=False,
        )

        input_height = int(cfg.data.n_mels) if not bool(cfg.data.get("resize_enabled", True)) else int(cfg.data.resize_height)
        token_h = conv2d_token_count(input_height, patch_size[0], patch_stride[0])
        if token_h <= 0:
            raise ValueError("Patch configuration produces no tokens. Check patch_size and patch_stride.")
        self.token_h = token_h

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if pooling == "cls" else None
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

    def _build_positional_encoding(self, length: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / self.embed_dim)
        )
        pos = torch.zeros(1, length, self.embed_dim, device=device)
        pos[:, :, 0::2] = torch.sin(position * div_term)
        pos[:, :, 1::2] = torch.cos(position * div_term)
        return pos

    def _token_mask(self, token_h: int, token_w: int, lengths: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
        if lengths is None:
            return None
        valid_w = torch.clamp((lengths - self.patch_size[1]) // self.patch_stride[1] + 1, min=0, max=token_w)
        width_idx = torch.arange(token_w, device=device).unsqueeze(0)
        mask_2d = width_idx.unsqueeze(1) >= valid_w.view(-1, 1, 1)
        return mask_2d.expand(-1, token_h, -1).reshape(lengths.size(0), token_h * token_w)

    def _encode_tokens(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.patch_embed(x)
        token_h, token_w = x.shape[2], x.shape[3]
        token_mask = self._token_mask(token_h, token_w, lengths, x.device)
        x = x.flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
            if token_mask is not None:
                cls_mask = torch.zeros(token_mask.size(0), 1, dtype=torch.bool, device=x.device)
                token_mask = torch.cat([cls_mask, token_mask], dim=1)
        x = x + self._build_positional_encoding(x.size(1), x.device)
        x = self.encoder(x, src_key_padding_mask=token_mask)
        if token_mask is not None:
            x = x.masked_fill(token_mask.unsqueeze(-1), 0.0)
        return self.norm(x), token_mask

    def get_embedding(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        tokens, token_mask = self._encode_tokens(x, lengths)
        if self.pooling_type == "cls":
            return tokens[:, 0]
        if self.pooling_type == "mean":
            if token_mask is None:
                return tokens.mean(dim=1)
            valid = (~token_mask).unsqueeze(-1)
            return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        pooled, _ = self.attentive_pool(tokens, token_mask)
        return pooled

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        embedding = self.get_embedding(x, lengths)
        return self.classifier(self.dropout(embedding))
