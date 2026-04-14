import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.transformer_blocks import AttentivePooling, ConformerBlock
from src.utils.registry import register_model


class ConvStemBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int], dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@register_model("cnn_conformer")
class CNNConformerSER(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        num_classes = 8
        stem_channels = list(cfg.model.get("stem_channels", [32, 64]))
        embed_dim = int(cfg.model.get("embed_dim", 192))
        num_heads = int(cfg.model.get("num_heads", 4))
        num_layers = int(cfg.model.get("num_layers", 4))
        ffn_dim = int(cfg.model.get("ffn_dim", embed_dim * 4))
        conv_kernel_size = int(cfg.model.get("conv_kernel_size", 15))
        dropout = float(cfg.model.get("dropout", 0.2))
        pooling = str(cfg.model.get("pooling", "attention"))

        layers = []
        in_channels = 1
        for idx, out_channels in enumerate(stem_channels):
            stride = (2, 2) if idx == 0 else (2, 1)
            layers.append(ConvStemBlock(in_channels, out_channels, stride=stride, dropout=dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.project = nn.Linear(stem_channels[-1], embed_dim)
        self.pos_dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [
                ConformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pooling_type = pooling
        self.attentive_pool = AttentivePooling(embed_dim) if pooling == "attention" else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.freq_pool(x).squeeze(2).transpose(1, 2)
        x = self.pos_dropout(self.project(x))
        for block in self.encoder:
            x = block(x)
        return self.norm(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        seq = self._encode_sequence(x)
        if self.pooling_type == "mean":
            return seq.mean(dim=1)
        pooled, _ = self.attentive_pool(seq)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.get_embedding(x)
        return self.classifier(self.dropout(embedding))
