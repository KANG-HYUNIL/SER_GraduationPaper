import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.cnn_conformer_blocks import CNNConformerBlock
from src.models.transformer_blocks import (
    AttentivePooling,
    ChannelLayerNorm2d,
    apply_channel_mask_2d,
    apply_sequence_mask,
    downsample_lengths_1d,
    lengths_to_padding_mask,
)
from src.utils.registry import register_model


def downsample_size_2d(size: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1) -> int:
    return ((size + (2 * padding) - dilation * (kernel_size - 1) - 1) // stride) + 1


class ConvStemBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int], dropout: float):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = ChannelLayerNorm2d(out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = ChannelLayerNorm2d(out_channels)
        self.act2 = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def _apply_mask(self, x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        return apply_channel_mask_2d(x, lengths)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.conv1(x)
        lengths = downsample_lengths_1d(lengths, kernel_size=3, stride=self.conv1.stride[1], padding=1)
        x = self._apply_mask(x, lengths)
        x = self.norm1(x)
        x = self.act1(x)
        x = self._apply_mask(x, lengths)

        x = self.conv2(x)
        lengths = downsample_lengths_1d(lengths, kernel_size=3, stride=self.conv2.stride[1], padding=1)
        x = self._apply_mask(x, lengths)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self._apply_mask(x, lengths)
        return x, lengths


class FlattenFrequencyProjector(nn.Module):
    def __init__(self, in_channels: int, remaining_freq: int, embed_dim: int, dropout: float):
        super().__init__()
        self.remaining_freq = int(remaining_freq)
        input_dim = int(in_channels) * self.remaining_freq
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, freq_bins, time_steps = x.shape
        if freq_bins != self.remaining_freq:
            raise ValueError(
                f"FlattenFrequencyProjector expected freq_bins={self.remaining_freq}, got {freq_bins}. "
                "Check n_mels and CNN stem stride settings."
            )
        flattened_freq = x
        x = x.permute(0, 3, 1, 2).reshape(batch_size, time_steps, channels * freq_bins)
        x = self.proj(self.norm(x))
        return self.dropout(x), flattened_freq


class LayerWeightedSum(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layer_outputs: list[torch.Tensor]) -> torch.Tensor:
        if len(layer_outputs) != self.logits.numel():
            raise ValueError(
                f"LayerWeightedSum expected {self.logits.numel()} layer outputs, got {len(layer_outputs)}."
            )
        weights = torch.softmax(self.logits, dim=0)
        stacked = torch.stack(layer_outputs, dim=0)
        return torch.sum(stacked * weights.view(-1, 1, 1, 1), dim=0)


@register_model("cnn_conformer")
class CNNConformerSER(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        num_classes = 8
        stem_channels = list(cfg.model.get("stem_channels", [32, 64]))
        embed_dim = int(cfg.model.get("embed_dim", 192))
        num_heads = int(cfg.model.get("num_heads", 4))
        num_layers = int(cfg.model.get("num_layers", 8))
        ffn_dim = int(cfg.model.get("ffn_dim", embed_dim * 4))
        conv_kernel_size = int(cfg.model.get("conv_kernel_size", 31))
        dropout = float(cfg.model.get("dropout", 0.1))
        stem_dropout = float(cfg.model.get("stem_dropout", dropout))
        projector_dropout = float(cfg.model.get("projector_dropout", dropout))
        input_dropout = float(cfg.model.get("input_dropout", dropout))
        encoder_dropout = float(cfg.model.get("encoder_dropout", dropout))
        classifier_dropout = float(cfg.model.get("classifier_dropout", dropout))
        pooling = str(cfg.model.get("pooling", "attention"))
        attention_type = str(cfg.model.get("attention_type", "relative"))
        max_relative_position = int(cfg.model.get("max_relative_position", 128))
        stem_strides = [[int(v) for v in pair] for pair in cfg.model.get("stem_strides", [[2, 2], [2, 2]])]
        layer_fusion = str(cfg.model.get("layer_fusion", "last"))
        conv_module_type = str(cfg.model.get("conv_module_type", "single"))
        multiscale_kernel_sizes = [int(v) for v in cfg.model.get("multiscale_kernel_sizes", [15, 31])]
        n_mels = int(cfg.data.get("n_mels", 80))
        if len(stem_strides) != len(stem_channels):
            raise ValueError("model.stem_strides must match the number of stem_channels stages.")
        if layer_fusion not in {"last", "learned_sum", "last2_mean"}:
            raise ValueError(f"Unsupported layer_fusion: {layer_fusion}")

        layers = []
        in_channels = 1
        remaining_freq = n_mels
        for out_channels, raw_stride in zip(stem_channels, stem_strides):
            stride = (int(raw_stride[0]), int(raw_stride[1]))
            layers.append(ConvStemBlock(in_channels, out_channels, stride=stride, dropout=stem_dropout))
            remaining_freq = downsample_size_2d(remaining_freq, kernel_size=3, stride=stride[0], padding=1)
            in_channels = out_channels
        if remaining_freq <= 0:
            raise ValueError(f"Invalid remaining frequency dimension: {remaining_freq} for n_mels={n_mels}.")

        self.features = nn.ModuleList(layers)
        self.remaining_freq = remaining_freq
        self.frequency_projector = FlattenFrequencyProjector(
            stem_channels[-1],
            remaining_freq=self.remaining_freq,
            embed_dim=embed_dim,
            dropout=projector_dropout,
        )
        self.pos_dropout = nn.Dropout(input_dropout)
        self.encoder = nn.ModuleList(
            [
                CNNConformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    conv_kernel_size=conv_kernel_size,
                    dropout=encoder_dropout,
                    attention_type=attention_type,
                    max_relative_position=max_relative_position,
                    conv_module_type=conv_module_type,
                    multiscale_kernel_sizes=multiscale_kernel_sizes,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_fusion = layer_fusion
        self.layer_fuser = LayerWeightedSum(num_layers) if layer_fusion == "learned_sum" else None
        self.norm = nn.LayerNorm(embed_dim)
        self.pooling_type = pooling
        self.attentive_pool = AttentivePooling(embed_dim) if pooling == "attention" else None
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.capture_visualizations = False
        self._visual_cache: dict[str, torch.Tensor] = {}

    def enable_visualization_capture(self, enabled: bool) -> None:
        self.capture_visualizations = bool(enabled)
        if not enabled:
            self._visual_cache = {}

    def _cache_visual(self, key: str, tensor: torch.Tensor) -> None:
        if self.capture_visualizations:
            self._visual_cache[key] = tensor.detach().cpu()

    def get_visualization_payload(self) -> dict[str, torch.Tensor] | None:
        if not self._visual_cache:
            return None
        return dict(self._visual_cache)

    def _encode_sequence(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if lengths is not None:
            lengths = lengths.to(x.device)
        self._cache_visual("spectrogram", x[0, 0])

        for block in self.features:
            x, lengths = block(x, lengths)
        self._cache_visual("cnn_feature_map", x[0].mean(dim=0))

        x, flattened_feature_map = self.frequency_projector(x)
        self._cache_visual("frequency_feature_map", flattened_feature_map[0].mean(dim=0))

        key_padding_mask = lengths_to_padding_mask(lengths, x.size(1))
        x = apply_sequence_mask(self.pos_dropout(x), key_padding_mask)

        layer_outputs: list[torch.Tensor] = []
        for block in self.encoder:
            x = block(x, key_padding_mask=key_padding_mask)
            x = apply_sequence_mask(x, key_padding_mask)
            layer_outputs.append(x)

        if self.layer_fuser is not None:
            x = self.layer_fuser(layer_outputs)
        elif self.layer_fusion == "last2_mean":
            if len(layer_outputs) >= 2:
                x = 0.5 * (layer_outputs[-1] + layer_outputs[-2])
            elif layer_outputs:
                x = layer_outputs[-1]
        elif layer_outputs:
            x = layer_outputs[-1]

        x = self.norm(apply_sequence_mask(x, key_padding_mask))
        return apply_sequence_mask(x, key_padding_mask), key_padding_mask

    def get_embedding(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        seq, key_padding_mask = self._encode_sequence(x, lengths)
        if self.pooling_type == "mean":
            if key_padding_mask is None:
                pooled = seq.mean(dim=1)
                attention_weights = seq.norm(dim=-1)
            else:
                valid = (~key_padding_mask).unsqueeze(-1)
                pooled = (seq * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
                attention_weights = (~key_padding_mask).float()
            self._cache_visual("attention_weights", attention_weights[0])
            return pooled

        pooled, weights = self.attentive_pool(seq, key_padding_mask)
        self._cache_visual("attention_weights", weights[0, :, 0])
        return pooled

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        embedding = self.get_embedding(x, lengths)
        return self.classifier(self.dropout(embedding))
