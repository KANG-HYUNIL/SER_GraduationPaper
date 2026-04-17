import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.hierarchical_window_blocks import (
    PatchMerging2D,
    SpatialProjector,
    WindowTransformerBlock2D,
    apply_spatial_mask_2d,
    lengths_to_2d_valid_mask,
)
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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.conv1(x)
        lengths = downsample_lengths_1d(lengths, kernel_size=3, stride=self.conv1.stride[1], padding=1)
        x = apply_channel_mask_2d(x, lengths)
        x = self.norm1(x)
        x = self.act1(x)
        x = apply_channel_mask_2d(x, lengths)

        x = self.conv2(x)
        lengths = downsample_lengths_1d(lengths, kernel_size=3, stride=self.conv2.stride[1], padding=1)
        x = apply_channel_mask_2d(x, lengths)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = apply_channel_mask_2d(x, lengths)
        return x, lengths


@register_model("hierarchical_window_transformer")
class HierarchicalWindowTransformerSER(nn.Module):
    # Current codebase gap note:
    # - Previous version collapsed frequency via AdaptiveAvgPool2d and ran 1D time windows only.
    # - Speech Swin-Transformer keeps a 2D time-frequency grid and uses shifted windows + patch merging.
    # - DWFormer emphasizes window design that preserves informative regions before late aggregation.
    # This refactor therefore preserves 2D feature maps through both stages and only collapses frequency at the final pooling step.
    def __init__(self, cfg: DictConfig):
        super().__init__()

        num_classes = 8
        stem_channels = list(cfg.model.get("stem_channels", [32, 64]))
        stage_dims = list(cfg.model.get("stage_dims", [128, 192]))
        stage_depths = list(cfg.model.get("stage_depths", [2, 2]))
        num_heads = list(cfg.model.get("num_heads", [4, 8]))
        window_sizes = list(cfg.model.get("window_sizes", [8, 8]))
        ffn_ratio = float(cfg.model.get("ffn_ratio", 2.0))
        dropout = float(cfg.model.get("dropout", 0.2))
        pooling = str(cfg.model.get("pooling", "attention"))
        use_shift = bool(cfg.model.get("use_shifted_windows", True))

        if not (len(stage_dims) == len(stage_depths) == len(num_heads) == len(window_sizes) == 2):
            raise ValueError("hierarchical_window_transformer expects exactly 2 stages.")
        if any(dim % heads != 0 for dim, heads in zip(stage_dims, num_heads)):
            raise ValueError("Each stage dim must be divisible by its num_heads.")

        stem_layers = []
        in_channels = 1
        remaining_freq = int(cfg.data.get("n_mels", 80))
        for out_channels in stem_channels:
            stride = (2, 2)
            stem_layers.append(ConvStemBlock(in_channels, out_channels, stride=stride, dropout=dropout))
            remaining_freq = downsample_size_2d(remaining_freq, kernel_size=3, stride=stride[0], padding=1)
            in_channels = out_channels
        self.features = nn.ModuleList(stem_layers)
        self.remaining_freq = remaining_freq

        self.stage0_projector = SpatialProjector(stem_channels[-1], stage_dims[0], dropout=dropout)
        self.pos_dropout = nn.Dropout2d(dropout)

        stage1_blocks = []
        for block_idx in range(stage_depths[0]):
            win = (int(window_sizes[0]), int(window_sizes[0]))
            shift = (win[0] // 2, win[1] // 2) if use_shift and block_idx % 2 == 1 else (0, 0)
            stage1_blocks.append(
                WindowTransformerBlock2D(
                    embed_dim=stage_dims[0],
                    num_heads=num_heads[0],
                    ffn_dim=int(stage_dims[0] * ffn_ratio),
                    dropout=dropout,
                    window_size=win,
                    shift_size=shift,
                )
            )
        self.stage1 = nn.ModuleList(stage1_blocks)
        self.downsample = PatchMerging2D(stage_dims[0], stage_dims[1])

        stage2_blocks = []
        for block_idx in range(stage_depths[1]):
            win = (int(window_sizes[1]), int(window_sizes[1]))
            shift = (win[0] // 2, win[1] // 2) if use_shift and block_idx % 2 == 1 else (0, 0)
            stage2_blocks.append(
                WindowTransformerBlock2D(
                    embed_dim=stage_dims[1],
                    num_heads=num_heads[1],
                    ffn_dim=int(stage_dims[1] * ffn_ratio),
                    dropout=dropout,
                    window_size=win,
                    shift_size=shift,
                )
            )
        self.stage2 = nn.ModuleList(stage2_blocks)
        self.norm = nn.LayerNorm(stage_dims[-1])
        self.pooling_type = pooling
        self.attentive_pool = AttentivePooling(stage_dims[-1]) if pooling == "attention" else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(stage_dims[-1], num_classes)

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

        x = self.stage0_projector(x, lengths=lengths)
        x = self.pos_dropout(x)
        spatial_valid_mask = lengths_to_2d_valid_mask(lengths, x.size(2), x.size(3), x.device)
        x = apply_spatial_mask_2d(x, spatial_valid_mask)
        self._cache_visual("frequency_feature_map", x[0].mean(dim=0))

        for block in self.stage1:
            x = block(x, valid_mask=spatial_valid_mask)

        x, spatial_valid_mask, lengths = self.downsample(x, valid_mask=spatial_valid_mask, lengths=lengths)
        x = apply_spatial_mask_2d(x, spatial_valid_mask)
        for block in self.stage2:
            x = block(x, valid_mask=spatial_valid_mask)

        x = apply_spatial_mask_2d(x, spatial_valid_mask)
        x = x.mean(dim=2).transpose(1, 2)
        key_padding_mask = lengths_to_padding_mask(lengths, x.size(1))
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
