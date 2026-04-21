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


def _downsample_time_lengths(
    lengths: torch.Tensor | None, kernel_size: int, stride: int, padding: int = 0, dilation: int = 1
) -> torch.Tensor | None:
    return downsample_lengths_1d(lengths, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)


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


class SequenceProjector(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(int(input_dim))
        self.proj = nn.Linear(int(input_dim), int(embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.norm(x)))


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


class StandardCNNFrontEnd(nn.Module):
    def __init__(self, n_mels: int, stem_channels: list[int], stem_strides: list[list[int]], stem_dropout: float, projector_dropout: float, embed_dim: int):
        super().__init__()
        layers = []
        in_channels = 1
        remaining_freq = int(n_mels)
        for out_channels, raw_stride in zip(stem_channels, stem_strides):
            stride = (int(raw_stride[0]), int(raw_stride[1]))
            layers.append(ConvStemBlock(in_channels, out_channels, stride=stride, dropout=stem_dropout))
            remaining_freq = downsample_size_2d(remaining_freq, kernel_size=3, stride=stride[0], padding=1)
            in_channels = out_channels
        if remaining_freq <= 0:
            raise ValueError(f"Invalid remaining frequency dimension: {remaining_freq} for n_mels={n_mels}.")
        self.features = nn.ModuleList(layers)
        self.remaining_freq = remaining_freq
        self.projector = FlattenFrequencyProjector(
            stem_channels[-1],
            remaining_freq=self.remaining_freq,
            embed_dim=embed_dim,
            dropout=projector_dropout,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        for block in self.features:
            x, lengths = block(x, lengths)
        seq, flattened_feature_map = self.projector(x)
        visuals = {
            "cnn_feature_map": x[0].mean(dim=0),
            "frequency_feature_map": flattened_feature_map[0].mean(dim=0),
        }
        return seq, lengths, visuals


class LightStemFrontEnd(nn.Module):
    def __init__(self, n_mels: int, channels: int, stride: list[int], stem_dropout: float, projector_dropout: float, embed_dim: int):
        super().__init__()
        conv_stride = (int(stride[0]), int(stride[1]))
        self.block = ConvStemBlock(1, int(channels), stride=conv_stride, dropout=stem_dropout)
        remaining_freq = downsample_size_2d(int(n_mels), kernel_size=3, stride=conv_stride[0], padding=1)
        self.remaining_freq = remaining_freq
        self.projector = FlattenFrequencyProjector(
            int(channels),
            remaining_freq=self.remaining_freq,
            embed_dim=embed_dim,
            dropout=projector_dropout,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        x, lengths = self.block(x, lengths)
        seq, flattened_feature_map = self.projector(x)
        visuals = {
            "cnn_feature_map": x[0].mean(dim=0),
            "frequency_feature_map": flattened_feature_map[0].mean(dim=0),
        }
        return seq, lengths, visuals


class NoStemPatchFrontEnd(nn.Module):
    def __init__(self, n_mels: int, time_patch: int, embed_dim: int, projector_dropout: float, norm_variant: str = "layernorm"):
        super().__init__()
        self.time_patch = int(time_patch)
        self.norm_variant = str(norm_variant)
        self.patch_proj = nn.Conv2d(1, int(embed_dim), kernel_size=(int(n_mels), self.time_patch), stride=(int(n_mels), self.time_patch), bias=False)
        if self.norm_variant == "layernorm":
            self.norm = nn.LayerNorm(int(embed_dim))
        elif self.norm_variant == "batchnorm":
            self.norm = nn.BatchNorm1d(int(embed_dim))
        elif self.norm_variant == "instancenorm":
            self.norm = nn.InstanceNorm1d(int(embed_dim), affine=False)
        elif self.norm_variant == "identity":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported nostem_patch norm_variant: {self.norm_variant}")
        self.dropout = nn.Dropout(projector_dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        x = self.patch_proj(x)
        lengths = _downsample_time_lengths(lengths, kernel_size=self.time_patch, stride=self.time_patch, padding=0)
        patch_map = x[0].mean(dim=0)
        seq = x.squeeze(2).transpose(1, 2)
        if self.norm_variant == "layernorm":
            seq = self.norm(seq)
        elif self.norm_variant in {"batchnorm", "instancenorm"}:
            seq = self.norm(seq.transpose(1, 2)).transpose(1, 2)
        else:
            seq = self.norm(seq)
        seq = self.dropout(seq)
        visuals = {
            "cnn_feature_map": patch_map,
            "frequency_feature_map": patch_map,
        }
        return seq, lengths, visuals


class BandTokenFrontEnd(nn.Module):
    def __init__(self, n_mels: int, num_bands: int, embed_dim: int, projector_dropout: float, use_band_embedding: bool = True):
        super().__init__()
        self.num_bands = int(num_bands)
        if self.num_bands < 2:
            raise ValueError("band_token num_bands must be at least 2.")
        band_sizes = [int(n_mels) // self.num_bands] * self.num_bands
        for idx in range(int(n_mels) % self.num_bands):
            band_sizes[idx] += 1
        self.band_sizes = band_sizes
        self.projectors = nn.ModuleList(
            [SequenceProjector(input_dim=band_size, embed_dim=embed_dim, dropout=projector_dropout) for band_size in band_sizes]
        )
        self.band_embedding = nn.Embedding(self.num_bands, int(embed_dim)) if use_band_embedding else None

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        x = x.squeeze(1).transpose(1, 2)  # [B, T, F]
        band_inputs = torch.split(x, self.band_sizes, dim=-1)
        band_tokens = []
        for band_idx, (band_input, projector) in enumerate(zip(band_inputs, self.projectors)):
            token = projector(band_input)
            if self.band_embedding is not None:
                token = token + self.band_embedding.weight[band_idx].view(1, 1, -1)
            band_tokens.append(token)
        seq = torch.stack(band_tokens, dim=2).reshape(x.size(0), x.size(1) * self.num_bands, -1)
        if lengths is not None:
            lengths = lengths * self.num_bands
        band_map = torch.cat([band.mean(dim=-1, keepdim=True).transpose(0, 1) for band in band_inputs], dim=1).transpose(0, 1)
        visuals = {
            "cnn_feature_map": band_map,
            "frequency_feature_map": band_map,
        }
        return seq, lengths, visuals


def shrink_sequence(
    x: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if factor <= 1:
        return x, key_padding_mask
    batch_size, seq_len, dim = x.shape
    pad_len = (-seq_len) % factor
    if pad_len > 0:
        x = torch.cat([x, x.new_zeros(batch_size, pad_len, dim)], dim=1)
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat(
            [key_padding_mask, torch.ones(batch_size, pad_len, dtype=torch.bool, device=x.device)],
            dim=1,
        )
    elif key_padding_mask is None:
        key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)

    valid = (~key_padding_mask).float().unsqueeze(-1)
    new_len = x.size(1) // factor
    x_grouped = x.view(batch_size, new_len, factor, dim)
    valid_grouped = valid.view(batch_size, new_len, factor, 1)
    counts = valid_grouped.sum(dim=2).clamp_min(1.0)
    shrunk = (x_grouped * valid_grouped).sum(dim=2) / counts
    new_mask = valid_grouped.sum(dim=2).squeeze(-1).eq(0)
    return shrunk, new_mask


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
        backbone_variant = str(cfg.model.get("backbone_variant", "standard"))
        stem_strides = [[int(v) for v in pair] for pair in cfg.model.get("stem_strides", [[2, 2], [2, 2]])]
        layer_fusion = str(cfg.model.get("layer_fusion", "last"))
        conv_module_type = str(cfg.model.get("conv_module_type", "single"))
        multiscale_kernel_sizes = [int(v) for v in cfg.model.get("multiscale_kernel_sizes", [15, 31])]
        n_mels = int(cfg.data.get("n_mels", 80))
        sequence_shrinking_cfg = cfg.model.get("sequence_shrinking", {})
        if backbone_variant == "standard" and len(stem_strides) != len(stem_channels):
            raise ValueError("model.stem_strides must match the number of stem_channels stages.")
        if layer_fusion not in {"last", "learned_sum", "last2_mean"}:
            raise ValueError(f"Unsupported layer_fusion: {layer_fusion}")
        self.backbone_variant = backbone_variant
        self.sequence_shrinking_enabled = bool(sequence_shrinking_cfg.get("enabled", False))
        self.sequence_shrinking_factor = int(sequence_shrinking_cfg.get("factor", 2))
        self.sequence_shrinking_layers = {int(v) for v in sequence_shrinking_cfg.get("at_layers", [])}
        layer_dim_schedule = [int(v) for v in cfg.model.get("layer_dim_schedule", [])]
        layer_ffn_schedule = [int(v) for v in cfg.model.get("layer_ffn_schedule", [])]
        if layer_dim_schedule and len(layer_dim_schedule) != num_layers:
            raise ValueError("model.layer_dim_schedule length must match model.num_layers.")
        if layer_ffn_schedule and len(layer_ffn_schedule) != num_layers:
            raise ValueError("model.layer_ffn_schedule length must match model.num_layers.")
        self.layer_dims = layer_dim_schedule if layer_dim_schedule else [embed_dim] * num_layers
        self.layer_ffn_dims = layer_ffn_schedule if layer_ffn_schedule else [ffn_dim] * num_layers
        if any(dim % num_heads != 0 for dim in self.layer_dims):
            raise ValueError("Every layer dimension in model.layer_dim_schedule must be divisible by model.num_heads.")
        if layer_fusion != "last" and len(set(self.layer_dims)) > 1:
            raise ValueError("layer_fusion modes other than 'last' are only supported for uniform layer_dim_schedule.")

        if self.backbone_variant == "standard":
            self.front_end = StandardCNNFrontEnd(
                n_mels=n_mels,
                stem_channels=stem_channels,
                stem_strides=stem_strides,
                stem_dropout=stem_dropout,
                projector_dropout=projector_dropout,
                embed_dim=embed_dim,
            )
        elif self.backbone_variant == "lightstem":
            light_cfg = cfg.model.get("lightstem", {})
            self.front_end = LightStemFrontEnd(
                n_mels=n_mels,
                channels=int(light_cfg.get("channels", stem_channels[-1] if stem_channels else 64)),
                stride=[int(v) for v in light_cfg.get("stride", [2, 1])],
                stem_dropout=stem_dropout,
                projector_dropout=projector_dropout,
                embed_dim=embed_dim,
            )
        elif self.backbone_variant == "nostem_patch":
            patch_cfg = cfg.model.get("nostem_patch", {})
            self.front_end = NoStemPatchFrontEnd(
                n_mels=n_mels,
                time_patch=int(patch_cfg.get("time_patch", 4)),
                embed_dim=embed_dim,
                projector_dropout=projector_dropout,
                norm_variant=str(patch_cfg.get("norm_variant", "layernorm")),
            )
        elif self.backbone_variant == "band_token":
            band_cfg = cfg.model.get("band_token", {})
            self.front_end = BandTokenFrontEnd(
                n_mels=n_mels,
                num_bands=int(band_cfg.get("num_bands", 4)),
                embed_dim=embed_dim,
                projector_dropout=projector_dropout,
                use_band_embedding=bool(band_cfg.get("use_band_embedding", True)),
            )
        else:
            raise ValueError(f"Unsupported backbone_variant: {self.backbone_variant}")
        first_layer_dim = int(self.layer_dims[0])
        self.input_projection = (
            nn.Identity()
            if embed_dim == first_layer_dim
            else nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, first_layer_dim))
        )
        self.pos_dropout = nn.Dropout(input_dropout)
        self.encoder = nn.ModuleList(
            [
                CNNConformerBlock(
                    embed_dim=layer_dim,
                    num_heads=num_heads,
                    ffn_dim=layer_ffn_dim,
                    conv_kernel_size=conv_kernel_size,
                    dropout=encoder_dropout,
                    attention_type=attention_type,
                    max_relative_position=max_relative_position,
                    conv_module_type=conv_module_type,
                    multiscale_kernel_sizes=multiscale_kernel_sizes,
                )
                for layer_dim, layer_ffn_dim in zip(self.layer_dims, self.layer_ffn_dims)
            ]
        )
        self.encoder_transitions = nn.ModuleList()
        for in_dim, out_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            if int(in_dim) == int(out_dim):
                self.encoder_transitions.append(nn.Identity())
            else:
                self.encoder_transitions.append(
                    nn.Sequential(
                        nn.LayerNorm(int(in_dim)),
                        nn.Linear(int(in_dim), int(out_dim)),
                    )
                )
        self.layer_fusion = layer_fusion
        self.layer_fuser = LayerWeightedSum(num_layers) if layer_fusion == "learned_sum" else None
        self.norm = nn.LayerNorm(int(self.layer_dims[-1]))
        self.pooling_type = pooling
        self.attentive_pool = AttentivePooling(int(self.layer_dims[-1])) if pooling == "attention" else None
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(int(self.layer_dims[-1]), num_classes)

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
        x, lengths, visuals = self.front_end(x, lengths)
        for key, value in visuals.items():
            self._cache_visual(key, value)

        x = self.input_projection(x)
        key_padding_mask = lengths_to_padding_mask(lengths, x.size(1))
        x = apply_sequence_mask(self.pos_dropout(x), key_padding_mask)

        layer_outputs: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.encoder, start=1):
            x = block(x, key_padding_mask=key_padding_mask)
            x = apply_sequence_mask(x, key_padding_mask)
            if self.sequence_shrinking_enabled and layer_idx in self.sequence_shrinking_layers:
                x, key_padding_mask = shrink_sequence(x, key_padding_mask, self.sequence_shrinking_factor)
                x = apply_sequence_mask(x, key_padding_mask)
            layer_outputs.append(x)
            if layer_idx < len(self.encoder):
                x = self.encoder_transitions[layer_idx - 1](x)
                x = apply_sequence_mask(x, key_padding_mask)

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
