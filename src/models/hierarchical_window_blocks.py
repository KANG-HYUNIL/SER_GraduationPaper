import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer_blocks import ChannelLayerNorm2d, apply_channel_mask_2d


def apply_spatial_mask_2d(x: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return x
    return x.masked_fill(~valid_mask.unsqueeze(1), 0.0)


def lengths_to_2d_valid_mask(lengths: torch.Tensor | None, freq_size: int, time_size: int, device: torch.device) -> torch.Tensor | None:
    if lengths is None:
        return None
    time_idx = torch.arange(time_size, device=device).unsqueeze(0)
    valid_time = time_idx < lengths.unsqueeze(1)
    return valid_time.unsqueeze(1).expand(-1, freq_size, -1)


class SpatialProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = ChannelLayerNorm2d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self.proj(x)
        x = apply_channel_mask_2d(x, lengths)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return apply_channel_mask_2d(x, lengths)


class Mlp2D(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 3, 1)
        y = self.net(self.norm(y))
        return y.permute(0, 3, 1, 2)


class WindowAttention2D(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, invalid_token_mask: torch.Tensor | None = None) -> torch.Tensor:
        x, _ = self.attn(x, x, x, key_padding_mask=invalid_token_mask, need_weights=False)
        return self.dropout(x)


class WindowTransformerBlock2D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        window_size: tuple[int, int],
        shift_size: tuple[int, int] = (0, 0),
    ):
        super().__init__()
        self.window_size = (int(window_size[0]), int(window_size[1]))
        self.shift_size = (int(shift_size[0]), int(shift_size[1]))
        self.norm1 = ChannelLayerNorm2d(embed_dim)
        self.attn = WindowAttention2D(embed_dim, num_heads, dropout)
        self.mlp = Mlp2D(embed_dim, ffn_dim, dropout)

    def _apply_shift(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        shift_f, shift_t = self.shift_size
        if shift_f <= 0 and shift_t <= 0:
            return x, valid_mask

        x = F.pad(x, (0, shift_t, 0, shift_f))
        x = x[:, :, shift_f:, shift_t:]
        if valid_mask is not None:
            valid_mask = F.pad(valid_mask, (0, shift_t, 0, shift_f), value=False)
            valid_mask = valid_mask[:, shift_f:, shift_t:]
        return x, valid_mask

    def _undo_shift(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
        original_freq: int,
        original_time: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        shift_f, shift_t = self.shift_size
        if shift_f <= 0 and shift_t <= 0:
            return x[:, :, :original_freq, :original_time], None if valid_mask is None else valid_mask[:, :original_freq, :original_time]

        x = F.pad(x, (shift_t, 0, shift_f, 0))
        x = x[:, :, :original_freq, :original_time]
        if valid_mask is not None:
            valid_mask = F.pad(valid_mask, (shift_t, 0, shift_f, 0), value=False)
            valid_mask = valid_mask[:, :original_freq, :original_time]
        return x, valid_mask

    def _pad_to_window(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
        _, _, freq_size, time_size = x.shape
        pad_f = (-freq_size) % self.window_size[0]
        pad_t = (-time_size) % self.window_size[1]
        if pad_f or pad_t:
            x = F.pad(x, (0, pad_t, 0, pad_f))
            if valid_mask is not None:
                valid_mask = F.pad(valid_mask, (0, pad_t, 0, pad_f), value=False)
        return x, valid_mask, pad_f, pad_t

    def _partition_windows(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
        batch_size, channels, freq_size, time_size = x.shape
        win_f, win_t = self.window_size
        num_freq_windows = freq_size // win_f
        num_time_windows = time_size // win_t

        x = x.view(batch_size, channels, num_freq_windows, win_f, num_time_windows, win_t)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(batch_size * num_freq_windows * num_time_windows, win_f * win_t, channels)

        invalid_token_mask = None
        if valid_mask is not None:
            mask = valid_mask.view(batch_size, num_freq_windows, win_f, num_time_windows, win_t)
            mask = mask.permute(0, 1, 3, 2, 4).reshape(batch_size * num_freq_windows * num_time_windows, win_f * win_t)
            invalid_token_mask = ~mask

        return x, invalid_token_mask, num_freq_windows, num_time_windows

    def _merge_windows(
        self,
        windows: torch.Tensor,
        batch_size: int,
        channels: int,
        num_freq_windows: int,
        num_time_windows: int,
    ) -> torch.Tensor:
        win_f, win_t = self.window_size
        x = windows.view(batch_size, num_freq_windows, num_time_windows, win_f, win_t, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch_size, channels, num_freq_windows * win_f, num_time_windows * win_t)
        return x

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        original_freq, original_time = x.size(2), x.size(3)

        x = self.norm1(x)
        x = apply_spatial_mask_2d(x, valid_mask)
        x, shifted_mask = self._apply_shift(x, valid_mask)
        x, shifted_mask, pad_f, pad_t = self._pad_to_window(x, shifted_mask)

        windows, invalid_token_mask, num_freq_windows, num_time_windows = self._partition_windows(x, shifted_mask)
        windows = self.attn(windows, invalid_token_mask=invalid_token_mask)
        x = self._merge_windows(windows, residual.size(0), residual.size(1), num_freq_windows, num_time_windows)

        if pad_f or pad_t:
            x = x[:, :, : x.size(2) - pad_f if pad_f else x.size(2), : x.size(3) - pad_t if pad_t else x.size(3)]
            if shifted_mask is not None:
                shifted_mask = shifted_mask[:, : x.size(2), : x.size(3)]

        x, _ = self._undo_shift(x, shifted_mask, original_freq, original_time)
        x = residual + x
        x = apply_spatial_mask_2d(x, valid_mask)
        x = x + self.mlp(x)
        return apply_spatial_mask_2d(x, valid_mask)


class PatchMerging2D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim * 4)
        self.reduction = nn.Linear(in_dim * 4, out_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        _, _, freq_size, time_size = x.shape
        pad_f = freq_size % 2
        pad_t = time_size % 2
        if pad_f or pad_t:
            x = F.pad(x, (0, pad_t, 0, pad_f))
            if valid_mask is not None:
                valid_mask = F.pad(valid_mask, (0, pad_t, 0, pad_f), value=False)

        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)

        if valid_mask is not None:
            m0 = valid_mask[:, 0::2, 0::2]
            m1 = valid_mask[:, 1::2, 0::2]
            m2 = valid_mask[:, 0::2, 1::2]
            m3 = valid_mask[:, 1::2, 1::2]
            valid_mask = m0 | m1 | m2 | m3

        x = x.permute(0, 2, 3, 1)
        x = self.reduction(self.norm(x))
        x = x.permute(0, 3, 1, 2)

        if lengths is not None:
            lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        x = apply_spatial_mask_2d(x, valid_mask)
        return x, valid_mask, lengths
