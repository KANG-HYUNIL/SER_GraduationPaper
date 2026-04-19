import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer_blocks import ChannelLayerNorm2d, apply_channel_mask_2d


def apply_spatial_mask_2d(x: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return x
    return x.masked_fill(~valid_mask.unsqueeze(1), 0.0)


def lengths_to_2d_valid_mask(
    lengths: torch.Tensor | None,
    freq_size: int,
    time_size: int,
    device: torch.device,
) -> torch.Tensor | None:
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


class RelativeWindowAttention2D(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, window_size: tuple[int, int], dropout: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.window_size = (int(window_size[0]), int(window_size[1]))
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        relative_bias_size = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(relative_bias_size, self.num_heads))
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        invalid_token_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_windows, token_count, _ = x.shape
        qkv = self.qkv(x).reshape(batch_windows, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        relative_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        relative_bias = relative_bias.view(token_count, token_count, self.num_heads).permute(2, 0, 1).contiguous()
        attn = attn + relative_bias.unsqueeze(0)

        if attn_mask is not None:
            num_windows = attn_mask.size(0)
            if batch_windows % num_windows != 0:
                raise ValueError("Window batch is not divisible by the attention mask window count.")
            batch_size = batch_windows // num_windows
            attn = attn.view(batch_size, num_windows, self.num_heads, token_count, token_count)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(batch_windows, self.num_heads, token_count, token_count)

        if invalid_token_mask is not None:
            attn = attn.masked_fill(invalid_token_mask.unsqueeze(1).unsqueeze(2), torch.finfo(attn.dtype).min)

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_windows, token_count, self.embed_dim)
        out = self.proj_dropout(self.proj(out))
        if invalid_token_mask is not None:
            out = out.masked_fill(invalid_token_mask.unsqueeze(-1), 0.0)
        return out


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
        self.attn = RelativeWindowAttention2D(embed_dim, num_heads, self.window_size, dropout)
        self.mlp = Mlp2D(embed_dim, ffn_dim, dropout)

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

    def _build_shift_mask(self, freq_size: int, time_size: int, device: torch.device) -> torch.Tensor:
        shift_f, shift_t = self.shift_size
        if shift_f == 0 and shift_t == 0:
            return None

        mask = torch.zeros((1, 1, freq_size, time_size), device=device)
        freq_slices = (
            slice(0, -self.window_size[0]),
            slice(-self.window_size[0], -shift_f),
            slice(-shift_f, None),
        )
        time_slices = (
            slice(0, -self.window_size[1]),
            slice(-self.window_size[1], -shift_t),
            slice(-shift_t, None),
        )

        region_id = 0
        for freq_slice in freq_slices:
            for time_slice in time_slices:
                mask[:, :, freq_slice, time_slice] = region_id
                region_id += 1

        mask_windows, _, _, _ = self._partition_windows(mask, None)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, torch.finfo(mask.dtype).min)
        return attn_mask.masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        original_freq, original_time = x.size(2), x.size(3)

        x = self.norm1(x)
        x = apply_spatial_mask_2d(x, valid_mask)
        x, padded_mask, _, _ = self._pad_to_window(x, valid_mask)

        shift_f = min(self.shift_size[0], self.window_size[0] - 1)
        shift_t = min(self.shift_size[1], self.window_size[1] - 1)
        attn_mask = None
        if shift_f > 0 or shift_t > 0:
            x = torch.roll(x, shifts=(-shift_f, -shift_t), dims=(2, 3))
            padded_mask = None if padded_mask is None else torch.roll(padded_mask, shifts=(-shift_f, -shift_t), dims=(1, 2))
            attn_mask = self._build_shift_mask(x.size(2), x.size(3), x.device)

        windows, invalid_token_mask, num_freq_windows, num_time_windows = self._partition_windows(x, padded_mask)
        windows = self.attn(windows, invalid_token_mask=invalid_token_mask, attn_mask=attn_mask)
        x = self._merge_windows(windows, residual.size(0), residual.size(1), num_freq_windows, num_time_windows)

        if shift_f > 0 or shift_t > 0:
            x = torch.roll(x, shifts=(shift_f, shift_t), dims=(2, 3))

        x = x[:, :, :original_freq, :original_time]
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


class BridgeContext2D(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_tokens: int, dropout: float):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.query_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, freq_size, time_size = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        key_padding_mask = None if valid_mask is None else ~valid_mask.reshape(batch_size, freq_size * time_size)
        query = self.query_tokens.expand(batch_size, -1, -1)
        bridge, _ = self.cross_attn(
            self.query_norm(query),
            self.kv_norm(tokens),
            self.kv_norm(tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        gate = self.gate(bridge.mean(dim=1)).unsqueeze(-1).unsqueeze(-1)
        x = x * (1.0 + gate)
        return apply_spatial_mask_2d(x, valid_mask), bridge


class BridgeProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.Tanh(),
        )

    def forward(self, bridge_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(bridge_tokens.mean(dim=1)).unsqueeze(-1).unsqueeze(-1)
