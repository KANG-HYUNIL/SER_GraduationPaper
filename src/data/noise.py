from __future__ import annotations

import math

import torch


def parse_snr_db(value) -> float | None:
    if isinstance(value, str) and value.lower() == "clean":
        return None
    return float(value)


def signal_power(waveform: torch.Tensor) -> torch.Tensor:
    return waveform.float().pow(2).mean().clamp_min(1e-12)


def _torch_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _randn_like(waveform: torch.Tensor, seed: int) -> torch.Tensor:
    generator = _torch_generator(seed, waveform.device)
    return torch.randn(waveform.shape, generator=generator, device=waveform.device, dtype=waveform.dtype)


def _colored_noise_like(waveform: torch.Tensor, seed: int, beta: float) -> torch.Tensor:
    white = _randn_like(waveform, seed).float()
    freq = torch.fft.rfft(white, dim=-1)
    bins = torch.arange(freq.shape[-1], device=waveform.device, dtype=torch.float32).clamp_min(1.0)
    scale = bins.pow(-float(beta) / 2.0)
    scale[0] = 0.0
    colored = torch.fft.irfft(freq * scale, n=waveform.shape[-1], dim=-1)
    return colored.to(dtype=waveform.dtype)


def _babble_like(waveform: torch.Tensor, seed: int, speakers: int) -> torch.Tensor:
    generator = _torch_generator(seed, waveform.device)
    source = waveform.float()
    mixed = torch.zeros_like(source)
    speakers = max(2, int(speakers))
    length = source.shape[-1]
    for _ in range(speakers):
        shift = int(torch.randint(0, max(1, length), (1,), generator=generator, device=waveform.device).item())
        gain = float(torch.empty((1,), device=waveform.device).uniform_(0.65, 1.15, generator=generator).item())
        mixed = mixed + gain * torch.roll(source, shifts=shift, dims=-1)
    mixed = mixed / float(speakers)
    mixed = mixed + 0.05 * _colored_noise_like(waveform, seed + 17, beta=1.0).float()
    return mixed.to(dtype=waveform.dtype)


def _cafe_like(waveform: torch.Tensor, seed: int, transient_count: int) -> torch.Tensor:
    generator = _torch_generator(seed, waveform.device)
    pink = _colored_noise_like(waveform, seed, beta=1.0).float()
    brown = _colored_noise_like(waveform, seed + 1, beta=2.0).float()
    noise = 0.7 * pink + 0.3 * brown
    length = waveform.shape[-1]
    transient_count = max(0, int(transient_count))
    for _ in range(transient_count):
        center = int(torch.randint(0, max(1, length), (1,), generator=generator, device=waveform.device).item())
        width = int(torch.randint(max(8, length // 800), max(16, length // 120), (1,), generator=generator, device=waveform.device).item())
        start = max(0, center - width // 2)
        end = min(length, start + width)
        if end <= start:
            continue
        window = torch.hann_window(end - start, periodic=False, device=waveform.device, dtype=torch.float32)
        gain = float(torch.empty((1,), device=waveform.device).uniform_(0.2, 0.7, generator=generator).item())
        noise[..., start:end] = noise[..., start:end] + gain * window
    return noise.to(dtype=waveform.dtype)


def generate_noise_like(
    waveform: torch.Tensor,
    noise_type: str,
    seed: int,
    babble_speakers: int = 4,
    cafe_transient_count: int = 6,
) -> torch.Tensor:
    noise_type = str(noise_type).lower()
    if noise_type == "white":
        return _randn_like(waveform, seed)
    if noise_type == "pink":
        return _colored_noise_like(waveform, seed, beta=1.0)
    if noise_type in {"brown", "brownian"}:
        return _colored_noise_like(waveform, seed, beta=2.0)
    if noise_type in {"babble", "speech", "speech_like"}:
        return _babble_like(waveform, seed, speakers=babble_speakers)
    if noise_type in {"cafe", "street", "background"}:
        return _cafe_like(waveform, seed, transient_count=cafe_transient_count)
    raise ValueError(f"Unsupported noise_type: {noise_type}")


def add_noise_at_snr(
    waveform: torch.Tensor,
    noise_type: str,
    snr_db,
    seed: int,
    babble_speakers: int = 4,
    cafe_transient_count: int = 6,
) -> torch.Tensor:
    parsed_snr = parse_snr_db(snr_db)
    if parsed_snr is None:
        return waveform

    noise = generate_noise_like(
        waveform,
        noise_type=noise_type,
        seed=seed,
        babble_speakers=babble_speakers,
        cafe_transient_count=cafe_transient_count,
    ).float()
    waveform_float = waveform.float()
    target_noise_power = signal_power(waveform_float) / math.pow(10.0, parsed_snr / 10.0)
    scale = torch.sqrt(target_noise_power / signal_power(noise))
    mixed = waveform_float + scale * noise
    return mixed.clamp(-1.0, 1.0).to(dtype=waveform.dtype)
