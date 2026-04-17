import torch
import torchaudio.transforms as T
import torch.nn.functional as F
from omegaconf import DictConfig


class AudioPipeline:
    def __init__(self, cfg: DictConfig):
        self.sample_rate = cfg.sample_rate
        self.duration = cfg.duration
        self.n_mels = cfg.n_mels
        self.n_fft = cfg.n_fft
        self.hop_length = cfg.hop_length
        self.target_length = int(self.sample_rate * self.duration)
        self.normalize = cfg.normalize
        self.resize_enabled = bool(cfg.get("resize_enabled", True))
        self.resize_height = cfg.get("resize_height", 128)
        self.resize_width = cfg.get("resize_width", 512)

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def transform(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        if self.normalize:
            mean = log_mel_spec.mean()
            std = log_mel_spec.std()
            log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)

        if not self.resize_enabled:
            return log_mel_spec

        log_mel_spec = log_mel_spec.unsqueeze(0)
        log_mel_spec = F.interpolate(
            log_mel_spec,
            size=(self.resize_height, self.resize_width),
            mode="bicubic",
            align_corners=False,
        )
        return log_mel_spec.squeeze(0)
