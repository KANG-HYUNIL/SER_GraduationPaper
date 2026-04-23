from __future__ import annotations

import logging

import torch

from src.data.dataset import RavdessDataset
from src.data.noise import add_noise_at_snr, parse_snr_db

logger = logging.getLogger(__name__)


class NoisyRavdessDataset(RavdessDataset):
    def __init__(
        self,
        cfg,
        transform=None,
        noise_type: str = "white",
        snr_db="clean",
        seed: int = 42,
        babble_speakers: int = 4,
        cafe_transient_count: int = 6,
    ):
        super().__init__(cfg, transform=transform)
        self.cache_features = False
        self.noise_type = str(noise_type)
        self.snr_db = snr_db
        self.seed = int(seed)
        self.babble_speakers = int(babble_speakers)
        self.cafe_transient_count = int(cafe_transient_count)

    def _load_feature(self, idx: int) -> tuple[torch.Tensor, int]:
        wav_path = self.files[idx]

        import soundfile as sf

        try:
            waveform_np, sample_rate = sf.read(wav_path)
            waveform = torch.from_numpy(waveform_np).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()
        except Exception as exc:
            logger.error("Failed to read %s: %s", wav_path, exc)
            return torch.zeros(1, 80, 1), 1

        if parse_snr_db(self.snr_db) is not None:
            waveform = add_noise_at_snr(
                waveform,
                noise_type=self.noise_type,
                snr_db=self.snr_db,
                seed=self.seed + int(idx),
                babble_speakers=self.babble_speakers,
                cafe_transient_count=self.cafe_transient_count,
            )

        feature = self.transform.transform(waveform, sample_rate) if self.transform else waveform
        return feature, int(feature.shape[-1])
