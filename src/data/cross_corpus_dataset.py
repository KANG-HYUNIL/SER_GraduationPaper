from __future__ import annotations

import glob
import logging
import os
from collections import Counter

import torch
from torch.utils.data import Dataset

from src.data.transforms import AudioPipeline

logger = logging.getLogger(__name__)

COMMON_6CLASS_NAMES = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
COMMON_6CLASS_TO_INDEX = {name: idx for idx, name in enumerate(COMMON_6CLASS_NAMES)}

RAVDESS_TO_COMMON_6CLASS = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
}

CREMAD_TO_COMMON_6CLASS = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fearful",
    "DIS": "disgust",
}


class BaseCrossCorpusDataset(Dataset):
    def __init__(self, transform: AudioPipeline | None = None, cache_features: bool = True):
        self.files: list[str] = []
        self.labels: list[int] = []
        self.actor_ids: list[int] = []
        self.transform = transform
        self.cache_features = bool(cache_features)
        self._feature_cache: dict[int, tuple[torch.Tensor, int]] = {}

    def __len__(self) -> int:
        return len(self.files)

    def _load_feature(self, idx: int) -> tuple[torch.Tensor, int]:
        wav_path = self.files[idx]

        import soundfile as sf

        waveform_np, sample_rate = sf.read(wav_path)
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

        feature = self.transform.transform(waveform, sample_rate) if self.transform else waveform
        return feature, int(feature.shape[-1])

    def get_feature(self, idx: int) -> tuple[torch.Tensor, int]:
        if self.cache_features and idx in self._feature_cache:
            feature, length = self._feature_cache[idx]
            return feature.clone(), int(length)

        feature, length = self._load_feature(idx)
        if self.cache_features:
            self._feature_cache[idx] = (feature.clone(), int(length))
        return feature, int(length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        label = self.labels[idx]
        feature, length = self.get_feature(idx)
        return feature, torch.tensor(label, dtype=torch.long), length

    def get_actor_id(self, idx: int) -> int:
        return int(self.actor_ids[idx])

    def _log_summary(self, dataset_name: str) -> None:
        logger.info("Loaded %s %s files.", len(self.files), dataset_name)
        counts = Counter(self.labels)
        distribution = {COMMON_6CLASS_NAMES[k]: v for k, v in sorted(counts.items())}
        logger.info("%s class distribution: %s", dataset_name, distribution)
        actor_counts = Counter(self.actor_ids)
        logger.info("%s actor distribution: %s", dataset_name, dict(sorted(actor_counts.items())))


class RavdessSixClassDataset(BaseCrossCorpusDataset):
    def __init__(self, root_path: str, transform: AudioPipeline | None = None, cache_features: bool = True):
        super().__init__(transform=transform, cache_features=cache_features)
        self.root_path = root_path
        self._load_dataset()

    def _load_dataset(self) -> None:
        search_path = os.path.join(self.root_path, "Actor_*", "*.wav")
        files = glob.glob(search_path)
        if not files:
            logger.error("No RAVDESS wav files found in %s", search_path)
            return

        for fpath in files:
            filename = os.path.basename(fpath)
            parts = filename.split("-")
            if len(parts) != 7:
                continue
            emotion_code = parts[2]
            actor_code = parts[6].split(".")[0]
            emotion_name = RAVDESS_TO_COMMON_6CLASS.get(emotion_code)
            if emotion_name is None:
                continue
            self.files.append(fpath)
            self.labels.append(COMMON_6CLASS_TO_INDEX[emotion_name])
            self.actor_ids.append(int(actor_code))
        self._log_summary("RAVDESS-6class")


class CremaDSixClassDataset(BaseCrossCorpusDataset):
    def __init__(self, root_path: str, transform: AudioPipeline | None = None, cache_features: bool = True):
        super().__init__(transform=transform, cache_features=cache_features)
        self.root_path = root_path
        self._load_dataset()

    def _candidate_patterns(self) -> list[str]:
        base = self.root_path
        return [
            os.path.join(base, "AudioWAV", "*.wav"),
            os.path.join(base, "AudioWAV", "*", "*.wav"),
            os.path.join(base, "*.wav"),
            os.path.join(base, "*", "*.wav"),
        ]

    def _load_dataset(self) -> None:
        files: list[str] = []
        for pattern in self._candidate_patterns():
            files.extend(glob.glob(pattern))
        files = sorted(set(files))
        if not files:
            logger.error("No CREMA-D wav files found under %s", self.root_path)
            return

        for fpath in files:
            filename = os.path.basename(fpath)
            stem, _ = os.path.splitext(filename)
            parts = stem.split("_")
            if len(parts) != 4:
                continue
            actor_id, _, emotion_code, _ = parts
            emotion_name = CREMAD_TO_COMMON_6CLASS.get(emotion_code)
            if emotion_name is None:
                continue
            try:
                actor_numeric = int(actor_id)
            except ValueError:
                continue
            self.files.append(fpath)
            self.labels.append(COMMON_6CLASS_TO_INDEX[emotion_name])
            self.actor_ids.append(actor_numeric)
        self._log_summary("CREMA-D-6class")
