import glob
import logging
import os

import torch
from torch.utils.data import Dataset

from src.data.transforms import AudioPipeline

logger = logging.getLogger(__name__)

EMOTION_MAP = {
    "01": 0,
    "02": 1,
    "03": 2,
    "04": 3,
    "05": 4,
    "06": 5,
    "07": 6,
    "08": 7,
}

INV_EMOTION_MAP = {v: k for k, v in EMOTION_MAP.items()}


class RavdessDataset(Dataset):
    def __init__(self, cfg, transform: AudioPipeline = None):
        self.root_path = cfg.dataset_path
        self.files = []
        self.labels = []
        self.actor_ids = []
        self.transform = transform
        self.cache_features = bool(cfg.get("cache_features", True))
        self._feature_cache: dict[int, tuple[torch.Tensor, int]] = {}
        self._load_dataset()

    def _load_dataset(self):
        search_path = os.path.join(self.root_path, "Actor_*", "*.wav")
        files = glob.glob(search_path)

        if not files:
            logger.error("No wav files found in %s. Check your path!", search_path)
            return

        for fpath in files:
            try:
                filename = os.path.basename(fpath)
                parts = filename.split("-")
                if len(parts) != 7:
                    continue

                emotion_code = parts[2]
                actor_code = parts[6].split(".")[0]
                if emotion_code in EMOTION_MAP:
                    self.files.append(fpath)
                    self.labels.append(EMOTION_MAP[emotion_code])
                    self.actor_ids.append(int(actor_code))
            except Exception as exc:
                logger.warning("Error parsing file %s: %s", fpath, exc)

        logger.info("Loaded %s files from %s", len(self.files), self.root_path)
        from collections import Counter

        counts = Counter(self.labels)
        distribution = {INV_EMOTION_MAP[k]: v for k, v in counts.items()}
        logger.info("Class Distribution: %s", distribution)
        actor_counts = Counter(self.actor_ids)
        logger.info("Actor Distribution: %s", dict(sorted(actor_counts.items())))

    def __len__(self):
        return len(self.files)

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


def build_chunk_spans(length: int, chunk_frames: int, hop_frames: int) -> list[tuple[int, int]]:
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive.")
    if hop_frames <= 0:
        raise ValueError("hop_frames must be positive.")
    if length < chunk_frames:
        raise ValueError(
            f"Utterance length {length} is shorter than chunk_frames={chunk_frames}. "
            "Choose a smaller chunk size or denser log-Mel parameters."
        )

    last_start = length - chunk_frames
    starts = list(range(0, last_start + 1, hop_frames))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + chunk_frames) for start in starts]


class ChunkedTrainDataset(Dataset):
    def __init__(self, base_dataset: RavdessDataset, indices, chunk_frames: int, hop_frames: int, include_actor_id: bool = False):
        self.base_dataset = base_dataset
        self.indices = [int(idx) for idx in indices]
        self.chunk_frames = int(chunk_frames)
        self.hop_frames = int(hop_frames)
        self.include_actor_id = bool(include_actor_id)
        self.chunk_index: list[tuple[int, int, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        for utterance_idx in self.indices:
            _, length = self.base_dataset.get_feature(utterance_idx)
            for start, end in build_chunk_spans(length, self.chunk_frames, self.hop_frames):
                self.chunk_index.append((utterance_idx, start, end))

    def __len__(self) -> int:
        return len(self.chunk_index)

    def __getitem__(self, idx: int):
        utterance_idx, start, end = self.chunk_index[idx]
        feature, _ = self.base_dataset.get_feature(utterance_idx)
        chunk = feature[..., start:end]
        label = self.base_dataset.labels[utterance_idx]
        if self.include_actor_id:
            actor_id = self.base_dataset.get_actor_id(utterance_idx)
            return chunk.contiguous(), torch.tensor(label, dtype=torch.long), torch.tensor(actor_id, dtype=torch.long)
        return chunk.contiguous(), torch.tensor(label, dtype=torch.long)


class UtteranceChunkDataset(Dataset):
    def __init__(self, base_dataset: RavdessDataset, indices, chunk_frames: int, hop_frames: int):
        self.base_dataset = base_dataset
        self.indices = [int(idx) for idx in indices]
        self.chunk_frames = int(chunk_frames)
        self.hop_frames = int(hop_frames)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        utterance_idx = self.indices[idx]
        feature, length = self.base_dataset.get_feature(utterance_idx)
        chunks = [feature[..., start:end] for start, end in build_chunk_spans(length, self.chunk_frames, self.hop_frames)]
        label = self.base_dataset.labels[utterance_idx]
        return torch.stack(chunks, dim=0), torch.tensor(label, dtype=torch.long), utterance_idx


def collate_fixed_chunks(batch):
    if len(batch[0]) == 3:
        features, labels, actor_ids = zip(*batch)
        return torch.stack(features, dim=0), torch.stack(labels, dim=0), torch.stack(actor_ids, dim=0)
    features, labels = zip(*batch)
    return torch.stack(features, dim=0), torch.stack(labels, dim=0)


class TrainSubsetWithActor(Dataset):
    def __init__(self, base_dataset: RavdessDataset, indices):
        self.base_dataset = base_dataset
        self.indices = [int(idx) for idx in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        feature, label, length = self.base_dataset[base_idx]
        actor_id = self.base_dataset.get_actor_id(base_idx)
        return feature, label, length, torch.tensor(actor_id, dtype=torch.long)


def collate_with_actor(batch):
    features, labels, lengths, actor_ids = zip(*batch)
    return (
        torch.stack(features, dim=0),
        torch.stack(labels, dim=0),
        torch.as_tensor(lengths, dtype=torch.long),
        torch.stack(actor_ids, dim=0),
    )


def collate_utterance_chunks(batch):
    if len(batch) != 1:
        raise ValueError("UtteranceChunkDataset requires batch_size=1 to avoid cross-utterance padding.")
    chunks, label, utterance_idx = batch[0]
    return chunks, label.unsqueeze(0), torch.tensor([utterance_idx], dtype=torch.long)
