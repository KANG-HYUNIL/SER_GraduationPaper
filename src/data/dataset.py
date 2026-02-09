import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset
from src.data.transforms import AudioPipeline
import logging

# Set up logging
logger = logging.getLogger(__name__)

# RAVDESS Emotion Mapping
EMOTION_MAP = {
    '01': 0, # neutral
    '02': 1, # calm
    '03': 2, # happy
    '04': 3, # sad
    '05': 4, # angry
    '06': 5, # fearful
    '07': 6, # disgust
    '08': 7  # surprised
}

INV_EMOTION_MAP = {v: k for k, v in EMOTION_MAP.items()}

class RavdessDataset(Dataset):
    def __init__(self, cfg, transform: AudioPipeline = None):
        """
        Args:
            cfg: Hydra config for data (contains dataset_path)
            transform: Instance of AudioPipeline
        """
        self.root_path = cfg.dataset_path
        self.files = []
        self.labels = []
        self.actor_ids = []
        self.transform = transform
        
        self._load_dataset()

    def _load_dataset(self):
        # Scan for all .wav files in Actor_* subdirectories
        search_path = os.path.join(self.root_path, "Actor_*", "*.wav")
        files = glob.glob(search_path)
        
        if not files:
            logger.error(f"No wav files found in {search_path}. Check your path!")
            return

        for fpath in files:
            try:
                # Filename example: 03-01-01-01-01-01-01.wav
                filename = os.path.basename(fpath)
                parts = filename.split('-')
                
                if len(parts) != 7:
                    continue
                
                emotion_code = parts[2] # 3rd part is emotion
                actor_code = parts[6].split('.')[0] # 7th part is actor, remove .wav

                if emotion_code in EMOTION_MAP:
                    self.files.append(fpath)
                    self.labels.append(EMOTION_MAP[emotion_code])
                    self.actor_ids.append(int(actor_code)) # Store as int for easier grouping
            except Exception as e:
                logger.warning(f"Error parsing file {fpath}: {e}")

        logger.info(f"Loaded {len(self.files)} files from {self.root_path}")
        
        # Log class distribution
        from collections import Counter
        counts = Counter(self.labels)
        distribution = {INV_EMOTION_MAP[k]: v for k, v in counts.items()}
        logger.info(f"Class Distribution: {distribution}")
        
        actor_counts = Counter(self.actor_ids)
        logger.info(f"Actor Distribution: {dict(sorted(actor_counts.items()))}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wav_path = self.files[idx]
        label = self.labels[idx]

        # Load Audio using soundfile directly (bypass torchaudio backend issue)
        import soundfile as sf
        try:
            # sf.read returns:
            # - waveform_np: (Time,) for mono, (Time, Channels) for stereo
            # - sample_rate: int
            waveform_np, sample_rate = sf.read(wav_path)
            
            # soundfile output is usually Float64, PyTorch needs Float32
            waveform = torch.from_numpy(waveform_np).float()
            
            # waveform.ndim means "Number of Dimensions" (rank)
            # Case 1: Mono audio -> Shape is (Time,) e.g. (48000,) -> ndim=1
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0) 
                # unsqueeze(0) adds a dimension at index 0.
                # (48000,) -> (1, 48000) : (Channel, Time)
                
            # Case 2: Stereo audio -> Shape is (Time, Channels) e.g. (48000, 2) -> ndim=2
            else:
                waveform = waveform.t() 
                # .t() transposes the matrix.
                # (Time, Channels) -> (Channels, Time) : (2, 48000)
                
        except Exception as e:
            logger.error(f"Failed to read {wav_path}: {e}")
            # return zero tensor to avoid crash, or filter out
            return torch.zeros(1, 16000), torch.tensor(label, dtype=torch.long)

        # Apply Transforms
        if self.transform:
            feature = self.transform.transform(waveform, sample_rate)
        else:
            feature = waveform 

        return feature, torch.tensor(label, dtype=torch.long)

def collate_dynamic_padding(batch):
    """
    Collate function to handle variable length audio/spectrograms.
    Args:
        batch: List of tuples (feature, label)
    Returns:
        padded_features: (Batch, Channel, Freq, Max_Time)
        labels: (Batch,)
    """
    # 1. Separate features and labels
    features, labels = zip(*batch)
    
    # feature shape: (1, n_mels, Time)
    # We need to find the max Time in this batch
    max_time = max([f.shape[2] for f in features])
    
    padded_features = []
    for f in features:
        # f: (1, 128, T)
        current_time = f.shape[2]
        pad_amount = max_time - current_time
        
        # F.pad format for 3D tensor: (pad_left, pad_right, pad_top, pad_bottom, ...)
        # We only pad the last dimension (Time) on the right
        if pad_amount > 0:
            # (pad_last_dim_left, pad_last_dim_right)
            padded_f = torch.nn.functional.pad(f, (0, pad_amount), "constant", 0.0)
        else:
            padded_f = f
            
        padded_features.append(padded_f)
        
    # Stack them
    padded_features = torch.stack(padded_features)
    labels = torch.stack(labels)
    
    return padded_features, labels


