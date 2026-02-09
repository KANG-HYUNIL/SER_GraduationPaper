import hydra
from omegaconf import DictConfig
import torch
import torchaudio
import matplotlib.pyplot as plt
import os
from datetime import datetime
import glob
import random
import numpy as np
from src.data.transforms import AudioPipeline

def save_spectrogram_image(spec, save_path, title="Mel Spectrogram"):
    """
    Saves the spectrogram as an image.
    Args:
        spec (torch.Tensor): Log-Mel Spectrogram tensor (C, F, T)
        save_path (str): Path to save the image
        title (str): Title of the plot
    """
    if spec.dim() == 3:
        spec = spec.squeeze(0)  # Remove channel dim (1, F, T) -> (F, T)
    
    spec_np = spec.cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup Output Directory in Project Root
    # Hydra changes cwd to outputs/..., but we want to save to a fixed root dir for inspection
    import hydra.utils
    original_cwd = hydra.utils.get_original_cwd()
    output_dir = os.path.join(original_cwd, "mel_spectrogram_transform_testing")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving visualizations to: {output_dir}")

    # 2. Initialize Pipeline
    # We might want to see UN-NORMALIZED data to check the raw Mel structure first.
    # But checking what the model ACTUALLY sees (normalized) is also important.
    # Let's do both or stick to the configured one. User asked for "transformed".
    pipeline = AudioPipeline(cfg.data)
    
    # 3. Find Audio Files
    # Assuming standard RAVDESS structure or flat, we try to find .wav files
    # dataset_path is usually absolute due to hydra resolver, check config
    dataset_path = cfg.data.dataset_path
    
    # Simple recursive search for wavs
    search_pattern = os.path.join(dataset_path, "**", "*.wav")
    audio_files = glob.glob(search_pattern, recursive=True)
    
    if not audio_files:
        print(f"No .wav files found in {dataset_path}")
        return

    # Select fixed samples for consistency across runs
    num_samples = 5
    # Sort to ensure deterministic order
    audio_files.sort()
    selected_files = audio_files[:num_samples]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, file_path in enumerate(selected_files):
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        # Load waveform
        waveform, sr = torchaudio.load(file_path)
        
        # Apply Transform
        # Note: transform() includes normalization if cfg.normalize is True
        mel_spec = pipeline.transform(waveform, sr)
        
        # Save Name: Timestamp-Filename.png
        save_name = f"{timestamp}-{filename.replace('.wav', '')}.png"
        save_path = os.path.join(output_dir, save_name)
        
        save_spectrogram_image(mel_spec, save_path, title=f"Log-Mel Spectrogram: {filename}")
        print(f"Saved: {save_path}")

    print("Done.")

if __name__ == "__main__":
    main()
