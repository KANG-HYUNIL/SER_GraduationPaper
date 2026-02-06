import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
import csv
import logging
from src.utils.registry import get_model_class

logger = logging.getLogger(__name__)

def load_audio_files(path):
    """
    Load audio files from a file path or a directory.
    """
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        # Recursive search for .wav files
        files = glob.glob(os.path.join(path, "**/*.wav"), recursive=True)
        return files
    else:
        raise FileNotFoundError(f"Path not found: {path}")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device == "auto" else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Determine Checkpoint Path
    checkpoint_path = cfg.inference.model_checkpoint
    if not checkpoint_path or checkpoint_path == "???":
        # Fallback to default saved_models path
        import hydra.utils
        root_dir = hydra.utils.get_original_cwd()
        default_name = f"best_model_{cfg.model.name}.pt"
        checkpoint_path = os.path.join(root_dir, "saved_models", default_name)
        logger.info(f"No checkpoint provided. Using default: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at: {checkpoint_path}")
        return

    # 3. Load Model Architecture & Weights
    logger.info(f"Loading model: {cfg.model.name}")
    try:
        model_class = get_model_class(cfg.model.name)
        model = model_class(cfg).to(device)
        
        # Load weights
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 5. Load Audio Files
    try:
        audio_files = load_audio_files(cfg.inference.audio_path)
        logger.info(f"Found {len(audio_files)} audio files to process.")
    except Exception as e:
        logger.error(str(e))
        return

    if not audio_files:
        logger.warning("No .wav files found in the provided path.")
        return

    # 6. Inference Loop
    results = []
    emotion_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

    # Prepare specific output directory
    import hydra.utils
    # inference_outputs/YYYY-MM-DD_HH-MM-SS
    output_root = os.path.join(hydra.utils.get_original_cwd(), cfg.inference.output_dir)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(output_root, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    logger.info("Starting inference...")
    
    # Import torchaudio/librosa for manual preprocessing
    # Ideally should share code with transforms.py, but for stability within single script we duplicate basic logic
    import librosa
    import torchaudio.transforms as T
    
    with torch.no_grad():
        for file_path in audio_files:
             # Preprocess (Load -> LogMel)
            try:
                # Load audio
                y, sr = librosa.load(file_path, sr=cfg.data.sample_rate)
                
                waveform = torch.tensor(y).unsqueeze(0) # (1, Time)
                
                # Apply MelSpectrogram Transform
                mel_transform = T.MelSpectrogram(
                    sample_rate=cfg.data.sample_rate,
                    n_fft=cfg.data.n_fft,
                    hop_length=cfg.data.hop_length,
                    n_mels=cfg.data.n_mels
                )
                
                spec = mel_transform(waveform)
                spec = T.AmplitudeToDB(top_db=80)(spec) # Log-Mel
                
                # Add Batch & Channel dims (1, 1, F, T)
                input_tensor = spec.unsqueeze(0).to(device)
                
                # Forward
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1) # (1, 8)
                
                # Get Top-1
                prob_values, indices = torch.max(probs, 1)
                predicted_idx = indices.item()
                confidence = prob_values.item()
                predicted_label = emotion_names[predicted_idx]
                
                # Store Result
                file_name = os.path.basename(file_path)
                result_row = {
                    "filename": file_name,
                    "filepath": file_path,
                    "prediction": predicted_label,
                    "confidence": f"{confidence:.4f}",
                    "probabilities": str([f"{p:.4f}" for p in probs.cpu().numpy()[0]])
                }
                results.append(result_row)
                
                logger.info(f"[{file_name}] -> {predicted_label} ({confidence:.2%})")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    # 7. Save Results
    csv_path = os.path.join(save_dir, "inference_results.csv")
    if results:
        keys = results[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Saved results to: {csv_path}")
    else:
        logger.warning("No results to save.")
        
    logger.info("Inference Complete.")

if __name__ == "__main__":
    main()
