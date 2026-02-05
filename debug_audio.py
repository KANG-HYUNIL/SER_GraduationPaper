import torch
import torchaudio
import os
import soundfile as sf

def check_audio():
    print("\n" + "="*50)
    print(f"Torchaudio version: {torchaudio.__version__}")
    print("MPL: soundfile imported successfully.")
    
    # Try to load a file
    root = "src/$RVNS6MQ"
    target_file = None
    for r, d, f in os.walk(root):
        for file in f:
            if file.endswith(".wav"):
                target_file = os.path.join(r, file)
                break
        if target_file:
            break
            
    if not target_file:
        print("No wav file found.")
        return

    print(f"Testing file: {target_file}")

    # Test 1: SoundFile direct
    try:
        data, sr = sf.read(target_file)
        print(f"[SoundFile] Read Success! Shape: {data.shape}, SR: {sr}")
    except Exception as e:
        print(f"[SoundFile] Failed: {e}")

    # Test 2: Torchaudio default
    try:
        print("[Torchaudio] Attempting default load...")
        wav, sr = torchaudio.load(target_file)
        print(f"[Torchaudio] Default Success! Shape: {wav.shape}, SR: {sr}")
    except Exception as e:
        print(f"[Torchaudio] Default Failed: {e}")

    # Test 3: Torchaudio force soundfile
    try:
        print("[Torchaudio] Attempting backend='soundfile'...")
        wav, sr = torchaudio.load(target_file, backend="soundfile")
        print(f"[Torchaudio] Force SoundFile Success! Shape: {wav.shape}, SR: {sr}")
    except Exception as e:
        print(f"[Torchaudio] Force SoundFile Failed: {e}")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    check_audio()
