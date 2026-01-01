import librosa
import matplotlib.pyplot as plt
import pywt
import numpy as np
import os


#0. Load raw-audio(wav) file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "03-01-01-01-01-01-01.wav")
def load_audio(file_path = FILE_PATH, sr=22050):
    """
    Load an audio file (wav/mp3, etc.) and return the signal and sampling rate.

    Args:
        file_path (str): Path to the audio file.
        sr (int, optional): Target sampling rate (Hz). Default is 22050.

    Returns:
        tuple[np.ndarray, int]: (audio signal, sampling rate)
    """
    signal, sr = librosa.load(file_path, sr=sr)  
    return signal, sr

signal, sr = load_audio()

#1. raw-audio(wav) amplitude show
def plot_waveform(signal, sr):
    """
    Visualize the time-amplitude waveform of the input audio signal.

    Args:
        signal (np.ndarray): Audio signal.
        sr (int): Sampling rate (Hz).
    """
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(signal, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


# Unified plotting function for feature extraction results
def plot_feature_result(data, sr, kind="waveform", **kwargs):
    """
    Unified visualization for feature extraction results.
    Args:
        data: Feature extraction result (array or list)
        sr: Sampling rate (optional, for time axes)
        kind: Type of feature ('waveform', 'fourier', 'fft', 'stft', 'mel', 'wavelet', 'mfcc')
        kwargs: Additional plotting parameters
    """
    plt.figure(figsize=(14, 5))
    if kind == "waveform":
        librosa.display.waveshow(data, sr=sr)
        plt.title("Raw Audio Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
    elif kind == "fourier" or kind == "fft":
        freqs = np.fft.fftfreq(len(data), 1/sr) if sr else np.arange(len(data))
        plt.plot(freqs[:len(data)//2], np.abs(data)[:len(data)//2])
        plt.title("{} Spectrum".format(kind.upper()))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
    elif kind == "stft":
        stft_db = librosa.amplitude_to_db(np.abs(data), ref=np.max)
        librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
        plt.title("Short-Time Fourier Transform (STFT)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(format="%+2.0f dB")
    elif kind == "mel":
        mel_db = librosa.power_to_db(data, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
        plt.title("Mel-Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Mel Frequency")
        plt.colorbar(format="%+2.0f dB")
    elif kind == "wavelet":
        # Only plot approximation coefficients (cA)
        cA = data[0]
        plt.plot(cA)
        plt.title("Wavelet Transform (Approximation Coefficients)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
    elif kind == "mfcc":
        librosa.display.specshow(data, sr=sr, x_axis='time', cmap='coolwarm')
        plt.title("MFCCs")
        plt.xlabel("Time (s)")
        plt.ylabel("MFCC Coefficient Index")
        plt.colorbar()
    else:
        plt.plot(data)
        plt.title(kind)
    plt.tight_layout()
    plt.show()

# 1. Plot raw waveform
plot_feature_result(signal, sr, kind="waveform")




#2. Fourier Transform
def extract_fourier_transform(signal):
    """
    Extract frequency components of the signal using the Fourier Transform (Discrete Fourier Transform, DFT).

    Args:
        signal (np.ndarray): Audio signal.

    Returns:
        np.ndarray: Complex frequency spectrum.
    """
    spectrum = np.fft.fft(signal)
    return spectrum

fourier_spectrum = extract_fourier_transform(signal)
plot_feature_result(fourier_spectrum, sr, kind="fourier")

#3. Fast Fourier Transform (FFT)
def extract_fft(signal):
    """
    Compute the frequency components of the signal using Fast Fourier Transform (FFT).
    (np.fft.fft is based on the FFT algorithm)

    Args:
        signal (np.ndarray): Audio signal.

    Returns:
        np.ndarray: Complex frequency spectrum.
    """
    fft_spectrum = np.fft.fft(signal)
    return fft_spectrum

fft_spectrum = extract_fft(signal)
plot_feature_result(fft_spectrum, sr, kind="fft")


#4. Short-Time Fourier Transform (STFT)
def extract_stft(signal, sr, n_fft=2048, hop_length=512):
    """
    Perform Short-Time Fourier Transform (STFT) to obtain a time-frequency representation.
    The signal is divided into short segments and Fourier Transform is applied to each segment.

    Args:
        signal (np.ndarray): Audio signal.
        sr (int): Sampling rate (Hz).
        n_fft (int, optional): FFT window size. Default is 2048.
        hop_length (int, optional): Hop length between windows. Default is 512.

    Returns:
        np.ndarray: Complex STFT matrix.
    """
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    return stft

stft_result = extract_stft(signal, sr)
plot_feature_result(stft_result, sr, kind="stft")


#5. Mel-Spectrogram
def extract_mel_spectrogram(signal, sr, n_fft=2048, hop_length=512, n_mels=128):
    """
    Compute the Mel-Spectrogram of the signal.
    Converts the STFT result to the Mel scale to reflect human auditory characteristics.

    Args:
        signal (np.ndarray): Audio signal.
        sr (int): Sampling rate (Hz).
        n_fft (int, optional): FFT window size. Default is 2048.
        hop_length (int, optional): Hop length between windows. Default is 512.
        n_mels (int, optional): Number of Mel bands. Default is 128.

    Returns:
        np.ndarray: Mel-Spectrogram (before dB conversion).
    """
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return mel_spec

mel_spec = extract_mel_spectrogram(signal, sr)
plot_feature_result(mel_spec, sr, kind="mel")


#6. Wavelet Transform
def extract_wavelet_transform(signal, wavelet='db4', level=4):
    """
    Extract time-frequency information of the signal at multiple resolutions using Wavelet Transform.
    Uses multi-level DWT (Discrete Wavelet Transform) from the pywt library.

    Args:
        signal (np.ndarray): Audio signal.
        wavelet (str, optional): Type of wavelet. Default is 'db4'.
        level (int, optional): Decomposition level. Default is 4.

    Returns:
        list[np.ndarray]: List of (cA, cD) coefficients.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs

wavelet_coeffs = extract_wavelet_transform(signal)
plot_feature_result(wavelet_coeffs, sr, kind="wavelet")

#7. MFCCs
def extract_mfccs(signal, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCCs), which are representative features for timbre and characteristics of audio signals.

    Args:
        signal (np.ndarray): Audio signal.
        sr (int): Sampling rate (Hz).
        n_mfcc (int, optional): Number of MFCC coefficients. Default is 13.
        n_fft (int, optional): FFT window size. Default is 2048.
        hop_length (int, optional): Hop length between windows. Default is 512.

    Returns:
        np.ndarray: MFCC coefficient matrix.
    """
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs

mfccs = extract_mfccs(signal, sr)
plot_feature_result(mfccs, sr, kind="mfcc")



