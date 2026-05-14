# Sources
- librosa.feature.melspectrogram documentation: maps a power spectrogram to mel basis.
- librosa.power_to_db examples: converts mel spectrogram coefficients to dB scale for visualization.

## Manual rebuild draft

- `mel_vs_logmel_rebuild_zh.drawio`
  - Built as editable draw.io XML.
  - Visual target: STFT power spectrogram -> mel filter bank -> mel spectrogram -> log/dB compression -> log-Mel spectrogram.
  - Reference patterns: librosa `melspectrogram`, librosa `power_to_db`, Keras MelSpectrogram layer, and Hugging Face Audio Course explanations of log-mel spectrograms.
