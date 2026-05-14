# Sources
- SciPy signal.spectrogram documentation: supports magnitude and power spectral density modes.
- librosa power_to_db and melspectrogram documentation: power spectrograms are commonly converted to dB/log scale for display and modeling.

## Manual rebuild draft

- `spectrogram_forms_rebuild_zh.drawio`
  - Built as editable draw.io XML.
  - Visual target: one STFT result branching into magnitude spectrogram, power spectrogram, and log/dB spectrogram.
  - Reference patterns: SciPy `spectrogram` modes, SciPy ShortTimeFFT spectrogram definition, librosa `power_to_db` dynamic-range compression convention.
