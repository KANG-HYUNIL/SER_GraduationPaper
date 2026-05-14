# Feedback Diagram Reference Summary

All figures in this folder are redrawn/generated diagrams. No external paper figure was copied into the thesis PDF.

## Chapter 1

- `chapter1/ser_evolution/ser_evolution_timeline.pdf`
  - Zhang and Song, 2020, `zhang2020transfer`: traditional acoustic feature and transfer/subspace-learning pipeline.
  - Peng et al., 2021, `peng2021efficient`: multi-scale CNN and attention SER architecture.
  - Mirsamadi et al., 2017, `mirsamadi2017automatic`: RNN with local attention for utterance-level SER.
  - Liu et al., 2023, `liu2023dualrobustness`: Dual-Transformer-BiLSTM hybrid SER trend.
  - Vaswani et al., 2017, `vaswani2017attention`: original Transformer encoder-decoder architecture.
- `chapter1/acoustic_features/acoustic_feature_taxonomy.pdf`
  - SciPy spectrogram documentation: nonstationary signal frequency-content visualization.
  - librosa melspectrogram documentation: power spectrogram to mel basis and dB/log display.
  - `zhang2020transfer` and review citations already present in `ref.bib`: MFCC/eGeMAPS/traditional acoustic feature usage.

## Chapter 2

- `chapter2/fourier/ft_time_to_frequency.pdf`
  - Generated from Chapter 2 DFT formula and standard Fourier-domain interpretation.
- `chapter2/stft/ft_vs_stft.pdf`
  - SciPy spectrogram/STFT documentation: framewise Fourier analysis for time-varying frequency content.
- `chapter2/spectrogram_forms/spectrogram_forms.pdf`
  - SciPy spectrogram documentation: magnitude/power spectral representations.
  - librosa documentation: power-to-dB/log display conventions.
- `chapter2/logmel/mel_vs_logmel.pdf`
  - librosa melspectrogram documentation: STFT power spectrum mapped to mel basis.
  - librosa power-to-dB example: log/dB compression after mel projection.

## Chapter 3

- `chapter3/mlp/mlp_neural_network.pdf`
  - Generated from Chapter 3 MLP equations and standard fully connected neural network notation.
- `chapter3/cnn/cnn_local_feature_extraction.pdf`
  - `peng2021efficient`: CNN-based local acoustic representation and attention context.
- `chapter3/rnn/rnn_sequence_context.pdf`
  - `mirsamadi2017automatic`: RNN hidden states and local attention pooling.
  - `liu2023dualrobustness`: BiLSTM in hybrid SER model.
- `chapter3/attention/attention_qkv_weighting.pdf`
  - `vaswani2017attention`: scaled dot-product attention.
  - `mirsamadi2017automatic`: attention as salient-region weighting in SER.
- `chapter3/transformer/transformer_full_architecture.pdf`
  - `vaswani2017attention`, Figure 1 structure concept: encoder-decoder, self-attention, cross-attention, FFN.
  - Chapter 3 thesis text: SER-specific encoder pooling/classification path.

## Web Sources Consulted

- ResearchGate page for Peng et al., Efficient SER Using Multi-Scale CNN and Attention: https://www.researchgate.net/publication/352244759_Efficient_Speech_Emotion_Recognition_Using_Multi-Scale_CNN_and_Attention
- Microsoft Research page for Mirsamadi et al., Automatic SER Using RNNs with Local Attention: https://www.microsoft.com/en-us/research/video/automatic-speech-emotion-recognition-using-recurrent-neural-networks-local-attention/
- ResearchGate page for Zhang and Song, Transfer Sparse Discriminant Subspace Learning for Cross-Corpus SER: https://www.researchgate.net/publication/337453920_Transfer_Sparse_Discriminant_Subspace_Learning_for_Cross-Corpus_Speech_Emotion_Recognition
- CiNii page for Dual-TBNet: https://cir.nii.ac.jp/crid/1873679867843956608
- NeurIPS paper page for Attention Is All You Need: https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
- Google Research page for Conformer: https://research.google/pubs/conformer-convolution-augmented-transformer-for-speech-recognition/
- SciPy spectrogram documentation: https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.signal.spectrogram.html
- librosa melspectrogram documentation: https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
