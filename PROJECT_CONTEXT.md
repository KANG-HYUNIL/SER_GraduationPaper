# SER Graduation Paper Context

## Purpose
This document is the canonical context note for the `SER_GraduationPaper` project. It summarizes the current thesis draft, implemented code, experiment pipeline, and the main mismatches between the thesis narrative and the actual codebase.

## Project Identity
- Topic: Speech Emotion Recognition (SER) for human-computer interaction.
- Thesis language: Chinese main thesis template from Beijing Institute of Technology.
- Core dataset in code: RAVDESS.
- Current code focus: log-Mel based 2D CNN comparison with attention variants and lightweight CNN backbones.

## Thesis Status From `main.pdf`
- PDF read from: `LateX_Paper/undergraduate-thesis/undergraduate-thesis/main.pdf`
- Cover date: 2026-04-10.
- Chapter completion status:
  - Abstract: still template placeholder text.
  - Chapter 1: substantially written.
  - Chapter 2: substantially written.
  - Chapter 3: substantially written.
  - Chapter 4: only section headers exist; content is not written yet.
  - Conclusion: still template placeholder text.
  - Acknowledgements: still template placeholder text.
- Thesis framing:
  - Research background emphasizes affective computing and emotion-aware human-computer interaction.
  - Speech is treated as a low-cost signal carrying both semantic and non-semantic emotional cues.
  - Literature review covers classical ML, CNN/RNN/LSTM, attention, Transformer, noisy settings, low-resource settings, and cross-corpus generalization.
  - Chapter 2 discusses audio preprocessing, time-frequency analysis, perceptual features, variable-length handling, noise, and cross-speaker/cross-corpus issues.
  - Chapter 3 explains baseline neural models, attention mechanisms, and Transformer structure in detail.
  - Chapter 4 is planned around:
    - comprehensive experiments
    - noise-condition experiments
    - cross-corpus related experiments
    - finer-grained analysis

## Actual Implemented Code Summary

### End-to-End Pipeline
1. Load RAVDESS `.wav` files.
2. Parse emotion label and actor ID from filename.
3. Preprocess waveform into log-Mel spectrogram.
4. Resize spectrogram to fixed `128 x 512`.
5. Train one of several CNN-family classifiers.
6. Use `GroupKFold` by actor ID for speaker-independent validation.
7. Log metrics and artifacts to MLflow.
8. Export the best fold model for inference.

### Data Pipeline
- File: `src/data/dataset.py`
- Dataset class: `RavdessDataset`
- Label mapping:
  - `01 neutral`
  - `02 calm`
  - `03 happy`
  - `04 sad`
  - `05 angry`
  - `06 fearful`
  - `07 disgust`
  - `08 surprised`
- Actor ID is extracted from filename and used as `groups` in cross-validation.
- Audio loading uses `soundfile`, not `torchaudio`, inside `__getitem__`.

- File: `src/data/transforms.py`
- Class: `AudioPipeline`
- Processing steps:
  - resample to `16 kHz`
  - mix to mono
  - Mel spectrogram
  - amplitude to dB
  - instance-style normalization
  - bicubic resize to `(128, 512)`
- Important consequence:
  - Although comments mention variable-length handling, the current pipeline resizes every sample to a fixed shape.
  - This makes later dynamic padding logic conceptually unnecessary.

### Training and Evaluation
- File: `src/train.py`
- Cross-validation:
  - `GroupKFold(n_splits=5)`
  - groups = actor IDs
- Optimizer: Adam.
- Loss: CrossEntropyLoss.
- Logged artifacts:
  - learning curves
  - confusion matrix
  - class-wise accuracy plot
  - t-SNE plot
- Exports:
  - fold checkpoints to `weights/`
  - best demo model to `saved_models/best_model_{model.name}.pt`

### Inference
- File: `src/infer.py`
- Loads either:
  - checkpoint from config, or
  - fallback checkpoint from `saved_models/best_model_{model.name}.pt`
- Performs manual preprocessing with `librosa` and `torchaudio.transforms`.
- Outputs CSV into `inference_outputs/<timestamp>/inference_results.csv`

### Sanity/Verification Entry Point
- File: `src/main.py`
- Purpose:
  - load config
  - instantiate dataset and processor
  - instantiate model through registry
  - run one forward pass sanity check

### Visualization Helper
- File: `src/visualize_mel.py`
- Saves example transformed log-Mel images for inspection.

## Implemented Models

### `cnn_baseline`
- File: `src/models/base.py`
- 4 convolution blocks:
  - Conv2d -> BatchNorm -> ReLU -> MaxPool
- Optional spatial attention flag exists in code path.
- Adaptive average pooling to `4 x 4`.
- Linear classifier to 8 classes.

### `cnn_channel_attention`
- File: `src/models/channel_attention.py`
- Adds SE-style channel attention after each conv block.
- Final pooling to `4 x 4`, then linear classification.

### `cnn_spatial_attention`
- File: `src/models/spatial_attention.py`
- Adds CBAM-like spatial attention after each conv block.
- Final classifier uses global average pooling to `1 x 1`.

### `cnn_temporal_attention`
- File: `src/models/temporal_attention.py`
- CNN feature extractor followed by:
  - frequency pooling
  - 1D attention across time frames
  - weighted temporal context vector
  - final linear classification
- This is the closest implementation to a model that explicitly weights emotionally salient time regions.

### `mobilenet_v3_small`
- File: `src/models/mobilenetv3.py`
- Uses torchvision MobileNetV3 Small.
- First conv changed to single-channel input.
- Final classifier changed to 8 emotion classes.
- `weights=None`, so it is not using ImageNet pretrained weights.

### `efficientnet_lite0`
- File: `src/models/efficientnet_lite.py`
- Custom EfficientNet-Lite0 style implementation.
- Single-channel input, final classifier outputs 8 classes.

## Registry and Utility Modules
- `src/utils/registry.py`
  - simple decorator-based model registry
- `src/utils/metrics_eval.py`
  - accuracy, macro/weighted F1, UAR, WAR, MCC, Cohen's kappa, ECE
- `src/utils/metrics_stat.py`
  - McNemar significance test
- `src/utils/viz_curves.py`
  - calibration, ROC/PR, learning curves
- `src/utils/viz_heatmaps.py`
  - confusion matrix and attention overlay
- `src/utils/viz_embeddings.py`
  - t-SNE embedding plots
- `src/utils/viz_optuna.py`
  - Optuna visualization helpers

## Active Configuration Snapshot
- Main config: `src/configs/config.yaml`
- Current defaults:
  - data config: `default`
  - model config: `cnn_baseline`
- Training defaults:
  - seed `42`
  - batch size `32`
  - epochs `1`
  - learning rate `1e-4`
  - k-folds `5`
  - early stopping `10`
- Data config:
  - dataset path: `${hydra:runtime.cwd}/src/$RVNS6MQ`
  - sample rate: `16000`
  - duration: `3.0`
  - n_mels: `128`
  - n_fft: `1024`
  - hop_length: `512`
  - normalize: `true`
  - resize to `128 x 512`

## Critical Reality Check: Thesis vs Code

### What the thesis currently claims or emphasizes
- Strong attention to Transformer theory and related work.
- Planned experiments include noisy settings and cross-corpus settings.
- Methodological framing suggests model comparison beyond simple CNN baselines.

### What the code actually implements now
- RAVDESS-only pipeline.
- CNN-family baselines plus attention variants plus lightweight CNN backbones.
- No implemented Transformer model in `src/models`.
- No implemented cross-corpus training/evaluation pipeline.
- No implemented noise augmentation or explicit noisy-condition experiment pipeline.

### Practical interpretation
- The current codebase supports a thesis centered on:
  - RAVDESS-based SER
  - log-Mel spectrogram preprocessing
  - actor-independent evaluation with GroupKFold
  - comparison among baseline CNN, channel attention, spatial attention, temporal attention, MobileNetV3, and EfficientNet-Lite
- The current codebase does not yet support a thesis whose experimental core depends on:
  - Transformer results
  - DWFormer/Swin-Transformer reproduction
  - cross-corpus experiments
  - noise robustness experiments

## Important Code Issues / Risks
- `src/train.py` imports `collate_dynamic_padding`, but `src/data/dataset.py` only contains a commented-out stub.
  - This likely causes runtime failure unless another local version exists outside the tracked code.
- `src/data/transforms.py` already resizes every sample to a fixed size.
  - This conflicts with the need for dynamic padding.
- `src/infer.py` preprocessing is not fully shared with `AudioPipeline`.
  - Training and inference preprocessing may drift.
- `src/configs/data/default.yaml` points to `src/$RVNS6MQ`.
  - This looks like a temporary or copied dataset directory under `src`, not a clean dataset root.
- `train.epochs` is currently set to `1`.
  - This looks like a debug setting, not a real experiment setting.
- Several comments contain broken encoding artifacts.
  - Not functionally critical, but they reduce maintainability.

## Thesis Writing Guidance Based On Current Code
- If you continue writing immediately without major code changes, Chapter 4 should describe:
  - RAVDESS dataset only
  - log-Mel preprocessing and resize strategy
  - GroupKFold split by speaker/actor
  - baseline CNN vs attention variants vs lightweight CNN backbones
  - evaluation with accuracy, macro F1, confusion matrix, class-wise analysis, t-SNE
- If you want Chapter 3 and Chapter 4 to strongly feature Transformer experiments, you need to implement them first.
- If you want to keep the current code scope, reduce the thesis claim from "CNN vs Transformer" to "CNN baseline and attention-based/local lightweight architecture comparison under limited data."

## Key Files To Reopen Quickly
- Thesis PDF: `LateX_Paper/undergraduate-thesis/undergraduate-thesis/main.pdf`
- Training: `src/train.py`
- Inference: `src/infer.py`
- Dataset: `src/data/dataset.py`
- Preprocessing: `src/data/transforms.py`
- Baseline CNN: `src/models/base.py`
- Temporal attention: `src/models/temporal_attention.py`
- Config root: `src/configs/config.yaml`

## One-Paragraph Executive Summary
This project is currently a RAVDESS-based speech emotion recognition system built around log-Mel spectrogram inputs and several CNN-family classifiers, especially baseline CNN and channel/spatial/temporal attention variants. The thesis draft already contains substantial theoretical writing on SER, attention, and Transformer models, but the implemented experiments do not yet include Transformer, cross-corpus, or noise-robustness pipelines. The most accurate current framing is therefore a speaker-independent RAVDESS SER study comparing CNN and attention-based architectures under limited-data conditions, with Chapter 4 still needing to be written around the code that actually exists.
