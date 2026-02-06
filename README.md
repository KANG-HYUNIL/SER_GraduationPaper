# SER_GraduationPaper

## Notion Link
I wrote record of each task in my notion page
https://www.notion.so/Task-2d4f1f0daf36809aa0b3ea959040b44b?source=copy_link


# SER Graduation Paper Project (Kaiti)

## 📌 Project Overview
**Speech Emotion Recognition (SER)** system using **RAVDESS** dataset.
This project aims to propose a lightweight yet robust model for classifying 8 emotions from speech data, focusing on **Subject-Independent** evaluation using Group K-Fold Cross Validation.

The core contribution is comparing a strong **CNN Baseline** with **Attention-based models** (Temporal Attention & Channel Attention) to demonstrate interpretability and performance improvements on small datasets.

---

## 📂 Directory Structure

```bash
SER_GraduationPaper/
├── .agent/              # Agent workflows (if any)
├── src/
│   ├── configs/         # Hydra Configuration
│   │   ├── config.yaml  # Main config
│   │   ├── data/        # Data params
│   │   └── model/       # Model architectures
│   ├── data/            # Dataset & Transforms
│   │   ├── dataset.py   # RavdessDataset (Actor ID parsing)
│   │   └── transforms.py# Log-Mel Spectrogram Pipeline
│   ├── models/          # Neural Networks
│   │   ├── base.py      # Baseline CNN
│   │   ├── temporal_attention.py
│   │   └── channel_attention.py
│   ├── utils/           # Helpers
│   │   └── registry.py  # Model Registry Pattern
│   ├── train.py         # Training Loop (Group K-Fold)
│   └── main.py          # Verification & Entry point
├── outputs/             # Saved Models & Logs
└── requirements.txt     # Dependencies
```

---

## 📊 Dataset: RAVDESS
**Ryerson Audio-Visual Database of Emotional Speech and Song**
- **Type**: Speech Audio (.wav) at 16kHz
- **Input**: Log-Mel Spectrogram (128 mel-bins)
- **Classes**: 8 Emotions (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised)
- **Subjects**: 24 Actors (12 Male, 12 Female)
- **Evaluation Strategy**: **Group K-Fold (k=5)**
    - Ensures **Subject Independence** (Train set and Test set have different actors).
    - Prevents speaker bias (overfitting to specific voice characteristics).

---

## 🏗️ Model Architectures

### 1. Baseline: VGG-Style CNN
- **Structure**: 4 Blocks of `[Conv2d -> BN -> ReLU -> MaxPool]`.
- **Feature Extraction**: Hierarchical spatial features from Spectrogram.
- **Pooling**: AdaptiveAvgPool (Features -> Vector)
- **Classifier**: Linear Layer (Vector -> 8 Class Logits).

### 2. Proposed: CNN + Attention

#### A. Temporal Attention (Time-Axis)
- **Goal**: Identify "Which time segment has the strongest emotion?"
- **Mechanism**:
    1. Extract features `(C, F, T)` from CNN.
    2. Squeeze Frequency: `(C, T)`.
    3. **Query Network**: Computes scalar score for each time step `T`.
    4. **Softmax**: Normalizes scores to sum to 1.
    5. **Weighted Sum**: Context Vector = $\sum (Score_t \times Feature_t)$.
- **Benefit**: Focuses on emotional bursts, ignoring silence.

#### B. Channel Attention (SE-Block)
- **Goal**: "Which frequency pattern (Filter) is important?"
- **Mechanism (Squeeze-and-Excitation)**:
    1. **Squeeze**: Global Avg Pool to get `(C)` channel descriptor.
    2. **Excitation**: Reduced-dimension MLP learns channel importance weights `(0~1)`.
    3. **Scale**: Multiply original features by these weights.
- **Benefit**: Enhances key feature maps while suppressing noise.

---

## 📈 Evaluation Metrics
This project uses **Subject-Independent Group K-Fold (k=5)**. The reported results are the **average** of 5 folds.

1.  **Accuracy (ACC)**: Percentage of correctly predicted samples.
2.  **Macro-F1 Score**: Harmonic mean of Precision and Recall, averaged across all 8 classes (Robust to class imbalance).
3.  **Confusion Matrix**: Visualizes misclassification patterns (e.g., "Sad" vs "Calm"). Aggregated from all 5 folds to analyze global performance.

---

## 🚀 Usage

### 1. Installation
```bash
# Create Environment
conda create -n grad_paper_ser python=3.9
conda activate grad_paper_ser

# Install Dependencies
pip install -r requirements.txt
```

### 2. Configuration (`src/configs`)
Hydra is used for configuration management. You can override parameters via CLI.

## 🏃 Code Structure: Train vs Main

| Script | Role (역할) | Command | Details |
| :--- | :--- | :--- | :--- |
| **`src/main.py`** | **Verification (검증)** | `python src/main.py model=...` | 파이프라인(Data -> Model -> Output)이 에러 없이 연결되는지 **1회 실행(Smoke Test)**으로 확인. |
| **`src/train.py`** | **Training (실학습)** | `python src/train.py model=...` | **Group K-Fold** 전체 수행, **모델 저장**, **시각화**, **MLFlow 리포트** 등 모든 학습 과정 자동화. |

---

## 🎨 Visualization (Automated)
`src/train.py` runs generate high-quality plots in `outputs/` and MLFlow:

1.  **Learning Curves**: Trace `Train/Val Loss` & `Accuracy` per epoch to detect Overfitting/Underfitting.
2.  **Global Confusion Matrix**: Aggregated results from all 5 folds to analyze classification errors.
3.  **Class-wise Accuracy**: Bar chart showing which emotions are recognized best.
4.  **t-SNE Projection**: 2D scatter plot of the **High-dimensional Feature Space** (Output of CNN before Classifier).
    *   **Goal**: Visually confirm if the model essentially separates emotions into distinct clusters.

---

## ⚙️ Configuration (`src/configs/config.yaml`)

Manage all hyperparameters without changing code.
File: `src/configs/config.yaml`

```yaml
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S} # Auto-create unique folder per run

train:
  seed: 42                # Reproducibility
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  k_folds: 5              # Group K-Fold (Split by Actor ID)
  early_stopping: 10      # Stop if no improvement for 10 epochs
  device: "auto"          # auto / cuda / cpu
```

---

## 🚀 Usage

### 1. Installation
```bash
# Create Environment
conda create -n grad_paper_ser python=3.9
conda activate grad_paper_ser

# Install Dependencies
pip install -r requirements.txt
```

### 2. Training (Group K-Fold)
This command runs K-Fold training and saves artifacts to `outputs/YYYY-MM-DD/HH-MM-SS/`.
```bash
# Baseline
python src/train.py model=cnn_baseline

# Temporal Attention
python src/train.py model=cnn_temporal_attention

# Channel Attention
python src/train.py model=cnn_channel_attention
```

### 3. Inference / Demo
To use the best trained model for prediction:
```bash
# (Required) Must provide valid audio path
python src/infer.py audio_path="data/test_audio.wav"
```
*(See `saved_models/` for the best model from the latest training run.)*

### 4. Logging (MLFlow)
Metrics and artifacts (Confusion Matrix) are saved in `./mlruns` directory.
Run the UI to visualize:
```bash
mlflow ui
```

---

## 🔍 Experiments & Results
*(To be updated after running full experiments)*

| Model | Avg Accuracy | Avg F1-Score | Best Fold |
| :--- | :--- | :--- | :--- |
| **Baseline** | TB W | TB W | Fold X |
| **Temporal Attn** | TB W | TB W | Fold Y |
| **Channel Attn** | TB W | TB W | Fold Z |

> **Note**: "TB W" = To Be Written. Requires full training run.
