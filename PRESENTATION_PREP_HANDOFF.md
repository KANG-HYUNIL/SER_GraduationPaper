# 발표 증명 작업 인수인계 문서 (Agent Handoff)

> **작성일**: 2026-05-21  
> **목적**: 다른 Agent가 이 문서만 읽고도 발표 준비 작업(실험 재현 스크립트 구축)을 시작할 수 있도록 배경, 현황, 작업 목표, 파일 위치, 주의사항을 모두 기술한다.

---

## 1. 배경 및 목적

### 1.0 2026-05-21 재검증 업데이트

이 문서의 이전 버전에는 "단발성 훈련으로 재현"이라는 표현이 섞여 있었으나, 발표 요구와 현재 사용자의 목표는 **이미 존재하는 best checkpoint로 실제 test fold inference/evaluation을 즉석 실행해 결과 출처를 증명하는 것**이다. 따라서 후속 작업의 기준은 다음과 같이 정정한다.

- 훈련 재실행은 2분 발표 시연 목적에 맞지 않으므로 기본 목표에서 제외한다.
- `cnn_baseline`, `pure_transformer`, `cnn_conformer`는 기존 best checkpoint와 각 발표용 script 내부에 직접 적은 실험 설정값으로 RAVDESS fold 1 validation/test split만 평가한다.
- `noise robustness`는 기존 CNN-Conformer trial_0004 checkpoint로 노이즈 조건별 inference/evaluation만 재실행한다.
- `cross corpus`는 기존 6-class CNN-Conformer checkpoint(`artifacts/fold_1/best_model.pt`)로 source validation 및 CREMA-D target evaluation을 재실행하는 방향이 맞다. 단, 현재 로컬에는 `src/CREMA-D`가 없어 실행 가능 여부는 데이터 복구에 의존한다.
- source of truth 우선순위는 `outputs/*/resolved_config.yaml`, `trial_summary.json`, `summary_metrics.json`, 실제 checkpoint 파일, 관련 docs 순서로 둔다.

### 1.1 상황 요약

이 프로젝트는 Speech Emotion Recognition (SER) 학사 졸업논문 실험 코드다. 논문 Chapter 4에 CNN Baseline / Pure Transformer / CNN-Conformer 모델 비교 실험 결과, 노이즈 실험 결과, 크로스 코퍼스 실험 결과가 숫자로 기재되어 있다. 발표 심사에서 "그 숫자들이 진짜인가?"라는 질문에 대비해, **논문에 기재된 수치들을 직접 재현하여 보여줄 수 있는 환경**을 준비해야 한다.

### 1.2 발표에서 보여줄 것

1. 미리 훈련한 `.pt` checkpoint를 로드한다.
2. 해당 checkpoint의 원본 실험 설정과 동일한 전처리/모델 설정을 script 내부 상수로 적용한다.
3. 원래 실험과 같은 RAVDESS fold 1 validation/test split 또는 cross-corpus target set에서 inference/evaluation을 즉석 실행한다.
4. outputs/ 경로의 원본 결과 파일(metrics, confusion matrix 등)과 새 실행 결과가 같은 출처와 절차에서 나온 것임을 보여준다.

### 1.3 핵심 제약

- 모든 결과는 **fold 1 기준** (5-fold 중 1개만 실행)
- 실험은 Optuna 탐색이나 재훈련이 아니라 **기존 checkpoint 기반 inference/evaluation**으로 재현
- Hydra 설정 관리 방식 유지
- outputs/ 경로에 새 inference/evaluation 결과 파일 저장
- checkpoint는 기존 파일을 직접 참조한다. 필요 시 발표 편의를 위해 복사본을 만들 수 있으나, 원본 경로를 항상 함께 기록한다.

---

## 2. 논문에 기재된 실험 결과 (증명 대상)

### 2.1 모델 비교 결과 (fold 1 기준)

| 실험 축 | 모델 | Accuracy | Macro-F1 | UAR | 파라미터수 |
|---|---|---|---|---|---|
| CNN Baseline | cnn_baseline | 0.61667 | 0.62196 | 0.61563 | 1.41M |
| Pure Transformer | pure_transformer | 0.52000 | 0.51163 | 0.51250 | 4.21M |
| CNN-Conformer (main) | cnn_conformer | 0.70000 | 0.70563 | 0.70938 | 3.55M |

> ⚠️ 논문에서의 CNN-Conformer 수치는 `outputs/2026-04-21/18-47-36_cnn_conformer/optuna_trials/trial_0003`의 결과 (F1=0.70563, Acc=0.70000)

### 2.2 노이즈 실험 결과 (fold 1 기준, 재훈련 없이 inference/evaluation만)

- **기준 checkpoint**: `outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/artifacts/weights/best_model_fold1.pt`
- clean 기준: Accuracy=0.70333, F1=0.70042
- 가장 취약: pink noise -5dB → Accuracy=0.15000
- 가장 강건: babble noise -5dB → Accuracy=0.44333

> ⚠️ 노이즈 실험은 trial_0004(Acc=0.70333)의 checkpoint를 사용. trial_0003(F1=0.70563)의 checkpoint는 trial 폴더의 `artifacts/weights/`에는 없지만, run root의 `outputs/2026-04-21/18-47-36_cnn_conformer/weights/best_model_fold1.pt`에 존재한다.

### 2.3 크로스 코퍼스 결과 (RAVDESS→CREMA-D, fold 1 기준)

| 평가 집합 | Accuracy | Macro-F1 | UAR | ECE |
|---|---|---|---|---|
| RAVDESS 소스 검증셋 | 0.58182 | 0.56897 | 0.58333 | 0.22191 |
| CREMA-D 타겟 | 0.19377 | 0.09243 | 0.18936 | 0.56109 |

- **출처**: `outputs/2026-04-30/18-10-12_cross_corpus_cremad6/`
- 원래 실험 방식: CNN-Conformer 기본 config를 num_classes=6으로 바꿔 RAVDESS 6-class source에서 재훈련 후 CREMA-D target 평가
- 발표 재현 방식: 이미 남아 있는 `artifacts/fold_1/best_model.pt` checkpoint로 source/target evaluation만 재실행

---

## 3. 프로젝트 파일 구조 및 각 파일의 역할

### 3.1 루트 경로

```
c:\Users\hik88\Desktop\BIT_Uni\GraduationPaper\Project\SER_GraduationPaper\
```

이하 모든 경로는 이 루트 기준 상대 경로로 표기한다.

### 3.2 핵심 소스 파일

| 경로 | 역할 |
|---|---|
| `src/train.py` | 단발성 훈련 엔트리포인트 (Hydra 기반, k-fold CV 실행) |
| `src/optuna_search.py` | Optuna 탐색 엔트리포인트 (재현 작업에는 사용하지 않음) |
| `src/evaluate_noise_robustness.py` | 노이즈 실험 평가 스크립트 (재훈련 없이 checkpoint로만 실행) |
| `src/cross_corpus_eval.py` | 크로스 코퍼스 평가 스크립트 |
| `src/engine/trainer.py` | 훈련 루프, CV 실행, weights 저장, artifact 생성 |
| `src/models/cnn_conformer.py` | CNN-Conformer 모델 구현 |
| `src/models/pure_transformer.py` | Pure Transformer 모델 구현 |

### 3.3 Hydra 설정 파일

| 경로 | 역할 |
|---|---|
| `src/configs/config.yaml` | 메인 설정 (기본 model: cnn_baseline, epochs: 30, seed: 42) |
| `src/configs/data/default.yaml` | 데이터셋 경로, 오디오 파라미터 기본값 |
| `src/configs/model/cnn_baseline.yaml` | CNN Baseline 기본 config |
| `src/configs/model/pure_transformer.yaml` | Pure Transformer 기본 config |
| `src/configs/model/cnn_conformer.yaml` | CNN-Conformer 기본 config |
| `src/configs/noise/default.yaml` | 노이즈 실험 설정 |
| `src/configs/cross_corpus/default.yaml` | 크로스 코퍼스 실험 설정 |

### 3.4 실험 기록 문서 (docs/)

| 경로 | 내용 |
|---|---|
| `docs/KR_EXPERIMENT_FLOW_OVERVIEW.md` | 전체 실험 타임라인 (2026-04-14~04-22) |
| `docs/KR_MODELS_CNN_BASELINE.md` | CNN Baseline Optuna 결과 및 winner 설정 |
| `docs/KR_MODEL_PURE_TRANSFORMER.md` | Pure Transformer 실험 기록 및 winner 설정 |
| `docs/KR_MODEL_CNN_CONFORMER.md` | CNN-Conformer 메인 문서 (winner 설정 포함) |
| `docs/cnn_conformer/2026-04-22_overfitting_followup.md` | **CNN-Conformer F1=0.70563 trial_0003 결과** (핵심) |
| `docs/noise_robustness/KR_NOISE_ROBUSTNESS_EXPERIMENT_PLAN.md` | **노이즈 실험 설계 + 전체 결과표 + 사용한 checkpoint 경로** |
| `docs/cross_corpus/2026-04-30_RAVDESS_to_CREMAD_6class.md` | **크로스 코퍼스 실험 결과 기록** |

### 3.5 실험 outputs 현황

| 날짜 폴더 | 내용 |
|---|---|
| `outputs/2026-04-14/04-49-31_cnn_optuna_stage1_baselineTest/` | CNN Baseline Optuna (winner: trial_0023, F1=0.62196) |
| `outputs/2026-04-15/13-44-11_thesis_transformer_stage2_pure_transformer/` | Pure Transformer 실험 |
| `outputs/2026-04-21/18-47-36_cnn_conformer/optuna_trials/trial_0003/` | **CNN-Conformer F1=0.70563 winner** (`weights/best_model_fold1.pt`가 run root에 존재) |
| `outputs/2026-04-21/18-47-36_cnn_conformer/weights/best_model_fold1.pt` | **논문 최고점 CNN-Conformer checkpoint** (12.35MB, 존재 확인됨) |
| `outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/` | **노이즈 실험 기준 checkpoint** (Acc=0.70333) |
| `outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/artifacts/weights/best_model_fold1.pt` | **노이즈 실험에 실제 사용한 checkpoint 파일** (14.2MB, 존재 확인됨) |
| `outputs/2026-04-23/15-33-32_noise_eval_winner/` | 노이즈 실험 결과 |
| `outputs/2026-04-30/18-10-12_cross_corpus_cremad6/` | 크로스 코퍼스 실험 결과 |

---

## 4. weights/ 폴더 현황 및 분석

### 4.1 현재 상태

```
weights/
└── best_model_fold5.pt   (6.0 MB)
```

### 4.2 weights 저장 코드 분석 결과

`src/engine/trainer.py` 분석:
- **L649**: `weights_dir = ensure_artifact_dir(Path(artifact_root) / "weights")` → 각 실험의 `artifacts/weights/`에 저장
- **L711**: `best_model_path = weights_dir / f"best_model_fold{fold}.pt"` → fold 번호로 파일명 결정
- **L624~635** `copy_best_model_to_root()`: `save_best_to_root=True`일 때 `saved_models/best_model_{model_name}.pt`에 복사 → **`weights/`가 아니라 `saved_models/`에 저장**

### 4.3 weights/best_model_fold5.pt 출처 추론

- `copy_best_model_to_root`는 `saved_models/` 폴더에 저장하므로, `weights/`와 무관
- `weights/best_model_fold5.pt`는 **프로젝트 루트에서 직접 `train.py`를 실행했을 때** Hydra의 `chdir: false` 또는 manual 실행 상황에서 생성된 것으로 추정
- fold 번호가 5인 점 → 5-fold 전체를 돌렸을 때 마지막 fold 결과
- 파일 크기 6MB → CNN-Conformer(14MB)나 Pure Transformer보다 작음 → **CNN Baseline(1.41M 파라미터)의 fold 5 결과일 가능성이 높음**
- 또는 cross_corpus_eval이 save 구조 다를 수 있음 → 추가 확인 필요

> ❗ 이 파일은 현재 용도가 불분명하여 발표 재현 작업에서는 사용하지 않고, 새로 생성할 것을 권장

---

## 5. 각 실험의 Winner 하이퍼파라미터 (재현에 필요한 정확한 값)

### 5.1 CNN Baseline Winner (docs/KR_MODELS_CNN_BASELINE.md 기준)

```yaml
model: cnn_baseline
  hidden_dims: [32, 64, 256, 512]
  dropout: 0.33238

data:
  n_mels: 80
  n_fft: 1024
  hop_length: 160
  f_min: 0.0
  f_max: 6000.0
  normalize: true
  resize_enabled: true
  resize_height: 96
  resize_width: 512

train:
  learning_rate: 3.4129546471254387e-4
  weight_decay: 1.9338610496754583e-5
  batch_size: 16
  epochs: 30
  early_stopping: 10
  k_folds: 5
  folds_to_run: 1   # fold 1만 실행
  seed: 42
```

### 5.2 Pure Transformer Winner (outputs resolved_config 기준)

```yaml
model: pure_transformer
  embed_dim: 256
  num_layers: 5
  num_heads: 4
  ffn_dim: 1024   # ffn_ratio=4 기준
  patch_size: [32, 32]
  patch_stride: [8, 8]
  pooling: mean
  dropout: 0.2707

data:
  n_mels: 64
  n_fft: 2048
  hop_length: 160
  f_min: 20.0
  f_max: 6000.0
  normalize: true
  resize_enabled: false
  resize_height: 128
  resize_width: 512

train:
  batch_size: 16
  learning_rate: 0.0003897915827154378
  weight_decay: 0.0003121519880319906
  folds_to_run: 1
  seed: 42
```

> ✅ 실제 확인 파일: `outputs/2026-04-15/13-44-11_thesis_transformer_stage2_pure_transformer/optuna_trials/trial_0016/resolved_config.yaml` 및 `trial_summary.json`

### 5.3 CNN-Conformer 논문 최고점 Winner (trial_0003 resolved_config 기준)

```yaml
model: cnn_conformer
  backbone_variant: nostem_patch
  embed_dim: 192
  num_heads: 8
  num_layers: 4
  ffn_dim: 768
  conv_kernel_size: 31
  layer_fusion: last
  pooling: attention
  attention_type: relative
  nostem_patch:
    time_patch: 4
    norm_variant: layernorm
  sequence_shrinking:
    enabled: false
  dropout: 0.1617079225419117
  stem_dropout: 0.08180909155642152
  projector_dropout: 0.09301321323053058
  input_dropout: 0.09554709158757928
  encoder_dropout: 0.1617079225419117
  classifier_dropout: 0.22287375091519293

data:
  n_mels: 80
  n_fft: 1024
  hop_length: 160
  f_min: 0.0
  f_max: 6000.0
  normalize: true
  resize_enabled: false
  chunking:
    enabled: true
    chunk_frames: 48
    hop_frames: 12
    eval_hop_frames: 12
    aggregation_mode: confidence_weighted_logit
    topk_ratio: 0.75

train:
  batch_size: 12
  learning_rate: 9.093618341363527e-05
  weight_decay: 0.00018258230439200242
  mixup:
    enabled: true
    alpha: 0.4
  folds_to_run: 1
  seed: 42
```

> ✅ 논문 표의 최고점 `F1=0.70563 / Acc=0.70000 / UAR=0.70938`은 `outputs/2026-04-21/18-47-36_cnn_conformer/optuna_trials/trial_0003/` 기준이다. 이 trial의 `trial_summary.json`은 `best_model_path: weights\best_model_fold1.pt`를 가리키며, 실제 checkpoint는 `outputs/2026-04-21/18-47-36_cnn_conformer/weights/best_model_fold1.pt`에 존재한다.

### 5.3b CNN-Conformer 노이즈 기준 Winner (trial_0004 resolved_config 기준)

```yaml
checkpoint_path: outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/artifacts/weights/best_model_fold1.pt
resolved_config_path: outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/resolved_config.yaml
clean_metrics:
  accuracy: 0.7033333333333334
  f1_macro: 0.7004160371017489
  uar: 0.7
```

> 이 checkpoint는 trial별 artifacts 아래에 보존되어 있어 노이즈 실험의 재현 가능한 기준점으로 사용되었다.

### 5.4 노이즈 실험 (재훈련 없음, inference only)

```yaml
# 사용할 checkpoint
checkpoint_path: outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/artifacts/weights/best_model_fold1.pt
resolved_config_path: outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/resolved_config.yaml

# 노이즈 그리드
noise_types: [white, pink, babble, cafe]
snr_db: [clean, 20, 10, 5, 0, -5]
```

실행 명령 (docs/noise_robustness/KR_NOISE_ROBUSTNESS_EXPERIMENT_PLAN.md에 기재):
```powershell
.\.venv\Scripts\python.exe -m src.evaluate_noise_robustness `
  model=cnn_conformer `
  noise.eval.enabled=true `
  noise.eval.resolved_config_path=outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/resolved_config.yaml `
  noise.eval.checkpoint_path=outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/artifacts/weights/best_model_fold1.pt `
  noise.eval.noise_types=[white,pink,babble,cafe] `
  noise.eval.snr_db=[clean,20,10,5,0,-5] `
  noise.eval.save_condition_artifacts=true `
  noise.eval.output_dir=noise_eval_winner `
  experiment.name=noise_eval_winner
```

### 5.5 크로스 코퍼스 실험 (RAVDESS 6-class → CREMA-D)

```yaml
# CNN-Conformer 기본 config 재사용, num_classes=6으로 변경
# resolved_config 참조: outputs/2026-04-30/18-10-12_cross_corpus_cremad6/resolved_config.yaml
n_mels: 128
n_fft: 1024
hop_length: 512
folds_to_run: 1
```

실행 명령:
```powershell
.\.venv\Scripts\python.exe -m src.cross_corpus_eval `
  model=cnn_conformer `
  cross_corpus.enabled=true `
  cross_corpus.protocol=ravdess_to_cremad_6class `
  experiment.name=cross_corpus_cremad6 `
  experiment.tag=source_only_6class
```

---

## 6. 작업 목표 (다른 Agent가 수행해야 할 일)

### 6.1 재현용 inference/evaluation 스크립트 5개 준비

각 스크립트는 사용자가 긴 Hydra override 문자열을 입력하지 않아도 되도록, 파일 내부 상수로 data/model/train 설정과 checkpoint/output 명칭을 정의한다. 목표 스크립트는 `src/` 내부에 둔다.

1. `src/presentation_infer_cnn_baseline.py`
   - config 값: script 내부 `EXPERIMENT_CONFIG`
   - checkpoint: `outputs/2026-04-14/04-49-31_cnn_optuna_stage1_baselineTest/weights/best_model_fold1.pt`
   - split: RAVDESS GroupKFold fold 1 validation/test

2. `src/presentation_infer_pure_transformer.py`
   - config 값: script 내부 `EXPERIMENT_CONFIG`
   - checkpoint: `outputs/2026-04-15/13-44-11_thesis_transformer_stage2_pure_transformer/weights/best_model_fold1.pt`
   - split: RAVDESS GroupKFold fold 1 validation/test

3. `src/presentation_infer_cnn_conformer.py`
   - config 값: script 내부 `EXPERIMENT_CONFIG`
   - checkpoint: `outputs/2026-04-21/18-47-36_cnn_conformer/weights/best_model_fold1.pt`
   - split: RAVDESS GroupKFold fold 1 validation/test

4. `src/presentation_infer_noise_eval.py`
   - config 값: script 내부 `EXPERIMENT_CONFIG`, `NOISE_TYPES`, `SNR_DB`
   - checkpoint: `outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/artifacts/weights/best_model_fold1.pt`
   - behavior: 기존 `src.evaluate_noise_robustness`와 같은 조건 그리드로 clean/noisy fold 1 evaluation

5. `src/presentation_infer_cross_corpus.py`
   - config 값: script 내부 `EXPERIMENT_CONFIG`
   - checkpoint: `outputs/2026-04-30/18-10-12_cross_corpus_cremad6/artifacts/fold_1/best_model.pt`
   - behavior: source fold 1 validation 및 CREMA-D target evaluation
   - blocker: 현재 `src/CREMA-D`가 없으므로 target evaluation 실행 전 데이터셋 위치 복구 또는 경로 override가 필요하다.

### 6.2 공통 구현 방향

- 기존 `src/infer.py`는 임의 audio folder 분류용이라 실험 재현용으로는 부족하다. 새 공통 helper를 만들거나 각 스크립트가 `RavdessDataset`, `GroupKFold`, `AudioPipeline`, `build_criterion`, `evaluate`를 직접 사용해야 한다.
- RAVDESS clean 모델 3개는 `trainer.evaluate()`를 재사용해 `summary_metrics.json`, `predictions.csv`, `confusion_matrix.png`, `calibration_curve.png`, `roc_pr_curves.png`를 저장한다.
- 결과는 `outputs/YYYY-MM-DD/HH-MM-SS_<presentation_experiment_name>/` 아래에 저장한다.
- 단순 로그만 남기지 않고 `summary_metrics.csv/html`, `predictions.csv/html`, 조건별 noise table, confusion matrix, calibration curve, ROC/PR curve, manifest를 생성한다.
- chunking이 켜진 CNN-Conformer는 `UtteranceChunkDataset`과 `collate_utterance_chunks`를 사용해야 기존 실험과 동일하다.
- 데이터셋 경로는 실행 시 현재 repo 기준 `src/$RVNS6MQ`로 보정하되, 실험의 data 설정값 자체는 각 script 내부에 직접 둔다.
- checkpoint 복사는 필수가 아니다. 발표용 출력에는 원본 checkpoint 경로와 sha256/파일 크기/mtime을 함께 기록하는 편이 더 설득력 있다.

---

## 7. 현재 확인된 사항 vs 추가 확인 필요 사항

### 7.1 확인된 사항 ✅

| 항목 | 확인 내용 |
|---|---|
| CNN-Conformer 논문 최고점 checkpoint 존재 | `outputs/2026-04-21/18-47-36_cnn_conformer/weights/best_model_fold1.pt` (12.35MB) 존재 확인 |
| CNN-Conformer 노이즈 기준 checkpoint 존재 | `outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/artifacts/weights/best_model_fold1.pt` (14.25MB) 존재 확인 |
| CNN-Conformer winner 정확한 설정 | 논문 최고점은 2026-04-21 trial_0003, 노이즈 기준은 2026-04-22 trial_0004로 분리 |
| 노이즈 실험 실행 명령어 | `docs/noise_robustness/KR_NOISE_ROBUSTNESS_EXPERIMENT_PLAN.md`에 기재 |
| 크로스 코퍼스 실행 명령어 | `docs/cross_corpus/2026-04-30_RAVDESS_to_CREMAD_6class.md`에 기재 |
| 노이즈 실험 기존 결과 폴더 | `outputs/2026-04-23/15-33-32_noise_eval_winner/` 존재 |
| 크로스 코퍼스 기존 결과 폴더 | `outputs/2026-04-30/18-10-12_cross_corpus_cremad6/` 존재 |
| weights 저장 코드 위치 | `src/engine/trainer.py` L624~635, L649, L711 |
| 모든 실험은 fold 1 기준 | 사용자 확인 |

### 7.2 남은 리스크 / 추가 확인 필요 사항 ❓

| 항목 | 확인 방법 |
|---|---|
| `weights/best_model_fold5.pt`의 실제 정체 | 파일 크기(6MB) + fold 5 → CNN Baseline 또는 Pure Transformer의 fold 5 결과로 추정 |
| 데이터셋 경로 | `src/configs/data/default.yaml`에 `$RVNS6MQ` 폴더명으로 되어 있음. trial_summary에서 실제 경로는 `src/$RVNS6MQ`임 |
| CREMA-D 데이터셋 존재 여부 | 현재 `src/CREMA-D` 없음. cross-corpus target 재현 시 데이터셋 위치 복구 필요 |

---

## 8. 실행 환경

| 항목 | 값 |
|---|---|
| OS | Windows 10 |
| Python | Python 3.10.19 (`grad_paper_ser`) |
| 가상환경 실행 | `conda activate grad_paper_ser` 또는 `C:\Users\hik88\miniconda3\envs\grad_paper_ser\python.exe` |
| 프레임워크 | PyTorch, torchaudio |
| 설정 관리 | Hydra / OmegaConf |
| GPU | NVIDIA GeForce RTX 2060 (단일 카드) |
| 실행 위치 | 항상 프로젝트 루트(`SER_GraduationPaper/`)에서 실행 |

### 8.1 실제 실행 환경 활성화 명령

발표용 inference 스크립트는 기본 Python/base 환경이 아니라 Miniconda의 `grad_paper_ser` 환경에서 실행해야 한다. 현재 확인된 기본 `python`은 Python 3.13 계열이며 `torch`가 없어 발표용 inference 실행에 적합하지 않다.

권장 실행 방식:

```powershell
conda activate grad_paper_ser
cd C:\Users\hik88\Desktop\BIT_Uni\GraduationPaper\Project\SER_GraduationPaper
python -m src.presentation_infer_cnn_conformer
```

PowerShell에서 `conda activate`가 바로 동작하지 않으면 먼저 conda activate script를 로드한다.

```powershell
C:\Users\hik88\miniconda3\Scripts\activate
conda activate grad_paper_ser
cd C:\Users\hik88\Desktop\BIT_Uni\GraduationPaper\Project\SER_GraduationPaper
python -m src.presentation_infer_cnn_conformer
```

activate 없이 직접 실행할 수도 있다.

```powershell
cd C:\Users\hik88\Desktop\BIT_Uni\GraduationPaper\Project\SER_GraduationPaper
C:\Users\hik88\miniconda3\envs\grad_paper_ser\python.exe -m src.presentation_infer_cnn_conformer
```

---

## 9. 참고: outputs 날짜별 디렉토리 전체 목록

```
outputs/
├── 2026-02-07/
├── 2026-04-14/   ← CNN Baseline Optuna
├── 2026-04-15/   ← Pure Transformer + CNN-Conformer Round 1
├── 2026-04-16/   ← padding-safe 점검
├── 2026-04-17/   ← CNN-Conformer champion (F1=0.63168)
├── 2026-04-18/   ← Regularization HPO
├── 2026-04-19/   ← 구조 ablation + loss/sampler
├── 2026-04-20/   ← backbone redesign (nostem_patch 부상)
├── 2026-04-21/   ← nostem generalization → trial_0003 F1=0.70563 (18-47-36)
├── 2026-04-22/   ← speaker-invariant → trial_0004 Acc=0.70333 (02-33-33)
├── 2026-04-23/   ← 노이즈 실험 결과 (15-33-32_noise_eval_winner)
└── 2026-04-30/   ← 크로스 코퍼스 실험 (18-10-12_cross_corpus_cremad6)
```

---

## 10. 우선순위별 작업 순서

1. **[필수] 공통 presentation evaluation helper 설계**  
   - resolved config 로드, 현재 repo 데이터셋 경로 보정, checkpoint 로드, fold 1 split, metrics/artifacts 저장

2. **[필수] CNN Baseline inference 스크립트 작성**  
   - trial_0023 config + root checkpoint로 fold 1 evaluation

3. **[필수] Pure Transformer inference 스크립트 작성**  
   - trial_0016 config + root checkpoint로 fold 1 evaluation

4. **[필수] CNN-Conformer 논문 최고점 inference 스크립트 작성**  
   - 2026-04-21 trial_0003 config + root checkpoint로 fold 1 chunked evaluation

5. **[필수] Noise evaluation 스크립트 작성**  
   - 2026-04-22 trial_0004 checkpoint와 기존 노이즈 그리드 사용

6. **[조건부] Cross-corpus inference/evaluation 스크립트 작성**  
   - checkpoint와 config는 존재하지만 `src/CREMA-D`가 없어 target 평가 실행은 데이터 복구 후 가능

---

## 11. Presentation Inference Scripts 사용 정리

이 섹션은 발표용으로 추가한 5개 inference/evaluation 스크립트를 실제 코드 기준으로 정리한 것이다. 핵심 목적은 expected metric과 비교하는 것이 아니라, 이미 만들어진 checkpoint와 원 실험의 데이터/모델 설정을 코드 안에 고정해 두고, 발표 영상에서 "실험 결과가 실제 코드 실행과 저장된 checkpoint/artifact에서 나온다"는 점을 보여주는 것이다.

### 11.1 공통 실행 흐름

1. 각 `src/presentation_infer_*.py` 파일 안의 `EXPERIMENT_CONFIG` 상수에서 data/model/train 값을 직접 정의한다.
2. `src/presentation_eval_common.py`가 이 상수를 OmegaConf config로 변환하고, dataset path는 repo 내부 실제 위치로 다시 바인딩한다.
   - RAVDESS: `src/$RVNS6MQ`
   - CREMA-D cross-corpus target: `src/CREMA-D`
3. RAVDESS clean/noise 재현은 원 학습과 같은 `GroupKFold` 방식으로 fold 1 test split을 재구성한다.
4. checkpoint는 각 스크립트의 `checkpoint_path`에 있는 상대경로 파일을 사용한다.
5. 기존 구현체(`AudioPipeline`, dataset class, model builder, `trainer.evaluate`)를 재사용해서 inference/evaluation을 수행한다.
6. 실행 결과는 `outputs/YYYY-MM-DD/HH-MM-SS_<presentation_experiment_name>/` 형식으로 새 폴더를 만들고, metrics/predictions/config/manifest/plot artifact를 저장한다.

### 11.2 스크립트별 실행 명령

`grad_paper_ser` conda 환경을 활성화한 뒤 프로젝트 루트(`Project/SER_GraduationPaper`)에서 실행한다.

```powershell
conda activate grad_paper_ser
cd C:\Users\hik88\Desktop\BIT_Uni\GraduationPaper\Project\SER_GraduationPaper
```

```powershell
python -m src.presentation_infer_cnn_baseline
python -m src.presentation_infer_pure_transformer
python -m src.presentation_infer_cnn_conformer
python -m src.presentation_infer_noise_eval
python -m src.presentation_infer_cross_corpus
```

직접 파일 실행도 가능하도록 `sys.path` bootstrap을 넣어두었다.

```powershell
python src/presentation_infer_cnn_baseline.py
python src/presentation_infer_pure_transformer.py
python src/presentation_infer_cnn_conformer.py
python src/presentation_infer_noise_eval.py
python src/presentation_infer_cross_corpus.py
```

### 11.3 스크립트별 핵심 전제조건

| Script | Checkpoint | Dataset prerequisite | Notes |
|---|---|---|---|
| `src/presentation_infer_cnn_baseline.py` | `outputs/2026-04-14/04-49-31_cnn_optuna_stage1_baselineTest/weights/best_model_fold1.pt` | `src/$RVNS6MQ` | CNN baseline best checkpoint 기반 fold 1 test 재현 |
| `src/presentation_infer_pure_transformer.py` | `outputs/2026-04-15/13-44-11_thesis_transformer_stage2_pure_transformer/weights/best_model_fold1.pt` | `src/$RVNS6MQ` | Pure Transformer best checkpoint 기반 fold 1 test 재현 |
| `src/presentation_infer_cnn_conformer.py` | `outputs/2026-04-21/18-47-36_cnn_conformer/weights/best_model_fold1.pt` | `src/$RVNS6MQ` | CNN-Conformer main/generalization best checkpoint 기반 fold 1 test 재현 |
| `src/presentation_infer_noise_eval.py` | `outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/artifacts/weights/best_model_fold1.pt` | `src/$RVNS6MQ` | clean fold 1과 noise grid를 같은 checkpoint로 평가 |
| `src/presentation_infer_cross_corpus.py` | `outputs/2026-04-30/18-10-12_cross_corpus_cremad6/artifacts/fold_1/best_model.pt` | `src/$RVNS6MQ`, `src/CREMA-D` | source RAVDESS fold 1 validation + target CREMA-D full inference/evaluation. 현재 repo에는 `src/CREMA-D`가 없으므로 dataset을 배치해야 실행 가능 |

### 11.4 출력 폴더와 산출물

clean RAVDESS 계열 3개 스크립트는 다음 파일을 생성한다.

```text
outputs/YYYY-MM-DD/HH-MM-SS_<experiment_name>/
  manifest.json
  runtime_config_from_script.yaml
  summary_metrics.json
  summary_metrics.csv
  summary_metrics.html
  predictions.csv
  predictions.html
  artifact_index.json
  artifacts/
    fold_1_confusion_matrix.png
    fold_1_roc_pr_curves.png
    fold_1_calibration_curve.png
    fold_1_tsne.png
```

noise 스크립트는 condition별 결과와 요약 테이블을 함께 생성한다.

```text
outputs/YYYY-MM-DD/HH-MM-SS_presentation_noise_eval/
  manifest.json
  runtime_config_from_script.yaml
  noise_summary.json
  noise_summary.csv
  noise_summary.html
  clean/
    metrics.json
    predictions.csv
    predictions.html
  <noise_condition>/
    metrics.json
    predictions.csv
    predictions.html
  artifacts/
    clean_confusion_matrix.png
    <noise_condition>_confusion_matrix.png
```

cross-corpus 스크립트는 source와 target을 분리해서 저장한다.

```text
outputs/YYYY-MM-DD/HH-MM-SS_presentation_cross_corpus/
  manifest.json
  runtime_config_from_script.yaml
  cross_corpus_summary.json
  cross_corpus_summary.csv
  cross_corpus_summary.html
  source_val_predictions.csv
  source_val_predictions.html
  target_predictions.csv
  target_predictions.html
  artifacts/
    source_val_confusion_matrix.png
    target_confusion_matrix.png
```

### 11.5 코드 내부에서 수정할 핵심 위치

- 데이터/모델/학습 설정값: 각 `src/presentation_infer_*.py`의 `EXPERIMENT_CONFIG`
- checkpoint 상대경로: 각 script의 `CheckpointEvalSpec(checkpoint_path=...)`
- 출력 폴더 이름: 각 script의 `output_name`
- 공통 runtime 변환, fold split, checkpoint loading, artifact 저장: `src/presentation_eval_common.py`

실험 수치나 provenance를 보고 싶으면 `manifest.json`, `runtime_config_from_script.yaml`, `summary_metrics.html` 또는 `noise_summary.html`/`cross_corpus_summary.html`을 발표 영상에서 열어 보여주면 된다. `manifest.json`에는 checkpoint path, checkpoint sha256, source note, original result reference가 들어가므로 결과 출처 설명에 적합하다.

### 11.6 Pyrefly / editor import note

`py_compile`은 통과했지만 editor에서 `src.data.noise`를 못 찾는 문제는 단순 Python syntax 문제가 아니라 Pyrefly의 import root 해석 문제에 가깝다. `pyproject.toml`에 아래 설정을 추가했다.

```toml
[tool.pyrefly]
project-includes = ["src"]
search-path = ["."]
```

따라서 editor/Pyrefly server를 reload하거나 IDE를 재시작하면 `import src...` 계열 local module 해석은 정상화될 가능성이 높다. 단, 선택된 Python interpreter에 `torch`, `torchaudio`, `hydra-core`, `omegaconf` 같은 project dependency가 설치되어 있지 않으면 third-party missing import 경고는 별도로 남을 수 있다. 그 경우에는 프로젝트 실행에 사용하는 interpreter/venv를 editor에도 동일하게 지정해야 한다.

### 11.7 현재 검증 상태

- `py_compile` 정적 검증은 5개 script와 `src/presentation_eval_common.py` 모두 통과했다.
- 5개 script에 기록된 checkpoint path는 현재 repo에서 모두 존재함을 확인했다.
- `src/$RVNS6MQ` RAVDESS dataset은 존재하고 wav 파일이 확인되었다.
- `src/CREMA-D`는 현재 존재하지 않는다. cross-corpus script는 이 dataset을 배치한 뒤 실행해야 한다.
- 사용자 요청에 따라 실제 inference 실행은 수행하지 않았다.

---

## 12. 발표용 Inference 녹화 계획

이 계획은 2분 software demonstration 영상에 사용할 흐름이다. 발표 시간 제한 때문에 프로젝트 구조, checkpoint path, script 내부 설정, output name 설명은 영상에서 직접 보여주지 않는다. 영상은 terminal 실행 장면부터 시작하고, 실행 후 생성된 output folder의 결과 데이터와 artifact를 보여주는 방식으로 구성한다.

### 12.1 영상 목표

- 이미 학습된 checkpoint를 이용해 실제 inference/evaluation이 실행되는 장면을 보여준다.
- 실행 결과가 단순 terminal log로 끝나는 것이 아니라, `outputs/YYYY-MM-DD/HH-MM-SS_<experiment_name>/` 아래에 metrics, predictions, config, manifest, plot artifact로 저장되는 것을 보여준다.
- 교수의 요구사항인 "做出了一定的工作", "实验结果是真实的", "展示实验结果的来源"에 대응한다.

### 12.2 녹화 전 준비

1. `grad_paper_ser` conda 환경을 활성화하고 프로젝트 루트로 이동한다.

```powershell
conda activate grad_paper_ser
cd C:\Users\hik88\Desktop\BIT_Uni\GraduationPaper\Project\SER_GraduationPaper
```

2. PowerShell에서 `conda activate`가 바로 동작하지 않으면 아래처럼 activate script를 먼저 로드한다.

```powershell
C:\Users\hik88\miniconda3\Scripts\activate
conda activate grad_paper_ser
```

3. 녹화에 사용할 스크립트는 우선 `src.presentation_infer_cnn_conformer`를 권장한다.
   - thesis main result에 가장 가까운 CNN-Conformer checkpoint 기반 재현이다.
   - RAVDESS만 필요하므로 현재 repo 상태에서 cross-corpus보다 안정적으로 실행 가능하다.

4. terminal 글자 크기를 키우고, output folder를 바로 열 수 있도록 file explorer 또는 editor를 준비한다.

### 12.3 권장 녹화 흐름

| Time | 화면 | 내용 |
|---|---|---|
| 0:00-0:10 | Terminal | `grad_paper_ser` 환경이 활성화된 프로젝트 루트에서 inference 명령어 입력 준비 |
| 0:10-0:45 | Terminal | `python -m src.presentation_infer_cnn_conformer` 실행 장면 녹화 |
| 0:45-1:05 | Terminal | 실행 완료 후 출력된 output directory 경로 확인 |
| 1:05-1:25 | File explorer/editor | 생성된 `outputs/YYYY-MM-DD/HH-MM-SS_presentation_cnn_conformer_retained_checkpoint/` 폴더 열기 |
| 1:25-1:40 | Output files | `summary_metrics.html` 또는 `summary_metrics.csv`를 열어 accuracy/F1 등 결과 테이블 확인 |
| 1:40-1:52 | Output files | `predictions.html` 또는 `predictions.csv`를 열어 sample별 inference 결과 확인 |
| 1:52-2:00 | Artifacts | `artifacts/fold_1_confusion_matrix.csv/html` 등 결과 artifact 확인 |

### 12.4 실제 실행 명령

메인 추천 명령:

```powershell
conda activate grad_paper_ser
cd C:\Users\hik88\Desktop\BIT_Uni\GraduationPaper\Project\SER_GraduationPaper
python -m src.presentation_infer_cnn_conformer
```

activate 없이 직접 실행하는 대체 명령:

```powershell
cd C:\Users\hik88\Desktop\BIT_Uni\GraduationPaper\Project\SER_GraduationPaper
C:\Users\hik88\miniconda3\envs\grad_paper_ser\python.exe -m src.presentation_infer_cnn_conformer
```

대체로 보여줄 수 있는 추가 명령:

```powershell
python -m src.presentation_infer_cnn_baseline
python -m src.presentation_infer_pure_transformer
python -m src.presentation_infer_noise_eval
```

`src.presentation_infer_cross_corpus`는 현재 `src/CREMA-D` dataset이 없으면 target evaluation이 실행되지 않으므로, 발표 영상의 메인 demo로는 권장하지 않는다.

### 12.5 영상에서 보여줄 output 우선순위

1. `summary_metrics.html`
   - inference/evaluation 결과를 표로 보여주기 가장 좋다.
2. `predictions.html`
   - 실제 sample 단위 prediction이 생성되었음을 보여준다.
3. `artifacts/fold_1_confusion_matrix.csv` 또는 `artifacts/fold_1_confusion_matrix.html`
   - 결과가 confusion matrix artifact로도 저장되었음을 보여준다.
4. `runtime_config_from_script.yaml`
   - 필요 시 재현에 사용된 runtime config를 보여준다.
5. `manifest.json`
   - 필요 시 checkpoint path, checkpoint sha256, source note를 보여준다.

2분 제한 안에서는 1-3번만 보여주고, 4-5번은 질문 대응용 backup 자료로 준비한다.

### 12.6 영상 녹화 시 말할 수 있는 짧은 설명

```text
这里我展示的是已经训练完成的 CNN-Conformer 模型的 inference/evaluation 过程。
程序会加载保存好的 checkpoint，并按照实验时的 RAVDESS fold 1 test split 重新运行评估。
运行结束后，结果会自动保存到 outputs 目录，包括 metrics 表格、sample-level predictions 和 confusion matrix 等 artifact。
这些文件可以追溯到实际 checkpoint 和运行配置，用来说明实验结果是真实生成的。
```

한국어 의미:

```text
이미 학습이 끝난 CNN-Conformer checkpoint를 로드해서 inference/evaluation을 재실행한다.
원 실험의 RAVDESS fold 1 test split을 재구성하고, 결과는 outputs에 metrics, predictions, artifact 형태로 저장된다.
이 파일들을 통해 실험 결과의 출처와 실제 생성 과정을 보여준다.
```
