# CNN Conformer Experiment - 2026-04-21 Nostem Generalization

## 1. 문서 범위

- 대상 모델: `cnn_conformer`
- 문서 목적: `nostem_patch` 승자 backbone 위에서 overfitting 완화를 위한 downsizing / gradually shrinking 실험 계획과 결과 기록
- 현재 문서 상태: `reference`

## 2. 모델 스냅샷

### 2.1 한 줄 요약

이번 회차는 2026-04-20 backbone redesign에서 승리한 `nostem_patch`를 고정하고, 모델 용량 축소와 sequence shrinking을 통해 일반화를 더 끌어올리는 것을 목표로 했다.  
결론적으로 **overfitting 완화 신호는 일부 확인했지만, 2026-04-20 winner를 넘지는 못했고, 200+ trial 시점에서 broad search는 중단하는 편이 타당하다.**

### 2.2 핵심 구성 요소

| 항목 | 값 또는 설명 |
|---|---|
| 입력 표현 | `log-Mel spectrogram` |
| 승자 backbone | `nostem_patch` |
| 핵심 블록 | patch projection + Conformer encoder |
| 주 탐색축 | `embed_dim`, `num_layers`, `ffn_ratio`, `time_patch`, `sequence_shrinking` |
| 출력 pooling | `attention` |
| 분류 대상 | 8-class emotion recognition |

### 2.3 비교 관점

- 비교 대상 1: CNN baseline
- 비교 대상 2: 2026-04-17 conformer champion `0.63168`
- 비교 대상 3: 2026-04-20 backbone redesign winner `0.68536`

이 문서의 목적은 “새 backbone 발굴”이 아니라, **승자 backbone의 overfitting을 줄여 더 안정적이고 재현 가능한 성능으로 다듬는 것**이다.

## 3. 실험 라운드 기록

### 3.1 공통 고정 조건

| 분류 | 항목 | 값 | 비고 |
|---|---|---|---|
| 데이터 | dataset | RAVDESS | |
| log-Mel | `n_mels / n_fft / hop_length` | `80 / 1024 / 160` | backbone winner와 동일 |
| 학습 | epochs | `30` | 1차 screening |
| 평가 | folds | `1 fold` | |

### 3.2 탐색 공간 또는 실험 변수

| 항목 | 후보군 | 비고 |
|---|---|---|
| backbone | `nostem_patch` 고정 | winner 고정 |
| `embed_dim` | `[128, 160, 192]` | downsizing |
| `num_layers` | `[3, 4]` | downsizing |
| `ffn_ratio` | `[3, 4]` | downsizing |
| `time_patch` | `[2, 3, 4]` | token 수 단순화 |
| `sequence_shrinking` | `none`, `late_2x`, `progressive_2x` | gradually shrinking |
| `batch_size` | `[8, 12]` | generalization / memory tradeoff |
| `lr`, `wd`, dropout` | 좁은 범위 | winner 주변 미세조정 |

### 3.3 실행 명령

```powershell
.\.venv\Scripts\python.exe -m src.optuna_search model=cnn_conformer optuna=cnn_conformer_nostem_generalization optuna.enabled=true optuna.trials=30 train.epochs=30 train.folds_to_run=1 experiment.tag=nostem_generalization
```

### 3.4 회차별 실험 로그

| 회차 | 날짜 | 목적 | 설정 요약 | 결과 요약 | 산출 경로 |
|---|---|---|---|---|---|
| Round 7 | 2026-04-21 | `nostem_patch` overfitting 완화 | downsizing + sequence shrinking + patch simplification | backbone redesign winner 미회복, 추가 broad search 중단 | `../../outputs/2026-04-21/00-49-32_cnn_conformer` |

실제 확인 경로:

- 실험 루트: [../../outputs/2026-04-21/00-49-32_cnn_conformer](../../outputs/2026-04-21/00-49-32_cnn_conformer)
- Hydra 설정 스냅샷: [../../outputs/2026-04-21/00-49-32_cnn_conformer/.hydra/config.yaml](../../outputs/2026-04-21/00-49-32_cnn_conformer/.hydra/config.yaml)
- Optuna 로그: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_search.log](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_search.log)
- Study DB: [../../optuna_studies/cnn_conformer_optuna_nostem_generalization.db](../../optuna_studies/cnn_conformer_optuna_nostem_generalization.db)
- 최고 성능 체크포인트: [../../outputs/2026-04-21/00-49-32_cnn_conformer/weights/best_model_fold1.pt](../../outputs/2026-04-21/00-49-32_cnn_conformer/weights/best_model_fold1.pt)
- 제외한 smoke run: `../../outputs/2026-04-21/00-40-11_cnn_conformer`

### 3.5 주요 결과 요약

| Rank | Trial | F1-macro | Accuracy | UAR | 핵심 파라미터 요약 |
|---|---|---:|---:|---:|---|
| 1 | `trial_0001` | 0.67100 | 0.66000 | 0.66563 | `embed=192`, `layers=3`, `ffn=768`, `time_patch=4`, `late_2x shrinking`, `batch=12` |
| 2 | `trial_0031` | 0.66789 | 0.65333 | 0.64688 | `embed=192`, `layers=4`, `ffn=576`, `time_patch=2`, `late_2x shrinking`, `batch=12` |
| 3 | `trial_0003` | 0.66639 | 0.66333 | 0.66250 | `embed=128`, `layers=4`, `ffn=384`, `time_patch=2`, `no shrinking`, `batch=12` |
| 4 | `trial_0033` | 0.66605 | 0.66667 | 0.65625 | `embed=192`, `layers=4`, `ffn=576`, `time_patch=2`, `late_2x shrinking`, `batch=12` |
| 5 | `trial_0164` | 0.66472 | 0.66667 | 0.67500 | `embed=192`, `layers=4`, `ffn=576`, `time_patch=2`, `late_2x shrinking`, `batch=12` |

추가 집계:

- 상태 분포:
  - `COMPLETE`: 13
  - `PRUNED`: 213
  - `FAIL`: 1
- 확인된 최대 trial 번호: `226`
- best milestone:
  - `trial_0000`: `0.66319`
  - `trial_0001`: `0.67100`
- `trial_0001` 이후 225개 번호가 더 진행됐지만 objective best는 갱신되지 않았다.
- 2026-04-20 winner 대비:
  - 2026-04-20 best F1 `0.68536`
  - 2026-04-21 best F1 `0.67100`
  - 차이 `-0.01436`

대표 trial 바로가기:

- objective best 요약: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/trial_summary.json](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/trial_summary.json)
- objective best metrics: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/summary_metrics.json](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/summary_metrics.json)
- accuracy/UAR strong trial 요약: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/trial_summary.json](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/trial_summary.json)
- accuracy/UAR strong trial metrics: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/artifacts/summary_metrics.json](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/artifacts/summary_metrics.json)

## 4. 설계 배경 및 구현 메모

### 4.1 설계 배경

- 2026-04-20에서 `nostem_patch`가 `0.68536`까지 올라가며 backbone redesign 자체는 성공했다.
- 하지만 learning curve는 여전히 train/val gap이 남아 있었다.
- 따라서 다음 단계는 새 backbone 탐색이 아니라, 그 backbone 위에서 overfitting을 줄이는 것이다.

이번 회차에서 보는 개념:

- `downsizing`
  - `embed_dim`, `num_layers`, `ffn_dim`을 줄여 모델 용량을 줄임
- `gradually shrinking`
  - encoder 중간 이후 sequence 길이를 줄여 더 압축된 representation으로 유도
- `tokenization 단순화`
  - `time_patch`를 키워 token 수를 줄임

이번 회차에서 실제로 관찰된 패턴:

- 최고 objective trial은 `3-layer + time_patch=4 + late_2x shrinking`이었다.
- accuracy/UAR 상위 trial들은 `4-layer + ffn=576 + time_patch=2 + late_2x shrinking`에 모였다.
- 즉 “무조건 크게 줄일수록 좋다”보다 **기본 용량은 유지하되 일부 축만 줄이는 절충형**이 더 유력했다.

### 4.2 현재 코드 기준 구현

- 모델 코드: [../../src/models/cnn_conformer.py](../../src/models/cnn_conformer.py)
- 모델 설정: [../../src/configs/model/cnn_conformer.yaml](../../src/configs/model/cnn_conformer.yaml)
- Optuna 설정: [../../src/configs/optuna/cnn_conformer_nostem_generalization.yaml](../../src/configs/optuna/cnn_conformer_nostem_generalization.yaml)
- search logic: [../../src/optuna_search.py](../../src/optuna_search.py)

### 4.3 구현상 차이점 또는 주의점

- 이전 실험 명령어 재현성을 깨지 않도록 기존 preset은 수정하지 않고 새 preset만 추가했다.
- `sequence_shrinking`은 optional 옵션이며 기본값은 비활성화다.
- shrinking은 token sequence에 대해 평균 기반 압축을 적용한다.
- 이 실험은 `nostem_patch`만 고정하고 일반화 축만 비교한다.

## 5. 아티팩트 분석

### 5.1 대표 trial

- objective best trial:
  - 요약: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/trial_summary.json](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/trial_summary.json)
  - metrics: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/summary_metrics.json](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/summary_metrics.json)
  - artifact 폴더: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts)
- accuracy/UAR strong trial:
  - 요약: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/trial_summary.json](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/trial_summary.json)
  - metrics: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/artifacts/summary_metrics.json](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/artifacts/summary_metrics.json)
  - artifact 폴더: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/artifacts](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/artifacts)

### 5.2 항목별 해석

대표 artifact 링크:

- objective best learning curve: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/fold_1_learning_curve.png](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/fold_1_learning_curve.png)
- objective best confusion matrix: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/global_confusion_matrix.png](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/global_confusion_matrix.png)
- objective best calibration curve: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/global_calibration_curve.png](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/global_calibration_curve.png)
- objective best ROC/PR: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/global_roc_pr_curves.png](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/global_roc_pr_curves.png)
- objective best t-SNE: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/global_tsne_plot.png](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/global_tsne_plot.png)
- accuracy/UAR strong confusion matrix: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/artifacts/global_confusion_matrix.png](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/artifacts/global_confusion_matrix.png)
- accuracy/UAR strong calibration curve: [../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/artifacts/global_calibration_curve.png](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/artifacts/global_calibration_curve.png)

- learning curve:
  - [trial_0001 learning curve](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/fold_1_learning_curve.png) 기준으로 train loss는 `1.84 -> 0.39`, train accuracy는 `0.28 -> 0.84+`까지 상승한다.
  - 반면 val loss는 `1.7 -> 1.1대`에서 더 내려가지 않고, val accuracy는 `0.60~0.66` 부근에서 횡보한다.
  - 즉 downsizing과 shrinking이 있어도 **과적합 자체가 사라진 것은 아니고, onset만 늦춘 수준**이다.
- confusion matrix:
  - [trial_0001 confusion matrix](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0001/artifacts/global_confusion_matrix.png)에서 `neutral=0.75`, `calm=0.78`, `angry=0.80`, `surprised=0.70`은 양호하다.
  - 반면 `sad=0.53`, `fearful=0.53`, `disgust=0.57`은 여전히 약하다.
  - 주요 혼동은 `neutral -> sad 0.20`, `calm -> sad 0.15`, `disgust -> sad 0.20`, `surprised -> sad 0.23`, `fearful -> sad 0.10`으로, 이전 라운드와 동일하게 `sad` 축으로 끌리는 현상이 남는다.
- calibration:
  - objective best인 trial 1의 ECE는 `0.0710`이다.
  - [trial_0164 metrics](../../outputs/2026-04-21/00-49-32_cnn_conformer/optuna_trials/trial_0164/artifacts/summary_metrics.json)에서는 accuracy `0.6667`, UAR `0.6750`, ECE `0.0446`으로 더 안정적이다.
  - 즉 이번 라운드는 “best F1 1개”보다 **조금 낮은 F1 대신 calibration/UAR가 더 안정적인 조합**을 같이 확보했다는 의미가 있다.
- 탐색 축 해석:
  - `late_2x shrinking`은 최고 trial을 만들긴 했지만, complete 평균은 `0.6556`으로 `none`의 `0.6648`보다 높지 않았다.
  - 다만 `none` complete는 2개뿐이라, 현재 결과는 “late shrinking이 항상 유리”가 아니라 **살아남는 후보를 만들 수는 있지만 확실한 승리축은 아니다**로 해석하는 편이 맞다.
  - `time_patch=4`는 최고 objective trial을 만들었지만, complete 평균은 `time_patch=2`가 더 높았다. 따라서 patch 확대는 강한 기본값이 아니라 특정 조합에서만 유효하다.
  - `embed=128` / `ffn=384`의 강한 downsizing도 `trial_0003`에서 `0.66639`를 기록해 “작게 만들면 무조건 성능이 무너진다”는 해석은 틀렸다.

## 6. 종합 인사이트 및 다음 액션

### 6.1 현재 판단

이번 회차는 `nostem_patch` backbone의 승리를 더 믿을 수 있는지 확인하는 generalization 실험이었다.

현재 판단은 다음과 같다.

- `nostem_patch` backbone 자체는 여전히 유효하다.
- 그러나 이번 overfitting 완화 search는 2026-04-20 winner를 넘지 못했다.
- 더 중요한 점은 `trial_0001`이 아주 초반에 best를 찍은 뒤, 200개가 넘는 시도 동안 갱신되지 않았다는 것이다.
- 따라서 지금 search를 계속 넓게 돌리는 것은 효율이 낮고, **이 라운드는 여기서 중단하는 것이 맞다.**

### 6.2 다음 액션

이번 결과와 `ref.bib` 기준 관련 문헌을 묶으면, 후속 방향은 아래 3개가 가장 타당하다.

1. 구조적 tapering 실험
   - 근거: 이번 라운드에서 `embed=128`, `ffn=384`, `layers=3~4` 같은 축소형이 상위권에 계속 남았다.
   - 문헌 연결:
     - Peng et al., *Efficient Speech Emotion Recognition Using Multi-Scale CNN and Attention*
     - Gulati et al., *Conformer: Convolution-augmented Transformer for Speech Recognition*
   - 다음 실험 아이디어:
     - 층마다 동일 폭을 쓰지 않고 `192 -> 160 -> 128`처럼 점진 축소
     - FFN도 `768 -> 576 -> 384`처럼 layer-wise shrinking
     - 현재의 token shrinking보다 **채널/FFN shrinking**이 더 직접적인 overfitting 대응일 수 있다

2. mixup / label-uncertainty 계열 regularization
   - 근거: 현재는 train acc가 과하게 오르는데 val plateau가 일찍 고정된다.
   - 문헌 연결:
     - Kang et al., *Learning Robust Self-Attention Features for Speech Emotion Recognition with Label-Adaptive Mixup*
     - Prabhu et al., *End-to-End Label Uncertainty Modeling in Speech Emotion Recognition Using Bayesian Neural Networks and Label Distribution Learning*
   - 다음 실험 아이디어:
     - spectrogram mixup 또는 chunk-level mixup
     - 작은 `label_smoothing` 재도입
     - 목표는 구조를 크게 바꾸지 않고 decision boundary를 덜 날카롭게 만드는 것

3. normalization / speaker-invariant regularization
   - 근거: confusion이 `sad` 축으로 쏠리는 현상은 감정 cue보다 speaker/style 편차를 과하게 주워 담는 신호일 수 있다.
   - 문헌 연결:
     - Fan et al., *ISNet: Individual Standardization Network for Speech Emotion Recognition*
     - Lu et al., *Domain Invariant Feature Learning for Speaker-Independent Speech Emotion Recognition*
   - 다음 실험 아이디어:
     - input/token level instance normalization 강화
     - speaker-invariant auxiliary branch 또는 gradient reversal은 무거우니, 먼저 경량 normalization ablation부터

중단 결론:

- 현재 일반화 라운드는 “충분히 의미 있는 데이터”를 이미 확보했다.
- 추가 trial을 더 늘리기보다, 다음 라운드는 위 3개 중 하나를 좁은 가설로 새 문서/새 preset에서 다시 여는 것이 낫다.
- 우선순위는 다음과 같다.
  1. 구조적 tapering
  2. mixup + 작은 label smoothing
  3. normalization 강화

## 7. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-04-21 | `nostem_patch` overfitting 완화 실험 계획 문서 작성 |
| 2026-04-21 | 실제 실험 결과, artifact 링크, 중단 결론, 후속 overfitting 대응 방향 반영 |
