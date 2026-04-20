# CNN Conformer Experiment - 2026-04-20 Backbone Redesign

## 1. 문서 범위

- 대상 모델: `cnn_conformer` 계열 후속 backbone 재설계 실험
- 문서 목적: 기존 conformer 실험의 한계를 바탕으로, backbone 재설계 screening 결과와 그 해석을 정리
- 현재 문서 상태: `reference`

## 2. 모델 스냅샷

### 2.1 한 줄 요약

이번 라운드는 기존 `CNN stem -> flatten frequency -> Conformer` 구조가 plateau에 도달한 것으로 보고, **front-end 압축과 tokenization 방식 자체를 다시 여는 conformer backbone 재설계**를 목표로 한다.

### 2.2 핵심 구성 요소

| 항목 | 값 또는 설명 |
|---|---|
| 입력 표현 | `log-Mel spectrogram` |
| 고정 조건 | SSL 미사용, `CNN vs Transformer` 비교 주제 유지 |
| 주 비교축 | CNN stem 강도, stem 제거 여부, frequency tokenization 방식 |
| 유지할 것 | Conformer encoder 자체, attentive pooling 계열, log-Mel 중심 실험 프레임 |
| 바꿀 것 | front-end/backbone token 생성 방식 |

### 2.3 비교 관점

- 비교 대상 1: 2026-04-17 conformer champion
- 비교 대상 2: CNN baseline
- 비교 대상 3: 2026-04-19 구조 탐색 및 round2 결과

이 문서의 목적은 “새 모델을 하나 더 만드는 것”이 아니라, **현재 conformer 계열이 왜 막혔는지에 대한 구조적 가설을 실험 가능한 형태로 다시 세우는 것**이다.

## 3. 실험 라운드 기록

### 3.1 공통 고정 조건

| 분류 | 항목 | 값 | 비고 |
|---|---|---|---|
| 입력 | 표현 | `log-Mel spectrogram` | 유지 |
| 입력 | 기준 해상도 | `n_mels=80`, `hop_length=160` | 현재 champion 라인과 비교 가능하게 고정 |
| 학습 | 기본 epoch | `30` | 1차 구조 검증 |
| 평가 | 기본 fold | `1 fold` | 1차 구조 screening |
| 목표 | 기준선 | CNN baseline, conformer champion `0.63168` | |

### 3.2 이전 실험에서 이미 확인된 것

| 회차 | 결론 | 이번 회차에 주는 의미 |
|---|---|---|
| 2026-04-17 | best conformer champion `0.63168` | 현재 계열의 기준점 |
| 2026-04-19 구조 탐색 | `time_preserve_first` 우세, `multiscale conv` 열세 | 압축 완화는 의미 있었고, conv branch 증설은 의미가 적었음 |
| 2026-04-19 round2 | `learned_sum + focal_loss`가 상대 우세였지만 champion 미회복 | 학습 objective 조정만으로는 한계 |

### 3.3 이번 라운드에서 제외하는 반복 축

| 제외 축 | 이유 |
|---|---|
| `multiscale conv branch` | 이미 열세 확인 |
| `layer_fusion` 단독 탐색 | `last`, `learned_sum`, `last2_mean`까지 확인 |
| `loss/sampler` 단독 탐색 | plateau 확인 |
| SSL branch | 현 논문 주제 밖 |
| 대규모 multi-feature fusion | `CNN vs Transformer` 주제 흐림 |

### 3.4 이번 라운드의 후보 구조

| 후보 | 구조 요약 | 핵심 가설 | 구현 난이도 |
|---|---|---|---|
| `lightstem_conformer` | 2-stage stem을 1-stage light conv로 축소 | 초기 과압축이 병목이라면 완화형 stem이 이득 | 낮음 |
| `nostem_patch_conformer` | CNN stem 제거, patch/token projection 후 Conformer | CNN subsampling 없이도 감정 cue 보존이 더 나을 수 있음 | 중간 |
| `band_token_conformer` | full flatten 대신 대역별 tokenization | frequency structure를 더 보존하면 혼동 클래스가 줄 수 있음 | 중간~높음 |

### 3.5 실행 명령

```powershell
python -m src.optuna_search model=cnn_conformer optuna=cnn_conformer_backbone_redesign optuna.enabled=true optuna.trials=30 train.epochs=30 train.folds_to_run=1 experiment.tag=backbone_redesign
```

권장 해석:

- 이 명령은 `lightstem`, `nostem_patch`, `band_token` 3개 backbone 후보를 같은 round에서 비교한다.
- 다만 각 variant 내부 탐색 공간은 작게 제한해 “누가 살아남는가”를 먼저 판별하는 screening 실험으로 쓴다.

### 3.6 회차별 실험 로그

| 회차 | 날짜 | 목적 | 설정 요약 | 결과 요약 | 산출 경로 |
|---|---|---|---|---|---|
| Round 6 | 2026-04-20 ~ 2026-04-21 | conformer backbone 재설계 screening | `lightstem / nostem_patch / band_token` 후보 비교 | `nostem_patch` 승리, broad search 종료 | `../../outputs/2026-04-20/15-01-58_cnn_conformer` |

실제 확인 경로:

- 실험 루트: [../../outputs/2026-04-20/15-01-58_cnn_conformer](../../outputs/2026-04-20/15-01-58_cnn_conformer)
- Hydra 설정 스냅샷: [../../outputs/2026-04-20/15-01-58_cnn_conformer/.hydra/config.yaml](../../outputs/2026-04-20/15-01-58_cnn_conformer/.hydra/config.yaml)
- Optuna 로그: [../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_search.log](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_search.log)
- Study DB: [../../optuna_studies/cnn_conformer_optuna_backbone_redesign.db](../../optuna_studies/cnn_conformer_optuna_backbone_redesign.db)
- 최고 성능 체크포인트: [../../outputs/2026-04-20/15-01-58_cnn_conformer/weights/best_model_fold1.pt](../../outputs/2026-04-20/15-01-58_cnn_conformer/weights/best_model_fold1.pt)

### 3.7 주요 결과 요약

| Rank | Trial | F1-macro | Accuracy | UAR | backbone 후보 | 핵심 파라미터 요약 |
|---|---|---:|---:|---:|---|---|
| 1 | `trial_0003` | 0.68536 | 0.67333 | 0.66562 | `nostem_patch` | `time_patch=2`, `embed=192`, `layers=4`, `ffn=768`, `batch=8` |
| 2 | `trial_0017` | 0.68186 | 0.68333 | 0.68750 | `nostem_patch` | `time_patch=2`, `embed=192`, `layers=4`, `ffn=768`, `batch=12` |
| 3 | `trial_0010` | 0.66615 | 0.65333 | 0.65625 | `nostem_patch` | `time_patch=2`, `embed=192`, `layers=4`, `ffn=768`, `batch=12` |
| 4 | `trial_0012` | 0.65480 | 0.65667 | 0.65312 | `nostem_patch` | `time_patch=2`, `embed=192`, `layers=4`, `ffn=768`, `batch=12` |
| 5 | `trial_0002` | 0.65462 | 0.65333 | 0.63750 | `nostem_patch` | `time_patch=4`, `embed=192`, `layers=4`, `ffn=768`, `batch=12` |

추가 집계:

- study DB: `../../optuna_studies/cnn_conformer_optuna_backbone_redesign.db`
- 상태 분포:
  - `COMPLETE`: 11
  - `PRUNED`: 61
  - `RUNNING`: 1
- best milestone:
  - `trial_0000`: `0.54786`
  - `trial_0001`: `0.64848`
  - `trial_0002`: `0.65462`
  - `trial_0003`: `0.68536`
- `trial_0003` 이후:
  - 추가 `COMPLETE`: 7
  - 추가 `PRUNED`: 61
  - 새 최고 없음

대표 trial 바로가기:

- 1위 trial 요약: [../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/trial_summary.json](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/trial_summary.json)
- 1위 trial metrics: [../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/summary_metrics.json](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/summary_metrics.json)
- 2위 trial 요약: [../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0017/trial_summary.json](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0017/trial_summary.json)
- 2위 trial metrics: [../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0017/artifacts/summary_metrics.json](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0017/artifacts/summary_metrics.json)

## 4. 설계 배경 및 구현 메모

### 4.1 왜 backbone을 다시 열어야 하는가

기존 실험 흐름을 종합하면, 지금 막힌 지점은 encoder 깊이/regularization/loss보다 **입력 token을 만드는 front-end 구조**에 더 가깝다.

관찰 근거:

- `standard_4x`보다 `time_preserve_first`가 낫다.
- `multiscale conv`, `layer fusion` 확장, `loss/sampler` 변경은 champion을 넘지 못했다.
- round2 최고점도 기존 champion보다 낮았다.
- confusion matrix에서 `sad / calm / disgust` 계열 혼동이 남아 있다.

즉 다음 실험은 “현재 backbone을 조금씩 조정”하는 게 아니라, **front-end 압축과 tokenization 방식을 바꿔서 encoder에 들어가는 표현 자체를 바꾸는 것**이어야 한다.

### 4.2 근거 논문 및 자료

| 자료 | 핵심 내용 | 이번 실험에 주는 의미 |
|---|---|---|
| Gulati et al. 2020 Conformer | ASR용 convolution subsampling 기반 encoder | SER에선 같은 압축 철학이 과할 수 있음 |
| Li et al. 2019 self-attention SER | spectrogram 직접 사용 + self-attention + salient period focus | CNN stem이 필수가 아니라는 근거 |
| Akinpelu et al. 2024 ViT SER | patch-based extraction이 SER에서 유효 | patch/token front-end 실험 정당화 |
| Li et al. 2023 MSTR | SER는 다중 시간 스케일 local pattern이 중요 | 시간축을 덜 뭉개는 backbone 방향 지지 |
| He et al. 2023 Cross-Attention Transformer | 단일 input transformer의 limitation 지적 | full flatten 대신 구조적 분리 tokenization 정당화 |

자료 링크:

- Conformer: <https://interspeech2020.org/index.php?a=show&c=index&catid=418&id=1331&m=content>
- Self-attention SER 2019: <https://www.isca-archive.org/interspeech_2019/li19n_interspeech.html>
- ViT SER 2024: <https://www.nature.com/articles/s41598-024-63776-4>
- ViT SER open mirror: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11161461/>
- Multi-Scale Temporal Transformer: <https://www.isca-archive.org/interspeech_2023/li23m_interspeech.html>
- Cross-Attention Transformer for SER: <https://resourcecenter.ieee.org/conferences/icassp-2023/spsicassp23vid0405>

### 4.3 자료를 바탕으로 이번에 실제로 무엇을 하려는가

#### A. `lightstem_conformer`

- 무엇을 바꾸나:
  - 2-stage `ConvStemBlock`을 1-stage로 줄임
  - stride는 주파수 위주 압축, 시간축은 최대한 보존
- 어떤 근거에서 왔나:
  - 기존 실험에서 압축 완화가 유효
  - Conformer 원형의 ASR용 aggressive subsampling이 SER에는 과할 수 있음
- 무엇을 검증하나:
  - “CNN stem은 필요하지만 지금보다 훨씬 약해야 하는가?”

#### B. `nostem_patch_conformer`

- 무엇을 바꾸나:
  - CNN stem 제거
  - log-Mel을 patch/token으로 투영 후 Conformer encoder에 직접 투입
- 어떤 근거에서 왔나:
  - self-attention SER와 ViT SER 계열은 spectrogram/patch를 더 직접적으로 사용
- 무엇을 검증하나:
  - “초기 CNN 압축 자체가 병목인가?”

#### C. `band_token_conformer`

- 무엇을 바꾸나:
  - 현재의 `channels x freq` full flatten 대신
  - 저/중/고 대역 또는 band group별 token 생성
- 어떤 근거에서 왔나:
  - Cross-attention SER 논문은 단일 source/단일 표현의 한계를 지적
  - 본 실험에서는 multi-feature가 아니라 `log-Mel 내부 구조 분리` 정도만 사용
- 무엇을 검증하나:
  - “frequency structure를 덜 섞으면 confusion-heavy class가 나아지는가?”

### 4.4 구현상 주의점

- 6GB GPU 제약 때문에 `nostem_patch`는 token 수를 강하게 제한해야 한다.
- `band_token`은 메모리보다 구현 복잡도가 문제다.
- `lightstem`은 가장 안전한 첫 후보다.

### 4.5 현재 코드 기준 구현 영향 범위

예상 수정 파일:

- 모델 코드:
  - `../../src/models/cnn_conformer.py`
- 모델 설정:
  - `../../src/configs/model/cnn_conformer.yaml`
- Optuna 설정:
  - `../../src/configs/optuna/cnn_conformer_backbone_redesign.yaml`
- search logic:
  - `../../src/optuna_search.py`

실제 구현 반영 요약:

- `cnn_conformer`에 `backbone_variant` 분기 추가
- front-end 모듈화:
  - `StandardCNNFrontEnd`
  - `LightStemFrontEnd`
  - `NoStemPatchFrontEnd`
  - `BandTokenFrontEnd`
- Hydra model config에 variant별 설정 추가
- Optuna에 `backbone_variant` 및 variant별 좁은 후보군 추가

## 5. 아티팩트 분석

### 5.1 이번 문서의 출발점이 되는 기존 문서

- champion: [2026-04-17.md](./2026-04-17.md)
- 구조 탐색: [2026-04-19.md](./2026-04-19.md)
- round2 중단 결론: [2026-04-19_round2.md](./2026-04-19_round2.md)

### 5.2 항목별 해석

대표 artifact 링크:

- 1위 trial learning curve: [../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/fold_1_learning_curve.png](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/fold_1_learning_curve.png)
- 1위 trial confusion matrix: [../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/global_confusion_matrix.png](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/global_confusion_matrix.png)
- 1위 trial calibration curve: [../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/global_calibration_curve.png](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/global_calibration_curve.png)
- 1위 trial ROC/PR: [../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/global_roc_pr_curves.png](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/global_roc_pr_curves.png)
- 1위 trial t-SNE: [../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/global_tsne_plot.png](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/global_tsne_plot.png)
- 1위 trial attention map: [../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/fold_1_attention_map.png](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/fold_1_attention_map.png)
- 1위 trial feature map: [../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/fold_1_cnn_feature_map.png](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/fold_1_cnn_feature_map.png)
- 2위 trial confusion matrix: [../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0017/artifacts/global_confusion_matrix.png](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0017/artifacts/global_confusion_matrix.png)
- 2위 trial calibration curve: [../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0017/artifacts/global_calibration_curve.png](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0017/artifacts/global_calibration_curve.png)

- learning curve:
  - [trial_0003 learning curve](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/fold_1_learning_curve.png) 기준으로 train accuracy가 꾸준히 상승해 `0.85+`까지 가지만, validation accuracy는 `0.65~0.67` 부근에서 흔들린다.
  - 이전 conformer 라인과 비교하면 여전히 overfitting은 남아 있지만, validation plateau 자체가 더 높은 수준에서 형성된다.
- confusion matrix:
  - [trial_0003 confusion matrix](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/global_confusion_matrix.png) 기준으로 `happy=0.85`, `angry=0.80`, `calm=0.72`, `fearful=0.68`, `sad=0.60`으로 전반적으로 향상됐다.
  - 남는 병목은 `neutral -> sad (0.35)`, `disgust -> sad (0.23)`, `surprised -> sad (0.23)` 계열이다.
- calibration:
  - [trial_0003 calibration curve](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/global_calibration_curve.png)와 [summary metrics](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0003/artifacts/summary_metrics.json) 기준으로 ECE는 `0.0565`다.
  - [trial_0017 summary metrics](../../outputs/2026-04-20/15-01-58_cnn_conformer/optuna_trials/trial_0017/artifacts/summary_metrics.json)에서는 ECE가 `0.0465`로 더 낮고, accuracy/UAR는 약간 더 높다. 즉 최고 F1 trial과 최고 calibration trial이 다르다는 점도 확인된다.
  - 단순 peak 상승뿐 아니라 calibration도 개선된 점이 중요하다.
- backbone 비교 관찰:
  - `lightstem`: complete 2개, 최고 `0.5479`로 사실상 탈락
  - `band_token`: complete 없음, pruned만 존재해 현재 설계로는 열세
  - `nostem_patch`: complete 9개, 평균 `0.6594`, 최고 `0.6854`
  - 즉 이번 회차의 핵심 결론은 “새 backbone 방향이 필요하다”가 아니라, **그중에서도 `nostem_patch`가 분명한 승자**라는 점이다.

## 6. 종합 인사이트 및 다음 액션

### 6.1 현재 판단

이번 회차는 성공한 screening 실험이다.

- 기존 conformer champion `0.63168`을 분명히 넘겼다.
- broad search 목적은 이미 달성되었다.
- 따라서 이 study를 계속 broad backbone search로 끌고 갈 필요는 낮다.

즉 판단은 다음과 같다.

- `lightstem`은 탈락
- `band_token`은 현재 설계에서는 탈락
- `nostem_patch`는 유지 및 후속 generalization 실험으로 승격

### 6.2 다음 액션

- broad backbone search는 여기서 종료한다.
- 다음 실험은 `nostem_patch`를 고정하고 overfitting 완화를 위한 generalization 라운드로 넘어간다.
- 후속 계획 문서:
  - [2026-04-21_nostem_generalization.md](./2026-04-21_nostem_generalization.md)

## 7. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-04-20 | backbone 재설계 실험 계획 문서 작성 및 템플릿 형식 보강 |
| 2026-04-21 | 실제 결과, artifact 해석, winner backbone 판정, broad search 종료 결론 반영 |
