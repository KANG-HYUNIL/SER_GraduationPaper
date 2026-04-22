# Pure Transformer

## 1. 문서 범위

- 대상 모델: `pure_transformer`
- 목적: 순수 transformer 기준선의 구조, 탐색 설정, 결과 요약을 공통 템플릿으로 정리
- 상태: `reference`

## 2. 모델 스냅샷

### 2.1 한 줄 요약

`pure_transformer`는 spectrogram을 patch 단위로 잘라 token으로 만든 뒤 곧바로 transformer encoder에 넣는, CNN stem이 없는 순수 transformer 기준선이다.

### 2.2 핵심 구성 요소

| 항목 | 값 또는 설명 |
|---|---|
| 입력 표현 | log-Mel spectrogram |
| 핵심 블록 | patch embedding + transformer encoder |
| 주요 구조 파라미터 | winner 기준 `patch_size=32`, `patch_stride=8`, `embed_dim=256`, `num_layers=5`, `num_heads=4`, `ffn_dim=1024` |
| 출력 pooling | winner 기준 `mean` |
| 분류 대상 | 8-class SER |

### 2.3 비교 관점

- `vaswani2017attention`에 가장 가까운 구조적 기준선이다.
- CNN이나 window, conformer 계열 구조의 효용을 비교할 때의 출발점 역할을 한다.

## 3. 실험 라운드 기록

### 3.1 실행 메모

- `pure_transformer`는 순수 transformer 기준선으로 유지한다.
- 예시 실행 명령:

```powershell
python -m src.optuna_search model=pure_transformer experiment.family=pure_transformer experiment.name=pure_transformer_optuna train.device=cuda optuna.trials=24 train.epochs=15 train.folds_to_run=1
```

### 3.2 주요 결과 요약

실험 기준 시점: `2026-04-15 13:44:11`

| Rank | Trial | F1-macro | Accuracy | UAR | logmel_n_mels | logmel_n_fft | logmel_hop | logmel_f_min | logmel_f_max | logmel_normalize | train_batch_size | train_learning_rate | train_weight_decay | transformer_dropout | transformer_embed_dim | transformer_ffn_ratio | transformer_num_heads | transformer_num_layers | transformer_patch_size | transformer_patch_stride | transformer_pooling |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `trial_0016` | 0.51163 | 0.52000 | 0.51250 | 64 | 2048 | 160 | 20.0 | 6000.0 | True | 16 | 3.90e-4 | 3.12e-4 | 0.271 | 256 | 4 | 4 | 5 | 32 | 8 | mean |
| 2 | `trial_0153` | 0.48436 | 0.50333 | 0.48438 | 64 | 1024 | 160 | 20.0 | 8000.0 | True | 8 | 2.09e-4 | 4.70e-5 | 0.100 | 256 | 2 | 4 | 5 | 32 | 16 | cls |
| 3 | `trial_0042` | 0.48262 | 0.49000 | 0.48438 | 64 | 2048 | 256 | 50.0 | 8000.0 | True | 16 | 1.81e-4 | 4.87e-5 | 0.286 | 256 | 2 | 4 | 5 | 32 | 16 | cls |
| 4 | `trial_0156` | 0.48111 | 0.48667 | 0.47187 | 64 | 1024 | 160 | 20.0 | 8000.0 | True | 8 | 1.47e-4 | 3.88e-5 | 0.102 | 256 | 2 | 4 | 5 | 32 | 16 | cls |
| 5 | `trial_0154` | 0.47742 | 0.47667 | 0.47188 | 64 | 1024 | 160 | 20.0 | 8000.0 | True | 8 | 1.78e-4 | 4.41e-5 | 0.100 | 256 | 2 | 4 | 5 | 32 | 16 | cls |

### 3.3 최종 winner 구조

- winner trial: `trial_0016`
- `embed_dim=256`
- `num_heads=4`
- `num_layers=5`
- `ffn_dim=1024`
- `patch_size=32`
- `patch_stride=8`
- `pooling=mean`
- `dropout=0.271`

```mermaid
flowchart LR
    A[Log-Mel Spectrogram] --> B[Conv2d Patch Embedding\n32x32, stride 8]
    B --> C[Flatten Patch Tokens]
    C --> D[Sinusoidal Positional Encoding]
    D --> E[TransformerEncoder x5\nembed 256, heads 4, ffn 1024]
    E --> F[Masked Mean Pooling]
    F --> G[Dropout 0.271]
    G --> H[Linear 8-class]
```

## 4. 설계 배경 및 구현 메모

### 4.1 설계 배경

`pure_transformer`는 spectrogram을 patch 단위로 잘라 token으로 만든 뒤, 곧바로 transformer encoder에 넣는다. CNN stem이 없고, local pattern을 먼저 압축해 주는 강한 inductive bias도 없다. 이 구조는 `vaswani2017attention`의 가장 순수한 해석에 가깝다.

- 입력을 token으로 만든다.
- 모든 token이 서로 attention한다.
- FFN을 거치며 표현을 업데이트한다.
- 마지막 pooled representation으로 감정 클래스를 예측한다.

### 4.2 현재 코드 기준 구현

- 입력: log-Mel spectrogram
- patch 분할: `patch_size`, `patch_stride`
- 선형 임베딩: patch를 `embed_dim` 차원으로 투영
- encoder 반복: `num_layers`
- attention heads: `num_heads`
- FFN 차원: `ffn_dim`
- 출력 pooling: `attention`, `mean`, `cls`

로직 흐름은 다음과 같다.

1. spectrogram을 patch로 자른다.
2. 각 patch를 token embedding으로 바꾼다.
3. 모든 token이 서로 attention한다.
4. 여러 layer를 통과하며 전역 문맥을 누적한다.
5. 마지막 token 집합을 pooling해 하나의 utterance embedding으로 만든다.
6. classifier가 8개 감정 클래스를 예측한다.

### 4.3 장단점 및 SER 관점 해석

- 장점:
  - transformer 자체의 기준선으로 해석이 쉽다.
  - 전역 문맥을 가장 직접적으로 본다.
  - "CNN 없이도 되는가"를 보는 비교 기준이 된다.
- 약점:
  - local time-frequency pattern을 초기에 안정적으로 추출해 주는 구조가 없다.
  - sequence가 길어질수록 global attention 비용이 커진다.
  - 작은 SER 데이터셋에서는 학습 분산이 커질 수 있다.
- SER 관점 해석:
  - SER에서는 local cue가 중요한데, pure transformer는 그 cue를 직접 학습해야 한다.
  - 데이터 규모가 작고 사전학습이 없는 상황에서는 이 점이 약점이 되기 쉽다.
  - 그래서 이 모델은 기준선으로는 유용하지만 최종 성능 후보로는 다소 불리하다.

## 5. 아티팩트 분석

- 현재 문서에는 대표 trial artifact 분석을 별도로 정리하지 않았다.
- 필요 시 `../outputs/.../optuna_trials/.../artifacts/` 기준으로 후속 추가한다.

## 6. 종합 인사이트 및 다음 액션

- `pure_transformer`는 유지하되, 주력 모델로 확장하기보다 비교 기준선으로 두는 것이 적절하다.
- 이후 비교는 `cnn_conformer` 또는 window 계열 구조 대비 성능 격차와 그 원인을 설명하는 방향이 맞다.

## 7. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-04-19 | 공통 템플릿 기준으로 문서 구조 정리 |
