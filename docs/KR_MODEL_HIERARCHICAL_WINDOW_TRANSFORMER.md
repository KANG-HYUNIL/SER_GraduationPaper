# Hierarchical Window Transformer

## 1. 문서 범위

- 대상 모델: `hierarchical_window_transformer`
- 목적: 2-stage window 계열 SER 모델의 구조, 실험 결과, 한계 분석을 공통 템플릿으로 정리
- 상태: `reference`

## 2. 모델 스냅샷

### 2.1 한 줄 요약

`hierarchical_window_transformer`는 CNN stem으로 초기 시간-주파수 특징을 추출한 뒤, 2단계의 2D shifted window attention으로 지역 문맥을 계층적으로 확장하는 SER 모델이다.

### 2.2 핵심 구성 요소

| 항목 | 값 또는 설명 |
|---|---|
| 입력 표현 | log-Mel spectrogram |
| 핵심 블록 | CNN stem + 2-stage 2D shifted window attention |
| 주요 구조 파라미터 | `stem_channels`, `stage_dims`, `stage_depths`, `num_heads`, `window_sizes`, `ffn_ratio` |
| 출력 pooling | `attention`, `mean` |
| 분류 대상 | 8-class SER |

### 2.3 비교 관점

- `pure_transformer`처럼 처음부터 전역 attention을 쓰지 않고 계산량을 줄이려는 목적이 있었다.
- `cnn_conformer`처럼 시간축만 보는 구조가 아니라, 주파수축을 stage 후반까지 유지하면서 spectro-temporal 지역 구조를 직접 모델링하려는 모델이었다.
- 현재 구현은 Speech Swin-Transformer와 DWFormer에서 영감을 받았지만, 둘 중 하나를 그대로 재현한 모델은 아니다. 정확히는 "Swin 계열 아이디어를 SER 코드베이스에 맞게 단순화한 2-stage 2D window transformer"에 가깝다.

## 3. 실험 라운드 기록

### 3.1 주요 결과 요약

실험 경로: `../outputs/2026-04-16/01-09-06_hierarchical_window_cnnfixed_stage2`

| Rank | Trial | F1-macro | Accuracy | UAR | train_batch | train_lr | train_wd | window_stem_pair | window_stage_spec | window_depth_pair | window_size | window_ffn | window_dropout | window_pooling |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|
| 1 | `trial_0144` | 0.48765 | 0.50333 | 0.51250 | 8 | 1.97e-4 | 2.97e-5 | `[48, 64]` | `128x192_h4x8` | `2x3` | 12 | 2 | 0.108 | attention |
| 2 | `trial_0105` | 0.48235 | 0.51000 | 0.53125 | 8 | 2.30e-4 | 1.34e-5 | `[48, 64]` | `128x192_h4x8` | `2x3` | 12 | 2 | 0.178 | attention |
| 3 | `trial_0041` | 0.47917 | 0.49000 | 0.49062 | 8 | 2.25e-4 | 1.20e-5 | `[48, 64]` | `128x192_h4x8` | `2x3` | 12 | 2 | 0.100 | attention |
| 4 | `trial_0033` | 0.47570 | 0.50000 | 0.51250 | 8 | 2.20e-4 | 1.80e-5 | `[48, 64]` | `128x192_h4x8` | `2x3` | 12 | 2 | 0.101 | attention |
| 5 | `trial_0012` | 0.47271 | 0.48333 | 0.48750 | 16 | 2.08e-4 | 2.78e-5 | `[48, 64]` | `128x192_h4x8` | `2x3` | 12 | 2 | 0.103 | attention |

### 3.2 결과 요약 메모

- 현재 구조는 `cnn_conformer`의 0.63대 성능과 비교하면 상당한 격차가 있다.
- search 결과도 매우 좁은 한 지점으로 수렴했다.
- 상위 trial이 거의 모두 `stem=[48,64]`, `stage=[128,192]`, `depth=2x3`, `window=12`, `attention pooling` 근처에 몰렸다.

## 4. 설계 배경 및 구현 메모

### 4.1 현재 코드 기준 구조

관련 파일:

- `../src/models/hierarchical_window_transformer.py`
- `../src/models/hierarchical_window_blocks.py`
- `../src/configs/model/hierarchical_window_transformer.yaml`
- `../src/configs/optuna/hierarchical_window_cnnfixed.yaml`

처리 흐름은 다음과 같다.

1. 입력은 log-Mel spectrogram `[B, 1, F, T]`이다.
2. `ConvStemBlock` 두 개가 `stride=(2, 2)`로 시간축과 주파수축을 각각 4배 downsample한다.
3. `SpatialProjector`가 stem 출력을 `stage_dims[0]` 채널로 투영한다.
4. Stage 1에서 `WindowTransformerBlock2D`를 여러 층 반복한다.
5. `PatchMerging2D`가 2x2 이웃 patch를 합쳐 해상도를 줄이고 채널을 늘린다.
6. Stage 2에서 다시 `WindowTransformerBlock2D`를 반복한다.
7. 마지막에 주파수축 평균을 거친 뒤 시간축 utterance pooling을 수행한다.
8. classifier가 8개 감정을 분류한다.

즉, "2-stage 2D shifted-window backbone + late temporal pooling" 구조다.

### 4.2 실제 SOTA 원본과의 차이

#### Speech Swin-Transformer와의 차이

Speech Swin-Transformer는 4-stage 구조이며, time-domain patch 분할과 stage별 patch merging을 통해 receptive field를 점진적으로 넓힌다. 반면 현재 `hierarchical_window_transformer`는 다음 차이가 있다.

- 4-stage가 아니라 2-stage다.
- 논문처럼 relative position bias를 두지 않고, PyTorch `nn.MultiheadAttention` 기반의 단순 window attention이다.
- shifted window도 Swin의 cyclic-shift + mask가 아니라 padding 기반 이동에 가깝다.
- window 크기가 stage별로 충분히 세분화되어 있지 않다.

#### DWFormer와의 차이

DWFormer는 더 강한 아이디어를 사용한다.

- 입력으로 `Pre-trained WavLM-Large` feature를 사용
- 중요도 기반 dynamic window split
- local window attention + cross-window interaction

현재 `hierarchical_window_transformer`에는 다음이 없다.

- SSL feature backbone
- dynamic window partition
- local/global dynamic window interaction block

따라서 DWFormer와는 구조적 거리가 더 멀다.

### 4.3 왜 성능이 낮은가

1. 전역 정보 주입이 약하다.  
현재 구조는 local window와 shifted window를 거의 기계적으로만 사용한다. `cnn_conformer`처럼 전체 시간축 attention을 쓰는 구조보다 utterance-level 정서를 모으는 능력이 약하다.

2. Swin의 핵심 구현 요소가 일부 비어 있다.  
relative position bias와 정식 shifted-window attention mask가 없으면 window 경계 근처 모델링 이득이 약해진다.

3. stage 수가 적다.  
2-stage는 계산량 면에서는 유리하지만 hierarchical receptive field를 충분히 확장하기엔 부족할 수 있다.

4. 현재 search space가 구조적으로 좁다.  
결과가 한 지점으로만 몰린 것은 좋은 구조를 찾았다기보다, 실험 가능한 구조 다양성이 적었음을 시사한다.

## 5. 아티팩트 분석

- 현재 문서에는 대표 trial artifact를 별도 분석하지 않았다.
- 이 모델은 결과 요약과 구조적 한계 분석 위주로 유지한다.

## 6. 종합 인사이트 및 다음 액션

### 6.1 현재 판단

- 이 모델을 그대로 추가 Optuna로 오래 미는 것은 우선순위가 낮다.
- 구조적 약점이 분명해 후속 확장 모델의 출발점으로 보는 편이 맞다.

### 6.2 다음 액션

이번 작업에서 이 한계를 바탕으로 분기 모델 `bridged_window_transformer`를 추가했다. 핵심 차이는 다음과 같다.

- relative position bias를 포함한 window attention
- Swin 방식에 가까운 cyclic shift + attention mask
- rectangular window 탐색
- local window backbone은 유지하되, stage 사이와 최종 표현에 global bridge context를 주입

관련 문서:

- `./KR_MODEL_BRIDGED_WINDOW_TRANSFORMER.md`

즉, 기존 `hierarchical_window_transformer`는 "window 계열의 출발점"으로 두고, thesis의 확장 실험은 bridge 계열로 넘어가는 것이 적절하다.

## 7. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-04-19 | 공통 템플릿 기준으로 문서 구조 정리 |
