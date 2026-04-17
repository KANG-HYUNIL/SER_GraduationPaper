# CNN Conformer

## 모델 개요

`cnn_conformer`는 CNN stem으로 지역적인 시간-주파수 패턴을 먼저 추출하고, 이후 Conformer block으로 장기 시간 문맥을 통합하는 SER 모델이다. 이번 개정의 핵심 목표는 원조 Conformer 논문(Gulati et al., 2020)의 설계 철학에 더 가깝게 돌아가면서, 기존 구현에서 발생하던 과도한 주파수 축 붕괴를 제거하는 것이다.

핵심 구성 요소는 다음과 같다.

- CNN stem: `stem_channels`
- Conformer 차원: `embed_dim`
- 레이어 수: `num_layers`
- 헤드 수: `num_heads`
- FFN 차원: `ffn_dim`
- Conformer conv kernel: `conv_kernel_size`
- utterance pooling: `attention` 또는 `mean`

## 기존 Optuna 결과 Top 5

실험 경로: `outputs/2026-04-15/20-13-31_thesis_transformer_stage2_cnn_conformer`

| Rank | Trial | F1-macro | Accuracy | UAR | logmel_n_mels | logmel_n_fft | logmel_hop | logmel_normalize | train_batch | train_lr | train_wd | conformer_stem | conformer_embed_dim | conformer_layers | conformer_heads | conformer_ffn | conformer_kernel | conformer_dropout | conformer_pooling |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `trial_0095` | 0.57946 | 0.59000 | 0.57188 | 64 | 2048 | 256 | False | 32 | 1.57e-4 | 4.28e-4 | `[64, 96]` | 256 | 5 | 8 | 4 | 15 | 0.101 | attention |
| 2 | `trial_0011` | 0.57833 | 0.59667 | 0.58437 | 64 | 2048 | 256 | False | 32 | 2.78e-4 | 8.06e-4 | `[64, 96]` | 256 | 2 | 8 | 4 | 15 | 0.110 | attention |
| 3 | `trial_0094` | 0.57576 | 0.57333 | 0.55937 | 64 | 2048 | 256 | False | 32 | 1.73e-4 | 4.29e-4 | `[64, 96]` | 256 | 5 | 8 | 4 | 15 | 0.101 | attention |
| 4 | `trial_0014` | 0.57116 | 0.57333 | 0.55625 | 64 | 2048 | 256 | False | 32 | 1.96e-4 | 2.44e-4 | `[64, 96]` | 256 | 3 | 8 | 4 | 15 | 0.166 | attention |
| 5 | `trial_0042` | 0.56654 | 0.57667 | 0.56250 | 64 | 2048 | 256 | False | 32 | 2.47e-4 | 8.06e-4 | `[64, 96]` | 256 | 2 | 8 | 4 | 15 | 0.112 | attention |

## padding-safe 적용 후 Optuna 결과 Top 5

실험 경로: `outputs/2026-04-16/15-04-48_cnn_conformer_padding_safe_stage2`

| Rank | Trial | F1-macro | Accuracy | UAR | logmel_n_mels | logmel_n_fft | logmel_hop | logmel_normalize | train_batch | train_lr | train_wd | conformer_stem | conformer_embed_dim | conformer_layers | conformer_heads | conformer_ffn | conformer_kernel | conformer_dropout | conformer_pooling |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `trial_0038` | 0.49378 | 0.50667 | 0.49688 | 96 | 2048 | 512 | False | 32 | 3.37e-4 | 1.64e-6 | `[96, 96]` | 128 | 2 | 4 | 512 | 31 | 0.122 | attention |
| 2 | `trial_0041` | 0.45532 | 0.49000 | 0.49062 | 96 | 2048 | 512 | False | 32 | 3.72e-4 | 2.51e-4 | `[64, 96]` | 128 | 2 | 4 | 512 | 31 | 0.123 | mean |
| 3 | `trial_0040` | 0.45020 | 0.49667 | 0.48750 | 96 | 2048 | 512 | False | 32 | 3.55e-4 | 1.36e-4 | `[64, 96]` | 128 | 2 | 4 | 512 | 31 | 0.119 | mean |
| 4 | `trial_0035` | 0.44603 | 0.48333 | 0.47188 | 64 | 1024 | 512 | False | 32 | 2.26e-4 | 2.04e-5 | `[96, 96]` | 128 | 3 | 8 | 256 | 31 | 0.157 | attention |
| 5 | `trial_0043` | 0.44521 | 0.48000 | 0.47813 | 96 | 2048 | 512 | False | 32 | 3.49e-4 | 1.24e-4 | `[64, 96]` | 128 | 2 | 4 | 512 | 31 | 0.132 | mean |

## 수정 전후 차이

### 설정 차이

`trial_0095`와 `trial_0038`는 둘 다 `resize_enabled=false`인 가변 길이 log-Mel 실험이지만, 입력과 모델 용량은 다르다.

- 입력 표현
  - 이전 best: `n_mels=64`, `hop_length=256`, `f_min=20`
  - padding-safe best: `n_mels=96`, `hop_length=512`, `f_min=50`
- 모델 용량
  - 이전 best: `embed_dim=256`, `num_layers=5`, `num_heads=8`, `ffn_dim=1024`
  - padding-safe best: `embed_dim=128`, `num_layers=2`, `num_heads=4`, `ffn_dim=512`
- convolution 시간 범위
  - 이전 best: `conv_kernel=15`
  - padding-safe best: `conv_kernel=31`

즉, 성능 차이를 단순히 padding-safe 처리만으로 해석하면 안 되고, search space가 더 작은 모델과 더 거친 시간 해상도로 수렴한 영향도 함께 봐야 한다.

### 설계 내부 차이

이번 SOTA-faithful 개정 이전의 중간 구조는 주파수 축을 `AdaptiveAvgPool2d`로 줄인 뒤 일부 band만 남겨 토큰화하는 방식이었다. 이 방식은 예전 구현보다 낫지만, 여전히 원조 Conformer의 convolution subsampling 철학과는 다르다.

현재 구조는 다음처럼 바뀌었다.

- CNN stem의 두 블록 모두 `stride=(2, 2)`를 사용한다.
  - 시간축과 주파수축을 균형 있게 4배 압축한다.
  - 이전처럼 한 축만 강하게 줄이는 비대칭 stride를 쓰지 않는다.
- `AdaptiveAvgPool2d((freq_bins, None))`를 제거했다.
  - 이제 stem 출력의 남아 있는 전체 주파수 차원을 그대로 유지한다.
  - 마지막 특징맵을 `channels x remaining_freq`로 펼친 뒤 `Linear`로 `embed_dim`에 투영한다.
- `remaining_freq`는 `n_mels`와 stem stride를 기준으로 동적으로 계산한다.
  - `n_mels=80`이면 현재 stem 설정에서 `80 -> 40 -> 20`으로 줄어든다.
  - 따라서 투영 입력 차원은 `stem_channels[-1] x 20`이 된다.
- `key_padding_mask`는 시간축 길이만 따라가며 유지된다.
  - 각 conv 직후 `apply_channel_mask_2d`
  - 투영 후 `apply_sequence_mask`
  - 각 Conformer block 뒤 다시 `apply_sequence_mask`

즉, 이번 개정의 핵심은 “주파수 band를 일부만 남기는 구조”에서 “CNN이 만들어낸 남은 주파수 해상도를 전부 보존한 뒤 토큰화하는 구조”로 바뀐 점이다.

## 원조 SOTA 설계로의 회귀 및 주파수 정보 보존(Flatten) 방식 적용

이번 변경은 Gulati et al., 2020의 원조 Conformer 논문이 제시한 세 가지 축을 가능한 한 현재 SER 코드베이스 안으로 가져오려는 작업이다.

- 80채널 filterbank를 기본 전제로 둔다.
- convolution subsampling으로 시퀀스 길이를 줄인 뒤 Conformer encoder를 적용한다.
- 모델 크기는 small~medium 범위(`d_model=144/192/256`, `heads=4/8`, `layers=4~16`, `FFN=4x`)를 중심으로 탐색한다.

참고한 원문과 근거는 다음과 같다.

- Gulati et al., 2020, *Conformer: Convolution-augmented Transformer for Speech Recognition*
  - 80-channel filterbank
  - convolution subsampling
  - small/medium 모델에서 `encoder dim=144/256`, `heads=4`, `kernel size≈32`, `FFN expansion=4`
- `ref.bib`
- MFA-Conformer
- Co-Attention 기반 다중 음향 정보 융합 논문

이번 구조는 원문을 그대로 복제한 ASR encoder는 아니지만, SER 환경에서 다음 부분을 충실히 반영했다.

- stem은 주파수와 시간을 함께 줄인다.
- 토큰화 전에 주파수 정보를 평균으로 무너뜨리지 않는다.
- Conformer에 들어가는 입력 차원을 stem 출력 해상도에 맞춰 정합시킨다.
- self-attention은 `cnn_conformer` 전용 relative positional MHSA로 분리한다.

### relative positional MHSA 적용 방식 비교

`cnn_conformer`에만 원조 Conformer 계열 attention을 넣기 위해 세 가지 방식을 검토했다.

- 공용 `transformer_blocks.py`의 `ConformerBlock`을 직접 수정
  - 장점: 코드 중복이 적다.
  - 단점: `hierarchical_window_transformer`, 다른 Transformer 계열까지 같이 영향을 받아 회귀 위험이 크다.
- `cnn_conformer.py` 내부에 attention 구현을 전부 인라인으로 작성
  - 장점: 한 파일에서 다 보인다.
  - 단점: 모델 파일이 과도하게 비대해지고, block 단위 재사용과 실험 분기가 불편하다.
- `cnn_conformer` 전용 block 파일을 분리하고 Hydra로 attention 종류를 관리
  - 장점: 공용 block와 격리되어 다른 모델에 영향이 없다.
  - 장점: `model.attention_type=relative/absolute`로 바로 ablation 가능하다.
  - 단점: 파일이 하나 더 생긴다.

최종적으로 세 번째 방식을 채택했다. 현재 구현은 [cnn_conformer_blocks.py](C:\Users\KANG\Desktop\BIT_Uni_record\GraudationPaper\SER_GraduationPaper\src\models\cnn_conformer_blocks.py)에 전용 relative positional MHSA를 두고, [cnn_conformer.yaml](C:\Users\KANG\Desktop\BIT_Uni_record\GraudationPaper\SER_GraduationPaper\src\configs\model\cnn_conformer.yaml)에서 `attention_type`, `max_relative_position`을 Hydra로 관리한다.

### 코드 변경 상세

대상 파일: [cnn_conformer.py](C:\Users\KANG\Desktop\BIT_Uni_record\GraudationPaper\SER_GraduationPaper\src\models\cnn_conformer.py)

- 두 개의 `ConvStemBlock` 모두 `stride=(2, 2)`로 통일
- `FrequencyBandProjector` 제거
- `FlattenFrequencyProjector` 추가
  - stem 마지막 출력의 전체 주파수 차원을 flatten
  - `LayerNorm -> Linear(embed_dim)` 순서로 투영
- `remaining_freq`를 `n_mels`와 stem 설정으로부터 동적으로 계산
- conv 직후와 encoder 내부에서 padding mask 재적용
- `cnn_conformer_blocks.py` 추가
  - `RelativePositionMultiHeadAttention`
  - `CNNConformerBlock`
- `model.attention_type`
  - 기본값 `relative`
  - 필요 시 `absolute`로 바꿔 기존 MHA와 직접 비교 가능
- `model.max_relative_position`
  - 상대 위치 bias가 보는 최대 거리 제어

대상 파일: [optuna_search.py](C:\Users\KANG\Desktop\BIT_Uni_record\GraudationPaper\SER_GraduationPaper\src\optuna_search.py)

- `embed_dim`: `[144, 192, 256]`
- `num_layers`: `[4, 8, 12, 16]`
- `num_heads`: `[4, 8]`
- `conv_kernel_size`: `[15, 31]`
- `ff_expansion_factor`: `4` 고정
- `freq_bins` 탐색 제거

대상 파일: [default.yaml](C:\Users\KANG\Desktop\BIT_Uni_record\GraudationPaper\SER_GraduationPaper\src\configs\optuna\default.yaml)

- `cnn_conformer` 탐색 공간을 원조 Conformer small~medium 축으로 재조정
- `logmel.n_mels_choices`에 `80` 포함 유지

## 왜 주파수 Flatten이 필요한가

SER에서는 특정 시간 구간에서 어떤 주파수대가 동시에 활성화되는지가 중요하다. 기존의 강한 pooling은 다음 정보를 너무 빨리 잃게 만든다.

- formant 상대 위치
- 고주파 마찰음의 에너지
- 저주파와 고주파의 동시 발현 패턴
- 감정별 스펙트럼 대비

Flatten 방식은 이 정보를 전부 영구 보존하는 것은 아니지만, 적어도 Conformer encoder에 들어가기 전 단계에서 주파수 구조를 평균 하나로 붕괴시키지 않는다는 점에서 훨씬 안전하다.


## SOTA-Faithful Refined 탐색 실험 (2026-04-16)
### 실험 개요
이전 실험들에서 발생한 정보 소실 문제를 근본적으로 해결하기 위해, 원조 Conformer 논문(Gulati et al., 2020)의 설계를 최대한 온전히 반영한 실험군이다. 오디오 전처리(Log-Mel)는 학술적 표준 수치로 완전히 고정하고, 남은 Optuna 자원을 오로지 **모델 아키텍처의 깊이와 넓이 최적화**에만 집중 투자한다.


### 명령어
python -m src.optuna_search model=cnn_conformer experiment.family=cnn_conformer experiment.name=cnn_conformer_fixed_logmel_sota_relative train.device=cuda train.epochs=30 train.folds_to_run=1 train.num_workers=0 optuna.trials=30 +optuna.search_space.logmel.enabled=false data.n_mels=80 data.n_fft=512 data.hop_length=160 data.f_min=0.0 data.f_max=8000.0 data.normalize=true model.attention_type=relative model.max_relative_position=128


### 고정 파라미터 (Log-Mel & 기본 설정)
원조 논문의 하이퍼파라미터를 따르되, 구현 편의성과 현재 하드웨어 환경을 고려하여 일부 조정하였다.
| 분류 | 파라미터 | 고정 값 | 비고 |
| :--- | :--- | :--- | :--- |
| **Log-Mel** | `n_mels` | 80 | 원조 논문 표준 |
| | `n_fft` | 512 | (논문 400ms 근사치) 효율 최적화 |
| | `hop_length` | 160 | 10ms stride (표준) |
| | `f_min / f_max` | 0 / 8000 | 전대역 정보 수용 |
| | `normalize` | True | Global CMVN 대용 |
| **모델 구조** | `ffn_ratio` | 4 | Conformer 표준 확장 비율 |
| | `attention_type` | relative | 상대 위치 인코딩 적용 |
| | `max_relative_pos`| 128 | 약 2.5초 문맥 커버 |



### Optuna 탐색 후보군 (모델 아키텍처 집중)
전처리 조합이 사라진 만큼, 모델의 성능을 결정짓는 핵심 변수들을 더 넓고 깊게 탐색한다.
| 파라미터 | 탐색 범위 (Candidate) | 비고 |
| :--- | :--- | :--- |
| `embed_dim` | [144, 192, 256] | 모델의 넓이 (Small ~ Medium) |
| `num_layers` | [4, 8, 12, 16] | 모델의 깊이 (Encoder Blocks) |
| `num_heads` | [4, 8] | 멀티헤드 어텐션 병렬 수 |
| `conv_kernel` | [15, 31] | Depthwise Conv 시간축 시야 |
| `stem_channels`| [32, 48, 64, 96] | CNN Stem 특징 추출 깊이 |
| `learning_rate`| 1e-4 ~ 3e-3 | 학습 속도 (Log-scale) |



### 실험 결과 (Top 5)
| Rank | Trial | F1-macro | Accuracy | embed_dim | layers | heads | kernel | LR |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | - | - | - | - | - | - | - | - |
| 2 | - | - | - | - | - | - | - | - |
| 3 | - | - | - | - | - | - | - | - |
| 4 | - | - | - | - | - | - | - | - |
| 5 | - | - | - | - | - | - | - | - |


