# Hierarchical Window Transformer

## 모델 개요

`hierarchical_window_transformer`는 CNN stem으로 초기 시간-주파수 패턴을 추출한 뒤, 계층형 window attention으로 지역 문맥을 쌓아 올리는 SER 모델이다. 이번 개편의 목적은 기존의 1D 시간 윈도우 기반 구현에서 벗어나, Speech Swin-Transformer와 DWFormer가 강조하는 “주파수 축을 보존한 상태의 window modeling”에 더 가깝게 맞추는 것이다.

현재 구조의 핵심은 다음과 같다.

- CNN stem: `stem_channels`
- stage 1: 2D shifted window attention
- patch merging: stage 1과 stage 2 사이의 계층형 다운샘플링
- stage 2: 더 넓은 receptive field를 가진 2D shifted window attention
- 최종 pooling: 주파수 평균을 마지막에만 수행하고 시간축에서 utterance pooling

## 기존 Optuna 결과 Top 5

실험 경로: `outputs/2026-04-16/01-09-06_hierarchical_window_cnnfixed_stage2`

| Rank | Trial | F1-macro | Accuracy | UAR | train_batch | train_lr | train_wd | window_stem_pair | window_stage_spec | window_depth_pair | window_size | window_ffn | window_dropout | window_pooling |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|
| 1 | `trial_0144` | 0.48765 | 0.50333 | 0.51250 | 8 | 1.97e-4 | 2.97e-5 | `[48, 64]` | `128x192_h4x8` | `2x3` | 12 | 2 | 0.108 | attention |
| 2 | `trial_0105` | 0.48235 | 0.51000 | 0.53125 | 8 | 2.30e-4 | 1.34e-5 | `[48, 64]` | `128x192_h4x8` | `2x3` | 12 | 2 | 0.178 | attention |
| 3 | `trial_0041` | 0.47917 | 0.49000 | 0.49062 | 8 | 2.25e-4 | 1.20e-5 | `[48, 64]` | `128x192_h4x8` | `2x3` | 12 | 2 | 0.100 | attention |
| 4 | `trial_0033` | 0.47570 | 0.50000 | 0.51250 | 8 | 2.20e-4 | 1.80e-5 | `[48, 64]` | `128x192_h4x8` | `2x3` | 12 | 2 | 0.101 | attention |
| 5 | `trial_0012` | 0.47271 | 0.48333 | 0.48750 | 16 | 2.08e-4 | 2.78e-5 | `[48, 64]` | `128x192_h4x8` | `2x3` | 12 | 2 | 0.103 | attention |

## 수정 후 Optuna 결과 기록

아직 이번 2D window 개편 버전으로 Optuna 학습 실험은 수행하지 않았다. 현재 서버에서 다른 실험이 동작 중이므로, 이번 작업에서는 실제 학습 대신 shape 정합성과 파이프라인 호환성만 검증했다.

| Rank | Trial | F1-macro | Accuracy | UAR | train_batch | train_lr | train_wd | window_stem_pair | window_stage_spec | window_depth_pair | window_size | window_ffn | window_dropout | window_pooling |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|
| 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## 기존 구현의 한계

기존 구현은 stem 이후 `AdaptiveAvgPool2d`로 주파수 축을 소수의 band로 줄인 다음, 사실상 1D 시간 시퀀스로 window attention을 수행했다. 이 구조의 문제는 다음과 같다.

- window attention이 시간축 기준 local sequence 모델처럼 동작한다.
- 같은 시간 구간 내의 주파수 조합 정보를 window 내부에서 직접 다루지 못한다.
- stage downsample 역시 `Conv1d`라서 Swin 계열의 patch merging과 성격이 다르다.

즉, 기존 구조는 이름은 window transformer이지만 실제로는 “주파수 collapse 이후의 1D shifted window”에 더 가까웠다.

## 논문 기준 분석과 채택 방안

참고 자료:

- `ref.bib`
- Wang et al., 2024, *Speech Swin-Transformer: Exploring a Hierarchical Transformer with Shifted Windows for Speech Emotion Recognition*
- Chen et al., 2023, *DWFormer: Dynamic Window Transformer for Speech Emotion Recognition*

핵심 해석은 다음과 같다.

- Speech Swin-Transformer는 spectrogram patch를 2D 시간-주파수 격자로 유지한 뒤, shifted window와 patch merging으로 계층형 표현을 만든다.
- DWFormer는 window 크기를 내용에 따라 더 유연하게 다루려는 접근이지만, 공통된 방향은 “중요 구간을 1D 평균으로 너무 빨리 붕괴시키지 않는다”는 점이다.

이번 코드베이스에서는 두 논문의 공통 철학을 반영하면서도 Optuna 파이프라인과 시각화 도구를 유지해야 했기 때문에, 다음 설계를 채택했다.

- **채택:** 2D time-frequency shifted window + patch merging
- **미채택:** 주파수 flatten 후 1D window

채택 이유:

- Speech Swin-Transformer와 더 직접적으로 정합된다.
- 주파수 축을 stage 전반에서 유지할 수 있다.
- shift가 시간축과 주파수축 모두에서 의미 있게 작동한다.
- stage 간 다운샘플링을 Swin식 patch merging으로 바꿀 수 있다.

## 수정 후 구조

이번 개편에서 바뀐 점은 다음과 같다.

- `FrequencyBandProjector` 제거
- `SpatialProjector` 추가
  - stem 출력의 2D feature map을 그대로 `stage_dims[0]` 채널로 투영
  - 주파수 평균 pooling 없음
- `WindowTransformerBlock2D` 추가
  - 2D window partition
  - 2축 shifted window
  - window 내부 attention
- `PatchMerging2D` 추가
  - 인접 2x2 patch를 채널로 합친 뒤 선형 축소
  - stage 1 -> stage 2 계층형 다운샘플링
- 최종 pooling만 late collapse
  - stage 2 이후에만 주파수 평균
  - utterance pooling은 기존 attention/mean 인터페이스 유지

즉, 이제 이 모델은 “주파수를 미리 접은 1D window transformer”가 아니라, “2D spectro-temporal grid 위에서 local-global receptive field를 키워 가는 구조”가 됐다.

## 파일 분리와 기존 실험 보호

이번 작업에서는 window 전용 block를 별도 파일로 분리했다.

- 새 파일: [hierarchical_window_blocks.py](C:\Users\KANG\Desktop\BIT_Uni_record\GraudationPaper\SER_GraduationPaper\src\models\hierarchical_window_blocks.py)
- 메인 모델: [hierarchical_window_transformer.py](C:\Users\KANG\Desktop\BIT_Uni_record\GraudationPaper\SER_GraduationPaper\src\models\hierarchical_window_transformer.py)

이렇게 한 이유는 다음과 같다.

- `cnn_conformer`, `pure_transformer`와 block 구현이 섞이지 않는다.
- window 관련 실험을 모델 단위로 독립적으로 수정할 수 있다.
- 이후 `dynamic window`, `rectangular window`, `cross-stage bridge` 같은 실험도 이 파일 안에서 확장하기 쉽다.

## Optuna 파이프라인 정합성

이번 구조에서는 `freq_bins`가 더 이상 존재하지 않으므로, Optuna 쪽도 같이 정리했다.

- [optuna_search.py](C:\Users\KANG\Desktop\BIT_Uni_record\GraudationPaper\SER_GraduationPaper\src\optuna_search.py)
  - `suggest_hierarchical_window_params()`에서 `window_freq_bins` 제거
- [default.yaml](C:\Users\KANG\Desktop\BIT_Uni_record\GraudationPaper\SER_GraduationPaper\src\configs\optuna\default.yaml)
- [hierarchical_window_cnnfixed.yaml](C:\Users\KANG\Desktop\BIT_Uni_record\GraudationPaper\SER_GraduationPaper\src\configs\optuna\hierarchical_window_cnnfixed.yaml)
  - `hierarchical_window.freq_bin_choices` 제거
- [hierarchical_window_transformer.yaml](C:\Users\KANG\Desktop\BIT_Uni_record\GraudationPaper\SER_GraduationPaper\src\configs\model\hierarchical_window_transformer.yaml)
  - `freq_bins` 제거

즉, 현재 Optuna는 새 모델 인터페이스와 충돌하지 않는다.

## 검증 결과

실제 학습은 의도적으로 수행하지 않았다. 대신 `torch.randn` 기반의 shape 검증만 수행했다.

검증 조건:

- 입력: `x.shape = (2, 1, 80, 96)`
- 길이: `lengths = [96, 61]`
- 설정: `stage_dims=[128,192]`, `stage_depths=[2,2]`, `window_sizes=[8,8]`

검증 결과:

- `logits.shape == (2, 8)`
- `embedding.shape == (2, 192)`
- stem 이후 남는 주파수 해상도: `20`

즉, 표준 `n_mels=80` 입력에서 새 구조는 에러 없이 끝까지 요약 벡터를 만든다.

## 후속 실험 제안

- `window_sizes=[6,6]`, `[8,8]`, `[12,12]` 비교
- `stage_dims=[96,128]` vs `[128,192]` 비교
- `patch merging` 유지 vs `Conv1d downsample` 회귀 비교
- `shifted window on/off` ablation

현재 상태는 “구조 재설계 완료, shape 검증 완료, 실제 학습 미실행”이다.
