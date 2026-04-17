# CNN Baseline 정리

## 상태

- 기존 기준선 모델
- `outputs/2026-04-14/04-49-31_cnn_optuna_stage1_baselineTest` 기준 최고 `f1_macro = 0.62196`
- 현재는 추가 Optuna 실험 대상에서 제외

## 왜 baseline이 강했는가

- 데이터셋이 작아 CNN의 local inductive bias가 유리했다
- 고정 resize 입력에서 CNN이 time-frequency local pattern을 안정적으로 학습했다
- pure transformer보다 파라미터 효율과 학습 안정성이 좋았다

## 이전 Optuna 탐색 파라미터

### 모델 파라미터

- `hidden_dims`
- `dropout`

### 학습 파라미터

- `learning_rate`
- `weight_decay`
- `batch_size`

### log-Mel 파라미터

- `n_mels`
- `n_fft`
- `hop_length`
- `normalize`
- `resize_height`
- `resize_width`
- `f_min`
- `f_max`

## Top 5 trial

| Rank | Trial | F1-macro | Accuracy | UAR | hidden_dims | dropout | n_mels | n_fft | hop | normalize | resize | f_min | f_max | batch | lr | wd |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `trial_0023` | 0.62196 | 0.61667 | 0.61563 | `[32, 64, 256, 512]` | 0.33238 | 80 | 1024 | 160 | true | `96x512` | 0 | 6000 | 16 | 3.41e-4 | 1.93e-5 |
| 2 | `trial_0075` | 0.58003 | 0.60333 | 0.59688 | `[32, 32, 96, 512]` | 0.37040 | 128 | 1024 | 160 | true | `96x512` | 20 | 6000 | 16 | 1.59e-4 | 3.65e-5 |
| 3 | `trial_0060` | 0.57599 | 0.59667 | 0.58125 | `[32, 32, 96, 512]` | 0.32769 | 80 | 1024 | 160 | true | `96x512` | 50 | 6000 | 16 | 2.04e-4 | 5.47e-5 |
| 4 | `trial_0072` | 0.57566 | 0.60667 | 0.58750 | `[32, 32, 96, 512]` | 0.34417 | 80 | 1024 | 160 | true | `96x512` | 20 | 6000 | 16 | 1.55e-4 | 3.01e-5 |
| 5 | `trial_0065` | 0.57488 | 0.59667 | 0.59375 | `[32, 32, 160, 512]` | 0.34306 | 80 | 1024 | 160 | true | `96x512` | 20 | 6000 | 16 | 2.58e-4 | 1.19e-4 |

## 최종 선택 파라미터

| 항목 | 값 |
|---|---|
| 선택 trial | `trial_0023` |
| hidden_dims | `[32, 64, 256, 512]` |
| dropout | `0.33238` |
| learning_rate | `3.4129546471254387e-4` |
| weight_decay | `1.9338610496754583e-5` |
| batch_size | `16` |
| n_mels | `80` |
| n_fft | `1024` |
| hop_length | `160` |
| normalize | `true` |
| resize_height | `96` |
| resize_width | `512` |
| f_min | `0.0` |
| f_max | `6000.0` |
| accuracy | `0.61667` |
| f1_macro | `0.62196` |
| uar | `0.61563` |

## 해석

- 상위권 trial은 거의 모두 `batch_size=16`, `n_fft=1024`, `hop_length=160`, `resize=96x512`, `normalize=true`에 몰렸다.
- 채널 구성은 `32`로 시작하고 마지막 stage를 `512`로 유지하는 얕은-중간 깊이 구조가 강했다.
- transformer 계열 후속 실험에서 이 조합은 비교 기준점으로 의미가 있다.
