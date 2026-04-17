# 데이터 정리

## 데이터셋

- 데이터: `RAVDESS`
- 클래스: 8감정
- 화자 수: 24명
- 현재 평가: `GroupKFold`, 다만 transformer Optuna는 시간 제약 때문에 기본 `folds_to_run=1`

## 현재 입력 처리

- 공통 입력 특징: log-Mel spectrogram
- transformer 계열:
  - `resize`를 강제하지 않음
  - 시간축 길이를 유지한 채 batch별 max width로 padding
  - 모델에 `lengths`와 padding mask를 함께 전달
- 기존 CNN baseline:
  - 과거 실험에서는 `resize_height`, `resize_width` 기반 고정 입력 사용
  - 현재는 추가 Optuna 실험 대상에서 제외

## transformer Optuna에서 탐색하는 log-Mel 파라미터

- `n_mels`
- `n_fft`
- `hop_length`
- `normalize`
- `f_min`
- `f_max`

`resize_height`, `resize_width`는 transformer Optuna 후보군에서 제거했다.
