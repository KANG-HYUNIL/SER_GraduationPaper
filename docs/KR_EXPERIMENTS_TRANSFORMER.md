# Transformer 실험 운영 메모

## 기본 원칙

현재 transformer 계열 실험은 다음 원칙을 따른다.

- 기본 평가는 `GroupKFold`
- Optuna 탐색 단계에서는 시간 제약 때문에 `folds_to_run=1`
- transformer 계열은 CNN baseline보다 학습 분산이 커서 trial 예산을 너무 작게 잡으면 비효율적
- 반대로 `COMPLETE trial` 목표를 너무 크게 잡으면 prune이 많을수록 총 실행 trial 수가 급격히 늘어난다

즉, transformer Optuna에서는 "적당히 많은 trial"과 "짧고 빠른 탐색" 사이의 균형이 중요하다.

## 현재 기본 실행

```powershell
python -m scripts.run_transformer_optuna_suite --device cuda
```

기본값은 다음과 같다.

- 모델:
  - `pure_transformer`
  - `cnn_conformer`
  - `hierarchical_window_transformer`
- `trials=24`
- `epochs=15`
- `folds_to_run=1`
- `max_parallel=1`

## 왜 이 기본값을 쓰는가

- RTX 2060 6GB 환경에서 trial 하나를 너무 길게 잡으면 전체 벽시계 시간이 크게 늘어난다.
- transformer 계열은 baseline보다 variance가 커서 trial 수가 너무 적어도 손해다.
- 그래서 기본은 `1-fold 빠른 탐색 -> 상위 후보만 재검증` 구조로 운용한다.

## 모델별 기본 해석

- `pure_transformer`
  - 순수 transformer 기준선
- `cnn_conformer`
  - local CNN + global attention hybrid 기준선
- `hierarchical_window_transformer`
  - shifted window + hierarchical downsampling 기반의 메인 실험 구조

## 단계별 권장 프로토콜

### 1차 빠른 탐색

- `trials=12~24`
- `epochs=8~15`
- `folds_to_run=1`

목적은 "될 법한 영역"을 빠르게 찾는 것이다.

### 2차 안정화 탐색

- `trials=20~32`
- `epochs=12~18`
- `folds_to_run=1`

목적은 1차에서 살아남은 후보 영역을 조금 더 정교하게 보는 것이다.

### 최종 검증

- 상위 3~5개 조합만 별도 재실행
- `epochs=24~30`
- 필요 시 `folds_to_run=3` 또는 전체 fold

최종 검증 단계에서는 Optuna를 더 크게 돌리기보다, 이미 선별된 조합을 공정하게 다시 평가하는 편이 낫다.

## 병렬 실행 주의

코드상 병렬 실행은 가능하다.

```powershell
python -m scripts.run_transformer_optuna_suite --device cuda --max-parallel 3
```

하지만 `RTX 2060 6GB` 환경에서는 실제로는 `max_parallel=1`이 가장 안전하고, 많아도 `2` 정도가 한계에 가깝다.

## 개별 Optuna 실행 예시

### Pure Transformer

```powershell
python -m src.optuna_search model=pure_transformer experiment.family=pure_transformer experiment.name=pure_transformer_optuna train.device=cuda optuna.trials=24 train.epochs=15 train.folds_to_run=1
```

### CNN Conformer

```powershell
python -m src.optuna_search model=cnn_conformer experiment.family=cnn_conformer experiment.name=cnn_conformer_optuna train.device=cuda optuna.trials=24 train.epochs=15 train.folds_to_run=1
```

### Hierarchical Window Transformer

```powershell
python -m src.optuna_search model=hierarchical_window_transformer experiment.family=hierarchical_window_transformer experiment.name=hierarchical_window_transformer_optuna train.device=cuda optuna.trials=24 train.epochs=15 train.folds_to_run=1
```

## Hierarchical Window Transformer 전용 실험

현재 메인 실험은 `hierarchical_window_transformer`에 집중하는 쪽이 더 효율적이다. 다른 모델에서 이미 어느 정도 탐색을 마쳤다면, 이제는 frontend를 넓게 흔드는 것보다 window-hierarchy 구조 자체에 trial 예산을 집중하는 것이 낫다.

### 현재 채택한 운영 방침

- log-Mel은 CNN baseline 최종 조합으로 고정
- Optuna는 hierarchical-window 본체와 학습 파라미터만 탐색
- `window_size`는 탐색 후보에 포함
- `30 epochs`, `30 COMPLETE trials`, `folds_to_run=1`

### log-Mel 고정값

고정값은 [KR_MODELS_CNN_BASELINE.md](KR_MODELS_CNN_BASELINE.md)의 최종 선택 trial을 따른다.

- `n_mels=80`
- `n_fft=1024`
- `hop_length=160`
- `normalize=true`
- `f_min=0.0`
- `f_max=6000.0`

주의할 점은 transformer 계열은 현재 `resize_enabled=False`로 동작한다는 것이다. 따라서 CNN baseline의 `96x512 resize`는 그대로 쓰지 않고, log-Mel의 핵심 spectral 설정만 고정한다.

### 왜 log-Mel을 고정하는가

- 이번 실험의 핵심 질문은 frontend보다 hierarchical-window 구조다.
- 30 COMPLETE trial 예산으로 frontend와 backbone을 동시에 크게 열면 각 축이 너무 얇게 샘플링된다.
- CNN stem 계열에서 이미 강했던 log-Mel 조합을 비교 기준점으로 고정하면, 이번 실험 결과를 더 해석하기 쉬워진다.

### 왜 `window_size`를 탐색하는가

`window_size`는 local attention의 범위를 직접 결정한다.

- 너무 작으면 local detail은 잘 보지만 문맥 확장이 느리다.
- 너무 크면 global attention과의 차별성이 약해지고 비용이 커진다.

`wang2024swin`과 `chen2023dwformer`는 모두 speech SER에서 local window 설계가 핵심이라는 점을 보여 준다. 현재 저장소는 이 논문들을 그대로 재현한 구조는 아니지만, shifted window와 hierarchical context expansion이라는 핵심 원리를 반영한다. 따라서 `window_size`를 2-stage window transformer의 핵심 탐색 축으로 보는 것이 맞다.

### 현재 전용 Optuna profile

새 전용 profile:

- `src/configs/optuna/hierarchical_window_cnnfixed.yaml`

이 profile의 특징은 다음과 같다.

- log-Mel sampling 비활성화
- CNN baseline 기반 고정 log-Mel 적용
- `warmup_steps=3`
- hierarchical window 구조는 유효 조합만 샘플링
- 중복 stem pair와 불가능한 stage/head 조합을 미리 제거

### 실행 명령

```powershell
python -m src.optuna_search optuna=hierarchical_window_cnnfixed model=hierarchical_window_transformer experiment.family=hierarchical_window_transformer experiment.name=hierarchical_window_cnnfixed_stage2 train.device=cuda train.epochs=30 train.folds_to_run=1
```

필요하면 study 이름과 storage는 override로 분리한다.

```powershell
python -m src.optuna_search optuna=hierarchical_window_cnnfixed model=hierarchical_window_transformer experiment.family=hierarchical_window_transformer experiment.name=hierarchical_window_cnnfixed_stage2 train.device=cuda train.epochs=30 train.folds_to_run=1 optuna.study_name=hierarchical_window_cnnfixed_20260416 optuna.storage=sqlite:///optuna_studies/hierarchical_window_cnnfixed_20260416.db
```

## 현재 코드 기준 Optuna 동작 메모

### prune이 바로 안 걸리는 이유

현재 pruning은 epoch 종료 후 validation score가 나온 시점에만 판단한다.

- `trial.report(score, step=global_step)`
- `trial.should_prune()`

또한 `MedianPruner`의 warmup step이 존재한다. 따라서 학습 시작 직후 trial이 즉시 잘리지 않는다.

### `trials`의 의미

현재 `optuna.trials`는 "총 시도 횟수"가 아니라 "COMPLETE trial 목표 개수"다.

즉,

- `trials=30`
- PRUNED가 많이 발생

인 상황이면 실제 총 trial 수는 30보다 훨씬 커질 수 있다.

이 점 때문에 탐색 공간을 무작정 크게 여는 것은 비효율적이다.

## Hierarchical Window 탐색 공간을 좁힌 이유

이번 전용 profile에서는 다음 방향으로 search space를 좁혔다.

- log-Mel은 고정
- `stem_pair`는 유효 조합만 사용
- `stage_dims`와 `num_heads`는 유효 튜플만 사용
- `depth_pair`는 2-stage 구조에서 현실적인 깊이만 사용
- `window_size`는 논문 취지와 현재 하드웨어 제약을 반영한 소수 후보만 사용
- `batch_size`, `lr`, `weight_decay`, `dropout` 범위도 좁혔다

핵심은 "prune으로 나중에 버릴 후보를 미리 제거"하는 것이다.
