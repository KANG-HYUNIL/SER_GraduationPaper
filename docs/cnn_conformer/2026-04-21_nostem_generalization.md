# CNN Conformer Experiment - 2026-04-21 Nostem Generalization

## 1. 문서 범위

- 대상 모델: `cnn_conformer`
- 문서 목적: `nostem_patch` 승자 backbone 위에서 overfitting 완화를 위한 downsizing / gradually shrinking 실험 계획과 결과 기록
- 현재 문서 상태: `active`

## 2. 모델 스냅샷

### 2.1 한 줄 요약

이번 회차는 2026-04-20 backbone redesign에서 승리한 `nostem_patch`를 고정하고, 모델 용량 축소와 sequence shrinking을 통해 일반화를 더 끌어올리는 것을 목표로 한다.

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
| Round 7 | 2026-04-21~ | `nostem_patch` overfitting 완화 | downsizing + sequence shrinking + patch simplification | 실험 전 | `../../outputs/...` |

### 3.5 주요 결과 요약

실험 완료 후 기입:

| Rank | Trial | F1-macro | Accuracy | UAR | 핵심 파라미터 요약 |
|---|---|---:|---:|---:|---|
| 1 | `trial_0000` | | | | |

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

실험 완료 후 기입:

- trial 요약: `../../outputs/.../trial_summary.json`
- metrics: `../../outputs/.../summary_metrics.json`
- artifact 폴더: `../../outputs/.../artifacts/`

### 5.2 항목별 해석

실험 완료 후 아래 관점으로 채운다.

- downsizing이 train/val gap을 줄였는지
- `sequence_shrinking`이 val loss 흔들림을 줄였는지
- `time_patch` 증가가 성능을 깎지 않고 generalization을 올렸는지
- winner backbone의 confusion-heavy class가 더 안정화되는지

## 6. 종합 인사이트 및 다음 액션

### 6.1 현재 판단

이번 회차는 `nostem_patch` backbone의 승리를 더 믿을 수 있는지 확인하는 generalization 실험이다.

### 6.2 다음 액션

- downsizing이 유효하면 이후 기본 conformer scale을 축소한다.
- `sequence_shrinking`이 유효하면 gradually shrinking을 정식 구조로 승격한다.
- 둘 다 무효면 새로운 regularization보다 patch/token 구조를 다시 검토한다.

## 7. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-04-21 | `nostem_patch` overfitting 완화 실험 계획 문서 작성 |
