# CNN Conformer Experiment - 2026-04-22 Overfitting Follow-up

## 1. 문서 범위

- 대상 모델: `cnn_conformer`
- 문서 목적: 2026-04-18 ~ 2026-04-21 실험 기록을 바탕으로, 이미 검증한 과적합 대응과 아직 시도하지 않은 대응을 분리하고 후속 실험 우선순위를 정리
- 현재 문서 상태: `active`

## 2. 모델 스냅샷

### 2.1 한 줄 요약

이 문서는 `nostem_patch` 기반 CNN-Conformer가 보이는 과적합 문제를 해결하기 위해, 이미 실패하거나 효과가 제한적이었던 축은 제외하고, **겹치지 않는 후속 실험 축만 다시 설계**하기 위한 문서다.

### 2.2 핵심 구성 요소

| 항목 | 값 또는 설명 |
|---|---|
| 입력 표현 | `log-Mel spectrogram` |
| 현재 승자 backbone | `nostem_patch` |
| 현재 병목 | train/val gap, `sad` 축 혼동, winner 미회복 |
| 후속 목적 | overfitting 완화 + 2026-04-20 winner 재도전 |
| 분류 대상 | 8-class emotion recognition |

### 2.3 비교 관점

- 비교 대상 1: 2026-04-20 backbone redesign winner `F1 0.68536`
- 비교 대상 2: 2026-04-21 nostem generalization best `F1 0.67100`
- 비교 대상 3: 2026-04-18 regularization HPO

이 문서의 목적은 “새 backbone을 또 만드는 것”이 아니라, **지금 확인된 winner backbone 위에서 과적합 해결을 위해 무엇을 더 해볼 수 있는지 정리하는 것**이다.

## 3. 실험 라운드 기록

### 3.1 공통 고정 조건

| 분류 | 항목 | 값 | 비고 |
|---|---|---|---|
| 데이터 | dataset | RAVDESS | 동일 |
| log-Mel | `n_mels / n_fft / hop_length` | `80 / 1024 / 160` | 동일 |
| backbone | front-end | `nostem_patch` 우선 | winner 유지 |
| 평가 | 기준 metric | `F1-macro`, `Accuracy`, `UAR`, `ECE` | 동시 확인 |

### 3.2 이미 한 실험과 결론

| 회차 | 이미 시도한 것 | 결과 | 후속 실험에서의 처리 |
|---|---|---|---|
| 2026-04-18 | dropout / SpecAugment / `label_smoothing` 중심 regularization HPO | champion 미회복, underfitting / calibration 악화 | 같은 축 반복 금지 |
| 2026-04-19 | subsampling / layer fusion / multiscale conv | `time_preserve_first`만 의미, multiscale은 열세 | multiscale 반복 금지 |
| 2026-04-19 round2 | loss / sampler / layer fusion 결합 | champion 미회복 | loss/sampler 반복 우선순위 낮음 |
| 2026-04-20 | backbone redesign | `nostem_patch` winner | 유지 |
| 2026-04-21 | downsizing + `sequence_shrinking` + patch simplification | best `0.67100`, winner 미회복 | 일부 신호만 확인, 다음 축은 더 좁게 |

### 3.3 구조적 tapering, downsizing, shrinking의 차이

| 개념 | 의미 | 이번 저장소에서 이미 했는가 | 차이점 |
|---|---|---|---|
| `downsizing` | 모델 전체 크기를 한 번에 줄임 | 예 | `embed_dim`, `layers`, `ffn_ratio`를 전역적으로 줄이는 방식 |
| `sequence shrinking` | encoder 중간에 token 길이를 줄임 | 예 | 시간축 token 수를 줄이는 방식 |
| `structural tapering` | 층이 깊어질수록 폭을 점진적으로 줄임 | 아니오 | 전역 축소가 아니라 **layer-wise shrinking** |

핵심 차이:

- 박사 제안의 `downsizing`은 “모델 전체를 더 작게 만들자”에 가깝다.
- 박사 제안의 `shrinking`은 “중간 표현이나 token 길이를 줄여 일반화를 유도하자”에 가깝다.
- 여기서 말하는 `structural tapering`은 그 둘과 겹치지 않는다.
  - 예: `192, 192, 160, 128`
  - 또는 FFN을 `768, 768, 576, 384`
  - 즉 앞단은 표현력을 유지하고, 뒤로 갈수록 압축하는 방식이다.

### 3.4 박사 제안이 효과가 있었는가

결론은 “완전 실패”는 아니지만 “충분한 승리”도 아니다.

- `downsizing`:
  - 효과가 전혀 없지는 않았다.
  - 2026-04-21 `trial_0003`이 `embed=128`, `ffn=384`로도 `F1 0.66639`를 냈다.
  - 즉 작은 모델도 경쟁력은 있었다.
  - 하지만 2026-04-20 winner `0.68536`을 넘지는 못했다.
- `sequence shrinking`:
  - 최고 objective trial `trial_0001`은 `late_2x shrinking`을 사용했다.
  - 따라서 “아예 의미 없다”는 해석은 틀리다.
  - 다만 complete 평균 기준으로는 `none`이 더 나빠 보이지 않았고, 압도적인 우세 축도 아니었다.
  - 즉 “후보군 하나로는 살아남았지만, 정답 축으로 확정할 정도는 아니다.”

정리하면:

- 박사 제안은 **의미 있는 탐색 방향이었다.**
- 다만 이번 실험 데이터 기준으로는 “winner를 갱신한 명확한 해결책”까지는 아니었다.

### 3.5 label smoothing은 이미 했는가

이미 했다.

- [2026-04-18.md](./2026-04-18.md)
  - `label_smoothing=0.1`이 상위권에 반복적으로 선택됐지만 champion peak를 회복하지 못했다.
- [2026-04-19.md](./2026-04-19.md)
  - `label smoothing 0.0`이 `0.05`보다 평균과 최고점 모두 우세했다.
- [2026-04-21_nostem_generalization.md](./2026-04-21_nostem_generalization.md)
  - 이번 generalization round는 `label_smoothing=0.0` 고정으로 갔다.

따라서 단독 `label_smoothing` 재시도는 중복 가능성이 높다.  
다음 라운드에서 쓰더라도 **mixup 같은 새 축의 보조 옵션** 정도로만 다루는 편이 맞다.

### 3.6 후속 실험 후보

| 항목 | 후보군 | 비고 |
|---|---|---|
| 구조 | `structural_tapering` | 새 축 |
| regularization | `mixup` on/off | 아직 안 함 |
| normalization | `token_norm_variant` | 아직 안 함 |

### 3.7 권장 실행 순서

1. `structural_tapering`
2. `mixup`
3. `token/input normalization`

동시에 세 축을 한 번에 다 섞는 것보다, 각 축을 분리한 1차 screening이 더 낫다.  
이유는 현재 과적합 문제가 “무엇이 실제로 먹히는지”가 아직 불분명하기 때문이다.

### 3.8 이번 라운드 구현 전략

이번에는 단일 Optuna study 안에서 전략 분기형으로 screening한다.

| 항목 | 구현 방식 |
|---|---|
| Optuna study | 하나 |
| Hydra preset | 하나 |
| 전략 분기 | `tapering`, `mixup`, `normalization` |
| 기존 재현성 | 기존 preset 유지, 새 preset만 추가 |

즉 trial마다 다음 중 한 전략만 선택한다.

- `tapering`
  - layer-wise `channel/ffn shrinking`
- `mixup`
  - spectrogram-level input mixup
- `normalization`
  - `nostem_patch` token normalization variant

### 3.9 실행 명령

```powershell
.\.venv\Scripts\python.exe -m src.optuna_search model=cnn_conformer optuna=cnn_conformer_overfit_screening optuna.enabled=true optuna.trials=36 train.epochs=30 train.folds_to_run=1 experiment.tag=overfit_screening
```

### 3.10 구현 반영 상태

이번 라운드 설계는 문서 제안에 그치지 않고 코드에 반영되었다.

| 구분 | 반영 파일 | 내용 |
|---|---|---|
| 모델 설정 | [../src/configs/model/cnn_conformer.yaml](../src/configs/model/cnn_conformer.yaml) | `layer_dim_schedule`, `layer_ffn_schedule`, `nostem_patch.norm_variant` 추가 |
| 공통 학습 설정 | [../src/configs/config.yaml](../src/configs/config.yaml) | `train.mixup` 기본값 추가 |
| Optuna preset | [../src/configs/optuna/cnn_conformer_overfit_screening.yaml](../src/configs/optuna/cnn_conformer_overfit_screening.yaml) | `tapering` / `mixup` / `normalization` 단일 study 검색 공간 추가 |
| 모델 구현 | [../src/models/cnn_conformer.py](../src/models/cnn_conformer.py) | layer-wise tapering, token normalization variant, schedule transition 지원 |
| 학습 루프 | [../src/engine/trainer.py](../src/engine/trainer.py) | optional mixup 경로 추가 |
| 검색 엔트리 | [../src/optuna_search.py](../src/optuna_search.py) | 전략 분기형 search space, Hydra merge, legacy-compatible trial override 추가 |

기존 preset은 그대로 유지된다.  
즉 이전 명령어는 바꾸지 않아도 동일한 실험 경로를 다시 실행할 수 있어야 한다.

### 3.11 Smoke Test 결과

| 검증 대상 | 명령 | 결과 |
|---|---|---|
| `tapering` 분기 | `model=cnn_conformer optuna=cnn_conformer_overfit_screening ... experiment.tag=smoke_tapering optuna.search_space.cnn_conformer.overfit_strategy_choices=[tapering]` | 통과 |
| `mixup` 분기 | `model=cnn_conformer optuna=cnn_conformer_overfit_screening ... experiment.tag=smoke_mixup_seq optuna.search_space.cnn_conformer.overfit_strategy_choices=[mixup]` | 통과 |
| `normalization` 분기 | `model=cnn_conformer optuna=cnn_conformer_overfit_screening ... experiment.tag=smoke_norm_seq optuna.search_space.cnn_conformer.overfit_strategy_choices=[normalization]` | 통과 |
| 기존 preset 재현 | `model=cnn_conformer optuna=cnn_conformer_nostem_generalization ... experiment.tag=smoke_legacy_repro` | 통과 |

관찰 메모:

- 새 전략 smoke는 `1 trial / 1 epoch / folds_to_run=1` 조건에서 모두 학습 루프까지 정상 진입했다.
- 병렬 smoke 실행 시 MLflow SQLite migration 충돌이 1회 발생했다. 이는 코드 구조 문제가 아니라 DB 초기화 동시 접근 문제였고, 순차 실행에서는 재현되지 않았다.
- tapering 초기 smoke에서 `layer_fusion` 참조 순서 버그가 발견되었고, `cnn_conformer.py` 초기화 검증 순서를 수정하여 해결했다.

### 3.12 Actual Run Result

| 항목 | 값 |
|---|---|
| 실행 경로 | [../../outputs/2026-04-21/18-47-36_cnn_conformer](../../outputs/2026-04-21/18-47-36_cnn_conformer) |
| Optuna DB | [../../optuna_studies/cnn_conformer_optuna_overfit_screening.db](../../optuna_studies/cnn_conformer_optuna_overfit_screening.db) |
| 총 trial | 64 |
| COMPLETE | 10 |
| PRUNED | 53 |
| FAIL | 1 |
| 최고 trial | `trial_0003` |
| 최고 F1-macro | `0.705629` |
| 최고 Accuracy | `0.700000` |
| 최고 UAR | `0.709375` |
| 최고 ECE | `0.207591` |

Top complete trials:

| 순위 | trial | 전략 | F1-macro | Accuracy | UAR | ECE | 핵심 설정 |
|---|---|---|---|---|---|---|---|
| 1 | [trial_0003](../../outputs/2026-04-21/18-47-36_cnn_conformer/optuna_trials/trial_0003/trial_summary.json) | `mixup` | `0.70563` | `0.70000` | `0.70938` | `0.20759` | `time_patch=4`, `mixup alpha=0.4`, `layernorm`, no shrinking |
| 2 | [trial_0001](../../outputs/2026-04-21/18-47-36_cnn_conformer/optuna_trials/trial_0001/trial_summary.json) | `normalization` | `0.68728` | `0.68667` | `0.67813` | `0.07966` | `time_patch=2`, `batchnorm`, no mixup |
| 3 | [trial_0002](../../outputs/2026-04-21/18-47-36_cnn_conformer/optuna_trials/trial_0002/trial_summary.json) | `mixup` | `0.67739` | `0.68000` | `0.68125` | `0.19000` | `time_patch=4`, `mixup alpha=0.4` |
| 4 | [trial_0041](../../outputs/2026-04-21/18-47-36_cnn_conformer/optuna_trials/trial_0041/trial_summary.json) | `tapering` | `0.66888` | `0.67333` | `0.66563` | `0.04561` | `late_2x`, `flat_192 + mild_taper FFN` |

전략별 요약:

| 전략 | complete 수 | best F1 | 평균 complete F1 | 해석 |
|---|---|---|---|---|
| `mixup` | 2 | `0.70563` | `0.69151` | 최고점 확보, 성능 이득은 가장 큼 |
| `normalization` | 2 | `0.68728` | `0.66783` | 최고점은 mixup보다 낮지만 calibration은 가장 안정적 |
| `tapering` | 6 | `0.66888` | `0.66215` | 과적합 억제는 일부 보였으나 winner 갱신 실패 |

### 3.13 Stop Decision

이 round는 중단하는 편이 맞다.

- 최고점 `0.70563`이 `trial_0003`에서 매우 이르게 나왔고, 이후 59개 추가 trial 동안 갱신되지 않았다.
- 최근 complete trial도 `trial_0041`, `trial_0042`, `trial_0062` 수준으로 `0.656~0.669` 범위에 머물렀다.
- pruned trial의 최고 중간값도 `mixup 0.5919`, `normalization 0.6020`, `tapering 0.6445` 수준이라 현재 search space 안에서 late breakthrough 가능성은 낮다.
- 따라서 같은 preset을 더 늘리는 것보다, winner branch를 기준선으로 고정하고 새 일반화 축을 추가하는 것이 낫다.

## 4. 설계 배경 및 구현 메모

### 4.1 설계 배경

이전 기록과 `ref.bib`를 합치면, 지금 필요한 것은 “더 센 일반 규제”가 아니라 **표현력은 남기되 과적합만 줄이는 정밀한 구조 제어**다.

참고 근거:

- Gulati et al., *Conformer: Convolution-augmented Transformer for Speech Recognition*
  - Conformer는 원래 ASR용 대형 encoder 성격이 강하다.
- Peng et al., *Efficient Speech Emotion Recognition Using Multi-Scale CNN and Attention*
  - SER에서는 효율성과 과도하지 않은 구조가 중요할 수 있다.
- Kang et al., *Learning Robust Self-Attention Features for Speech Emotion Recognition with Label-Adaptive Mixup*
  - mixup 계열 regularization이 attention 기반 SER에서 직접적으로 쓰인다.
- Prabhu et al., *End-to-End Label Uncertainty Modeling in Speech Emotion Recognition Using Bayesian Neural Networks and Label Distribution Learning*
  - SER 라벨 경계의 불확실성을 다루는 방향이 유효하다.
- Fan et al., *ISNet: Individual Standardization Network for Speech Emotion Recognition*
  - 개인차 보정은 SER에서 의미 있는 축이다.
- Lu et al., *Domain Invariant Feature Learning for Speaker-Independent Speech Emotion Recognition*
  - speaker-invariant regularization은 speaker leakage 완화에 유효하다.

### 4.2 이번에 제외하는 중복 축

- 단독 `label_smoothing`
- 단독 dropout 확장
- 단독 SpecAugment 확장
- 기존 `sequence_shrinking_choices`만 재탐색
- 기존 `embed_dim / layers / ffn_ratio`만 다시 넓게 Optuna

이 축들은 이미 했거나, 추가 이득이 작다는 근거가 문서에 남아 있다.

### 4.3 후속 실험 설계안

#### A. Structural Tapering

- 목적:
  - 박사 제안의 downsizing을 “전역 축소”에서 “층별 점진 축소”로 바꾼다.
- 예시:
  - encoder dim schedule: `[192, 192, 160, 128]`
  - ffn schedule: `[768, 768, 576, 384]`
- 기대 효과:
  - 초반 층은 감정 cue를 충분히 읽고
  - 후반 층은 압축된 표현으로 과적합을 줄인다.

구현 메모:

- `layer_dim_schedule`
- `layer_ffn_schedule`
- schedule이 비어 있으면 기존 uniform encoder와 완전히 동일하게 동작

#### B. Mixup Regularization

- 목적:
  - decision boundary를 완만하게 만들고, train/val gap을 줄인다.
- 형태:
  - spectrogram mixup
  - 또는 chunk-level mixup
- 주의:
  - `label_smoothing` 단독 재시도는 금지
  - 필요하면 `mixup + smoothing 0.02~0.05`처럼 보조 옵션으로만 사용

구현 메모:

- trainer 단계에서 optional mixup 적용
- `train.mixup.enabled=false`면 기존 학습 루프와 동일

#### C. Token / Input Normalization

- 목적:
  - speaker/style 편차를 줄여 `sad` 쏠림 혼동을 완화한다.
- 형태:
  - per-sample input normalization 강화
  - patch projection 직후 token normalization variant
- 기대 효과:
  - speaker-specific amplitude/style bias를 줄이고 감정 cue에 더 집중

구현 메모:

- `nostem_patch.norm_variant`
  - `layernorm`
  - `batchnorm`
  - `instancenorm`
- 기본값은 기존과 같은 `layernorm`

## 5. 아티팩트 분석

### 5.1 대표 근거 문서

- regularization 실패: [2026-04-18.md](./2026-04-18.md)
- 구조 탐색: [2026-04-19.md](./2026-04-19.md)
- backbone redesign winner: [2026-04-20_backbone_redesign.md](./2026-04-20_backbone_redesign.md)
- generalization round: [2026-04-21_nostem_generalization.md](./2026-04-21_nostem_generalization.md)

### 5.2 항목별 해석

- 이미 확인된 사실 1:
  - 강한 일반 regularization만으로는 해결되지 않았다.
- 이미 확인된 사실 2:
  - 구조를 너무 크게 유지하면 train/val gap이 빠르게 벌어진다.
- 이미 확인된 사실 3:
  - 무조건 작은 모델이 답도 아니다.
- 구조적 해석:
  - 필요한 것은 “전역 축소”보다 “표현력은 일부 유지하고 후반만 줄이는 방식”일 가능성이 높다.
- 다음 액션 해석:
  - 따라서 next round는 `structural tapering`이 1순위다.
  - mixup과 normalization은 그 다음 축으로 분리 검증한다.

## 6. 종합 인사이트 및 다음 액션

### 6.1 현재 판단

현재까지의 데이터 기준으로 다음 결론이 가장 타당하다.

- 박사 제안의 downsizing / shrinking은 시도할 가치가 있었고, 부분적으로 유효했다.
- 하지만 그 자체만으로는 winner 갱신에 실패했다.
- `label_smoothing`은 이미 실험했고, 반복 우선순위가 낮다.
- 다음은 “이미 한 것의 반복”이 아니라, **겹치지 않는 새 overfitting 대응 축**으로 넘어가야 한다.

### 6.2 다음 액션

우선순위:

1. `structural_tapering`
2. `mixup`
3. `token/input normalization`

권장 전략:

- 1차: single-Optuna 전략 분기형 screening
- 2차: 가장 유효한 축만 winner backbone과 결합
- 3차: 그때도 안 되면 새로운 transformer 계열로 이동 검토

## 7. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-04-22 | overfitting 후속 실험 방향 문서 초안 작성 |
| 2026-04-22 | single-Optuna screening 설계와 실행 명령 반영 |
