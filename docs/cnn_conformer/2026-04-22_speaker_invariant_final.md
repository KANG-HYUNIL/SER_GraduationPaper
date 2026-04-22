# CNN Conformer Experiment - 2026-04-22 Speaker-Invariant Final Round

## 1. 문서 범위

- 대상 모델: `cnn_conformer`
- 문서 목적: 2026-04-22 overfitting screening 종료 후, winner branch를 고정한 상태에서 마지막 일반화 개선 축으로 `speaker-invariant adversarial regularization` 실험을 설계하고 구현 범위를 기록
- 현재 문서 상태: `active`

## 2. 모델 개요

### 2.1 한 줄 요약

이번 라운드는 새 backbone을 다시 여는 실험이 아니다.  
`trial_0003` winner의 `nostem_patch + mixup` 구성을 기준선으로 고정하고, **RAVDESS actor split에서의 speaker overfitting을 직접 줄이는 lightweight adversarial head**를 붙여 일반화 성능을 더 끌어올리는 것이 목표다.

### 2.2 기준선

| 기준선 | 값 |
|---|---|
| 기반 run | [../../outputs/2026-04-21/18-47-36_cnn_conformer](../../outputs/2026-04-21/18-47-36_cnn_conformer) |
| 기준 trial | [trial_0003](../../outputs/2026-04-21/18-47-36_cnn_conformer/optuna_trials/trial_0003/trial_summary.json) |
| backbone | `nostem_patch`, `time_patch=4` |
| regularization | `mixup alpha=0.4` |
| F1-macro | `0.705629` |
| Accuracy | `0.700000` |
| UAR | `0.709375` |

## 3. 실험 설계

### 3.1 왜 이 축을 마지막으로 보나

기존 기록을 종합하면:

- `dropout / SpecAugment / label_smoothing` 단독 확장은 champion 회복에 실패했다.
- `loss / sampler / layer fusion` 조합도 plateau였다.
- `downsizing / shrinking / tapering`은 일부 일반화 이득은 있었지만 winner를 넘지 못했다.
- 반면 이번 screening에서는 `mixup`이 가장 높은 F1을 만들었다.

즉 현재 CNN-Conformer의 핵심 문제는 단순 capacity 문제가 아니라, **speaker-specific shortcut을 타면서도 일부 mixup으로만 버티는 상태**로 해석하는 것이 가장 자연스럽다.

### 3.2 논문 근거

핵심 근거:

- Lu et al., *Domain Invariant Feature Learning for Speaker-Independent Speech Emotion Recognition*  
  speaker-induced domain shift를 줄이기 위해 speaker 정보를 혼동시키는 adversarial discriminator를 사용한다.  
  출처: [TUM publication page](https://portal.fis.tum.de/en/publications/domain-invariant-feature-learning-for-speaker-independent-speech-)  

- Fan et al., *ISNet: Individual Standardization Network for Speech Emotion Recognition*  
  개인 차이 때문에 생기는 representation deviation이 SER 일반화를 해치며, individual-agnostic representation이 필요하다고 본다.  
  출처: [ResearchGate abstract page](https://www.researchgate.net/publication/363046658_ISNet_Individual_Standardization_Network_for_Speech_Emotion_Recognition)

- Gao et al., *Adversarial Domain Generalized Transformer for Cross-Corpus Speech Emotion Recognition*  
  adversarial learning으로 non-affective information을 제거하는 방향이 transformer 계열에서도 유효하다고 본다.  
  출처: [CiNii entry](https://cir.nii.ac.jp/crid/1360302866839799296)

- Kang et al., *Learning Robust Self-Attention Features for Speech Emotion Recognition with Label-Adaptive Mixup*  
  mixup 계열 regularization이 self-attention 기반 SER에서 실제로 효과적이라는 직접 근거다. 이번 라운드는 이 winner line을 유지한다.  
  출처: [CatalyzeX abstract page](https://www.catalyzex.com/paper/learning-robust-self-attention-features-for)

해석:

- `mixup` winner는 유지할 가치가 있다.
- 다만 mixup만으로는 speaker shortcut을 직접 제거하지 못하므로, 마지막 라운드는 **emotion branch는 유지하고 speaker branch만 적대적으로 억제**하는 방향이 가장 논리적이다.
- full DIFL/ADoGT처럼 multi-discriminator, cross-corpus UDA, pretrained fusion까지 가면 현재 학부 논문 범위와 6GB GPU 제약을 넘기므로, 이번 구현은 **lightweight GRL speaker head**까지만 가져간다.

### 3.3 이번 라운드의 실험 축

| 축 | 후보 | 목적 |
|---|---|---|
| `speaker_adversarial.enabled` | `false`, `true` | adversarial branch 자체 효과 확인 |
| `speaker_adversarial.loss_weight` | `0.05`, `0.1`, `0.2` | emotion loss 대비 규제 강도 |
| `speaker_adversarial.grl_lambda` | `0.5`, `1.0` | feature reversal 강도 |
| `speaker_adversarial.hidden_dim` | `64`, `128` | auxiliary head 용량 |
| `speaker_adversarial.dropout` | `0.1`, `0.2` | auxiliary head regularization |
| `nostem_patch.norm_variant` | `layernorm`, `batchnorm` | screening에서 보였던 normalization 보조 효과 재확인 |
| `mixup.alpha` | `0.3`, `0.4`, `0.5` | winner 주변 재탐색 |

고정 조건:

- `backbone_variant=nostem_patch`
- `time_patch=4`
- `stem_strides=[[2,1],[2,2]]`
- `embed_dim=192`, `num_layers=4`, `ffn_dim=768`
- `conv_kernel=31`, `layer_fusion=last`, `pooling=attention`
- `chunk_frames=48`, `confidence_weighted_logit`
- `loss=cross_entropy`, `sampler=random`

### 3.4 구현 방식

모델 자체 backbone은 바꾸지 않는다.

1. `cnn_conformer.get_embedding()`으로 utterance/chunk embedding을 뽑는다.
2. 메인 emotion classifier는 기존과 동일하게 학습한다.
3. train only auxiliary branch로 `speaker_head`를 추가한다.
4. `Gradient Reversal Layer (GRL)`를 통해 speaker classification loss가 backbone에는 반대로 전달되게 만든다.
5. 총 loss는 아래 형태다.

```text
L_total = L_emotion + lambda_speaker * L_speaker_adv
```

여기서 `L_speaker_adv`는 speaker head 입장에서는 speaker를 맞히도록 최소화되지만, backbone 입장에서는 GRL 때문에 speaker를 구분하기 어렵게 만드는 방향으로 작동한다.

### 3.5 기대 효과

- actor-specific timbre, loudness habit, articulation habit 같은 non-affective shortcut을 줄인다.
- GroupKFold의 unseen actor validation에서 F1/UAR를 더 안정적으로 올릴 가능성이 있다.
- normalization winner가 보여준 calibration 이득과 mixup winner가 보여준 accuracy/F1 이득을 동시에 흡수할 여지가 있다.

## 4. 구현 반영 파일

| 구분 | 파일 | 내용 |
|---|---|---|
| 공통 학습 설정 | [../src/configs/config.yaml](../src/configs/config.yaml) | `train.speaker_adversarial` 기본값 추가 |
| dataset | [../src/data/dataset.py](../src/data/dataset.py) | actor id를 train batch로 넘길 수 있도록 dataset/collate 확장 |
| trainer | [../src/engine/trainer.py](../src/engine/trainer.py) | GRL, `SpeakerAdversary`, auxiliary loss, trial-local weights 저장 추가 |
| optuna search | [../src/optuna_search.py](../src/optuna_search.py) | speaker adversarial search space merge 추가 |
| optuna preset | [../src/configs/optuna/cnn_conformer_speaker_invariant_final.yaml](../src/configs/optuna/cnn_conformer_speaker_invariant_final.yaml) | final round 검색 공간 정의 |

## 5. 실행 명령

```powershell
.\.venv\Scripts\python.exe -m src.optuna_search model=cnn_conformer optuna=cnn_conformer_speaker_invariant_final optuna.enabled=true optuna.trials=30 train.epochs=30 train.folds_to_run=1 experiment.tag=speaker_invariant_final
```

## 6. 실험 로그 기록

### 6.1 공통 고정 조건

| 분류 | 항목 | 값 | 비고 |
|---|---|---|---|
| 데이터 | dataset | RAVDESS | actor-group split |
| log-Mel | `n_mels / n_fft / hop_length` | `80 / 1024 / 160` | screening winner line 유지 |
| backbone | front-end | `nostem_patch` | `time_patch=4` 중심 |
| chunking | `frames / hop / aggregation` | `48 / 12 / confidence_weighted_logit` | 고정 |
| optimizer | epochs / folds | `30 / 1` | Optuna screening |

### 6.2 실행 회차

| 회차 | 날짜 | 목적 | 설정 요약 | 결과 요약 | 산출 경로 |
|---|---|---|---|---|---|
| Round 8-1 | 2026-04-22 | mixup winner에 `speaker_adversarial` 보조 branch를 추가해 unseen actor 일반화가 개선되는지 검증 | `mixup alpha`, `speaker_adversarial on/off`, `loss_weight`, `GRL lambda`, `hidden_dim`, `norm_variant` 탐색 | 출력 폴더 기준 complete 4개, 최고 `F1 0.70042`, 최고점은 `speaker_adversarial=false` | [../../outputs/2026-04-22/02-33-33_cnn_conformer](../../outputs/2026-04-22/02-33-33_cnn_conformer) |

### 6.3 주요 결과 요약

| Rank | Trial | F1-macro | Accuracy | UAR | ECE | 핵심 파라미터 요약 |
|---|---|---:|---:|---:|---:|---|
| 1 | [trial_0004](../../outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/trial_summary.json) | 0.70042 | 0.70333 | 0.70000 | 0.22023 | `speaker_adv=false`, `mixup=0.4`, `layernorm` |
| 2 | [trial_0003](../../outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0003/trial_summary.json) | 0.69110 | 0.67333 | 0.66875 | 0.23241 | `speaker_adv=true`, `mixup=0.5`, `loss_weight=0.2`, `hidden=128`, `layernorm` |
| 3 | [trial_0001](../../outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0001/trial_summary.json) | 0.67084 | 0.67333 | 0.68125 | 0.17531 | `speaker_adv=false`, `mixup=0.3`, `layernorm` |
| 4 | [trial_0002](../../outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0002/trial_summary.json) | 0.65893 | 0.66000 | 0.65625 | 0.17464 | `speaker_adv=true`, `mixup=0.5`, `loss_weight=0.2`, `hidden=64`, `batchnorm` |

### 6.4 전략 비교

| 비교축 | complete 수 | best F1 | 해석 |
|---|---|---|---|
| `speaker_adversarial=false` | 2 | `0.70042` | 현재 run 내부 winner |
| `speaker_adversarial=true` | 2 | `0.69110` | 새 branch는 baseline을 넘지 못함 |

추가 상태 메모:

- 출력 폴더 기준 complete trial summary는 4개다.
- 공유 study DB 전체 상태는 `COMPLETE 6 / PRUNED 110 / FAIL 1`이다.
- complete 2개는 같은 study 이름을 공유한 사전 smoke run 기록이므로, 본실험 폴더 해석에서는 제외한다.
- 본실험 폴더만 보면 complete 수는 적지만, pruning 규모까지 포함하면 plateau 판단에는 충분한 신호가 쌓였다.

기준선 비교:

- 2026-04-21 overfitting screening winner: `F1 0.70563`
- 2026-04-22 final round current best: `F1 0.70042`
- 차이: `-0.00521`

즉 이번 final round는 speaker adversarial 추가 여부를 떠나, 아직 screening winner를 회복하지 못했다.

## 7. Artifact 분석

### 7.1 대상 trial

- 최고 trial summary: [../../outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/trial_summary.json](../../outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/trial_summary.json)
- 최고 trial artifact 폴더: [../../outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/artifacts](../../outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/artifacts)
- adversarial 최고 trial summary: [../../outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0003/trial_summary.json](../../outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0003/trial_summary.json)
- adversarial 최고 trial artifact 폴더: [../../outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0003/artifacts](../../outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0003/artifacts)

### 7.2 항목별 해석

- learning curve:
  - `speaker_adversarial=true` trial들은 train loss가 크게 남아 있는 초반에는 val F1가 빠르게 오르지만, 0.69 부근에서 더 이상 screening winner를 넘는 상승이 보이지 않았다.
  - log 상으로 adversarial trial은 emotion head가 안정화되기 전 auxiliary loss가 학습을 더 어렵게 만드는 모습이 있다.

- confusion / macro-F1:
  - `trial_0004`가 `trial_0003`보다 accuracy와 UAR를 모두 앞선다.
  - 즉 adversarial branch가 특정 클래스만 살리고 다른 클래스를 희생하는 패턴도 현재는 보상되지 않았다.

- 추가 메모:
  - pruned trial 중 최고 intermediate도 약 `0.6587` 수준이라 남은 조합에서 late breakthrough가 나올 신호도 약하다.

- calibration:
  - `trial_0004` ECE `0.22023`, `trial_0003` ECE `0.23241`로 둘 다 calibration은 좋지 않다.
  - 이번 라운드는 speaker regularization이 calibration 안정화로도 이어지지 않았다.

- representation / feature map:
  - artifact는 모두 정상 생성되었지만, 현재 지표 수준에서는 새로운 speaker branch가 feature quality를 실질적으로 개선했다는 정량 증거가 없다.

## 8. 종합 인사이트 및 다음 액션

### 8.1 현재 판단

- 이번 lightweight `speaker-invariant adversarial regularization`은 **검증 가치는 있었지만 winner 갱신에는 실패했다.**
- 현재 complete trial 기준 최고점은 여전히 `speaker_adversarial=false`에서 나왔고, baseline mixup winner line이 더 강하다.
- 따라서 이 축은 “도입했으나 본 프로젝트 설정에서는 실효성이 약했다”로 정리하는 편이 맞다.

### 8.2 다음 액션

- 이 final round는 중단하고 결과를 기록한다.
- 이후 선택지는 두 가지다.
  - `논문 작성 시작`: 현재 `~0.70` 성능과 다수의 실패/성공 ablation을 근거로 비교·분석 중심 서술
  - `한 번 더 후속 실험`: overfitting 해결을 더 하되, speaker adversarial처럼 새 학습 branch를 늘리기보다 더 단순하고 방어적인 축만 제한적으로 시도

## 9. 검증 계획

1. `1 trial / 1 epoch / folds_to_run=1` smoke로 auxiliary path가 정상 동작하는지 확인
2. `speaker_adversarial=false`와 `true`가 모두 trial 생성되는지 확인
3. per-trial weights가 trial-local 디렉터리에 저장되는지 확인
4. 그 다음 30-trial 본실험 실행

## 10. 중단 기준

- best F1가 screening winner `0.70563`을 넘지 못하고
- `speaker_adversarial=true` trial이 baseline보다 일관되게 낮으며
- calibration 개선만 있고 F1/UAR 개선이 없으면

이 축은 종료하고, 문서 결론에서는 `mixup winner 유지 + adversarial regularization 비효율`로 정리한다.

## 11. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-04-22 | final round 설계 문서 초안 작성 |
| 2026-04-22 | `02-33-33_cnn_conformer` 실제 결과, top trial 표, artifact 해석, 중단 권고 반영 |
