# CNN Conformer Experiment - 2026-04-22 Speaker-Invariant Final Round

## 1. 문서 범위

- 대상 모델: `cnn_conformer`
- 문서 목적: 2026-04-22 overfitting screening 종료 후, winner branch를 고정한 상태에서 마지막 일반화 개선 축으로 `speaker-invariant adversarial regularization` 실험을 설계하고 구현 범위를 기록
- 현재 문서 상태: `planned`

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

## 6. 검증 계획

1. `1 trial / 1 epoch / folds_to_run=1` smoke로 auxiliary path가 정상 동작하는지 확인
2. `speaker_adversarial=false`와 `true`가 모두 trial 생성되는지 확인
3. per-trial weights가 trial-local 디렉터리에 저장되는지 확인
4. 그 다음 30-trial 본실험 실행

## 7. 중단 기준

- best F1가 screening winner `0.70563`을 넘지 못하고
- `speaker_adversarial=true` trial이 baseline보다 일관되게 낮으며
- calibration 개선만 있고 F1/UAR 개선이 없으면

이 축은 종료하고, 문서 결론에서는 `mixup winner 유지 + adversarial regularization 비효율`로 정리한다.
