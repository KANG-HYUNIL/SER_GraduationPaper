# CNN Conformer

## 1. 문서 범위

- 대상 모델: `cnn_conformer`
- 목적: CNN Conformer 계열의 구조 이해, 코드 기준 구현 해설, 날짜별 실험 문서 인덱스, 차후 실험 방향 정리
- 상태: `active`

이 문서는 더 이상 날짜별 실험 로그를 모두 누적하는 저장소가 아니다.  
실험 회차별 상세 결과는 하위 문서로 분리하고, 이 문서는 `cnn_conformer` 구조를 빠르게 이해하고 각 회차 문서로 이동하기 위한 기준 문서로 유지한다.

## 2. 모델 스냅샷

### 2.1 한 줄 요약

`cnn_conformer`는 최종 winner 기준으로, Log-Mel spectrogram을 `nostem_patch` 방식으로 시간 분할한 뒤 Conformer encoder로 장단기 문맥을 함께 읽어 감정을 분류하는 SER 모델이다.

### 2.2 핵심 구성 요소

| 항목 | 값 또는 설명 |
|---|---|
| 입력 표현 | `log-Mel spectrogram`, 기본 실험 축은 주로 `n_mels=80`, `hop_length=160` |
| 입력 stem | winner 기준 `nostem_patch` |
| 시퀀스 투영 | `LayerNorm + Linear` |
| 핵심 블록 | `CNNConformerBlock` = `FFN -> MHSA -> Conv -> FFN` |
| attention | 기본 `relative positional MHSA` |
| 계층 융합 | winner 기준 `last` |
| utterance pooling | winner 기준 `attention` |
| 출력 | 8-class emotion classifier |

### 2.2.1 최종 winner 설정

| 항목 | 값 |
|---|---|
| backbone | `nostem_patch` |
| `time_patch` | `4` |
| `norm_variant` | `layernorm` |
| `embed_dim` | `192` |
| `num_layers` | `4` |
| `num_heads` | `8` |
| `ffn_dim` | `768` |
| `conv_kernel_size` | `31` |
| `layer_fusion` | `last` |
| `pooling` | `attention` |
| regularization | `mixup alpha=0.4` |
| best metric | `F1 0.70563`, `Accuracy 0.70000`, `UAR 0.70938` |

### 2.3 프로젝트 내 비교 관점

- CNN baseline보다 local cue와 global context를 더 함께 다룰 수 있는 주력 후보 구조다.
- window transformer 계열보다 현재 저장소에서는 더 안정적으로 상위권 정확도를 형성했다.
- 다만 최근 구조 확장 실험에서 peak accuracy가 다시 내려간 사례가 있어, 무작정 구조를 늘리기보다 어떤 축이 실제로 유효한지 통제된 실험이 중요하다.

## 3. 아키텍처 상세

### 3.1 전체 흐름

1. 입력은 `[B, 1, F, T]` 형태의 spectrogram이다.
2. winner에서는 CNN stem을 쓰지 않고, `nostem_patch`가 전체 주파수 대역을 한 번에 덮는 시간 분할 token을 만든다.
3. `LayerNorm + Linear`로 token을 `embed_dim=192` 공간으로 투영한다.
4. 투영된 시퀀스를 `CNNConformerBlock` 4층에 통과시킨다.
5. winner에서는 `layer_fusion=last`를 사용해 마지막 층 출력만 사용한다.
6. final norm 후 `attention pooling`으로 utterance embedding을 만든다.
7. 분류기에서 8개 감정 클래스로 예측한다.

### 3.2 전체 파이프라인 Mermaid

```mermaid
flowchart LR
    A[Log-Mel Spectrogram\nB x 1 x F x T] --> B[NoStem Patch Projection\ntime_patch 4]
    B --> C[LayerNorm + Linear Projection\nembed 192]
    C --> D[Positional Dropout]
    D --> E[CNNConformerBlock x4\nheads 8, ffn 768, kernel 31]
    E --> F[Last Layer Output]
    F --> G[LayerNorm]
    G --> H[AttentivePooling]
    H --> I[Classifier]
    I --> J[8 Emotion Logits]
```

### 3.3 Conformer block 상세

winner에서 사용하는 `CNNConformerBlock`은 macaron-style FFN과 single convolution module을 유지하면서, `relative positional MHSA`를 사용한다.

```mermaid
flowchart TD
    X[Input Sequence] --> A[FFN 1]
    A --> B[Residual Add x0.5]
    B --> C[LayerNorm]
    C --> D[Self-Attention\nrelative]
    D --> E[Residual Add]
    E --> F[Conv Module\nsingle, kernel 31]
    F --> G[Residual Add]
    G --> H[FFN 2]
    H --> I[Residual Add x0.5]
    I --> J[Mask Re-apply]
    J --> K[Final LayerNorm]
```

### 3.4 winner 기준 front-end 해석

winner에서는 `stem_strides` 자체를 쓰지 않는다.  
즉, 표준 CNN stem 압축이 아니라 `nostem_patch`로 직접 시간축 token을 만든다.

```mermaid
flowchart LR
    A[n_mels x time] --> B[NoStem Patch\nfull-frequency kernel]
    B --> C[time_patch 4]
    C --> D[Sequence Tokens]
```

### 3.5 핵심 파라미터와 역할

| 파라미터 | 역할 | 실험상 의미 |
|---|---|---|
| `stem_channels` | CNN stem 폭 | 초기 local feature 추출량 |
| `stem_strides` | subsampling 강도 | 시간/주파수 해상도 보존 정도 |
| `embed_dim` | encoder 너비 | 표현력과 연산량의 핵심 축 |
| `num_layers` | encoder 깊이 | 문맥 누적량과 smoothing 강도 |
| `num_heads` | attention head 수 | 시간 의존성 분해 granularity |
| `conv_kernel_size` | depthwise conv 시야 | 감정의 짧은 burst vs 긴 prosody 포착 범위 |
| `layer_fusion` | 계층 결합 방식 | 저층 acoustic cue와 고층 context 결합 여부 |
| `conv_module_type` | single / multiscale | local receptive field 다양화 여부 |
| `pooling` | `attention` / `mean` | utterance-level 요약 방식 |
| `aggregation_mode` | chunk-level vote 방식 | 긴 발화 chunk 통합 안정성 |

### 3.6 코드 기준 구현 위치

- 모델 본체: [src/models/cnn_conformer.py](../src/models/cnn_conformer.py)
- 블록 구현: [src/models/cnn_conformer_blocks.py](../src/models/cnn_conformer_blocks.py)
- 공통 transformer 유틸: [src/models/transformer_blocks.py](../src/models/transformer_blocks.py)
- 모델 설정: [src/configs/model/cnn_conformer.yaml](../src/configs/model/cnn_conformer.yaml)
- Optuna 기본 설정: [src/configs/optuna/default.yaml](../src/configs/optuna/default.yaml)
- 구조 탐색 preset: [src/configs/optuna/cnn_conformer_structural.yaml](../src/configs/optuna/cnn_conformer_structural.yaml)

## 4. 논문 근거와 현재 구현의 대응

### 4.1 핵심 참고 논문

- Gulati et al. 2020, Conformer: local convolution과 global self-attention을 결합한 원형 구조
- Zou et al. 2022, co-attention 기반 multi-level acoustic information 융합
- Peng et al. 2021, multi-scale CNN과 attention의 SER 적용
- Pepino et al. 2021, wav2vec 2.0 embedding 기반 SER
- Morais et al. 2022, self-supervised speech feature의 SER 활용

관련 참고문헌:

- [ref.bib](../LateX_Paper/undergraduate-thesis/undergraduate-thesis/misc/ref.bib)

### 4.2 논문과 현재 코드의 정합성

- Conformer의 block ordering은 원형과 같은 큰 골격을 유지한다.
- 입력 front-end는 ASR용 raw feature stack이 아니라 SER용 2-stage CNN stem으로 재구성했다.
- 주파수 축을 평균 풀링하지 않고 flatten 후 projector에 넣는 방식으로, formant와 대역별 에너지 구조를 초기 단계에서 덜 잃도록 했다.
- attention은 Transformer-XL식 상대 위치 인코딩이 아니라 learnable relative bias embedding 기반 구현이다.
- convolution module은 현재 공용 `ConvModule`을 사용하고, 필요 시 multiscale branch를 붙이는 방식으로 변형 가능하게 만들었다.

### 4.3 실험 데이터와 맞물린 해석

- 2026-04-17 champion은 `kernel=31`, `attention pooling`, `embed_dim=256`, `layers=4`로 수렴했다.
- 2026-04-19 구조 탐색에서는 time-preserving subsampling이 peak를 만들었지만, `learned_sum`과 `multiscale conv`는 평균 안정성 또는 가설 검증 수준에 그쳤고 champion을 넘지 못했다.
- confusion matrix 기준 병목은 `sad`, `neutral`, `calm`, `happy`, `fearful`의 경계다. 반면 `angry`는 상대적으로 분리된다.
- 따라서 다음 실험은 “모델을 더 크게”보다 “경계가 겹치는 감정 구간을 더 잘 분리하도록” 가는 편이 타당하다.

## 5. 날짜별 실험 문서

| 날짜 | 문서 | 핵심 내용 |
|---|---|---|
| 2026-04-15 | [2026-04-15.md](./cnn_conformer/2026-04-15.md) | 초기 stage2 Optuna, baseline 수준의 첫 conformer 탐색 |
| 2026-04-16 | [2026-04-16.md](./cnn_conformer/2026-04-16.md) | padding-safe 회귀와 SOTA-faithful 재출발 |
| 2026-04-17 | [2026-04-17.md](./cnn_conformer/2026-04-17.md) | fixed log-Mel relative Conformer champion과 artifact 분석 |
| 2026-04-18 | [2026-04-18.md](./cnn_conformer/2026-04-18.md) | regularization HPO 실패와 underfitting 신호 |
| 2026-04-19 | [2026-04-19.md](./cnn_conformer/2026-04-19.md) | 구조 축 탐색과 subsampling/layer fusion/conv branch 검증 |
| 2026-04-19 Round 2 | [2026-04-19_round2.md](./cnn_conformer/2026-04-19_round2.md) | layer fusion + loss + sampler 결합 탐색 결과와 중단 결론 |
| 2026-04-20 Redesign | [2026-04-20_backbone_redesign.md](./cnn_conformer/2026-04-20_backbone_redesign.md) | conformer backbone 재설계 결과와 artifact 인사이트 |
| 2026-04-21 Generalization | [2026-04-21_nostem_generalization.md](./cnn_conformer/2026-04-21_nostem_generalization.md) | `nostem_patch` winner 기반 overfitting 완화 실험 계획 |
| 2026-04-22 Follow-up | [2026-04-22_overfitting_followup.md](./cnn_conformer/2026-04-22_overfitting_followup.md) | 중복 축 제외 후 overfitting 후속 실험 설계 |
| 2026-04-22 Final Round | [2026-04-22_speaker_invariant_final.md](./cnn_conformer/2026-04-22_speaker_invariant_final.md) | winner branch 고정 후 speaker-invariant adversarial regularization 최종 실험 |

## 6. 다음 실험 권장 방향

### 6.1 결론부터

현재 시점의 다음 라운드는 “구조를 더 복잡하게 추가”보다 아래 3갈래 중 하나를 명확히 선택하는 편이 적합하다.

1. champion 회복형 미세탐색
2. scratch conformer 재설계
3. 새로운 non-SSL transformer 계열

### 6.2 권장안 1: champion 회복형 미세탐색

고정 권장 축:

- `stem_strides=[[2,1],[2,2]]`
- `layer_fusion=last`
- `conv_module_type=single`
- `pooling=attention`
- `aggregation_mode=confidence_weighted_logit`
- `label_smoothing=0.0`
- `freq_mask_count=0`

근거:

- 2026-04-19 최고점 `trial_0043`은 구조 확장보다 위 조합에서 나왔다.
- 2026-04-18 regularization HPO는 강한 규제가 오히려 peak를 깎았다.
- 즉 현재 병목은 “더 많은 장치 추가”가 아니라 “승자 구조를 덜 흐리게 조율하는 것”에 가깝다.

### 6.3 권장안 2: scratch conformer 재설계

round2까지의 결과를 보면 현 구조의 국면은 어느 정도 정리됐다.

- 2026-04-17 champion `0.63168`
- 2026-04-19 구조 탐색 최고 `0.62017`
- 2026-04-19 round2 최고 `0.61282`

즉 현재 `CNN stem -> flatten frequency -> Conformer` 조합은 미세 설정 차이로 움직이더라도 큰 폭 개선이 어려운 상태다.  
conformer를 유지하되 새로 볼 가치가 있는 재설계 축은 다음 두 가지다.

- 앞단 CNN 압축 약화 또는 제거
  - 너무 이른 국소 압축이 감정 cue를 깎는지 검증
  - 예: `linear patch/projection -> conformer` 또는 `single-stage light conv -> conformer`
- frequency flatten 방식 재설계
  - 현재는 `channels x freq` 전체 flatten 후 projection인데, 이 방식이 주파수 구조를 보존하는 대신 과도한 차원 혼합을 만들 수 있다.
  - 예: `freq-aware projection`, `band-wise tokenization`, `time-first patch tokenization`

이 방향은 “conformer를 계속 보되, 현재 backbone 가정을 다시 열어보는 것”이다.

### 6.4 권장안 3: 새로운 non-SSL transformer 계열

`ref.bib` 기준으로, SSL 없이도 다음 후보들은 SER 특화 transformer 방향으로 의미가 있다.

- `Multi-Scale Temporal Transformer`
  - 기준 문헌: `li2023multiscaletransformer`
  - 장점: 현재 conformer에서 kernel/stride로 우회적으로 보던 시간 스케일 문제를 transformer 단계에서 직접 다룰 수 있다.
- `Multiple Acoustic Features + Cross-Attention Transformer`
  - 기준 문헌: `he2023multiple`, `zhao2025crosslingual`, `ryu2025pcm`
  - 장점: log-Mel 고정은 유지하되, pitch contour나 energy 같은 제한된 보조 acoustic feature를 추가해 transformer가 상호작용을 배우게 할 수 있다.
  - 주의: feature engineering이 과해지면 논문 주제가 흐려지므로 보조 feature는 최소화해야 한다.
- `Conditional Transformer`
  - 기준 문헌: `chung2025conditionaltransformer`
  - 장점: 감정 판별에 중요한 지역을 conditioning 또는 gating으로 강조하는 방향

이 중 현재 프로젝트와 논문 주제에 가장 잘 맞는 우선순위는 다음과 같다.

1. `Multi-Scale Temporal Transformer`
2. `log-Mel + pitch contour` 수준의 경량 cross-attention transformer
3. 조건부 transformer 계열

backbone 재설계 결과는 [2026-04-20_backbone_redesign.md](./cnn_conformer/2026-04-20_backbone_redesign.md)에, 그 후속 overfitting 완화 실험은 [2026-04-21_nostem_generalization.md](./cnn_conformer/2026-04-21_nostem_generalization.md)에 정리했다.

## 7. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-04-19 | 메인 문서를 구조 설명 중심 기준 문서로 재구성하고, 날짜별 하위 문서 분리 구조를 도입 |
