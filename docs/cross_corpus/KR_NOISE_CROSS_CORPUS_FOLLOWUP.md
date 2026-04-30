# 잡음 환경 및 교차 코퍼스 후속 실험 방향 정리

## 1. 문서 목적

- 현재 확보한 `CNN-Conformer` 대표 설정을 기준으로
  - 잡음 환경 실험을 더 확장해야 하는지
  - 교차 코퍼스 실험을 더 확장해야 하는지
  - 아니면 현재 결과를 논문 서술용 증거로 사용해도 되는지
  를 정리한다.
- 판단 근거는 다음 세 축을 함께 사용한다.
  - 현재 프로젝트 실험 결과
  - `ref.bib`에 수록된 관련 SER 논문
  - 실제 외부 논문 자료의 초록/설명

## 2. 현재 확보한 상태

### 2.1 In-corpus 대표 모델

- `RAVDESS` 내부에서 `CNN-Conformer`는 약 `0.70` 수준의 Accuracy와 Macro-F1을 기록한 대표 구성을 확보했다.
- 이 값은 동일 프로젝트 내 `CNN baseline`, `pure Transformer`보다 높은 구간으로 정리되어 있다.
- 따라서 후속 robustness 실험에서 기준 모델을 하나 고정해야 한다면, 현재로서는 `CNN-Conformer winner`를 사용하는 것이 가장 자연스럽다.

### 2.2 잡음 환경 실험

- 현재 잡음 실험은 `clean` 조건에서 선정된 대표 `CNN-Conformer`를 고정한 뒤,
- `white / pink / babble / cafe`
- `clean / 20dB / 10dB / 5dB / 0dB / -5dB`
  조건으로 평가만 수행하는 구조다.
- 이 설계는 “모델 자체를 다시 학습시키지 않고, 입력 조건 변화에 대한 안정성만 본다”는 점에서 robustness 관찰용 실험으로 성립한다.

### 2.3 교차 코퍼스 실험

- 현재 교차 코퍼스 실험은 `RAVDESS -> CREMA-D 6-class source-only baseline`이다.
- 중요한 점:
  - 기존 `8-class winner checkpoint`를 그대로 옮긴 것이 아니다.
  - `RAVDESS`와 `CREMA-D`의 공통 6개 감정으로 라벨 공간을 재정의하고,
  - `RAVDESS 6-class subset`에서 다시 학습한 뒤,
  - `CREMA-D 6-class` 전체에 직접 평가했다.
- 현재 기록된 결과:
  - source validation: Accuracy `0.58182`, Macro-F1 `0.56897`, UAR `0.58333`
  - target `CREMA-D`: Accuracy `0.19377`, Macro-F1 `0.09243`, UAR `0.18936`
  - source ECE `0.22191`
  - target ECE `0.56109`

## 3. 현재 winner를 고정해 noisy와 cross-corpus를 보는 방식이 맞는가

## 3.1 잡음 환경 실험

결론부터 말하면, **맞다**.

이유는 다음과 같다.

- 잡음 환경 절의 핵심 질문은 “어떤 구조가 가장 강한가”가 아니라 “선정된 대표 구조가 입력 오염에 얼마나 민감한가”이다.
- 이런 질문에는 모델을 다시 튜닝하지 않고 동일 checkpoint를 유지한 채 입력 조건만 바꾸는 방식이 더 해석하기 쉽다.
- `Selective Acoustic Feature Enhancement for Speech Emotion Recognition With Noisy Speech`는 잡음이 SER에 직접적인 성능 저하를 유발하며, 이를 완화하려면 별도의 noisy-aware enhancement 전략이 필요하다고 본다.
  - DOI: `10.1109/TASLP.2023.3340603`
  - 링크: https://pubmed.ncbi.nlm.nih.gov/39015743/
- `A Robust Pitch-Fusion Model for Speech Emotion Recognition in Tonal Languages`도 잡음 하 robustness를 별도 설계 축으로 다룬다.
  - DOI: `10.1109/ICASSP48485.2024.10448373`
- `Dual-TBNet` 역시 robustness를 일반 clean SER와 분리된 문제로 본다.
  - DOI: `10.1109/TASLP.2023.3282092`

즉, 잡음 환경 section에서 지금처럼 `winner 고정 -> noisy evaluation`으로 가는 것은 논리적으로 적절하다.

### 3.1.1 잡음 실험에서 추가 튜닝이 필요한 경우

다만 아래 주장을 하고 싶다면 별도 훈련 실험이 필요하다.

- “현재 구조는 잡음에 강하다”
- “현재 구조는 잡음 robustness를 개선했다”
- “노이즈 환경에서도 안정적이다”

이 경우에는 단순 evaluation만으로 부족하고, 아래 중 하나가 필요하다.

- noisy augmentation을 포함한 재학습
- speech enhancement 또는 selective feature enhancement 결합
- noise-aware multi-condition training

정리하면:

- **robustness 관찰**이 목적이면 지금 방식으로 충분하다.
- **robustness 개선 주장**이 목적이면 별도 학습 실험이 필요하다.

## 3.2 교차 코퍼스 실험

결론부터 말하면, **baseline으로는 맞지만, 개선 실험으로는 부족하다**.

이유는 다음과 같다.

- 교차 코퍼스 SER에서는 source-only baseline을 먼저 두는 것이 일반적이다.
- 그러나 실제 성능 개선은 대체로
  - transfer subspace learning
  - domain invariant feature learning
  - adversarial domain generalization
  - multi-task adaptation
  같은 별도 설계가 들어간다.

근거 논문:

- `Transfer Sparse Discriminant Subspace Learning for Cross-Corpus Speech Emotion Recognition`
  - DOI: `10.1109/TASLP.2019.2955252`
- `Nonnegative Matrix Factorization Based Transfer Subspace Learning for Cross-Corpus Speech Emotion Recognition`
  - DOI: `10.1109/TASLP.2020.3006331`
- `Domain Invariant Feature Learning for Speaker-Independent Speech Emotion Recognition`
  - DOI: `10.1109/TASLP.2022.3178232`
  - 링크: https://portal.fis.tum.de/en/publications/domain-invariant-feature-learning-for-speaker-independent-speech-/
- `Unsupervised Cross-Corpus Speech Emotion Recognition Using a Multi-Source Cycle-GAN`
  - DOI: `10.1109/TAFFC.2021.3095717`
- `Adversarial Domain Generalized Transformer for Cross-Corpus Speech Emotion Recognition`
  - DOI: `10.1109/TAFFC.2023.3290795`
  - 링크: https://cir.nii.ac.jp/crid/1360302866839799296
- `Multitask Transformer for Cross-Corpus Speech Emotion Recognition`
  - DOI: `10.1109/TAFFC.2025.3526592`

현재 결과가 매우 낮게 나온 이유를 단순히 “CNN-Conformer가 약하다”로 결론내리기 어려운 이유도 여기 있다.

- 현재 실험은 `fold 1`만 수행했다.
- `8-class in-corpus winner`를 직접 재사용하지 않았다.
- `cross-corpus 전용 adaptation`이 전혀 없다.
- 그럼에도 target 성능이 크게 무너졌다는 점은, **source-only 상태의 일반화는 약하다**는 사실을 보여준다.

즉:

- 지금 방식은 **교차 코퍼스 baseline**으로는 적합하다.
- 하지만 **교차 코퍼스 개선 실험**을 하고 싶다면 별도 설계가 필요하다.

## 4. 그럼 이후 무엇을 더 해야 하는가

## 4.1 논문 완성 우선일 때

현재 학사 논문 범위에서 가장 보수적이고 안전한 선택은 다음이다.

1. `CNN baseline vs pure Transformer vs CNN-Conformer`의 in-corpus 비교를 핵심 본선으로 둔다.
2. `CNN-Conformer winner`로 noisy robustness section을 유지한다.
3. `CNN-Conformer winner family`로 cross-corpus source-only baseline을 제시한다.
4. noisy와 cross-corpus는 “구조 일반화의 한계와 확장 가능성”을 보여주는 보조 section으로 위치시킨다.

이 경로의 장점:

- 현재까지 구현한 실험 자산을 그대로 활용할 수 있다.
- 각 section의 목적이 분리되어 해석이 깔끔하다.
- 학사 논문 분량과 시간에 비해 과도한 실험 확장을 피할 수 있다.

이 경로의 한계:

- cross-corpus section은 개선 결과가 아니라 baseline 관찰에 머문다.
- noisy section도 robustness characterization 수준에 머문다.

## 4.2 실험을 딱 하나만 더 한다면

가장 의미 있는 추가 실험은 **cross-corpus 전용 경량 adaptation 축 1개**다.

이유:

- noisy는 이미 대표 모델의 안정성 곡선을 충분히 보여주고 있다.
- 반면 cross-corpus는 현재 값이 매우 낮아, “왜 일반화가 무너지는가”를 한 단계 더 보여줄 여지가 있다.
- 관련 문헌도 교차 코퍼스 문제는 일반 hyperparameter tuning보다 domain shift 완화 설계가 더 직접적이라고 본다.

추천 우선순위:

1. `source-only 5-fold 평균` 완성
2. 그 다음 한 축만 추가

추가 축 후보:

- `도메인 불변 정규화/표준화`
  - 예: utterance-level CMVN, instance-style normalization, speaker/corpus 편향 완화용 normalization
  - 장점: 구현 부담이 가장 작다.
  - 단점: 개선 폭은 제한적일 수 있다.
- `간단한 domain adversarial head`
  - 장점: 문헌 근거가 가장 직접적이다.
  - 단점: 구현량이 늘어난다.
- `source + unlabeled target feature alignment`
  - 장점: cross-corpus 문제 정의에 가장 가깝다.
  - 단점: 논문 서술과 코드가 함께 커진다.

현재 프로젝트 여건을 고려하면, **한 번 더 한다면 normalization 또는 경량 adversarial adaptation 중 하나만** 고르는 편이 적절하다.

## 4.3 baseline 강화를 다시 해야 하는가

우선순위는 높지 않다.

이유:

- 현재 thesis의 중심 비교축은 `CNN vs Transformer 계열`이다.
- in-corpus 기준으로는 이미 `CNN-Conformer`가 대표 모델로 정리될 정도의 결과가 있다.
- 이 시점에서 baseline을 다시 크게 손보면 본선 비교축이 흔들릴 수 있다.

따라서 추가 시간을 쓰더라도,

- `baseline 재강화`
보다
- `cross-corpus 전용 일반화 축 1개`

가 더 논문 메시지에 도움이 된다.

## 5. 최종 권고

현재 상태에서의 최종 권고는 다음과 같다.

### 권고 A

- noisy 실험은 **지금 상태로 충분하다**.
- 본문에서는 “대표 구조의 입력 오염 민감도 관찰”로 서술한다.
- 별도 noisy 재학습은 하지 않아도 된다.

### 권고 B

- cross-corpus 실험은 **현재 source-only baseline으로 의미가 있다**.
- 다만 논문에서 “개선”을 보이고 싶다면, 추가 실험은 random tuning이 아니라 **domain shift 완화 축**으로 가야 한다.

### 권고 C

- 시간이 많지 않다면 여기서 실험을 멈추고 논문 작성으로 넘어가도 된다.
- 시간이 조금 더 있다면, `cross-corpus 5-fold 평균 + 경량 adaptation 1개`까지만 추가하는 것이 가장 효율적이다.

## 6. 참고 자료

- CREMA-D dataset paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/
- Cross-corpus SER review: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.784514/full
- Selective Acoustic Feature Enhancement for Noisy SER: https://pubmed.ncbi.nlm.nih.gov/39015743/
- Domain Invariant Feature Learning: https://portal.fis.tum.de/en/publications/domain-invariant-feature-learning-for-speaker-independent-speech-/
- Adversarial Domain Generalized Transformer: https://cir.nii.ac.jp/crid/1360302866839799296
