# 논문 Chapter 4 계획

## 목적

현재 시점에서 Chapter 4 전체를 미리 완성하는 것이 목적이 아니다.  
지금 실제로 작성 대상인 부분은 `4.1`과 `4.2`이며, `4.3`과 `4.4`는 후속 실험을 어떤 규칙으로 수행할지 미리 정의하는 수준까지만 계획한다.

현재 진행 중인 모델 실험들은 아래 목적에 대응한다.

- `cnn` 계열: 4.1, 4.2의 기준선 형성
- `pure transformer`: 순수 전역 attention 구조의 한계 확인
- `cnn_conformer`: local-global 결합 구조의 강점과 과적합 한계 확인
- `hierarchical window transformer`: window 기반 local modeling의 가능성과 한계 확인
- `bridged window transformer`: window 구조의 global context 부족을 보완하는 창신적 확장 시도

즉 현재 Chapter 4의 핵심 서사는 `가설 기반 구조 탐색 -> 수정 -> 해석`이다.

## 현재 작성 범위

### 4.1 실험 파라미터 및 설정

4.1은 공정 비교를 위한 공통 조건을 선언하는 절이다.  
이 절에서는 모델별 우열을 말하는 것이 아니라, 이후 결과 해석이 가능하도록 실험 조건을 고정하고 비교 규칙을 명시한다.

4.1에 포함할 내용:

- 데이터셋
  - RAVDESS
  - 8개 감정 클래스
  - actor 기반 subject-independent 평가
- 데이터 분할
  - GroupKFold
  - actor ID 기준 분리
- 입력 표현
  - log-Mel spectrogram
  - sample rate, n_mels, n_fft, hop_length
  - normalize 여부
  - resize 또는 dynamic padding 처리 방식
  - chunking 사용 여부
- 평가 지표
  - Accuracy
  - UAR
  - F1-macro
  - ECE
- 공통 학습 설정
  - optimizer
  - epoch
  - early stopping
  - batch size
  - folds_to_run
- Optuna 탐색 원칙
  - 1차는 1-fold 기반 빠른 구조 탐색
  - 상위 조합만 재학습 및 재검증

4.1에서 강조할 문장 방향:

- 입력 표현과 평가 지표는 최대한 공통으로 유지하고 모델 구조 차이만 비교한다.
- 현재 구조 탐색 단계에서는 log-Mel 조건을 고정하여 backbone 차이를 우선 확인한다.

### 4.2 종합 실험

4.2는 현재 연구의 본론이다.  
이 절에서는 모든 실험을 나열하는 것이 아니라, 구조 가설이 어떻게 수정되었는지가 드러나도록 결과를 배열한다.

권장 흐름:

1. CNN 계열 기준선 확보
2. pure transformer 도입
3. cnn_conformer로 local-global 결합 구조 검증
4. hierarchical window transformer로 window 기반 구조 검증
5. bridged window transformer로 window 구조 보완 시도
6. 현재 시점 최종 비교 및 해석

각 모델군은 같은 서술 형식으로 정리한다.

1. 설계 가설
2. 핵심 구조
3. 대표 결과
4. 해석

이렇게 써야 4.2가 모델 소개가 아니라, 연구자가 구조를 어떻게 수정해 갔는지 보여주는 절이 된다.

#### 4.2에서 다룰 모델별 역할

`CNN 계열`

- 기준선 성능 확보
- 안정적인 입력 설정과 기본 분류 성능 확인
- Transformer 계열 비교의 출발점 제공

`Pure Transformer`

- CNN 없이 전역 self-attention만으로 감정 분류가 가능한지 검토
- local inductive bias 부족 여부를 확인

`CNN Conformer`

- CNN의 local bias와 Transformer/Conformer의 long-range modeling 결합 가설 검증
- 현재 최고 성능 후보이지만 과적합 및 calibration 한계를 같이 서술

`Hierarchical Window Transformer`

- 2D local window와 stage 구조가 효율성과 성능을 동시에 줄 수 있는지 검토
- 성능이 낮았던 원인을 global context 부족 관점에서 해석

`Bridged Window Transformer`

- window backbone은 유지하되, cross-scale bridge context로 global emotion context를 보완한다는 가설 제시
- 기존 hierarchical window의 한계에 대한 수정 실험으로 배치

#### 4.2에서 필요한 표와 그림

- 모델군별 대표 성능 비교 표
- 최고 성능 trial 요약 표
- confusion matrix
- calibration curve
- t-SNE
- attention map 또는 feature map

4.2의 목적은 단순 최고 성능 제시가 아니라 아래 질문에 답하는 것이다.

- 어떤 구조가 현재 데이터 조건에서 가장 유리했는가
- 어떤 구조가 과적합 또는 일반화 한계를 보였는가
- 다음 수정 실험이 왜 필요했는가

#### 추가 검증: 우승 백본 확정 후 절제 연구 (Ablation Study) 및 성능 극대화

4.2에서 두 모델(Conformer vs Window) 중 성능과 방어력이 가장 우수한 최종 백본이 확정되면, 다음의 최적화 스텝과 Ablation Study를 수행하여 해당 구조의 '설계적 타당성'을 과학적으로 입증한다. 

- **Ablation 1 (주파수 보존 효과 확인)**: `FlattenFrequencyProjector`를 제거하고 구형 모델들처럼 `AdaptiveAvgPool2d`(주파수 평균치 병합)를 사용할 때성능(포먼트 정보 손실)이 얼마나 떨어지는지 비교.
- **Ablation 2 (어텐션 메커니즘 확인)**: `Relative Position MHSA`를 `Absolute MHSA`로 변경해 보며, 발화 속도가 다른 음성에서 '상대적 위치 정보'가 얼마나 중요한지 입증.
- **Ablation 3 (규제 기법 및 HPO 결과)**: 구조를 고정한 뒤, Optuna를 활용해 국소적 HPO(LR, Epoch 등)를 수행하고 최적화. 동시에 `SpecAugment`와 `Label Smoothing` 적용 시 모델의 확신도 보정(Calibration ECE 감소)과 과적합 방어 효과가 어떻게 나타나는지 검증.
- **Ablation 4 (시간축 압축 전략)**: `Attentive Pooling` vs 단순 `Mean Pooling` 성능 비교.

#### 최후의 영점 조절 (Final Hyperparameter Optimization, HPO)

Ablation Study까지 통과하여 '구조적 타당성'이 증명된 최고의 모델(Architecture)에 대해, 구조 크기(Layer, 차원 등)를 고정해 둔 상태로 '훈련 방법론(Training Recipe)' 자체만을 극한으로 최적화한다. (학위논문 및 최신 SER/ASR 논문들의 필수 통과 의례)

최종 튜닝 대상 파라미터는 구조 변경 없이 다음 5가지로 한정하여 검색 공간(Search Space)을 좁히고 효율을 극대화한다:
1. **Learning Rate (학습률)**: 최고 성능 도달 속도 및 안정성 (`1e-4` ~ `5e-4` 구간 정밀 타격).
2. **Weight Decay (L2 정규화)**: 파라미터의 비대화(Overfitting) 방지 강도 튜닝 (`1e-5` ~ `1e-3`).
3. **Dropout 비율 (세부 부위별)**: 구조 전체 통일이 아닌 `Input Dropout`, `Attention Dropout`, `FFN Dropout` 등 각 부위별 비율을 개별적으로 미세 조정.
4. **Label Smoothing Factor**: 모델의 과잉 확신(Overconfidence) 방지 수치 조절 (보통 `0.05` ~ `0.15` 사이 탐색).
5. **SpecAugment Masking 강도**: 주파수(Freq) 및 시간(Time) 마스킹 영역의 개수(Count)와 너비(Width) 강도 조절.

이 최종 튜닝을 통해 뽑은 "Silver Bullet(은탄환)" 세팅의 모델이 본 논문이 내세우는 최종 SOTA 후보(Champion Model)가 된다.

## 4.3과 4.4의 현재 위치

지금은 4.3과 4.4를 본문처럼 자세히 쓰는 단계가 아니다.  
현재는 `후속 실험 설계 원칙`만 미리 정의해 두는 단계다.

즉 지금 해야 할 일은 아래 두 가지다.

- 4.3, 4.4를 당장 작성하지 않는다.
- 대신 나중에 어떤 방식으로 확장할지 규칙을 먼저 정한다.

## 4.3 잡음 조건 실험 계획

### 현재 결론

잡음 실험은 `새 구조 탐색`이 아니라, `4.2에서 선정된 최고 clean-condition 모델의 robustness 검증`으로 보는 것이 맞다.

### 일반적인 논문 전개 방식

SER의 noisy-condition 연구들은 보통 아래 순서를 따른다.

1. clean 조건에서 backbone 또는 기본 모델을 먼저 확정
2. 그 backbone을 유지한 상태에서 noise augmentation, enhancement, robustness 기법을 비교
3. noisy 조건에서 전체 구조 탐색을 다시 처음부터 반복하지 않음

참고:

- MetricAug, Interspeech 2023  
  https://www.isca-archive.org/interspeech_2023/wu23c_interspeech.html
- RL-based augmentation for noise robust SER, Interspeech 2024  
  https://www.isca-archive.org/interspeech_2024/ranjan24_interspeech.html
- Speech Enhancement Preprocessing for SER in Realistic Noisy Conditions, Interspeech 2020  
  https://www.isca-archive.org/interspeech_2020/zhou20g_interspeech.html

### 우리 논문에서의 4.3 계획

후속 4.3은 아래 원칙으로 진행한다.

1. 4.2에서 선정된 최고 성능 모델 1개를 주 backbone으로 선택
2. 필요하면 generalization 성향이 다른 보조 모델 1개만 추가 비교
3. backbone 구조는 고정
4. 변경 대상은 noise 대응 요소로 한정
   - noise augmentation
   - SpecAugment 강도
   - enhancement preprocessing
5. SNR 및 noise type 프로토콜을 명시적으로 고정

즉 4.3은 `최고 backbone을 noisy 환경으로 확장하는 장`으로 설계한다.

## 4.4 Cross-Corpus 실험 계획

### 현재 결론

cross-corpus 실험도 `새 구조 탐색`이 아니라, `4.2에서 검증된 backbone의 domain shift generalization 평가`로 보는 것이 맞다.

### 일반적인 논문 전개 방식

Cross-corpus SER 논문들은 대체로 아래 원칙을 따른다.

1. source corpus에서 학습
2. source validation으로 모델 선택
3. target corpus는 최종 평가에 사용
4. 필요 시 adaptation 여부를 별도 실험으로 분리

즉 target test를 보면서 구조를 다시 최적화하는 방식은 피하는 편이 일반적이다.

참고:

- Cross-Corpus SER Using SSL Representations, Applied Sciences 2023  
  https://www.mdpi.com/2076-3417/13/16/9062
- Cross-corpus SER using subspace learning and domain adaptation, EURASIP 2022  
  https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-022-00264-5
- Cross Corpus SER using transfer learning and Wav2Vec2 fusion, KBS 2023  
  https://doi.org/10.1016/j.knosys.2023.110814

### 우리 논문에서의 4.4 계획

후속 4.4는 아래 원칙으로 진행한다.

1. 4.2에서 선택된 최고 backbone 1개를 기본 모델로 사용
2. 필요하면 일반화 성향이 다른 보조 backbone 1개만 추가
3. source-domain validation 기준으로만 설정 선택
4. target-domain test는 최종 평가에서만 사용
5. adaptation이 들어간다면 zero-shot과 adaptation을 분리해서 비교

즉 4.4는 `clean within-corpus에서 검증한 backbone을 cross-corpus로 확장하는 장`으로 설계한다.

## 현재 작업 우선순위

지금 실제로 해야 할 일은 아래 순서다.

1. 4.1 초안 작성
2. 4.2 초안 작성
3. 4.2에 들어갈 표와 그림 고정
4. `bridged_window_transformer` 결과 확인
5. `cnn_conformer` regularization 결과 확인
6. 그 다음 4.3, 4.4 실험 프로토콜 확정

## 한 줄 요약

현재 Chapter 4는 `4.1과 4.2를 먼저 완성하는 단계`이며, 4.3과 4.4는 지금 쓰는 장이 아니라 `4.2에서 고른 최적 backbone을 이후 noisy 환경과 cross-corpus 환경으로 어떻게 확장할지 정의하는 계획 장치`로 유지한다.
