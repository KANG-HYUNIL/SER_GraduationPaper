# Transformer 계열 모델 정리

이 문서는 현재 코드베이스에 있는 세 가지 transformer 계열 모델을 초보자도 이해할 수 있게 설명한다.

- `pure_transformer`
- `cnn_conformer`
- `hierarchical_window_transformer`

설명은 아래 두 축을 함께 따른다.

- 현재 저장소의 실제 구현 코드
- `ref.bib`에 포함된 transformer 계열 핵심 논문

직접적으로 참고한 논문은 다음과 같다.

- `vaswani2017attention`
  - Transformer의 기본 self-attention 개념
- `wang2024swin`
  - shifted window 기반 speech SER 모델
- `chen2023dwformer`
  - speech SER에서 window 기반 attention을 동적으로 다루는 접근
- `li2023multiscaletransformer`
  - speech SER에서 시간 축 문맥을 여러 스케일로 다루는 필요성

`cnn_conformer`의 경우 현재 `ref.bib`에는 Conformer 원 논문이 직접 들어 있지 않다. 따라서 이 문서의 `cnn_conformer` 설명은 현재 구현 코드와 transformer 기반 SER 문헌의 공통 구조를 바탕으로 정리한다.

## 한눈에 비교

| 모델 | 입력 초반 처리 | attention 범위 | local pattern 처리 | 메모리/연산 특성 | 현재 역할 |
|---|---|---|---|---|---|
| `pure_transformer` | patch embedding만 사용 | 전체 token에 global attention | 약함 | 가장 무거운 편 | 순수 transformer 기준선 |
| `cnn_conformer` | CNN stem 사용 | 전체 time sequence에 global attention | 강함 | 중간 | 안정적인 hybrid 기준선 |
| `hierarchical_window_transformer` | CNN stem 사용 | window 단위 local attention 후 계층적 확장 | 강함 | 가장 현실적 | 현재 메인 실험 구조 |

## Transformer가 뭔가

Transformer는 입력을 여러 개의 token으로 나눈 뒤, 각 token이 다른 token을 얼마나 참고해야 하는지를 attention으로 계산하는 구조다. `Attention Is All You Need`에서는 이 self-attention이 RNN 없이도 긴 거리 문맥을 직접 연결할 수 있다는 점이 핵심이었다.

중요한 직관은 다음과 같다.

- token은 입력을 쪼갠 작은 단위다.
- self-attention은 각 token이 다른 token들을 가중합해 새 표현을 만드는 연산이다.
- 모든 token이 서로 직접 attention하면 전역 문맥을 잘 보지만, 비용이 커진다.
- 그래서 speech SER에서는 "얼마나 넓게 보느냐"와 "local cue를 얼마나 잘 살리느냐"가 구조 차이를 만든다.

SER에서는 감정 단서가 짧은 burst, formant 변화, 에너지 윤곽, 발화 말미의 prosody처럼 국소적으로 나타나는 경우가 많다. 이 때문에 이미지나 대규모 NLP처럼 pure transformer를 그대로 쓰면 항상 유리하지 않다.

## 1. Pure Transformer

### 핵심 아이디어

`pure_transformer`는 spectrogram을 patch 단위로 잘라 token으로 만든 뒤, 곧바로 transformer encoder에 넣는다. CNN stem이 없고, local pattern을 먼저 압축해 주는 강한 inductive bias도 없다.

이 구조는 `vaswani2017attention`의 가장 순수한 해석에 가깝다.

- 입력을 token으로 만든다.
- 모든 token이 서로 attention한다.
- FFN을 거치며 표현을 업데이트한다.
- 마지막 pooled representation으로 감정 클래스를 예측한다.

### 현재 코드 기준 구조

- 입력: log-Mel spectrogram
- patch 분할: `patch_size`, `patch_stride`
- 선형 임베딩: patch를 `embed_dim` 차원으로 투영
- encoder 반복: `num_layers`
- attention heads: `num_heads`
- FFN 차원: `ffn_dim`
- 출력 pooling: `attention`, `mean`, `cls`

### 로직 흐름

1. spectrogram을 patch로 자른다.
2. 각 patch를 token embedding으로 바꾼다.
3. 모든 token이 서로 attention한다.
4. 여러 layer를 통과하며 전역 문맥을 누적한다.
5. 마지막 token 집합을 pooling해 하나의 utterance embedding으로 만든다.
6. classifier가 8개 감정 클래스를 예측한다.

### 장점

- transformer 자체의 기준선으로 해석이 가장 쉽다.
- 전역 문맥을 가장 직접적으로 본다.
- "CNN 없이도 되는가"를 보는 비교 기준이 된다.

### 약점

- local time-frequency pattern을 초기에 안정적으로 추출해 주는 구조가 없다.
- sequence가 길어질수록 global attention 비용이 커진다.
- 작은 SER 데이터에서는 학습 분산이 커지기 쉽다.

### SER 관점 해석

SER에서는 local cue가 중요한데, pure transformer는 그 cue를 직접 학습해야 한다. 데이터 규모가 작고 사전학습이 없는 상황에서는 이 점이 약점이 되기 쉽다. 그래서 이 모델은 "가장 개념적으로 순수한 기준선"으로는 좋지만, 실제 최종 성능 후보로는 다소 불리하다.

## 2. CNN Conformer

### 핵심 아이디어

`cnn_conformer`는 앞단의 CNN stem으로 local time-frequency pattern을 먼저 추출하고, 그다음 time-major sequence를 만들어 attention으로 전역 시간 문맥을 본다. 여기서 핵심은 attention만 쓰지 않고 convolution 성분을 block 안에 함께 넣는 hybrid 발상이다.

현재 저장소 구현은 "CNN으로 local cue를 먼저 안정화하고, 그 뒤 attention으로 긴 시간 문맥을 본다"는 구조적 장점을 노린다.

### 현재 코드 기준 구조

- CNN stem: `stem_channels`
- projection: CNN feature를 `embed_dim`으로 투영
- conformer-style block 반복: `num_layers`
- attention heads: `num_heads`
- convolution kernel: `conv_kernel_size`
- FFN 차원: `ffn_dim`
- 출력 pooling: `attention`, `mean`

### 로직 흐름

1. spectrogram에 CNN stem을 적용한다.
2. 주파수 축을 줄이고 시간 축 중심 sequence로 바꾼다.
3. 각 time step 표현을 transformer/conformer block에 넣는다.
4. self-attention이 긴 시간 의존성을 본다.
5. convolution branch가 local temporal continuity를 보강한다.
6. pooled representation으로 감정을 분류한다.

### 왜 pure transformer보다 현실적인가

- CNN stem이 먼저 local edge, burst, formant 변화 같은 cue를 압축한다.
- attention에 들어가기 전 sequence가 더 정제되어 있다.
- local modeling과 global modeling이 분업된다.

### 장점

- 작은 데이터셋에서 pure transformer보다 더 안정적일 가능성이 크다.
- CNN의 local bias와 transformer의 long-range modeling을 같이 쓴다.
- 하드웨어 부담도 pure transformer보다 다루기 쉽다.

### 약점

- 여전히 global attention이므로 sequence가 길면 비용이 커진다.
- 구조가 pure transformer보다 복잡해 해석성이 약간 떨어진다.
- 구현에 따라 convolution branch와 attention branch의 균형이 민감할 수 있다.

### SER 관점 해석

speech SER에서는 local acoustic cue와 시간적 문맥이 모두 중요하다. `cnn_conformer`는 그 둘을 가장 정석적으로 절충하는 구조다. 그래서 현재 코드베이스에서도 안정적인 hybrid 기준선으로 보는 것이 맞다.

## 3. Hierarchical Window Transformer

### 핵심 아이디어

`hierarchical_window_transformer`는 CNN stem으로 local pattern을 먼저 뽑은 뒤, 전체 sequence에 한 번에 attention하지 않고 작은 window 안에서만 attention한다. 이후 downsampling으로 sequence 길이를 줄이고, 다음 stage에서 다시 window attention을 수행한다.

이 아이디어는 `wang2024swin`과 `chen2023dwformer`의 핵심 문제의식과 맞닿아 있다.

- speech SER에서 local region은 중요하다.
- 모든 frame/token을 한 번에 global attention으로 묶으면 비용이 크다.
- 그래서 작은 window에서 출발해 점진적으로 더 넓은 문맥을 보도록 만든다.

### 현재 코드 기준 구조

- CNN stem: `stem_channels`
- stage 1 transformer:
  - `stage_dims[0]`
  - `stage_depths[0]`
  - `num_heads[0]`
  - `window_sizes[0]`
- sequence downsample
- stage 2 transformer:
  - `stage_dims[1]`
  - `stage_depths[1]`
  - `num_heads[1]`
  - `window_sizes[1]`
- shifted window:
  - 홀수 block에서 `shift_size = window_size // 2`
- output pooling:
  - `attention` 또는 `mean`

### shifted window가 왜 필요한가

window attention만 쓰면, 서로 다른 window에 들어간 token끼리는 직접 정보를 섞기 어렵다. `wang2024swin` 계열의 핵심은 다음 block에서 window를 절반만큼 옮겨서, 이전에는 다른 window에 있던 token들이 새 block에서는 같은 window 안에 들어오게 만드는 것이다.

즉,

- block A: 고정된 window 안에서 local attention
- block B: window를 반 칸 옮겨 다시 local attention

이 과정을 반복하면, global attention을 쓰지 않고도 정보가 점차 더 넓게 섞인다.

### 현재 구현의 실제 로직 흐름

1. 입력 spectrogram을 CNN stem에 통과시킨다.
2. 주파수 축을 average pooling으로 줄여 time-major sequence로 바꾼다.
3. `input_proj`로 stage 1 차원으로 투영한다.
4. stage 1 block에서 sequence를 `window_size` 단위로 자른다.
5. 각 window 안에서만 self-attention을 수행한다.
6. 다음 block에서는 window를 절반 이동한 뒤 다시 attention한다.
7. stage 1이 끝나면 `SequenceDownsample`이 stride 2로 sequence 길이를 줄인다.
8. stage 2에서 더 높은 차원 표현으로 다시 window attention을 수행한다.
9. 마지막 sequence를 pooling해 utterance embedding을 만든다.
10. classifier가 감정을 예측한다.

### 왜 이 구조가 SER에 맞는가

- local cue를 유지한다.
- CNN stem이 time-frequency local pattern을 먼저 요약한다.
- window attention이 짧은 감정 단서를 밀도 있게 본다.
- downsampling 뒤에는 더 긴 receptive field를 적은 비용으로 확보한다.

즉, "처음부터 전역을 다 보는 구조"가 아니라 "가까운 구간부터 정교하게 보고, stage를 거치며 문맥 범위를 넓히는 구조"다.

### 왜 하드웨어 친화적인가

- global attention은 대략 sequence length의 제곱 비용으로 커진다.
- window attention은 각 window 내부에서만 attention하므로 비용이 훨씬 작다.
- downsampling 이후 stage 2에서는 sequence 길이가 더 짧아진다.

그래서 `RTX 2060 6GB` 같은 제한된 환경에서도 pure transformer보다 훨씬 현실적이다.

### window size를 왜 탐색하는가

window size는 "한 번의 local attention이 보는 범위"를 정한다.

- 너무 작으면:
  - local detail은 잘 보지만 문맥 연결이 느리다.
- 너무 크면:
  - local window의 장점이 줄고 비용이 다시 커진다.

`wang2024swin`과 `chen2023dwformer`는 모두 speech SER에서 local window 설계 자체가 중요하다는 점을 보여 준다. 다만 현재 저장소의 구현은 그 논문들을 그대로 재현한 것이 아니라, shifted window와 hierarchical downsampling이라는 핵심 원리를 가져와 단순화한 2-stage 구조다. 따라서 현재 실험에서의 `window_size` 후보 범위는 "논문이 강조한 local window 원리"와 "현재 3초 RAVDESS, 16kHz, CNN stem 이후 sequence 길이"를 함께 고려한 공학적 탐색 범위로 보는 것이 맞다.

## 세 모델을 어떻게 해석해야 하나

### `pure_transformer`

- 가장 순수한 전역 attention 기준선
- 해석은 쉽지만 SER에서는 불리할 수 있음

### `cnn_conformer`

- local CNN + global attention의 안정적 절충
- 성능과 안정성 균형이 좋음

### `hierarchical_window_transformer`

- local window에서 시작해 계층적으로 문맥 범위를 넓힘
- 제한된 GPU 환경에서 가장 현실적인 transformer 계열 후보

## 현재 프로젝트에서의 역할 정리

- 메인 비교축:
  - `cnn_conformer` vs `hierarchical_window_transformer`
- 기준선:
  - `pure_transformer`

즉, 현재 실험의 핵심 질문은 "speech SER에서 local cue를 더 잘 살리면서도 transformer의 문맥 modeling을 유지하려면, global attention hybrid가 나은가, hierarchical window가 나은가"로 정리할 수 있다.
