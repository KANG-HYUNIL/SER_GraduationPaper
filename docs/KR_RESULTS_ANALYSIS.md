# 결과 분석

## 2026_04_16 

### 2. 성능 병목 및 설계 결함 분석 (가변 길이 처리와 Transformer 구조의 충돌)

실제 실험 코드(`src/models/*.py` 및 `src/optuna_search.py`)와 `ref.bib`에 기재된 주요 논문들(Speech Swin-Transformer, DWFormer 등)의 이론을 비교·분석한 결과, "가변 길이 처리(Variable-length + Padding)"를 도입한 것이 오히려 성능 하락의 가장 큰 원인으로 작용했음을 확인했습니다. 치명적인 두 가지 병목 원인은 다음과 같습니다.

1. **가변 패딩(Zero-Padding)과 CNN Stem의 BatchNorm 충돌**
   - 현재 모든 Transformer 계열(`cnn_conformer`, `hierarchical_window_transformer`)의 앞단에 2D CNN 기반의 `ConvStemBlock`이 사용되고 있습니다. 
   - `resize_enabled = False`로 인해 배치 내의 짧은 오디오들은 긴 오디오 길이에 맞춰 0으로 패딩(Zero-Padding)됩니다. 그런데 `BatchNorm2d` 연산은 패딩된 0값들까지 모두 포함하여 배치 전체의 평균과 분산을 계산합니다.
   - 배치마다 패딩되는 양이 다르기 때문에, BatchNorm의 통계량이 극심하게 왜곡(Skew)되며 특징 추출이 매우 불안정해집니다. 반면, 과거의 **CNN Baseline은 Fixed Resize(96x512)** 를 사용했기 때문에 패딩 없이 순수한 데이터만으로 학습되어 안정적인 성능(0.62)을 낼 수 있었습니다.

2. **Shifted Window 패딩 순환(Roll) 버그 (Temporal Discontinuity)**
   - `hierarchical_window_transformer.py`의 `WindowTransformerBlock`을 보면, Shifted Window를 구현하기 위해 `torch.roll(..., shifts=-self.shift_size, dims=1)` 함수를 사용해 시간 축을 이동시킵니다.
   - 문제는 "가변 길이" 상황에서 패딩된 시퀀스를 그대로 Roll 해버리면, **마지막에 위치하던 '패딩 영역'과 그 뒤로 밀려난 시퀀스의 '맨 처음 유효 영역(Start of speech)'이 같은 윈도우 안에 묶여버리는 대참사**가 발생한다는 것입니다.
   - 비록 `key_padding_mask`가 같이 롤링되어 0 패딩을 무시하도록 설정되어 있다고 해도, 원래 오디오의 앞부분(시간 = 0)과 끝부분(시간 = 끝)이 같은 윈도우에 담겨 서로 Attention 되기 때문에 시간적 연속성이 완전히 붕괴됩니다. Swin-Transformer 논문에서는 이를 방지하기 위해 경계선을 넘지 못하도록 하는 복잡한 'Shift Attention Mask' 기법을 필수적으로 적용해야 한다고 제안하지만, 현재 코드에는 이 마스킹 로직이 누락되어 있습니다. 이 때문에 계층적 윈도우 모델이 가장 치명적인 성능 저하(0.487)를 겪은 것입니다.

### 3. 채택한 개선 방향

현재 프로젝트는 **가변 길이 입력을 유지하면서 Zero-Padding을 padding-safe 하게 처리하는 방향**을 채택합니다.

- `ConvStemBlock`의 `BatchNorm2d`는 패딩 비율에 따라 통계량이 흔들리므로 제거하고, 패딩 통계에 독립적인 정규화와 단계별 마스킹으로 교체합니다.
- `cnn_conformer` 내부 Conformer convolution branch의 `BatchNorm1d` 역시 동일한 문제를 일으키므로 함께 제거하고, temporal mask를 끝까지 전달합니다.
- `hierarchical_window_transformer.py`의 shifted window는 `torch.roll` 기반 순환 이동을 제거하고, `F.pad` 기반의 물리적 shift와 padding mask 확장으로 경계 혼합을 방지합니다.
- Optuna는 구조적으로 불가능하거나 메모리상 성립하지 않는 trial을 학습 시작 전에 preflight 단계에서 바로 prune 하도록 보강합니다.
