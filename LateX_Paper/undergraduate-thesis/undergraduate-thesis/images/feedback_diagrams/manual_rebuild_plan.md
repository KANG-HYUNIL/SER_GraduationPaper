# 피드백 다이어그램 직접 재제작 계획서

작성 목적: 기존 `images/feedback_diagrams` 내부의 자동 생성 다이어그램 품질이 낮으므로, 사용자가 직접 draw.io에서 재제작할 때 필요한 자료 조사 방향, 배치 위치, 도형 구성, 검색 키워드를 정리한다. 이 문서는 새 다이어그램을 만들지 않고, 재제작 설계만 제공한다.

주의 원칙:

- 외부 논문 figure는 그대로 복사하지 말고, 논문 구조를 참고하여 직접 재도식화한다.
- 단순한 큰 사각형 내부에 텍스트만 넣는 방식은 피한다.
- 각 그림은 “입력 형태”, “변환 연산”, “중간 표현”, “출력”이 눈으로 보이도록 만든다.
- 논문 본문에는 중국어 캡션을 사용한다. draw.io 내부 텍스트도 가능하면 중국어 중심으로 둔다.
- 각 다이어그램은 독립 폴더에 원본 `.drawio`, exported `.pdf`, 필요한 캡처/참고 이미지, `sources.md`를 함께 둔다.

현재 기존 자동 생성물:

- `images/feedback_diagrams/chapter1/ser_evolution/ser_evolution_timeline.*`
- `images/feedback_diagrams/chapter1/acoustic_features/acoustic_feature_taxonomy.*`
- `images/feedback_diagrams/chapter2/fourier/ft_time_to_frequency.*`
- `images/feedback_diagrams/chapter2/stft/ft_vs_stft.*`
- `images/feedback_diagrams/chapter2/spectrogram_forms/spectrogram_forms.*`
- `images/feedback_diagrams/chapter2/logmel/mel_vs_logmel.*`
- `images/feedback_diagrams/chapter3/mlp/mlp_neural_network.*`
- `images/feedback_diagrams/chapter3/cnn/cnn_local_feature_extraction.*`
- `images/feedback_diagrams/chapter3/rnn/rnn_sequence_context.*`
- `images/feedback_diagrams/chapter3/attention/attention_qkv_weighting.*`
- `images/feedback_diagrams/chapter3/transformer/transformer_full_architecture.*`

## 0. Chapter 1 기존 연구流程图 확대

대상 그림:

- `images/1_1_BaseLine.drawio.pdf`
- 현재 삽입 위치: `chapters/1_chapter1.tex`, `研究内容` 내부, `\includegraphics[page=1,width=\textwidth]{images/1_1_BaseLine.drawio.pdf}`

목적:

- 그림 원본 draw.io가 없으므로, 기존 PDF가 깨지지 않는 범위에서 LaTeX 표시 크기만 키운다.
- 원본이 PDF라면 보통 벡터 기반이므로 `width` 확대는 이미지 자체를 흐리게 만들 가능성이 낮다.

권장 수정 방식:

```tex
\begin{figure}[htbp]
    \centering
    \makebox[\textwidth][c]{%
        \includegraphics[page=1,width=1.08\textwidth,trim=3mm 3mm 3mm 3mm,clip]{images/1_1_BaseLine.drawio.pdf}
    }
    \caption{研究流程图：涵盖数据预处理、卷积神经网络模型与Transformer并行对比及结果分析}
    \label{fig:chapter1_research_flow}
\end{figure}
```

검토 포인트:

- `1.08\textwidth`가 너무 크면 `1.04\textwidth`로 줄인다.
- PDF 내부 여백이 크면 `trim`을 조정한다.
- 본문 밖으로 튀어나오면 `\makebox[\textwidth][c]{...}`를 유지하되 `width`만 낮춘다.

## 1. Chapter 1, 1.3.1 语音情绪识别发展历程图

대상 위치:

- `chapters/1_chapter1.tex`
- `\section{研究现状及发展趋势}`
- `\subsection{语音情绪识别研究的发展历程}`
- 현재 자동 생성 그림: `images/feedback_diagrams/chapter1/ser_evolution/ser_evolution_timeline.pdf`

목적:

- “传统特征工程 / 机器学习 / CNN / RNN / Attention / Transformer / Conformer 또는 SSL”로 이어지는 발전 흐름을 한눈에 보여준다.
- 1.3.1 본문이 이미 네 단계 발전 흐름을 설명하므로, 이 그림은 문단 설명을 시각적으로 압축하는 역할이다.

권장 구조:

- 가로 타임라인 형태.
- 각 시대 노드는 단순 박스가 아니라 “작은 내부 구조”를 가진 미니 파이프라인으로 구성한다.
- 각 노드 하단에는 대표 논문/개념을 작은 글씨로 둔다.

권장 draw.io 구성:

1. 좌측: `语音波形`
   - 작은 waveform 선.
   - 아래 텍스트: `原始语音信号`
2. 1단계: `人工声学特征 + 传统分类器`
   - waveform -> MFCC/eGeMAPS feature vector 막대 -> SVM/HMM classifier 작은 기어 또는 결정 경계.
   - 하단: `MFCC / pitch / energy / eGeMAPS`
3. 2단계: `谱图输入 + CNN局部建模`
   - 작은 log-Mel heatmap -> 여러 겹 feature map -> convolution kernel.
   - 하단: `Log-Mel / Spectrogram -> CNN`
4. 3단계: `RNN / BiLSTM时序建模`
   - 프레임 시퀀스 `x1, x2, x3...` -> 반복 셀 여러 개 -> hidden states.
   - 하단: `temporal context`
5. 4단계: `Attention / Transformer`
   - Q/K/V 작은 행렬, attention heatmap, encoder stack.
   - 하단: `self-attention / global context`
6. 선택 확장: `Conformer / SSL`
   - attention branch + convolution branch가 합쳐지는 작은 구조.
   - 하단: `local + global / pretrained representation`

참고 논문 및 시각 참고:

- `zhang2020transfer`: 전통적 feature/subspace learning 흐름 참고. DOI `10.1109/TASLP.2019.2955252`.
- `peng2021efficient`: Multi-scale CNN + attention 구조 참고. DOI `10.1109/ICASSP43922.2021.9413554`.
- `mirsamadi2017automatic`: RNN hidden states + local attention pooling 참고. DOI `10.1109/ICASSP.2017.7952552`.
- `liu2023dualrobustness`: Dual-Transformer-BiLSTM 계열 hybrid 흐름 참고. DOI `10.1109/TASLP.2023.3282092`.
- `vaswani2017attention`: Transformer encoder-decoder figure 참고.
- `gulati2020conformer`: Conformer의 convolution + self-attention 결합 개념 참고.

검색 키워드:

- `speech emotion recognition traditional features SVM HMM pipeline figure`
- `Transfer Sparse Discriminant Subspace Learning Cross-Corpus Speech Emotion Recognition Figure 1`
- `Efficient Speech Emotion Recognition Multi-Scale CNN Attention Figure 1`
- `Automatic Speech Emotion Recognition RNN Local Attention Figure 1`
- `Dual-TBNet speech emotion recognition framework figure`
- `Attention Is All You Need Figure 1 Transformer architecture`
- `Conformer convolution augmented transformer architecture figure`

권장 파일 구조:

- `images/feedback_diagrams_manual/chapter1/ser_evolution/ser_evolution_rebuild.drawio`
- `images/feedback_diagrams_manual/chapter1/ser_evolution/ser_evolution_rebuild.pdf`
- `images/feedback_diagrams_manual/chapter1/ser_evolution/sources.md`

## 2. Chapter 1, 1.3.3 声学特征与信号处理方法分类图

대상 위치:

- `chapters/1_chapter1.tex`
- `\subsection{声学特征与信号处理方法的研究现状}`
- 현재 자동 생성 그림: `images/feedback_diagrams/chapter1/acoustic_features/acoustic_feature_taxonomy.pdf`

목적:

- “语音信号에서 어떤 정보가 특징으로 바뀌는지”를 보여준다.
- MEL, MFCC, pitch, energy, prosody, spectrogram, log-Mel spectrogram이 서로 다른 층위의 특징이라는 점을 비교한다.

권장 구조:

- 좌측에 `语音波形`.
- 중앙에서 3갈래로 분기:
  - `时域/韵律特征`: energy, zero-crossing, duration, speaking rate.
  - `频域/谱特征`: pitch, formant, spectral centroid.
  - `时频表示`: spectrogram, mel spectrogram, log-Mel spectrogram, MFCC.
- 우측에 `机器学习分类器`와 `深度学习模型` 두 갈래를 둔다.

도형 구성:

- waveform은 직접 선으로 그린다.
- `能量`은 막대그래프 아이콘.
- `音高`는 상승/하강 pitch curve.
- `频谱`는 세로 peak가 있는 frequency plot.
- `Mel滤波器组`은 삼각 필터 여러 개.
- `Log-Mel谱图`은 작은 heatmap.
- MFCC는 feature vector 막대 + DCT 화살표.

본문과의 연결:

- 1.3.3 본문은 “人为设计声学特征”에서 “时频表示”로 이동하는 흐름을 설명한다.
- 그림은 해당 문단 뒤, `随着深度学习的发展...` 문단 다음에 배치하는 것이 자연스럽다.

검색 키워드:

- `speech acoustic features pitch energy formant MFCC diagram`
- `speech feature extraction pipeline MFCC mel spectrogram pitch energy`
- `eGeMAPS acoustic features speech emotion recognition diagram`
- `mel filter bank triangular filters diagram`
- `log mel spectrogram feature extraction pipeline`

참고 자료:

- `ma2023review`: affective computing feature/representation 논의 근거.
- Hugging Face Audio Course: waveform, spectrogram, mel spectrogram 설명과 시각 예시.
- librosa `melspectrogram`, `power_to_db` 문서: mel projection과 log/dB 변환 설명.

## 3. Chapter 2, Fourier Transform 时域到频域图

대상 위치:

- `chapters/2_chapter2.tex`
- `\subsection{时频变换与频谱分析}`
- `(3) 傅里叶变换：从时域到频域的基础` 문단 근처.
- 현재 자동 생성 그림: `images/feedback_diagrams/chapter2/fourier/ft_time_to_frequency.pdf`

목적:

- 시간 영역 신호가 Fourier transform을 거쳐 주파수 성분 분포로 바뀌는 모습을 보여준다.

대응 피드백:

- 요구사항 5번에 대응한다. 즉 `(3) 傅里叶变换：从时域到频域的基础` 부분에서 일반 raw 신호가 Fourier transform 이후 주파수 영역의 peak/spectrum으로 바뀌는 모습을 한눈에 보여 달라는 요청이다.
- 단순 텍스트 박스가 아니라, 좌측에는 실제 진폭 파형 형태, 우측에는 frequency-domain peak 형태를 배치해야 한다.
- 이 그림은 “傅里叶变换은 전체 신호를 주파수 성분으로 분해하지만 시간 위치 정보는 직접 보여주지 않는다”는 본문 설명을 시각적으로 보조한다.

권장 구조:

- 좌측: 시간축 waveform.
  - x축 `时间`, y축 `幅度`.
  - 두세 개의 sine wave가 합쳐진 모양.
- 중앙: 큰 화살표 `傅里叶变换`.
  - 화살표 위에 `整体频率分解`.
- 우측: frequency spectrum.
  - x축 `频率`, y축 `幅度`.
  - 특정 주파수 위치에 peak 2-3개.
- 하단: `时域保留时间变化`, `频域显示成分强弱`, `不显示局部发生时刻`.

도형 구성:

- waveform은 draw.io freehand/curve 또는 polyline.
- spectrum은 얇은 vertical line peak 여러 개.
- 축은 검은 선과 작은 tick mark.
- 화살표는 직선 굵은 화살표 하나만 사용.

검색 키워드:

- `time domain to frequency domain Fourier transform diagram`
- `simple time domain vs frequency domain Fourier transform SVG`
- `audio signal Fourier transform spectrum example`
- `FFT waveform frequency spectrum diagram`

참고 자료:

- Wikimedia Commons `Simple time domain vs frequency domain.svg`는 좋은 시각 참고 자료다. 그대로 복사하지 말고, 유사 구조를 직접 재도식화한다.
- Chapter 2의 DFT/FFT 공식과 설명을 근거로 한다.

## 4. Chapter 2, FT vs STFT 对比图

대상 위치:

- `chapters/2_chapter2.tex`
- `(4) 短时傅里叶变换：时频分析的桥梁`
- 현재 자동 생성 그림: `images/feedback_diagrams/chapter2/stft/ft_vs_stft.pdf`

목적:

- FT는 전체 신호를 한 번에 주파수로 바꾸고, STFT는 짧은 창을 이동시키며 시간별 주파수 변화를 얻는다는 차이를 보여준다.

대응 피드백:

- 요구사항 6번에 대응한다. 즉 `短时傅里叶变换：时频分析的桥梁` 부분에서 FT와 STFT의 차이를 좌우 또는 상하 비교로 보여 달라는 요청이다.
- 핵심은 FT가 하나의 전체 spectrum을 만들고, STFT는 이동 창을 통해 여러 시간 구간의 spectrum을 만들며 최종적으로 time-frequency heatmap을 형성한다는 점이다.
- 그림 내부에는 sliding window, overlapping frames, per-frame FFT, spectrogram grid가 반드시 포함되어야 한다.

권장 구조:

- 상단 또는 좌측: `傅里叶变换`
  - 긴 waveform 전체에 하나의 bracket.
  - 화살표 -> 하나의 frequency spectrum.
  - 주석: `全局频率成分`
- 하단 또는 우측: `短时傅里叶变换`
  - waveform 위에 여러 개의 overlapping windows.
  - 각 window에서 작은 spectrum으로 이동.
  - 최종적으로 spectrogram heatmap.
  - 주석: `时间-频率二维表示`

도형 구성:

- overlapping window는 투명한 노란 직사각형 여러 개.
- STFT 결과는 heatmap grid로 표현한다.
- FT 결과는 peak spectrum으로 표현한다.
- FT와 STFT를 나란히 놓아 비교성이 강해야 한다.

검색 키워드:

- `short time Fourier transform sliding window diagram`
- `FT vs STFT spectrogram diagram`
- `STFT windowing time frequency representation figure`
- `spectrogram sliding window Fourier transform`

참고 자료:

- SciPy `spectrogram` 문서: STFT 기반으로 time-frequency representation을 만드는 함수 설명.
- Hugging Face Audio Course: waveform에서 spectrogram으로 이동하는 직관적 설명.

## 5. Chapter 2, 频谱图三种形式 비교图

대상 위치:

- `chapters/2_chapter2.tex`
- `频谱图：时频表示的三种形式`
- 현재 자동 생성 그림: `images/feedback_diagrams/chapter2/spectrogram_forms/spectrogram_forms.pdf`

목적:

- magnitude spectrogram, power spectrogram, log spectrogram의 차이를 시각적으로 비교한다.

대응 피드백:

- 요구사항 4번에 대응한다. 즉 `频谱图：时频表示的三种形式` 부분에서 power spectrogram, log spectrogram 등 본문이 다루는 spectrogram 종류의 차이를 눈으로 비교할 수 있게 하라는 요청이다.
- 동일한 STFT 결과에서 `|X|`, `|X|^2`, `log(|X|^2)` 또는 dB scaling이 어떻게 다른 표시 결과를 만드는지 한 그림 안에 보여야 한다.
- 단순히 세 개 이름을 나열하지 말고, 같은 time-frequency grid가 scale 변환에 따라 대비와 dynamic range가 달라진다는 시각적 차이를 표현해야 한다.

권장 구조:

- 좌측: 같은 STFT complex output `X(t, f)`.
- 우측에 세 갈래:
  - `幅度谱图 |X|`
  - `功率谱图 |X|^2`
  - `对数谱图 log(|X|^2 + ε)` 또는 `dB`.
- 세 결과는 같은 시간-주파수 heatmap 모양이지만 색상 대비가 다르게 표현되어야 한다.

도형 구성:

- 공통 입력은 작은 spectrogram grid + complex notation.
- magnitude는 부드러운 색상.
- power는 강한 peak와 넓은 dynamic range.
- log/dB는 contrast가 압축된 형태.
- 아래에 colorbar mini scale 3개를 둔다.

검색 키워드:

- `magnitude spectrogram power spectrogram log spectrogram comparison`
- `spectrogram magnitude power dB scale comparison`
- `scipy spectrogram mode magnitude psd power spectrum`
- `librosa power_to_db spectrogram comparison`

참고 자료:

- SciPy `signal.spectrogram` 문서: `mode='magnitude'`, `mode='psd'`, power spectrum 관련 설명.
- librosa `power_to_db` 문서: power spectrogram을 dB/log scale로 바꾸는 계산 설명.

## 6. Chapter 2, Mel vs Log-Mel 비교图

대상 위치:

- `chapters/2_chapter2.tex`
- `\subsection{感知特征提取}`
- `\textbf{(2) 对数梅尔谱图：听觉感知与时频表示的融合}`
- 현재 자동 생성 그림: `images/feedback_diagrams/chapter2/logmel/mel_vs_logmel.pdf`

목적:

- mel spectrogram은 mel filter bank로 주파수 축을 감각적으로 압축한 결과이고, log-Mel spectrogram은 mel power를 log/dB로 압축한 결과라는 차이를 보여준다.

대응 피드백:

- 요구사항 7번에 대응한다. 즉 `对数梅尔谱图：听觉感知与时频表示的融合` 부분에서 일반 mel spectrogram과 log-Mel spectrogram의 차이를 한눈에 볼 수 있게 하라는 요청이다.
- 그림은 `STFT功率谱 -> Mel滤波器组 -> Mel谱图 -> log/dB压缩 -> 对数梅尔谱图`의 계산 흐름을 보여야 한다.
- 특히 Mel 변환은 frequency axis를 사람 청각에 가까운 mel scale로 바꾸는 단계이고, log 변환은 power dynamic range를 압축하는 단계라는 차이를 분리해서 보여야 한다.

권장 구조:

- `STFT功率谱` -> `Mel滤波器组` -> `Mel谱图` -> `log / dB压缩` -> `对数梅尔谱图`.
- 하단에 “线性频率 -> Mel频率”, “功率动态范围 -> 对数动态范围”를 분리해서 표시한다.

도형 구성:

- mel filter bank는 삼각형 필터 여러 개.
- mel spectrogram과 log-Mel spectrogram은 같은 grid지만 log-Mel 쪽 색상 대비가 더 균형 있게 보이도록 한다.
- `log` 변환은 압축 스프링 또는 curve 아이콘으로 표현한다.

검색 키워드:

- `mel filter bank triangular filters diagram`
- `mel spectrogram vs log mel spectrogram comparison`
- `librosa melspectrogram power_to_db example`
- `Hugging Face audio course mel spectrogram log mel`

참고 자료:

- librosa `melspectrogram`: magnitude/power spectrum을 mel basis로 mapping.
- librosa `power_to_db`: `10 * log10(S / ref)` 방식의 dB 변환.
- Hugging Face Audio Course: mel spectrogram과 log-mel spectrogram의 교육용 설명.

## 7. Chapter 3, MLP 구조图

대상 위치:

- `chapters/3_chapter3.tex`
- `\subsection{全连接神经网络模型}`
- 현재 자동 생성 그림: `images/feedback_diagrams/chapter3/mlp/mlp_neural_network.pdf`

목적:

- 고정 길이 feature vector가 여러 fully connected layer를 지나 class logits로 바뀌는 구조를 보여준다.

권장 구조:

- 입력: `x1, x2, ..., xd` feature vector.
- hidden layer 1, hidden layer 2는 작은 원 노드로 구성.
- 모든 인접 layer 노드를 선으로 연결.
- 각 layer 사이에 `W`, `b`, `σ` 표시.
- 출력: 8개 emotion class logits 막대.

도형 구성:

- 원형 neuron 4-6개씩 3열.
- 선은 너무 많으면 희미한 회색으로.
- 일부 대표 연결만 진하게 표시해 “fully connected” 느낌을 준다.
- activation은 작은 `ReLU` 또는 `σ` 박스로 표시.

검색 키워드:

- `multi layer perceptron diagram neurons weights bias activation`
- `fully connected neural network architecture diagram`
- `MLP input hidden output layer diagram`

본문 연결:

- MLP 공식 `h = σ(Wx+b)` 바로 뒤나 앞에 배치하는 것이 적절하다.

## 8. Chapter 3, CNN局部特征提取图

대상 위치:

- `chapters/3_chapter3.tex`
- `\subsection{卷积神经网络模型}`
- 현재 자동 생성 그림: `images/feedback_diagrams/chapter3/cnn/cnn_local_feature_extraction.pdf`

목적:

- log-Mel spectrogram에서 작은 convolution kernel이 국소 time-frequency pattern을 훑고 feature map을 만드는 과정을 보여준다.

권장 구조:

- 입력: 작은 log-Mel heatmap.
- kernel: 3x3 또는 5x5 작은 격자, 입력 위에 투명하게 overlay.
- feature maps: 여러 장 겹쳐진 plane.
- pooling: 크기가 줄어든 feature map.
- classifier: 작은 output vector.

도형 구성:

- feature map은 실제 CNN figure처럼 얇은 직사각형을 여러 장 겹쳐 표현한다.
- kernel 이동은 점선 화살표로 표시.
- local receptive field를 강조하려면 입력 heatmap의 한 영역만 밝게 표시한다.

검색 키워드:

- `CNN convolution kernel feature map diagram`
- `convolutional neural network spectrogram feature extraction diagram`
- `speech emotion recognition CNN log mel spectrogram architecture`
- `Efficient Speech Emotion Recognition Multi-Scale CNN Attention Figure 1`

참고 자료:

- `peng2021efficient`: multi-scale CNN로 audio/text representation을 추출하는 구조.
- `zhang2023cnn` 또는 본문 내 CNN 관련 인용은 국소 feature extraction 설명 근거로 사용 가능.

## 9. Chapter 3, RNN / BiLSTM时序建模图

대상 위치:

- `chapters/3_chapter3.tex`
- `\subsection{循环神经网络模型及其变体}`
- 현재 자동 생성 그림: `images/feedback_diagrams/chapter3/rnn/rnn_sequence_context.pdf`

목적:

- 음성 frame sequence가 시간 순서대로 hidden state로 누적되고, BiLSTM은 양방향 문맥을 결합한다는 점을 보여준다.

권장 구조:

- 입력: `x1, x2, x3, x4, ...` frame sequence.
- RNN/LSTM cell을 시간축에 따라 반복 배치.
- hidden state `h1, h2, h3...`를 연결.
- BiLSTM이면 forward arrow와 backward arrow를 다른 색으로 표시.
- 마지막에 pooling 또는 attention으로 utterance representation 생성.

도형 구성:

- 반복 cell 내부에는 작은 gate 표시 `i, f, o`를 넣으면 LSTM 느낌이 난다.
- forward는 파란 화살표, backward는 주황 화살표.
- time axis를 하단에 얇게 표시한다.

검색 키워드:

- `RNN sequence hidden state diagram`
- `LSTM gates diagram input forget output`
- `BiLSTM speech emotion recognition architecture figure`
- `Automatic Speech Emotion Recognition RNN Local Attention Figure 1`
- `Dual-TBNet BiLSTM speech emotion recognition framework`

참고 자료:

- `mirsamadi2017automatic`: RNN hidden states와 local attention pooling.
- `liu2023dualrobustness`: CNN/Transformer/BiLSTM 계열 hybrid 흐름.

## 10. Chapter 3, Attention / QKV 加权图

대상 위치:

- `chapters/3_chapter3.tex`
- `\section{注意力机制}`
- 현재 자동 생성 그림: `images/feedback_diagrams/chapter3/attention/attention_qkv_weighting.pdf`
- 기존 수동 QKV 그림도 있음: `images/chapter3/chapter3_attention_qkv.drawio.pdf`

목적:

- Q, K, V가 어떻게 만들어지고, QK^T가 attention score가 되며, softmax 후 V를 가중합한다는 흐름을 시각화한다.

권장 구조:

- 입력 sequence `X`를 작은 token matrix로 표시.
- 세 갈래 선형 변환:
  - `Q = XW_Q`
  - `K = XW_K`
  - `V = XW_V`
- `QK^T / sqrt(d_k)`는 matrix multiplication 블록으로 표시.
- softmax는 heatmap matrix로 표시.
- heatmap matrix와 `V`를 곱해 output matrix.

도형 구성:

- Q/K/V는 색이 다른 작은 행렬.
- attention score는 정사각형 heatmap으로 표현한다.
- 행렬 곱 연산은 `×`, scaling은 `/√d_k`, softmax는 S-curve 또는 작은 함수 블록.
- 최종 output은 value 색과 attention heatmap 색이 섞인 matrix.

검색 키워드:

- `scaled dot product attention Q K V diagram`
- `query key value attention matrix multiplication diagram`
- `Attention Is All You Need scaled dot-product attention figure`
- `self attention QKV visualization`

참고 자료:

- `vaswani2017attention`: scaled dot-product attention 공식과 multi-head attention 구조.
- `mirsamadi2017automatic`: SER에서 attention이 중요한 time region을 weighting하는 사용 맥락.

## 11. Chapter 3, Transformer整体架构图

대상 위치:

- `chapters/3_chapter3.tex`
- `\section{Transformer结构}`
- 현재 자동 생성 그림: `images/feedback_diagrams/chapter3/transformer/transformer_full_architecture.pdf`

목적:

- 원본 Transformer encoder-decoder 구조와 SER에서 사용하는 encoder-centered classification path를 함께 이해하게 한다.

권장 구조:

- 좌측 큰 블록: `Encoder Stack`.
  - input embedding + positional encoding.
  - repeated encoder layer:
    - multi-head self-attention
    - add & norm
    - feed-forward
    - add & norm
- 우측 작은 블록: `Decoder Stack`은 원본 구조 참고용으로 연하게 표시.
  - masked self-attention
  - encoder-decoder attention
  - feed-forward.
- 하단 또는 우측 별도 branch:
  - `SER分类路径: Encoder output -> pooling/CLS -> linear -> emotion`.

도형 구성:

- 원본 Figure 1처럼 encoder와 decoder를 세로 stack 형태로 둔다.
- 논문 본문이 SER classification을 다루므로 decoder는 너무 강조하지 않는다.
- SER branch는 별도 색으로 처리하여 “本文使用的简化方向”임을 보인다.

검색 키워드:

- `Attention Is All You Need Figure 1 Transformer model architecture`
- `Transformer encoder decoder architecture diagram`
- `Transformer encoder classification head diagram`
- `speech emotion recognition transformer encoder classification architecture`

참고 자료:

- `vaswani2017attention`: 원본 encoder-decoder architecture.
- Chapter 3 본문: SER에서는 encoder output에 pooling/classification head를 연결한다고 설명함.
- `chen2023dwformer`, `wang2024swin`: SER에서 Transformer 변형이 입력 범위와 계층 구조를 조정한다는 근거.

## 12. 참고 자료 접근 링크

논문 및 문서:

- Zhang and Song, 2020, `zhang2020transfer`, DOI: `10.1109/TASLP.2019.2955252`
  - 검색: https://www.researchgate.net/publication/337453920_Transfer_Sparse_Discriminant_Subspace_Learning_for_Cross-Corpus_Speech_Emotion_Recognition
- Peng et al., 2021, `peng2021efficient`, arXiv:
  - https://arxiv.org/abs/2106.04133
- Mirsamadi et al., 2017, `mirsamadi2017automatic`, SigPort/PDF:
  - https://sigport.org/documents/automatic-speech-emotion-recognition-using-recurrent-neural-networks-local-attention
  - https://sigport.org/sites/default/files/docs/icassp2017_1.pdf
- Vaswani et al., 2017, `vaswani2017attention`, NeurIPS/PDF:
  - https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf
  - https://arxiv.org/abs/1706.03762
- Gulati et al., 2020, `gulati2020conformer`:
  - https://research.google/pubs/conformer-convolution-augmented-transformer-for-speech-recognition/
- SciPy spectrogram:
  - https://docs.scipy.org/doc/scipy-1.12.0/reference/generated/scipy.signal.spectrogram.html
- librosa melspectrogram:
  - https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
- librosa power_to_db:
  - https://librosa.org/doc/latest/generated/librosa.power_to_db.html
- Hugging Face Audio Course, audio data and mel spectrogram:
  - https://huggingface.co/learn/audio-course/chapter1/audio_data

## 13. 재제작 우선순위

1. Chapter 1 `ser_evolution`: 박사 피드백에서 “발전 동향 한눈에 보기” 요구가 직접적이므로 최우선.
2. Chapter 3 `Transformer`, `Attention QKV`: 박사 피드백의 핵심인 attention/Transformer 이해 보강과 직접 연결.
3. Chapter 3 `CNN`, `RNN`, `MLP`: 모델 구조 부분의 텍스트 의존도를 낮추는 데 필요.
4. Chapter 2 `FT`, `STFT`, `spectrogram forms`, `Mel vs Log-Mel`: 신호처리 이론 파트를 시각적으로 보강.
5. Chapter 1 `acoustic_feature_taxonomy`: 1.3.3의 feature 설명을 압축하는 보조 그림.
6. 기존 `1_1_BaseLine.drawio.pdf` 확대: 원본 재제작이 아니라 LaTeX 표시 조정으로 해결 가능.
