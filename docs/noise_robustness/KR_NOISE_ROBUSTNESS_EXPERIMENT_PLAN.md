# 잡음 환경 강건성 실험 설계

## 1. 문서 범위

- 문서 대상 모델명: `cnn_conformer`
- 문서 목적: clean 조건에서 선택된 최고 성능 모델을 고정한 뒤, 잡음 종류와 SNR 조건에 따른 성능 저하를 측정한다.
- 현재 문서 상태: `completed`
- 작성 단계: 실험 설계, 코드 구축, 본 실험 결과 기록 완료

이 실험은 새로운 backbone을 찾기 위한 실험이 아니다. 현재 논문 흐름에서 `CNN baseline`, `pure Transformer`, `CNN-Conformer`를 비교한 뒤, 가장 높은 성능을 보인 `CNN-Conformer`의 실제 잡음 환경 민감도를 확인하기 위한 후속 평가로 둔다. 첫 단계에서는 학습 데이터를 바꾸지 않고 noisy evaluation만 수행한다.

## 2. 모델 요약

### 2.1 핵심 요약

clean 조건 최고 성능으로 확인된 `CNN-Conformer` checkpoint를 고정하고, 평가 fold의 파형에만 additive noise를 주입한다. 잡음은 log-Mel 변환 이후가 아니라 waveform 단계에서 섞기 때문에, 실제 녹음 환경에서 신호가 오염된 뒤 특징 추출이 일어나는 흐름과 더 가깝다.

### 2.2 선택된 winner 구성

| 항목 | 값 |
|---|---|
| 기준 모델 | `cnn_conformer` |
| 기준 실험 | `../../outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004` |
| checkpoint | `../../outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/artifacts/weights/best_model_fold1.pt` |
| clean Accuracy | 0.70333 |
| clean Macro-F1 | 0.70042 |
| backbone | `nostem_patch` |
| 입력 특징 | 80-bin log-Mel, `n_fft=1024`, `hop_length=160`, resize 미사용 |
| chunk 평가 | `chunk_frames=48`, `eval_hop_frames=12`, `confidence_weighted_logit` |
| 주요 구조 | `time_patch=4`, `embed_dim=192`, `num_layers=4`, `num_heads=8`, `ffn_dim=768`, `conv_kernel=31` |
| regularization | `mixup alpha=0.4`, dropout 범위는 winner resolved config 사용 |

주의: `../../outputs/2026-04-21/18-47-36_cnn_conformer/optuna_trials/trial_0003`도 Macro-F1 0.70563으로 기록되어 있으나, 해당 trial 폴더에는 trial별 checkpoint가 남아 있지 않고 root weights가 다른 trial 형태로 덮인 상태이다. 따라서 noisy evaluation의 재현 가능한 기준점은 trial별 checkpoint가 보존된 2026-04-22 trial 0004로 고정한다.

## 3. 실험 라운드 기록

### 3.1 공통 고정 조건

| 분류 | 항목 | 값 | 비고 |
|---|---|---|---|
| 데이터 | RAVDESS speech | 1440 files | 기존 actor-group split 유지 |
| 평가 fold | fold 1 | 기존 Optuna partial CV와 동일 | 사용자가 진행한 최고 성능 기록과 맞춤 |
| 입력 처리 | waveform noise 후 log-Mel | `AudioPipeline` 재사용 | 특징 추출 방식 고정 |
| 모델 | CNN-Conformer winner | 구조 및 checkpoint 고정 | 잡음 평가 중 재학습 없음 |
| 지표 | Accuracy, Macro-F1, UAR, ECE | clean 대비 delta 함께 저장 | 논문 표에는 핵심 지표만 사용 |
| SNR | clean, 20, 10, 5, 0, -5 dB | 저강도부터 강한 잡음까지 확인 | -5 dB는 extreme condition으로 해석 |
| 잡음 종류 | white, pink, babble, cafe | synthetic 생성 | 외부 noise corpus 없이 즉시 재현 가능 |

### 3.2 실험 조건 선택 근거

| 근거 자료 | 확인한 방식 | 이번 실험 반영 |
|---|---|---|
| Zhou et al., Interspeech 2020 | noisy SER에서 clean/noisy 조건과 enhancement preprocessing을 비교 | 먼저 clean winner를 고정하고 noisy test-only 평가 수행 |
| Ranjan et al., Interspeech 2024 | noise robust SER에서 noise augmentation의 효과를 평가 | 다음 단계 후보로 train-time noise augmentation을 남김 |
| Nam and Park, Applied Sciences 2024 | noisy SER에서 SNR grid와 noise source를 분리해 평가 | SNR 단계와 noise type을 명시적으로 분리 |
| Huang et al., Archives of Acoustics | clean train 후 white noise SNR test를 사용 | white noise를 가장 기본적인 기준 잡음으로 포함 |
| NIST MWA-SER | RAVDESS 기반 augmentation 연구 | RAVDESS에서도 잡음/증강 평가를 수행할 수 있다는 근거로 사용 |
| `../../ref.bib`의 `leem2024selective` 등 | noisy speech SER에서 feature enhancement와 robustness 논의 | 본 실험 후 성능 하락이 크면 enhancement/augmentation을 후속 후보로 둠 |

### 3.3 잡음 조건 정의

| 잡음명 | 구현 방식 | 해석 |
|---|---|---|
| `white` | 모든 주파수 대역에 균등한 synthetic noise | 기본 AWGN 민감도 확인 |
| `pink` | 저주파 성분이 상대적으로 큰 colored noise | 실제 환경 배경음에 가까운 저주파 오염 확인 |
| `babble` | 원 파형을 여러 시간 shift로 섞고 약한 colored noise 추가 | 여러 사람 말소리와 유사한 speech-like 간섭을 단순 근사 |
| `cafe` | pink/brown noise에 짧은 transient를 섞음 | 카페/거리 배경처럼 연속음과 순간음을 함께 근사 |

현재 구현은 외부 noise corpus가 아니라 synthetic noise이다. 장점은 즉시 재현 가능하고 repo 외부 파일 의존성이 없다는 점이다. 단점은 MUSAN, ESC-50, NoiseX92 같은 실제 잡음 corpus보다 현실성이 낮을 수 있다는 점이다. 논문에는 synthetic noisy condition으로 제한하여 서술한다.

### 3.4 실험 명령어

Smoke test:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_noise_robustness model=cnn_conformer noise.eval.enabled=true noise.eval.resolved_config_path=outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/resolved_config.yaml noise.eval.checkpoint_path=outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/artifacts/weights/best_model_fold1.pt noise.eval.noise_types=[white,babble,cafe] noise.eval.snr_db=[clean,20] noise.eval.save_condition_artifacts=false noise.eval.output_dir=noise_eval_smoke experiment.name=noise_smoke
```

본 실험:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_noise_robustness model=cnn_conformer noise.eval.enabled=true noise.eval.resolved_config_path=outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/resolved_config.yaml noise.eval.checkpoint_path=outputs/2026-04-22/02-33-33_cnn_conformer/optuna_trials/trial_0004/artifacts/weights/best_model_fold1.pt noise.eval.noise_types=[white,pink,babble,cafe] noise.eval.snr_db=[clean,20,10,5,0,-5] noise.eval.save_condition_artifacts=true noise.eval.output_dir=noise_eval_winner experiment.name=noise_eval_winner
```

### 3.5 구현 경로

| 역할 | 경로 | 내용 |
|---|---|---|
| Hydra config | `../../src/configs/noise/default.yaml` | noise eval grid와 출력 경로 관리 |
| noise generator | `../../src/data/noise.py` | SNR 기반 waveform additive noise |
| noisy dataset | `../../src/data/noisy_dataset.py` | 평가 시 idx별 deterministic noise 주입 |
| evaluation entrypoint | `../../src/evaluate_noise_robustness.py` | winner config/checkpoint load, 조건별 평가, CSV/JSON/artifact 저장 |

기존 학습 명령어는 `noise.eval.enabled=false`, `noise.train.enabled=false`가 기본값이므로 그대로 재현 가능하다. 잡음 평가는 별도 script로 분리되어 training loop를 변경하지 않는다.

## 4. 주요 결과 요약

### 4.1 Smoke test 결과

| 조건 | SNR | Accuracy | Macro-F1 | UAR | clean 대비 Accuracy |
|---|---:|---:|---:|---:|---:|
| clean | clean | 0.70333 | 0.70042 | 0.70000 | 0.00000 |
| white | 20 | 0.61000 | 0.56390 | 0.57810 | -0.09333 |
| babble | 20 | 0.59667 | 0.57410 | 0.57500 | -0.10667 |
| cafe | 20 | 0.63667 | 0.63600 | 0.62500 | -0.06667 |

Smoke test는 코드 동작 검증용이므로 최종 분석에는 본 실험 grid 전체 결과를 사용한다.

### 4.2 본 실험 결과 표

본 실험 결과는 `../../outputs/2026-04-23/15-33-32_noise_eval_winner/noise_eval_winner/noise_summary.csv`와 `../../outputs/2026-04-23/15-33-32_noise_eval_winner/noise_eval_winner/noise_summary.json`을 기준으로 정리하였다. 조건별 잡음은 파형 단계에서 주입되었으며, 모델 구조와 checkpoint는 clean winner로 고정하였다.

| 잡음 | SNR | Accuracy | Macro-F1 | UAR | clean 대비 Accuracy | clean 대비 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| clean | clean | 0.70333 | 0.70042 | 0.70000 | 0.00000 | 0.00000 |
| white | 20 | 0.61000 | 0.56389 | 0.57812 | -0.09333 | -0.13653 |
| white | 10 | 0.49000 | 0.44717 | 0.46562 | -0.21333 | -0.25324 |
| white | 5 | 0.40667 | 0.32933 | 0.38125 | -0.29667 | -0.37108 |
| white | 0 | 0.34000 | 0.24169 | 0.31875 | -0.36333 | -0.45873 |
| white | -5 | 0.29333 | 0.17877 | 0.27500 | -0.41000 | -0.52165 |
| pink | 20 | 0.60000 | 0.58425 | 0.58125 | -0.10333 | -0.11616 |
| pink | 10 | 0.36667 | 0.32255 | 0.34688 | -0.33667 | -0.37787 |
| pink | 5 | 0.27000 | 0.19737 | 0.25312 | -0.43333 | -0.50305 |
| pink | 0 | 0.21000 | 0.13333 | 0.19688 | -0.49333 | -0.56709 |
| pink | -5 | 0.15000 | 0.06243 | 0.14062 | -0.55333 | -0.63798 |
| babble | 20 | 0.59667 | 0.57407 | 0.57500 | -0.10667 | -0.12635 |
| babble | 10 | 0.54667 | 0.52658 | 0.52500 | -0.15667 | -0.17383 |
| babble | 5 | 0.51667 | 0.49893 | 0.49687 | -0.18667 | -0.20148 |
| babble | 0 | 0.45333 | 0.43736 | 0.43438 | -0.25000 | -0.26305 |
| babble | -5 | 0.44333 | 0.42124 | 0.42188 | -0.26000 | -0.27917 |
| cafe | 20 | 0.63667 | 0.63597 | 0.62500 | -0.06667 | -0.06445 |
| cafe | 10 | 0.58667 | 0.58083 | 0.57188 | -0.11667 | -0.11959 |
| cafe | 5 | 0.55667 | 0.55418 | 0.54688 | -0.14667 | -0.14623 |
| cafe | 0 | 0.45333 | 0.44479 | 0.43750 | -0.25000 | -0.25563 |
| cafe | -5 | 0.34000 | 0.30379 | 0.32500 | -0.36333 | -0.39663 |

### 4.3 결과 요약

- 20 dB 조건에서도 모든 잡음에서 clean 대비 성능 하락이 발생하였다. 가장 약한 하락은 `cafe` 20 dB의 Accuracy -0.06667, Macro-F1 -0.06445였고, `white`, `pink`, `babble`은 모두 Accuracy 기준 약 -0.09에서 -0.11 수준의 하락을 보였다.
- `pink` 잡음은 SNR이 낮아질수록 가장 급격한 성능 붕괴를 보였다. -5 dB에서 Accuracy 0.15000, Macro-F1 0.06243으로 떨어졌으며, clean 대비 Macro-F1 하락폭은 -0.63798이었다.
- `babble` 잡음은 20 dB에서는 큰 하락을 보였지만, 10 dB 이하에서는 상대적으로 완만하게 하락하였다. -5 dB에서도 Accuracy 0.44333, Macro-F1 0.42124를 유지하여 네 잡음 중 가장 높은 강건성을 보였다.
- `cafe` 잡음은 20, 10, 5 dB에서 비교적 안정적이었으나 0 dB 이하에서는 하락폭이 커졌다. 이는 연속 배경음과 순간 transient가 약한 조건에서는 제한적 영향만 주지만, 강한 조건에서는 시간적 감정 단서를 직접 가릴 수 있음을 시사한다.
- ECE는 SNR 하락에 따라 단조롭게 증가하지 않았다. 강한 잡음에서 모델 confidence가 낮아지거나 특정 class로 치우치면 ECE가 낮게 보일 수 있으므로, calibration만으로 잡음 강건성을 판단하지 않고 Accuracy, Macro-F1, confusion matrix를 함께 해석해야 한다.

## 5. 아티팩트 분석

### 5.1 저장되는 산출물

| 산출물 | 경로 | 해석 |
|---|---|---|
| 조건별 metric | `../../outputs/2026-04-23/15-33-32_noise_eval_winner/noise_eval_winner/<condition>/metrics.json` | 조건별 Accuracy, Macro-F1, UAR, ECE |
| 전체 요약 CSV | `../../outputs/2026-04-23/15-33-32_noise_eval_winner/noise_eval_winner/noise_summary.csv` | 논문 표 작성용 압축 데이터 |
| 전체 요약 JSON | `../../outputs/2026-04-23/15-33-32_noise_eval_winner/noise_eval_winner/noise_summary.json` | 후속 문서화용 원자료 |
| confusion matrix | `../../outputs/2026-04-23/15-33-32_noise_eval_winner/noise_eval_winner/<condition>/confusion_matrix.png` | 잡음 조건별 감정 혼동 변화 |
| calibration curve | `../../outputs/2026-04-23/15-33-32_noise_eval_winner/noise_eval_winner/<condition>/calibration_curve.png` | 잡음에서 confidence 과신 여부 |
| ROC/PR curve | `../../outputs/2026-04-23/15-33-32_noise_eval_winner/noise_eval_winner/<condition>/roc_pr_curves.png` | class-wise 판별력 변화 |
| 논문용 SNR curve | `../../LateX_Paper/undergraduate-thesis/undergraduate-thesis/images/chapter4_experiment_artifacts/noise_snr_accuracy_curve.png` | SNR별 Accuracy 하락 추세 |
| 논문용 worst confusion matrix | `../../LateX_Paper/undergraduate-thesis/undergraduate-thesis/images/chapter4_experiment_artifacts/noise_pink_m5_confusion_matrix.png` | 가장 큰 성능 붕괴 조건의 혼동 양상 |

### 5.2 분석 관점

- SNR curve: SNR이 낮아질수록 모든 잡음에서 성능이 하락한다. 하락폭은 `pink > white > cafe > babble` 순서로 강하게 나타났다.
- noise type 차이: `babble`은 speech-like 간섭임에도 강한 조건에서 상대적으로 덜 무너졌다. 현재 synthetic babble이 원 파형 shift 기반이라 실제 다중 화자 잡음보다 단순할 가능성이 있다.
- confusion matrix: `pink -5 dB`는 가장 큰 성능 붕괴 조건이므로 논문에서는 worst-case 혼동 양상 예시로 사용하기 적합하다.
- calibration: 강한 잡음에서 ECE가 낮아지는 조건이 있어 confidence 해석은 보조 지표로 제한한다.

## 6. 종합 인사이트 및 다음 액션

### 6.1 현재 판단

첫 라운드 결과만으로도 clean winner CNN-Conformer가 잡음 조건에서 어느 정도 민감한지 설명할 수 있다. 학사논문 범위에서는 최고 모델 1개를 고정하여 SNR/type grid를 평가한 실험이 보조 강건성 분석으로 충분한 가치가 있다. 31번 우수 논문도 모든 보조 분석에서 모든 후보 모델을 반복 비교하기보다는, 주 모델 또는 제안 모델을 중심으로 조건 변화와 ablation을 제시하는 흐름을 사용한다. 본 논문에서도 잡음 조건 실험은 모델 선정의 주 실험이 아니라 최종 선택 모델의 환경 변화 민감도 확인으로 배치하는 편이 자연스럽다.

다만 현재 잡음은 synthetic 조건이므로, “실제 환경에서 검증되었다”는 식의 표현은 피해야 한다. 논문에서는 “제한된 합성 잡음 조건에서의 관찰”로 방어적으로 서술하고, 추가 실험은 시간과 분량이 허용될 때만 수행하는 것이 적합하다.

### 6.2 후속 후보

| 후보 | 적용 시점 | 장점 | 주의점 |
|---|---|---|---|
| 추가 없음, 논문 작성 시작 | 현재 학사논문 마감과 본문 완성도가 더 중요할 때 | 실험 축을 더 늘리지 않고 clean 모델 비교와 최종 모델 강건성 분석을 정리 가능 | synthetic noise의 한계를 명확히 써야 함 |
| noise augmentation training | 잡음 강건성을 논문의 핵심 기여로 키울 때 | 구현 비용 낮고 관련 연구 근거가 많음 | clean 성능 희생 여부를 다시 학습/평가해야 하므로 실험량 증가 |
| external noise corpus 평가 | synthetic noise 한계가 심사상 약점으로 보일 때 | 현실성 상승 | 데이터 다운로드, 라이선스, 경로 관리, 실험 재현 문서가 추가로 필요 |
| speech enhancement preprocessing | noisy SER 자체를 후속 연구 축으로 확장할 때 | Zhou et al. 계열 연구와 직접 연결 | 현재 논문 주제인 CNN vs Transformer 비교에서 벗어날 수 있음 |

현재 권장안은 추가 학습 실험을 멈추고 논문 작성으로 넘어가는 것이다. 잡음 실험은 최종 모델의 부가 검증으로 충분히 활용하고, 후속 연구 항목에 noise augmentation과 external noise corpus 평가를 남기는 흐름이 가장 안정적이다.

## 7. 참고 링크

- Zhou et al., Interspeech 2020, Using Speech Enhancement Preprocessing for SER in Realistic Noisy Conditions: <https://www.isca-archive.org/interspeech_2020/zhou20g_interspeech.html>
- Ranjan et al., Interspeech 2024, Reinforcement Learning based Data Augmentation for Noise Robust SER: <https://www.isca-archive.org/interspeech_2024/ranjan24_interspeech.html>
- Nam and Park, Applied Sciences 2024, SER under Noisy Environments with SNR Down to -6 dB: <https://www.mdpi.com/2076-3417/14/12/5227>
- Huang et al., Archives of Acoustics, SER under White Noise: <https://acoustics.ippt.pan.pl/index.php/aa/article/view/308>
- NIST, Augmenting Deep Learning Models for SER: <https://www.nist.gov/publications/augmenting-deep-learning-models-speech-emotion-recognition>

## 8. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-04-23 | 잡음 환경 실험 설계 초안 작성 |
| 2026-04-23 | Hydra 기반 noisy evaluation 구현 및 smoke test 결과 기록 |
| 2026-04-23 | 본 실험 결과와 artifact 분석 기록 |
