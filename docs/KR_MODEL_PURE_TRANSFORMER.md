# Pure Transformer

## 모델 설명 (KR_MODELS_TRANSFORMERS.md 참고)

### 핵심 아이디어
`pure_transformer`는 spectrogram을 patch 단위로 잘라 token으로 만든 뒤, 곧바로 transformer encoder에 넣는다. CNN stem이 없고, local pattern을 먼저 압축해 주는 강한 inductive bias도 없다. 이 구조는 `vaswani2017attention`의 가장 순수한 해석에 가깝다.
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

### 장약점 및 SER 관점 해석
- **장점**: transformer 자체의 기준선으로 해석이 가장 쉽다. 전역 문맥을 가장 직접적으로 본다. "CNN 없이도 되는가"를 보는 비교 기준이 된다.
- **약점**: local time-frequency pattern을 초기에 안정적으로 추출해 주는 구조가 없다. sequence가 길어질수록 global attention 비용이 커진다. 작은 SER 데이터에서는 학습 분산이 커지기 쉽다.
- **SER 관점 해석**: SER에서는 local cue가 중요한데, pure transformer는 그 cue를 직접 학습해야 한다. 데이터 규모가 작고 사전학습이 없는 상황에서는 이 점이 약점이 되기 쉽다. 그래서 이 모델은 "가장 개념적으로 순수한 기준선"으로는 좋지만, 실제 최종 성능 후보로는 다소 불리하다.

## 실험 운영 메모 (KR_EXPERIMENTS_TRANSFORMER.md 참고)

- `pure_transformer`는 순수 transformer 기준선 역할을 한다.
- 예시 실행 명령어:
```powershell
python -m src.optuna_search model=pure_transformer experiment.family=pure_transformer experiment.name=pure_transformer_optuna train.device=cuda optuna.trials=24 train.epochs=15 train.folds_to_run=1
```

## Optuna 탐색 결과 Top 5 (실험 일시: 2026-04-15 13:44:11)

| Rank | Trial | F1-macro | Accuracy | UAR | logmel_n_mels | logmel_n_fft | logmel_hop | logmel_f_min | logmel_f_max | logmel_normalize | train_batch_size | train_learning_rate | train_weight_decay | transformer_dropout | transformer_embed_dim | transformer_ffn_ratio | transformer_num_heads | transformer_num_layers | transformer_patch_size | transformer_patch_stride | transformer_pooling |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `trial_0016` | 0.51163 | 0.52000 | 0.51250 | 64 | 2048 | 160 | 20.0 | 6000.0 | True | 16 | 3.90e-4 | 3.12e-4 | 0.271 | 256 | 4 | 4 | 5 | 32 | 8 | mean |
| 2 | `trial_0153` | 0.48436 | 0.50333 | 0.48438 | 64 | 1024 | 160 | 20.0 | 8000.0 | True | 8 | 2.09e-4 | 4.70e-5 | 0.100 | 256 | 2 | 4 | 5 | 32 | 16 | cls |
| 3 | `trial_0042` | 0.48262 | 0.49000 | 0.48438 | 64 | 2048 | 256 | 50.0 | 8000.0 | True | 16 | 1.81e-4 | 4.87e-5 | 0.286 | 256 | 2 | 4 | 5 | 32 | 16 | cls |
| 4 | `trial_0156` | 0.48111 | 0.48667 | 0.47187 | 64 | 1024 | 160 | 20.0 | 8000.0 | True | 8 | 1.47e-4 | 3.88e-5 | 0.102 | 256 | 2 | 4 | 5 | 32 | 16 | cls |
| 5 | `trial_0154` | 0.47742 | 0.47667 | 0.47188 | 64 | 1024 | 160 | 20.0 | 8000.0 | True | 8 | 1.78e-4 | 4.41e-5 | 0.100 | 256 | 2 | 4 | 5 | 32 | 16 | cls |
