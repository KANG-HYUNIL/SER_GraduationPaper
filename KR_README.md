# SER_GraduationPaper 한국어 문서 인덱스

문서를 모델/데이터/실험/분석 단위로 분리했다. 아래 문서를 기준으로 읽으면 된다.

## 문서 링크

- 데이터 정리: [KR_DATA.md](docs/KR_DATA.md)
- CNN baseline 요약: [KR_MODELS_CNN_BASELINE.md](docs/KR_MODELS_CNN_BASELINE.md)
- Transformer 모델 정리: [KR_MODELS_TRANSFORMERS.md](docs/KR_MODELS_TRANSFORMERS.md)
  - [Pure Transformer](docs/KR_MODEL_PURE_TRANSFORMER.md)
  - [CNN Conformer](docs/KR_MODEL_CNN_CONFORMER.md)
  - [Hierarchical Window Transformer](docs/KR_MODEL_HIERARCHICAL_WINDOW_TRANSFORMER.md)
- 결과 분석: [KR_RESULTS_ANALYSIS.md](docs/KR_RESULTS_ANALYSIS.md)

## 현재 실험 기준 요약

- transformer 실험 대상:
  - `pure_transformer`
  - `cnn_conformer`
  - `hierarchical_window_transformer`
- patch transformer:
  - 현재 하드웨어 제약으로 실험 대상에서 제외
- transformer 입력:
  - `resize` 없이 가변 길이 log-Mel 사용
  - batch padding + `lengths` mask 적용

## 기본 실행

```powershell
python -m scripts.run_transformer_optuna_suite --device cuda
```

현재 기본값:

- `trials=24`
- `epochs=15`
- `folds_to_run=1`
- `max_parallel=1`

병렬 실행:

```powershell
python -m scripts.run_transformer_optuna_suite --device cuda --max-parallel 3
```

주의:

- 코드상 병렬 실행은 가능하지만 `RTX 2060 6GB`에서는 실제로는 `1` 또는 `2`가 더 안전하다.
