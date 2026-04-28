# 교차 코퍼스 실험 설계

## 1. 문서 범위

- 문서 대상 모델명: `cnn_conformer`
- 문서 목적: `RAVDESS`에서 선택된 최고 성능 `CNN-Conformer` backbone을 유지한 상태에서 `CREMA-D`로 교차 코퍼스 일반화 성능을 평가하기 위한 실험 설계와 실행 절차를 고정한다.
- 현재 문서 상태: `active`
- 작성 단계: 설계 및 실행 코드 구축 완료, 실험 미실행

이 문서는 `RAVDESS -> CREMA-D` 교차 코퍼스 실험의 기준 문서다. 기존 clean 조건 주 실험은 `RAVDESS` 내부 사용자 무관 평가에 초점을 두고 있었고, 이 문서는 그 다음 단계로서 “선택된 backbone이 다른 코퍼스에서도 어느 정도 유지되는가”를 확인하기 위한 별도 실험 계획을 다룬다.

## 2. 모델 스냅샷

### 2.1 한 줄 요약

현재 교차 코퍼스 실험은 `CNN-Conformer winner backbone`을 유지한 채, `RAVDESS`와 `CREMA-D`의 공통 감정 6개만 남긴 `source-only train -> target-only test` 프로토콜로 일반화 성능을 측정하는 방향을 기본안으로 둔다.

### 2.2 핵심 구성 요소

| 항목 | 값 또는 설명 |
|---|---|
| 입력 표현 | 80-bin log-Mel |
| 핵심 블록 | `nostem_patch` 기반 `CNN-Conformer` |
| 주요 구조 파라미터 | `time_patch=4`, `embed_dim=192`, `num_layers=4`, `num_heads=8`, `ffn_dim=768`, `conv_kernel=31` |
| 출력 pooling | attention pooling |
| 분류 대상 | `neutral`, `happy`, `sad`, `angry`, `fearful`, `disgust` |

### 2.3 비교 관점

- 비교 대상은 `RAVDESS` 내부 성능이 아니라, `source corpus`에서 학습한 모델이 `target corpus`에서 어느 정도 유지되는지다.
- 이 실험은 새로운 backbone 탐색이 아니라, 현재 winner backbone의 코퍼스 이동 민감도를 측정하는 일반화 실험이다.
- 첫 라운드에서는 domain adaptation을 넣지 않고 `source-only baseline`을 세운다.

## 3. 실험 라운드 기록

### 3.1 공통 고정 조건

| 분류 | 항목 | 값 | 비고 |
|---|---|---|---|
| source 모델 | backbone | `cnn_conformer` winner 계열 | 구조 family 고정 |
| source corpus | `RAVDESS` | 기존 주 실험과 동일 | `src/$RVNS6MQ` |
| target corpus | `CREMA-D` | 6-class target test | `src/CREMA-D` |
| 입력 | log-Mel | 기존 main 실험과 동일 | 전처리 축 추가 변경 금지 |
| 샘플링 | 16kHz resample | 기존 pipeline 유지 | 데이터셋별 원본 차이 통일 |
| 평가 | Accuracy / Macro-F1 / UAR | 기존 본문 표와 동일 | target set 기준 보고 |
| 프로토콜 | source-only train, target-only test | adaptation 없음 | 첫 실험 기준선 |

### 3.2 탐색 공간 또는 실험 변수

| 항목 | 후보군 | 비고 |
|---|---|---|
| source corpus | `RAVDESS` | 현재 주 실험 코퍼스 |
| target corpus | `CREMA-D` | 1차 고정 |
| label set | 6-class (`neutral`, `happy`, `sad`, `angry`, `fearful`, `disgust`) | 공통 감정 집합 |
| 학습 방식 | source-only supervised | 첫 라운드에서는 domain adaptation 제외 |
| fold 수 | source actor-level 5-fold | target은 전체 평가셋으로 사용 |

### 3.3 실행 명령

실행 명령:

```powershell
.\.venv\Scripts\python.exe -m src.cross_corpus_eval model=cnn_conformer cross_corpus.enabled=true cross_corpus.protocol=ravdess_to_cremad_6class experiment.name=cross_corpus_cremad6 experiment.tag=source_only_6class
```

fold 수를 1개만 먼저 확인하고 싶으면:

```powershell
.\.venv\Scripts\python.exe -m src.cross_corpus_eval model=cnn_conformer cross_corpus.enabled=true cross_corpus.train.folds_to_run=1 cross_corpus.protocol=ravdess_to_cremad_6class experiment.name=cross_corpus_cremad6_smoke experiment.tag=fold1
```

### 3.4 회차별 실험 로그

| 회차 | 날짜 | 목적 | 설정 요약 | 결과 요약 | 산출 경로 |
|---|---|---|---|---|---|
| Round 0 | 2026-04-28 | 실험 설계 및 코드 구축 | `RAVDESS -> CREMA-D`, 6-class, source-only | 아직 미실행 | `./cross_corpus/KR_CROSS_CORPUS_EXPERIMENT_PLAN.md` |

### 3.5 주요 결과 요약

아직 미실행.

권장 표:

| Rank | Protocol | Macro-F1 | Accuracy | UAR | 핵심 설정 |
|---|---|---:|---:|---:|---|
| 1 | `RAVDESS -> CREMA-D (6-class)` | | | | `CNN-Conformer source-only baseline` |

## 4. 설계 배경 및 구현 메모

### 4.1 왜 6-class를 택하는가

교차 코퍼스 SER에서는 두 코퍼스가 공통으로 갖는 감정 집합만 남겨서 실험하는 방식이 매우 흔하다. `Deep Cross-Corpus Speech Emotion Recognition: Recent Advances and Perspectives`는 교차 코퍼스 SER에서 데이터셋 쌍마다 공통 감정만 남기는 방식이 일반적임을 정리하고 있으며, 여러 기존 연구들도 `4-class`, `5-class`, `6-class` 등 공통 집합 기반 실험을 사용한다. `A study on cross-corpus speech emotion recognition and data augmentation`는 `angry`, `happy`, `sad`, `neutral`의 4-class 설정을 사용했고, `Progressively Discriminative Transfer Network for Cross-Corpus Speech Emotion Recognition`과 `Cross-Corpus Speech Emotion Recognition Based on Multi-Task Learning and Subdomain Adaptation`도 데이터셋 쌍마다 공통 감정만 다시 선택한다.

이번 프로젝트에서는 `4-class`보다 `6-class`가 더 적합하다. 이유는 다음과 같다.

- 현재 주 실험 출발점이 `RAVDESS 8-class`이므로, 곧바로 `4-class`로 줄이면 감정 공간이 지나치게 축소된다.
- `CREMA-D`는 `RAVDESS`와 `neutral`, `happy`, `sad`, `angry`, `fearful`, `disgust`의 6개 감정을 공통으로 가진다.
- `6-class`는 `4-class`보다 감정 범위를 더 많이 유지하면서도, 여전히 코퍼스 간 공통 분모를 유지한다.

### 4.2 타깃 코퍼스로 CREMA-D를 택하는 이유

`CREMA-D`는 `RAVDESS`와 동일하게 영어 acted speech 계열이면서, 화자 수와 표본 수가 충분하고, 공통 감정 수가 6개로 비교적 넓다. `SAVEE`는 공통 감정 수는 많지만 화자 수가 4명으로 너무 적고, `IEMOCAP`은 연구적 의미는 크지만 접근과 라벨 정리 부담이 크다. 첫 번째 교차 코퍼스 기준선 실험으로는 `CREMA-D`가 가장 균형이 좋다.

### 4.3 CREMA-D 데이터셋 서술

| 항목 | 내용 |
|---|---|
| 정식 명칭 | Crowd-sourced Emotional Multimodal Actors Dataset (`CREMA-D`) |
| 주요 논문 | Cao et al., IEEE TAC 2014 |
| 언어 | English |
| 발화 수 | 7,442 clips |
| 화자 수 | 91 actors |
| 성비 | 48 male, 43 female |
| 연령 | 20 to 74 |
| 발화 문장 수 | 12 fixed sentences |
| 감정 수 | 6 emotions: anger, disgust, fear, happy, neutral, sad |
| 강도 표기 | low, medium, high, unspecified |
| 음성 형식 | processed `.wav` audio |
| 공식 저장 위치 | `AudioWAV` directory |
| 라이선스 | ODbL 1.0 계열, 공식 페이지 참고 |

공식 요약에 따르면, `CREMA-D`는 91명의 배우가 12개의 고정 문장을 6개 감정으로 연기한 멀티모달 감정 코퍼스이며, 음성-only 평가와 audio-visual 평가를 위해 crowd-sourcing 기반 perceptual rating도 함께 구축되었다. 본 실험은 이 가운데 processed audio인 `AudioWAV` 부분만 사용한다. 공식 설명은 다음 링크에서 확인할 수 있다.

- 공식 홈페이지: https://cheyneycomputerscience.github.io/CREMA-D/
- 원 논문 정보: https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/
- 보조 데이터 문서: https://audeering.github.io/datasets/datasets/crema-d.html

### 4.4 CREMA-D 다운로드 링크와 주의점

다운로드/접근 링크:

- 공식 홈페이지: https://cheyneycomputerscience.github.io/CREMA-D/
- 공식 GitHub: https://github.com/CheyneyComputerScience/CREMA-D
- 공식 GitLab mirror: https://gitlab.com/cheyneycomputerscience/CREMA-D

중요 주의사항:

- 공식 저장소는 `git-lfs`를 사용한다.
- GitHub의 일반 zip 다운로드만 받으면 실제 `.wav` 오디오 대신 LFS pointer 파일만 내려올 수 있다.
- 실제 실험에 필요한 것은 `AudioWAV` 아래의 실체 `.wav` 파일이다.

사용자 권장 절차:

1. `git lfs` 설치
2. 공식 GitHub 또는 GitLab mirror에서 `CREMA-D` clone
3. clone 완료 후 `AudioWAV` 아래에 실제 `.wav` 파일이 존재하는지 확인
4. 최종 폴더를 프로젝트의 `src/CREMA-D` 위치에 둠

### 4.5 실제 코드가 기대하는 폴더 구조

코드는 아래 네 가지 패턴을 자동 탐지한다.

| 허용 패턴 | 예시 |
|---|---|
| `src/CREMA-D/AudioWAV/*.wav` | `src/CREMA-D/AudioWAV/1001_DFA_ANG_XX.wav` |
| `src/CREMA-D/AudioWAV/*/*.wav` | `src/CREMA-D/AudioWAV/1001/1001_DFA_ANG_XX.wav` |
| `src/CREMA-D/*.wav` | `src/CREMA-D/1001_DFA_ANG_XX.wav` |
| `src/CREMA-D/*/*.wav` | `src/CREMA-D/1001/1001_DFA_ANG_XX.wav` |

공식 문서와 공개 예시에서 가장 자주 보이는 형식은 다음과 같다.

```text
CREMA-D/
  AudioWAV/
    1001_DFA_ANG_XX.wav
    1001_DFA_DIS_XX.wav
    ...
```

또는 다음처럼 actor별 하위 폴더가 있는 배포본도 존재한다.

```text
CREMA-D/
  AudioWAV/
    1001/
      1001_DFA_ANG_XX.wav
      1001_DFA_DIS_XX.wav
```

파일명 규칙은 `ActorID_SentenceCode_EmotionCode_IntensityCode.wav`다.

예:

- `1001_DFA_ANG_XX.wav`
  - `1001`: actor id
  - `DFA`: sentence code
  - `ANG`: anger
  - `XX`: unspecified intensity

### 4.6 현재 코드 기준 구현

관련 구현 경로:

- 데이터셋 로더: `../src/data/cross_corpus_dataset.py`
- 교차 코퍼스 설정: `../src/configs/cross_corpus/default.yaml`
- 메인 설정 병합: `../src/configs/config.yaml`
- 모델 class 수 동적화: `../src/models/cnn_conformer.py`
- 교차 코퍼스 실행 엔트리포인트: `../src/cross_corpus_eval.py`

구현 내용 요약:

- `RAVDESS 8-class` 중 `neutral`, `happy`, `sad`, `angry`, `fearful`, `disgust`만 남기는 source dataset loader 추가
- `CREMA-D` 파일명 규칙을 파싱해 6-class로 매핑하는 target dataset loader 추가
- `cnn_conformer`가 `cfg.model.num_classes`를 읽도록 수정
- source actor-level `GroupKFold` 학습 후 target full-set 평가를 수행하는 전용 script 추가

### 4.7 현재 라벨 매핑

| 공통 index | 공통 label | RAVDESS | CREMA-D |
|---|---|---|---|
| 0 | neutral | `01` | `NEU` |
| 1 | happy | `03` | `HAP` |
| 2 | sad | `04` | `SAD` |
| 3 | angry | `05` | `ANG` |
| 4 | fearful | `06` | `FEA` |
| 5 | disgust | `07` | `DIS` |

`RAVDESS`의 `calm`과 `surprised`는 `CREMA-D`에 직접 대응하는 공통 클래스가 없으므로 제외한다.

### 4.8 실험 프로토콜

권장 프로토콜은 다음과 같다.

1. `RAVDESS`에서 공통 6개 감정만 남긴 source subset을 구성한다.
2. `CNN-Conformer` winner backbone의 구조 파라미터를 유지하고 classifier 출력 차원만 6으로 변경한다.
3. source dataset에서 actor-level `GroupKFold`로 학습/검증을 수행한다.
4. 각 fold에서 source validation 기준 best epoch를 고른다.
5. best checkpoint를 이용해 `CREMA-D` 전체 6-class target set에서 평가한다.
6. fold별 target 성능과 평균 성능을 기록한다.

### 4.9 내가 해야 할 것

단순 다운로드 외에 추가로 필요한 작업은 아래와 같다.

1. `CREMA-D`를 실제 `.wav` 파일이 포함된 형태로 준비한다.
2. 폴더를 프로젝트 기준 `src/CREMA-D` 아래에 둔다.
3. `AudioWAV` 안의 파일이 실제 오디오인지 확인한다.
   - LFS pointer 텍스트 파일이면 안 된다.
4. 기존 `RAVDESS` source 폴더 `src/$RVNS6MQ`가 그대로 유지되는지 확인한다.
5. 필요하면 `cross_corpus.target.dataset_path`를 명령어에서 직접 덮어쓴다.

예:

```powershell
.\.venv\Scripts\python.exe -m src.cross_corpus_eval model=cnn_conformer cross_corpus.enabled=true cross_corpus.target.dataset_path=src/CREMA-D
```

## 5. 아티팩트 분석

### 5.1 대표 산출물

실행 후 저장 예정 산출물:

| 산출물 | 경로 |
|---|---|
| fold 요약 CSV | `../outputs/.../artifacts/cross_corpus_fold_summary.csv` |
| 전체 요약 JSON | `../outputs/.../artifacts/cross_corpus_summary.json` |
| source validation confusion matrix | `../outputs/.../artifacts/fold_*/source_val_confusion_matrix.png` |
| target confusion matrix | `../outputs/.../artifacts/fold_*/target_confusion_matrix.png` |
| calibration curve | `../outputs/.../artifacts/fold_*/target_calibration_curve.png` |
| ROC/PR | `../outputs/.../artifacts/fold_*/target_roc_pr_curves.png` |

### 5.2 항목별 해석

- target confusion matrix: 어떤 감정 쌍에서 코퍼스 이동이 크게 발생하는지 확인
- source vs target gap: backbone의 일반화 손실 폭 확인
- class-wise recall 차이: 특정 감정이 코퍼스 이동에 더 취약한지 확인

## 6. 종합 인사이트 및 다음 액션

### 6.1 현재 판단

현재 단계에서 가장 적합한 교차 코퍼스 실험은 `RAVDESS -> CREMA-D 6-class source-only baseline`이다. 이 실험은 새로운 적응 기법을 추가하지 않고도, 현재 winner backbone의 외부 코퍼스 일반화 정도를 비교적 명확하게 보여줄 수 있다.

### 6.2 다음 액션

- `CREMA-D` 실체 `.wav` 데이터 준비
- `src/CREMA-D` 폴더 배치
- fold 1 smoke 실행
- 전체 fold 실행
- 결과를 바탕으로 필요 시 후속 `domain adaptation` 실험 여부 판단

## 7. 참고 링크

- CREMA-D 공식 홈페이지: https://cheyneycomputerscience.github.io/CREMA-D/
- CREMA-D 논문: https://pmc.ncbi.nlm.nih.gov/articles/PMC4313618/
- CREMA-D 데이터 문서: https://audeering.github.io/datasets/datasets/crema-d.html
- SAVEE: https://kahlan.eps.surrey.ac.uk/savee/
- IEMOCAP: https://www.slrb.net/IEMOCAP.html
- Deep Cross-Corpus SER Review: https://pmc.ncbi.nlm.nih.gov/articles/PMC8666588/
- Cross-corpus SER with 4-class setup example: https://www.researchgate.net/publication/357733781_A_study_on_cross-corpus_speech_emotion_recognition_and_data_augmentation
- Progressively Discriminative Transfer Network for Cross-Corpus SER: https://pmc.ncbi.nlm.nih.gov/articles/PMC9407047/
- Cross-Corpus SER Based on Multi-Task Learning and Subdomain Adaptation: https://www.mdpi.com/1099-4300/25/1/124

## 8. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-04-28 | `6-class CREMA-D` 기준으로 교차 코퍼스 실험 설계 문서 갱신 |
| 2026-04-28 | 코드 구현 경로, 폴더 구조, 사용자 준비 절차, 실행 명령 추가 |
