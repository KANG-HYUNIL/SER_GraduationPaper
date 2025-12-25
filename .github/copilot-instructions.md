# Copilot Instructions for SER_GraduationPaper

## 프로젝트 개요
- 본 프로젝트는 음성(wav) 파일을 대상으로 다양한 Feature Extraction(특징 추출) 기법을 Python으로 구현하는 실습 중심의 연구입니다.
- 주요 목표: Fourier Transform, Fast Fourier Transform, Short-Time Fourier Transform, Mel-Spectrogram, Wavelet Transform, MFCCs 등 오디오 신호 처리 기법을 직접 구현 및 비교 분석.

## 주요 폴더 및 파일
- `Week1/feature_extraction.py`: 모든 feature extraction 관련 코드의 중심 파일. 각 기법별 함수가 이 파일에 구현됨.
- `README.md`: 프로젝트 개요 및 목적 설명. 워크플로우/실행 방법 등은 별도 문서로 관리 필요.

## 아키텍처 및 워크플로우
- **단일 Python 스크립트 기반**: 모든 feature extraction 함수는 하나의 파일(`Week1/feature_extraction.py`)에 구현.
- **실행 흐름**: wav 파일을 입력 받아, 각 기법별로 특징을 추출하고 결과를 시각화(예: matplotlib 등 사용).
- **보고서 작성**: 각 기법별로 개념 설명, 코드, 파라미터 설명, 결과 이미지(진폭, 변환 결과 등)를 포함.

## 개발 환경 및 의존성 관리
- **Anaconda**: Python 환경 및 패키지 관리에 사용. 환경 생성/패키지 설치는 conda 명령어로 수행.
- **PyTorch**: 설치 증명용으로만 사용하며, 실제 feature extraction에는 직접적 사용 없음.
- **필수 라이브러리**: `librosa`, `matplotlib`, `numpy`, `scipy`, `pywt` 등. requirements.txt에 명시.

### requirements.txt 예시
```
librosa
matplotlib
numpy
scipy
pywt
```

## 프로젝트별 관례 및 패턴
- **함수명/코드 스타일**: 각 feature extraction 함수는 `extract_<method>` 형태로 명명(예: `extract_fft`, `extract_mfccs`).
- **입력/출력**: 함수 입력은 wav 파일 경로, 출력은 numpy array 및 시각화 이미지.
- **파라미터 설명**: 각 함수 내 주요 파라미터는 docstring으로 설명.
- **보고서 작성**: 코드 실행 결과(이미지, 파라미터, 개념 설명)를 Markdown 또는 PDF로 정리.

## 실행 및 디버깅
- **실행 예시**: `python Week1/feature_extraction.py` (필요시 main 함수 구현)
- **디버깅**: 함수별로 독립적으로 테스트 가능하도록 설계. 예시 wav 파일을 활용한 단위 테스트 권장.

## 확장/통합 지침
- 새로운 feature extraction 기법 추가 시, 동일한 함수명 패턴 및 docstring 규칙 준수.
- 외부 라이브러리 추가 시, requirements.txt에 반드시 반영.
- 보고서에 각 기법별 결과 및 비교 분석 포함.

## 참고
- 주요 구현/실행/보고서 작성 패턴은 `Week1/feature_extraction.py`에 집중되어 있음.
- 프로젝트 구조가 단순하므로, 폴더/파일 추가 시 명확한 목적과 역할을 주석/README에 기록.
