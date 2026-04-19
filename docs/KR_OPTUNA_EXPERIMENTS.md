# Optuna 검색 및 Pruning 시스템 상세 가이드

본 문서는 현재 코드베이스에서 작동하는 Optuna 하이퍼파라미터 최적화(`src/optuna_search.py`)의 세부 로직, 특히 Pruning(가지치기)의 기준과 Trial 집계 방식을 상세히 설명합니다.

실제 실행 코드: [../src/optuna_search.py](../src/optuna_search.py)

---

## 1. Pruning(가지치기)의 두 가지 유형

코드베이스 내에는 두 가지 다른 목적을 가진 Pruning이 존재합니다.

### A. 파라미터 조합 기반 사전 Pruning (Pre-flight Pruning)
학습을 시작하기도 전에, 설정된 하이퍼파라미터 조합이 물리적/수학적으로 불가능하거나 비효율적인 경우 즉각적으로 잘라냅니다.

- **발생 시점**: 모델이나 데이터로더를 초기화하기 전, Optuna가 값을 Suggest하는 단계.
- **예시**:
  - `f_min >= f_max` 일 때
  - `hop_length >= n_fft` 일 때
  - Transformer 계열의 `embed_dim`이 `num_heads`로 나누어떨어지지 않을 때
  - 윈도우 스케일링 시 `stage2_dim < stage1_dim` 역행이 발생할 때
- **의의**: 불가능한 에러(Crash)를 방지하고 쓸데없는 초기화 시간을 아낍니다.

### B. 학습 경과 기반 조기 종료 Pruning (Epoch-based Pruning)
학습이 실제로 시작된 후 파라미터 조합이 정상이라도 성능이 저조할 때 멈춥니다. 

- **발생 시점**: 매 Epoch가 끝나고 Validation 과정 수행 이후 `trial.report()` 가 호출될 때.
- **사용 알고리즘**: `optuna.pruners.MedianPruner`
- **의의**: 이미 "가망이 없는" 것으로 판별된 Trial에 연산 자원(GPU 시간)을 끝까지 낭비하지 않도록 합니다.

---

## 2. Epoch 기반 Pruning의 발동 기준 (언제 잘리는가?)

실행 시 yaml 설정(`warmup_steps: 0`)에 기반하여 Median Pruner가 동작합니다.

### **기준 로직: 왜 1~2 Epoch는 봐주다가 중간에 자르는가?**
Median Pruner는 특정 에포크 $E$에서 현재 Trial의 최고 성능(보통 `f1_macro` 등 설정된 `objective_metric`)이 **'이전에 에포크 $E$를 통과했던 이전 Trial들의 점수 중간값(Median)'**보다 낮은지 확인합니다.

1. **학습 극초반 (Trial이 몇 개 없을 때)**
   - 비교할 과거 통계치(중간값)가 없기 때문에 Pruning이 거의 발생하지 않습니다.
   - 따라서 초반 Trial들은 끝까지(30 Epoch 등) 완주하며 기준 통계치를 쌓습니다.
2. **통계가 쌓인 이후의 Trial**
   - 앞선 Trial들의 데이터가 쌓인 후에는, 새로운 Trial이 Epoch 1이나 Epoch 2를 마쳤을 때 Validation 점수를 냅니다.
   - 이때 이 점수가 "이전 Trial들이 Epoch 1(또는 2)에서 받았던 점수의 평균/중간값" 보다 낮으면 **"아, 이건 초반 성장세가 너무 느려서 후반까지 가봤자 남들만큼 못 크겠네"**라고 판단하여 즉각 `TrialPruned` 에러를 띄웁니다.

---

## 3. Optuna의 Trial 집계 방식 (COMPLETE vs PRUNED)

```bash
python -m src.optuna_search ... optuna.trials=30
```

이 명령어를 실행할 때 지정하는 `optuna.trials=30`은 **"전체 시도 횟수"가 아니라 "완료(COMPLETE)된 횟수"를 의미합니다.**

- [../src/optuna_search.py](../src/optuna_search.py) 571번째 줄을 보면 아래와 같이 강제 설정되어 있습니다.
  ```python
  callbacks=[MaxTrialsCallback(target_complete_trials, states=(TrialState.COMPLETE,))]
  ```
- **즉, 명시된 목표치는 "살아남아 끝까지 완주한 우수한/정상적인 Trial의 수"입니다.**
- 만약 `optuna.trials=30`을 주었는데 학습 도중에 15개의 Trial이 Prune(가지치기) 당했다면, Optuna는 30개의 완료된 결과를 띄우기 위해 사실상 45번의 Trial을 시도하게 됩니다.
- 따라서 Prune이 많이 발생할수록 실제 탐색하는 조합의 스펙트럼은 명시된 숫자보다 훨씬 방대해집니다.

---

## 4. Pruning이 Bayesian Optimization에 미치는 영향

현재 Optuna는 기본적으로 `TPESampler` (Tree-structured Parzen Estimator)라는 베이지안 최적화 방식을 사용합니다. 

이때 중간에 잘려나간(Pruned) Trial들은 결과가 오염되거나 버려지는 것이 아닙니다. 
- Pruned Trial이 머물고 있는 하이퍼파라미터 조합은 **Bad Region (나쁜 구역)** 데이터로 Sampler 내부에 차곡차곡 저장됩니다.
- 학습이 진행될수록 샘플러는 "이 조합 근처로 가면 금방 잘리는구나"를 터득하고, 점차 **살아남았던(Completed되고 점수가 높았던) Good Region으로 샘플링 확률을 집중**시킵니다.
- **결론**: Prune은 오염이 아니라 Optuna가 다음 번 뽑기(Sampling)를 훨씬 똑똑하게 하도록 도와주는 가장 강력한 네거티브 피드백(Negative Feedback)입니다.
