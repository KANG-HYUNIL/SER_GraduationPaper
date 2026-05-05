# **4. 噪声鲁棒性评估 (Noise Robustness Evaluation)**

## **1.1 实验介绍与设计逻辑**

噪声鲁棒性评估是对已选定的 **CNN-Conformer Champion 模型**进行的辅助验证实验，目的是在不重新训练模型的条件下，测量合成噪声条件下的性能下降幅度。噪声注入位置为波形（waveform）阶段，即在对数梅尔特征提取之前混入，更接近真实录音环境中信号受到污染后再提取特征的实际流程。

**实验设计要点：**

1. **固定模型：** Champion checkpoint 完全固定，仅对评估集波形注入加性噪声（additive noise），不进行任何再训练。
2. **噪声类型：** 采用四类合成噪声——`white`（全频段均匀噪声）、`pink`（低频偏重彩色噪声）、`babble`（多人说话近似）、`cafe`（连续背景音与瞬时噪声混合）。
3. **SNR 条件：** 评估 `clean`、`20`、`10`、`5`、`0`、`−5 dB` 共六档信噪比，覆盖轻度至极端噪声。
4. **评估说明：** 本实验所用噪声均为合成方式生成，不依赖 MUSAN、ESC-50 等外部噪声语料库，结果反映合成噪声条件下的有限观察，论文中需明确说明此局限性。

---

## **2. 实验参数配置**

### **2.1 固定模型参数（Champion）**

| **参数名称** | **数值** | **说明** |
| --- | --- | --- |
| **基准模型** | CNN-Conformer | clean 条件最高性能模型 |
| **clean Accuracy** | 0.7033 | 基准性能参考 |
| **clean Macro-F1** | 0.7004 | 基准性能参考 |
| **输入特征** | 80-bin log-Mel | `n_fft=1024`, `hop_length=160`, resize 未启用 |
| **主要结构** | `nostem_patch`, `embed_dim=192`, `num_layers=4`, `num_heads=8`, `ffn_dim=768`, `conv_kernel=31` | — |

### **2.2 噪声评估条件设定**

| **参数名称** | **数值** | **说明** |
| --- | --- | --- |
| **评估数据** | RAVDESS speech, fold 1 | 300 条发话 |
| **噪声注入位置** | waveform 阶段 | 特征提取前混入 |
| **噪声类型** | white, pink, babble, cafe | 合成生成，无外部依赖 |
| **SNR 条件** | clean, 20, 10, 5, 0, −5 dB | 六档覆盖 |
| **评估指标** | Accuracy, Macro-F1, UAR, ECE | clean 对比 delta 同步记录 |

---

## **3. 实验结果摘要**

### **3.1 全条件性能对比表**

| **噪声类型** | **SNR (dB)** | **Accuracy** | **Macro-F1** | **UAR** | **Δ Accuracy** | **Δ Macro-F1** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | clean | 0.7033 | 0.7004 | 0.7000 | 0.0000 | 0.0000 |
| white | 20 | 0.6100 | 0.5639 | 0.5781 | −0.0933 | −0.1365 |
| white | 10 | 0.4900 | 0.4472 | 0.4656 | −0.2133 | −0.2532 |
| white | 5 | 0.4067 | 0.3293 | 0.3813 | −0.2967 | −0.3711 |
| white | 0 | 0.3400 | 0.2417 | 0.3188 | −0.3633 | −0.4587 |
| white | −5 | 0.2933 | 0.1788 | 0.2750 | −0.4100 | −0.5217 |
| pink | 20 | 0.6000 | 0.5843 | 0.5813 | −0.1033 | −0.1162 |
| pink | 10 | 0.3667 | 0.3225 | 0.3469 | −0.3367 | −0.3779 |
| pink | 5 | 0.2700 | 0.1974 | 0.2531 | −0.4333 | −0.5030 |
| pink | 0 | 0.2100 | 0.1333 | 0.1969 | −0.4933 | −0.5671 |
| **pink** | **−5** | **0.1500** | **0.0624** | **0.1406** | **−0.5533** | **−0.6380** |
| babble | 20 | 0.5967 | 0.5741 | 0.5750 | −0.1067 | −0.1263 |
| babble | 10 | 0.5467 | 0.5266 | 0.5250 | −0.1567 | −0.1738 |
| babble | 5 | 0.5167 | 0.4989 | 0.4969 | −0.1867 | −0.2015 |
| babble | 0 | 0.4533 | 0.4374 | 0.4344 | −0.2500 | −0.2631 |
| babble | −5 | 0.4433 | 0.4212 | 0.4219 | −0.2600 | −0.2792 |
| cafe | 20 | 0.6367 | 0.6360 | 0.6250 | −0.0667 | −0.0644 |
| cafe | 10 | 0.5867 | 0.5808 | 0.5719 | −0.1167 | −0.1196 |
| cafe | 5 | 0.5567 | 0.5542 | 0.5469 | −0.1467 | −0.1462 |
| cafe | 0 | 0.4533 | 0.4448 | 0.4375 | −0.2500 | −0.2556 |
| cafe | −5 | 0.3400 | 0.3038 | 0.3250 | −0.3633 | −0.3966 |

### **3.2 噪声类型鲁棒性排序（以 −5 dB 为基准）**

| **排名** | **噪声类型** | **−5 dB Accuracy** | **−5 dB Macro-F1** | **特征** |
| --- | --- | ---: | ---: | --- |
| 1（最强健） | babble | 0.4433 | 0.4212 | SNR 全段下降平缓 |
| 2 | cafe | 0.3400 | 0.3038 | 低强度稳定，高强度急降 |
| 3 | white | 0.2933 | 0.1788 | 中等 SNR 区间急速下降 |
| 4（最脆弱） | **pink** | **0.1500** | **0.0624** | 低频污染导致最剧烈崩溃 |

---

## **4. 关键实验产出**

**SNR 条件下各噪声类型 Accuracy 下降曲线**
本地路径：`LateX_Paper/undergraduate-thesis/undergraduate-thesis/images/chapter4_experiment_artifacts/noise_snr_accuracy_curve.png`

**Pink −5 dB 混淆矩阵（最差条件）**
本地路径：`LateX_Paper/undergraduate-thesis/undergraduate-thesis/images/chapter4_experiment_artifacts/noise_pink_m5_confusion_matrix.png`

**Clean 基准混淆矩阵**
本地路径：`outputs/2026-04-23/15-33-32_noise_eval_winner/noise_eval_winner/clean/confusion_matrix.png`

**Babble −5 dB 混淆矩阵（最强健条件）**
本地路径：`outputs/2026-04-23/15-33-32_noise_eval_winner/noise_eval_winner/babble_snrm5/confusion_matrix.png`

---

---

# **5. 跨语料库评估 (Cross-Corpus Evaluation)**

## **1.1 实验介绍与设计逻辑**

跨语料库评估是对 CNN-Conformer 在无任何域适应（domain adaptation）条件下，迁移至外部语料库的泛化能力进行的辅助验证实验。本实验以 `RAVDESS 6-class` 为源域进行训练，以 `CREMA-D 6-class` 为目标域进行仅评估，属于 **source-only baseline**，旨在确认当前 backbone 的跨语料库敏感度，并为后续适应策略提供对照基准。

**实验设计要点：**

1. **模型重新初始化：** 并非直接复用 RAVDESS 8-class champion checkpoint。本实验将 `cnn_conformer` 重新初始化，仅将 `num_classes` 改为 6，并在 RAVDESS 公共 6-class 子集上重新训练。
2. **目标域零接触：** 目标域 CREMA-D 标签在训练和模型选择阶段均不被使用，仅在最终评估阶段读取。
3. **Best epoch 选择：** 依据 source validation Macro-F1 选择最佳 epoch，目标域评估基于此 best model 展开。
4. **规模差异：** source 训练集 1,056 条，target 评估集 7,442 条，两者规模差异显著。

---

## **2. 实验参数配置**

### **2.1 数据与评估条件**

| **参数名称** | **数值** | **说明** |
| --- | --- | --- |
| **源域数据集** | RAVDESS 6-class subset | neutral, happy, sad, angry, fearful, disgust |
| **目标域数据集** | CREMA-D 6-class subset | 相同 6 类情绪 |
| **源域数据量** | 1,056 条 | fold 1 train split |
| **目标域数据量** | 7,442 条 | 全量评估，无筛选 |
| **训练 fold** | fold 1 | actor-level GroupKFold |
| **log-Mel 设定** | `n_mels=128`, `n_fft=1024`, `hop_length=512` | resolved config 基准 |
| **训练轮数 / early stopping** | 30 / 10 | source fold 基准 |
| **评估指标** | Accuracy, Macro-F1, UAR, ECE | source 与 target 均使用 |

### **2.2 模型结构参数**

| **参数名称** | **数值** | **说明** |
| --- | --- | --- |
| **backbone** | cnn_conformer | 基本配置，新初始化 |
| **出力类别数** | 6 | 公共 6-class 对齐 |
| **主要结构** | `embed_dim=192`, `num_layers=8`, `num_heads=4`, `ffn_dim=768`, `conv_kernel=31` | resolved config 记录值 |
| **池化方式** | attention pooling | — |
| **域适应** | 无 | source-only 基准线 |

---

## **3. 实验结果摘要**

### **3.1 Source vs Target 性能对比**

| **评估域** | **Accuracy** | **Macro-F1** | **UAR** | **ECE** |
| --- | ---: | ---: | ---: | ---: |
| **Source val（RAVDESS）** | 0.5818 | 0.5690 | 0.5833 | 0.2219 |
| **Target（CREMA-D）** | 0.1938 | 0.0924 | 0.1894 | 0.5611 |
| **Δ（Target − Source）** | **−0.3880** | **−0.4766** | **−0.3939** | **+0.3392** |

附加信息：`best_epoch = 20`；source train F1 后期升至 `0.9+`，source val F1 停滞于 `0.56` 附近，存在源域内过拟合迹象。

### **3.2 关键观察汇总**

| **观察事实** | **结构性解读** |
| --- | --- |
| Source val Macro-F1 0.5690 | Backbone 已对 6-class 情感边界形成基本建模能力，并非完全训练失败 |
| Target Macro-F1 0.0924（接近随机） | RAVDESS 表达风格无法在无适应条件下迁移至 CREMA-D 发话分布 |
| ECE 从 0.2219 升至 0.5611 | 不仅分类边界崩溃，模型置信度校准本身已不适配目标域分布 |
| Source 1,056 条 vs Target 7,442 条 | 小规模源域过拟合加剧目标域性能崩溃幅度 |

---

## **4. 关键实验产出**

**Source vs Target 性能差距对比图**
本地路径：`LateX_Paper/undergraduate-thesis/undergraduate-thesis/images/chapter4_experiment_artifacts/cross_corpus_source_target_gap.png`

**Target（CREMA-D）混淆矩阵**
本地路径：`LateX_Paper/undergraduate-thesis/undergraduate-thesis/images/chapter4_experiment_artifacts/cross_corpus_target_confusion_matrix.png`

**Source val（RAVDESS）混淆矩阵**
本地路径：`outputs/2026-04-30/18-10-12_cross_corpus_cremad6/artifacts/fold_1/source_val_confusion_matrix.png`

**Target（CREMA-D）ROC/PR 曲线**
本地路径：`outputs/2026-04-30/18-10-12_cross_corpus_cremad6/artifacts/fold_1/target_roc_pr_curves.png`
