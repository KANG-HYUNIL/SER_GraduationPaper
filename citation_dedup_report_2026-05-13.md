# Citation Deduplication Report - 2026-05-13

Rule applied: one physical line in each chapter `.tex` file is treated as one paragraph. Within that paragraph, later repetitions of an already appeared citation key were removed. If a later `\cite{...}` became empty, the whole citation command was removed.

Verification: `duplicate_problems=0` after scanning `1_chapter1.tex` through `4_chapter4.tex`.

## Guidance Files Updated

- `Project/SER_GraduationPaper/CLAUDE.md`
- `.claude/skills/thesis-polish-cn/SKILL.md`
- `.claude/skills/thesis_polish_cn/SKILL.md`

## Edited Chapter Paragraphs

### 1_chapter1.tex

- 1.1 Research background: L11, L13
- 1.2 Research significance: L29
- 1.3.1 SER development history: L39, L41, L43
- 1.3.2 SER datasets: L47, L49
- 1.3.3 Acoustic features and signal processing: L57, L59
- 1.3.4 Deep-learning-based SER models: L63, L65, L67
- 1.3.5 Attention mechanism and Transformer: L71, L73
- 1.3.6 Low-resource, noise, and cross-corpus generalization: L79, L81, L83
- 1.4.1 Evaluation methodology limitations: L94, L96, L98
- 1.4.2 Model complexity and data scale: L103, L105, L107
- 1.4.3 Acoustic and language feature fusion: L112, L116, L118, L120

### 2_chapter2.tex

- Signal processing / time-frequency transform and spectrum analysis: L148, L268, L270
- Signal processing / perceptual feature extraction: L318, L320
- Signal processing / feature post-processing and sequence length handling: L356, L420, L428
- Challenges overview: L456
- Challenge / non-stationary speech signal and temporal non-uniformity: L467
- Challenge / variable utterance length: L471
- Challenge / external environmental interference: L475
- Challenge / multi-level emotional cues and feature representation complexity: L479
- Challenge / user, language, and dataset distribution differences: L483
- Challenge / emotion data scale limitation and class imbalance: L487
- Existing technologies: L492, L494, L496, L498

### 3_chapter3.tex

- Basic model structures / CNN model: L66
- Attention mechanism overview: L120
- Attention mechanism / basic concepts and key elements: L126
- Attention mechanism / representative attention scoring methods: L202
- Attention mechanism / structural types and extensions: L252
- Transformer structure / overall Transformer architecture: L299
- Transformer structure / positional encoding mechanism: L368

### 4_chapter4.tex

- Experimental parameters and settings / comparison model structure design / pure Transformer model: L103
