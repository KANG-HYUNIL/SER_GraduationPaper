from __future__ import annotations

from pathlib import Path
import math
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    Circle,
    Rectangle,
    Polygon,
    Arc,
)
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "feedback_diagrams"
FONT = ROOT / "STXIHEI.TTF"


if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    mpl.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()

mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


INK = "#243447"
MUTED = "#607080"
BLUE = "#4E79A7"
TEAL = "#59A14F"
ORANGE = "#F28E2B"
RED = "#E15759"
PURPLE = "#8A6BBE"
YELLOW = "#EDC948"
GRAY = "#F3F5F7"
GRID = "#D9E1E8"


def setup(name: str, figsize=(12, 6.6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.965, name, ha="left", va="top", fontsize=17, color=INK, weight="bold")
    return fig, ax


def save(fig, folder: Path, stem: str, sources: str):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    (folder / "sources.md").write_text(sources.strip() + "\n", encoding="utf-8")
    (folder / f"{stem}.drawio").write_text(
        f"""<mxfile host="app.diagrams.net">
  <diagram name="{stem}">
    <mxGraphModel dx="1169" dy="827" grid="1" gridSize="10" page="1" pageWidth="1169" pageHeight="827">
      <root>
        <mxCell id="0"/><mxCell id="1" parent="0"/>
        <mxCell id="note" value="{stem}&#xa;Generated redraw source: see {stem}.pdf and sources.md" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#EEF3F7;strokeColor=#61788A;strokeWidth=2;fontSize=18;fontColor=#1F2B37;" vertex="1" parent="1">
          <mxGeometry x="80" y="80" width="980" height="180" as="geometry"/>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
""",
        encoding="utf-8",
    )


def arrow(ax, p1, p2, color=INK, lw=2.0, rad=0.0, style="-|>", mutation=14):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle=style,
            mutation_scale=mutation,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=3,
            shrinkB=3,
        )
    )


def box(ax, xy, wh, text, fc=GRAY, ec=MUTED, fontsize=10.5, radius=0.02, lw=1.5, weight="normal"):
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK, weight=weight)


def mini_wave(ax, x, y, w, h, color=BLUE, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 260)
    env = 0.15 + 0.85 * np.exp(-((t - 0.52) ** 2) / 0.08)
    sig = env * (np.sin(2 * np.pi * (7 * t + 2.5 * t**2)) + 0.25 * np.sin(2 * np.pi * 19 * t))
    sig += 0.05 * rng.normal(size=t.size)
    ax.plot(x + t * w, y + h / 2 + sig * h * 0.35, color=color, lw=1.6)
    ax.plot([x, x + w], [y + h / 2, y + h / 2], color="#CCD4DA", lw=0.8)


def heat(ax, x, y, w, h, cmap="magma", seed=0, mel=False, log=False):
    rng = np.random.default_rng(seed)
    xx = np.linspace(0, 1, 120)
    yy = np.linspace(0, 1, 70)
    X, Y = np.meshgrid(xx, yy)
    Z = 0.2 * rng.random(X.shape)
    for cx, cy, sx, sy, amp in [(0.25, 0.35, 0.08, 0.12, 1.2), (0.52, 0.62, 0.12, 0.08, 0.8), (0.74, 0.25, 0.1, 0.16, 0.9)]:
        Z += amp * np.exp(-((X - cx) ** 2 / sx**2 + (Y - cy) ** 2 / sy**2))
    if mel:
        Z = Z ** 0.75
    if log:
        Z = np.log1p(6 * Z)
    ax.imshow(Z, extent=(x, x + w, y, y + h), origin="lower", cmap=cmap, aspect="auto", interpolation="bilinear")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=MUTED, lw=1.1))


def spectrum(ax, x, y, w, h, color=ORANGE, seed=0):
    rng = np.random.default_rng(seed)
    f = np.linspace(0, 1, 180)
    z = 0.05 + 0.55 * np.exp(-((f - 0.16) ** 2) / 0.006) + 0.35 * np.exp(-((f - 0.36) ** 2) / 0.004)
    z += 0.18 * np.exp(-((f - 0.63) ** 2) / 0.01) + 0.03 * rng.random(f.size)
    ax.fill_between(x + f * w, y, y + z * h, color=color, alpha=0.28)
    ax.plot(x + f * w, y + z * h, color=color, lw=1.7)
    ax.plot([x, x + w], [y, y], color=MUTED, lw=0.9)


def ser_evolution():
    fig, ax = setup("语音情绪识别方法演进：从人工特征到注意力与Transformer", (13.2, 6.8))
    stages = [
        ("人工声学特征", "LLD / eGeMAPS\nMFCC、基频、能量", BLUE, "手工设计"),
        ("机器学习分类", "SVM / HMM / GMM\n子空间学习、迁移学习", TEAL, "浅层判别"),
        ("CNN局部模式", "Log-Mel / 谱图\n卷积核扫描时频纹理", ORANGE, "自动特征"),
        ("RNN时序上下文", "LSTM / BiLSTM\n局部注意力聚合", RED, "顺序建模"),
        ("Transformer范式", "Self-Attention\nConformer / SSL 表示", PURPLE, "全局关系"),
    ]
    xs = np.linspace(0.10, 0.90, len(stages))
    y = 0.54
    for i, (title, body, color, tag) in enumerate(stages):
        ax.add_patch(Circle((xs[i], y), 0.048, fc=color, ec="white", lw=2.4))
        ax.text(xs[i], y, str(i + 1), ha="center", va="center", color="white", fontsize=15, weight="bold")
        box(ax, (xs[i] - 0.085, 0.22), (0.17, 0.16), f"{title}\n{body}", fc="#FFFFFF", ec=color, fontsize=9.5, radius=0.015)
        ax.text(xs[i], 0.15, tag, ha="center", va="center", fontsize=10, color=color, weight="bold")
        if i < len(stages) - 1:
            arrow(ax, (xs[i] + 0.055, y), (xs[i + 1] - 0.055, y), color=MUTED, lw=2.2)
    mini_wave(ax, 0.065, 0.71, 0.12, 0.12, BLUE, 2)
    heat(ax, 0.40, 0.70, 0.14, 0.13, seed=3, log=True)
    for k in range(5):
        ax.add_patch(Rectangle((0.56 + k * 0.018, 0.71 + k * 0.006), 0.07, 0.10, fc="#F6D7BD", ec=ORANGE, lw=1))
    for j in range(5):
        box(ax, (0.69 + j * 0.035, 0.73), (0.026, 0.055), "", fc="#F9D8D8", ec=RED, radius=0.006)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        arrow(ax, (0.705 + a * 0.035, 0.76), (0.705 + b * 0.035, 0.76), color=RED, lw=1.4, mutation=10)
    for i in range(5):
        for j in range(5):
            alpha = 0.25 + 0.55 * np.exp(-abs(i - j) / 1.5)
            ax.add_patch(Rectangle((0.865 + i * 0.018, 0.72 + j * 0.018), 0.014, 0.014, fc=PURPLE, alpha=alpha, ec="none"))
    ax.text(0.49, 0.065, "底部标签表示每一阶段的核心变化；上部小图标表示对应论文中常见的结构表达方式", ha="center", fontsize=9.5, color=MUTED)
    save(
        fig,
        OUT / "chapter1" / "ser_evolution",
        "ser_evolution_timeline",
        """
# Sources
- Zhang and Song, 2020, Transfer Sparse Discriminant Subspace Learning for Cross-Corpus SER (`zhang2020transfer`): traditional feature/subspace-learning pipeline.
- Peng et al., 2021, Efficient SER Using Multi-Scale CNN and Attention (`peng2021efficient`): multi-scale CNN and attention-style acoustic/lexical modeling.
- Mirsamadi et al., 2017, Automatic SER Using RNNs with Local Attention (`mirsamadi2017automatic`): recurrent temporal modeling with local attention.
- Liu et al., 2023, Dual-TBNet (`liu2023dualrobustness`): CNN/Transformer/BiLSTM hybrid trend.
- Vaswani et al., 2017, Attention Is All You Need (`vaswani2017attention`): Transformer encoder-decoder self-attention paradigm.
""",
    )


def acoustic_features():
    fig, ax = setup("声学特征比较：从物理信号到可学习表示", (12.5, 7.2))
    mini_wave(ax, 0.05, 0.67, 0.25, 0.16, BLUE, 5)
    ax.text(0.175, 0.62, "原始波形\n振幅随时间变化", ha="center", fontsize=10, color=INK)
    arrow(ax, (0.31, 0.75), (0.40, 0.75), color=MUTED)
    heat(ax, 0.41, 0.64, 0.23, 0.22, seed=7, log=False)
    ax.text(0.525, 0.59, "谱图 / Mel / Log-Mel\n时频能量分布", ha="center", fontsize=10, color=INK)
    arrow(ax, (0.65, 0.75), (0.74, 0.75), color=MUTED)
    for i in range(14):
        h = 0.02 + 0.13 * (0.45 + 0.5 * np.sin(i * 0.85) ** 2)
        ax.add_patch(Rectangle((0.75 + i * 0.012, 0.66), 0.007, h, fc=PURPLE, ec=PURPLE, alpha=0.85))
    ax.text(0.84, 0.59, "MFCC / eGeMAPS\n紧凑统计特征", ha="center", fontsize=10, color=INK)
    rows = [
        ("频谱图", "线性频率轴", "保留细节多，维度较高", BLUE),
        ("Mel谱图", "听觉尺度压缩", "低频更细，高频聚合", TEAL),
        ("Log-Mel谱图", "Mel + 对数压缩", "动态范围更稳定，适合CNN/Transformer", ORANGE),
        ("MFCC", "Log-Mel + DCT", "低维倒谱系数，适合传统分类器", PURPLE),
        ("基频 / 能量 / 韵律", "时间轨迹统计", "直接反映语调、强度、语速", RED),
    ]
    y0 = 0.45
    for i, (a, b, c, color) in enumerate(rows):
        y = y0 - i * 0.07
        ax.add_patch(Circle((0.09, y + 0.017), 0.018, fc=color, ec="white", lw=1.2))
        ax.text(0.13, y + 0.017, a, ha="left", va="center", fontsize=10.5, color=INK, weight="bold")
        ax.text(0.33, y + 0.017, b, ha="left", va="center", fontsize=10, color=INK)
        ax.text(0.58, y + 0.017, c, ha="left", va="center", fontsize=10, color=MUTED)
        ax.plot([0.11, 0.92], [y - 0.015, y - 0.015], color=GRID, lw=0.8)
    ax.text(0.08, 0.50, "特征", fontsize=10, weight="bold", color=MUTED)
    ax.text(0.33, 0.50, "变换方式", fontsize=10, weight="bold", color=MUTED)
    ax.text(0.58, 0.50, "直观差异", fontsize=10, weight="bold", color=MUTED)
    save(
        fig,
        OUT / "chapter1" / "acoustic_features",
        "acoustic_feature_taxonomy",
        """
# Sources
- librosa melspectrogram documentation: STFT power spectrum mapped to a mel basis, commonly followed by dB/log display.
- SciPy spectrogram documentation: spectrogram visualizes frequency content change over time.
- Zhang and Song, 2020 (`zhang2020transfer`) and Ma et al. review references in `ref.bib`: traditional SER acoustic feature sets such as MFCC/eGeMAPS.
""",
    )


def ft_time_frequency():
    fig, ax = setup("傅里叶变换：时域波形到频域幅度谱", (11.8, 5.5))
    mini_wave(ax, 0.07, 0.30, 0.34, 0.34, BLUE, 10)
    ax.text(0.24, 0.22, "时域信号 x[n]\n振幅随时间起伏", ha="center", fontsize=11, color=INK)
    box(ax, (0.445, 0.43), (0.12, 0.10), "DFT / FFT", fc="#FFF2CC", ec=ORANGE, fontsize=12, weight="bold")
    arrow(ax, (0.41, 0.48), (0.445, 0.48), color=MUTED)
    arrow(ax, (0.565, 0.48), (0.62, 0.48), color=MUTED)
    spectrum(ax, 0.64, 0.31, 0.30, 0.28, ORANGE, 12)
    ax.text(0.79, 0.22, "频域幅度 |X[k]|\n峰值对应主要频率成分", ha="center", fontsize=11, color=INK)
    ax.text(0.50, 0.13, "一次傅里叶变换给出整段信号的频率组成，但会丢失频率随时间变化的位置关系", ha="center", fontsize=10, color=MUTED)
    save(
        fig,
        OUT / "chapter2" / "fourier",
        "ft_time_to_frequency",
        """
# Sources
- SciPy signal spectrogram documentation: Fourier transforms are used over segments to represent frequency content.
- Chapter 2 equations in this thesis: DFT and power spectrum definitions.
""",
    )


def ft_vs_stft():
    fig, ax = setup("FT 与 STFT：全局频谱和逐帧时频表示的差异", (12.2, 6.0))
    mini_wave(ax, 0.05, 0.63, 0.30, 0.16, BLUE, 21)
    ax.text(0.20, 0.58, "同一段非平稳语音", ha="center", fontsize=10, color=INK)
    arrow(ax, (0.36, 0.72), (0.48, 0.72), color=MUTED)
    spectrum(ax, 0.50, 0.61, 0.23, 0.17, ORANGE, 23)
    ax.text(0.615, 0.56, "FT：整段平均后的频谱", ha="center", fontsize=10, color=INK)
    for i in range(6):
        x = 0.08 + i * 0.042
        ax.add_patch(Rectangle((x, 0.33), 0.07, 0.13, fc="#E9F2FB", ec=BLUE, lw=1, alpha=0.75))
    arrow(ax, (0.35, 0.39), (0.48, 0.39), color=MUTED)
    heat(ax, 0.50, 0.25, 0.28, 0.22, seed=25, log=True)
    ax.text(0.64, 0.19, "STFT：分帧、加窗、逐帧FFT\n保留时间-频率二维结构", ha="center", fontsize=10, color=INK)
    box(ax, (0.81, 0.34), (0.13, 0.19), "关键差异\nFT: 频率组成\nSTFT: 频率随时间变化", fc="#F7F7F9", ec=MUTED, fontsize=10)
    save(
        fig,
        OUT / "chapter2" / "stft",
        "ft_vs_stft",
        """
# Sources
- SciPy signal spectrogram documentation: spectrograms visualize change of nonstationary signal frequency content over time.
- Chapter 2 STFT equations in this thesis: sliding window plus per-frame Fourier transform.
""",
    )


def spectrogram_forms():
    fig, ax = setup("频谱图三种形式：幅度、功率与对数功率", (12.8, 5.9))
    titles = ["幅度谱图 |X|", "功率谱图 |X|²", "对数谱图 log(|X|²+ε)"]
    cmaps = ["viridis", "inferno", "magma"]
    for i in range(3):
        x = 0.07 + i * 0.30
        heat(ax, x, 0.35, 0.23, 0.27, cmap=cmaps[i], seed=30, log=(i == 2))
        ax.text(x + 0.115, 0.30, titles[i], ha="center", fontsize=11, color=INK, weight="bold")
        desc = ["直接看振幅强度", "强调高能量区域", "压缩动态范围，弱能量更可见"][i]
        ax.text(x + 0.115, 0.245, desc, ha="center", fontsize=9.5, color=MUTED)
        if i < 2:
            arrow(ax, (x + 0.245, 0.49), (x + 0.285, 0.49), color=MUTED)
    ax.text(0.50, 0.16, "同一STFT结果可按不同数值尺度呈现；模型输入常根据任务选择功率或对数尺度。", ha="center", fontsize=10, color=MUTED)
    save(
        fig,
        OUT / "chapter2" / "spectrogram_forms",
        "spectrogram_forms",
        """
# Sources
- SciPy signal.spectrogram documentation: supports magnitude and power spectral density modes.
- librosa power_to_db and melspectrogram documentation: power spectrograms are commonly converted to dB/log scale for display and modeling.
""",
    )


def mel_vs_logmel():
    fig, ax = setup("Mel谱图与对数Mel谱图：听觉尺度与动态范围压缩", (12.2, 5.8))
    mini_wave(ax, 0.05, 0.64, 0.18, 0.14, BLUE, 40)
    arrow(ax, (0.24, 0.71), (0.33, 0.71), color=MUTED)
    for m in range(14):
        x0 = 0.34 + m * 0.012
        ax.add_patch(Polygon([[x0, 0.62], [x0 + 0.016, 0.80], [x0 + 0.032, 0.62]], closed=False, ec=TEAL, lw=1.1))
    ax.text(0.42, 0.56, "Mel滤波器组\n线性频谱 → Mel频带", ha="center", fontsize=10, color=INK)
    arrow(ax, (0.51, 0.71), (0.58, 0.71), color=MUTED)
    heat(ax, 0.59, 0.59, 0.16, 0.22, cmap="viridis", seed=41, mel=True, log=False)
    ax.text(0.67, 0.54, "Mel谱图\n能量仍可能高度集中", ha="center", fontsize=10, color=INK)
    arrow(ax, (0.76, 0.71), (0.82, 0.71), color=MUTED)
    box(ax, (0.79, 0.33), (0.08, 0.08), "log / dB", fc="#FFF2CC", ec=ORANGE, fontsize=10, weight="bold")
    heat(ax, 0.84, 0.59, 0.16, 0.22, cmap="magma", seed=41, mel=True, log=True)
    ax.text(0.92, 0.54, "Log-Mel谱图\n弱频带差异更可见", ha="center", fontsize=10, color=INK)
    ax.text(0.50, 0.20, "对数变换不是改变频带位置，而是压缩能量尺度，使局部纹理更适合后续学习。", ha="center", fontsize=10, color=MUTED)
    save(
        fig,
        OUT / "chapter2" / "logmel",
        "mel_vs_logmel",
        """
# Sources
- librosa.feature.melspectrogram documentation: maps a power spectrogram to mel basis.
- librosa.power_to_db examples: converts mel spectrogram coefficients to dB scale for visualization.
""",
    )


def mlp_diagram():
    fig, ax = setup("MLP / 全连接网络：固定长度特征向量到情绪类别", (12.0, 5.8))
    layers = [(0.12, 5, BLUE, "输入特征\nMFCC/能量/基频"), (0.35, 7, TEAL, "隐藏层1"), (0.58, 5, ORANGE, "隐藏层2"), (0.82, 4, RED, "Softmax\n情绪类别")]
    coords = []
    for x, n, color, label in layers:
        ys = np.linspace(0.30, 0.76, n)
        cur = []
        for y in ys:
            ax.add_patch(Circle((x, y), 0.024, fc="white", ec=color, lw=2))
            cur.append((x, y))
        coords.append(cur)
        ax.text(x, 0.20, label, ha="center", fontsize=10, color=INK)
    for a, b in zip(coords[:-1], coords[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0] + 0.025, q[0] - 0.025], [p[1], q[1]], color="#B8C3CC", lw=0.45, alpha=0.55)
    for x in [0.35, 0.58]:
        box(ax, (x - 0.055, 0.82), (0.11, 0.055), "W·x + b", fc="#EEF3F7", ec=MUTED, fontsize=9.5)
        box(ax, (x - 0.05, 0.10), (0.10, 0.05), "激活 φ", fc="#FFF2CC", ec=ORANGE, fontsize=9.5)
    arrow(ax, (0.40, 0.82), (0.53, 0.82), color=MUTED, mutation=10)
    save(
        fig,
        OUT / "chapter3" / "mlp",
        "mlp_neural_network",
        """
# Sources
- Chapter 3 MLP equations in this thesis.
- General neural-network notation; no external figure copied.
""",
    )


def cnn_diagram():
    fig, ax = setup("CNN：卷积核在Log-Mel谱图上提取局部时频模式", (12.5, 6.0))
    heat(ax, 0.06, 0.36, 0.22, 0.28, seed=55, log=True)
    ax.text(0.17, 0.29, "输入Log-Mel谱图\n时间 × 频带", ha="center", fontsize=10, color=INK)
    for dx, dy in [(0, 0), (0.012, 0.010), (0.024, 0.020)]:
        ax.add_patch(Rectangle((0.36 + dx, 0.42 + dy), 0.12, 0.14, fc="#FBE1C7", ec=ORANGE, lw=1.2))
    ax.add_patch(Rectangle((0.12, 0.47), 0.06, 0.07, fill=False, ec=YELLOW, lw=2.5))
    arrow(ax, (0.28, 0.50), (0.36, 0.50), color=MUTED)
    ax.text(0.42, 0.29, "多通道特征图\n局部纹理响应", ha="center", fontsize=10, color=INK)
    for i in range(4):
        ax.add_patch(Rectangle((0.58 + i * 0.014, 0.43 + i * 0.006), 0.10, 0.13, fc="#DFF0D8", ec=TEAL, lw=1.1))
    arrow(ax, (0.50, 0.50), (0.58, 0.50), color=MUTED)
    ax.text(0.65, 0.29, "池化/下采样\n保留显著响应", ha="center", fontsize=10, color=INK)
    for i in range(7):
        ax.add_patch(Rectangle((0.80, 0.37 + i * 0.026), 0.08, 0.017, fc=PURPLE, ec=PURPLE, alpha=0.7 + i * 0.03))
    arrow(ax, (0.71, 0.50), (0.80, 0.50), color=MUTED)
    ax.text(0.84, 0.29, "分类头\n情绪概率", ha="center", fontsize=10, color=INK)
    save(
        fig,
        OUT / "chapter3" / "cnn",
        "cnn_local_feature_extraction",
        """
# Sources
- Peng et al., 2021 (`peng2021efficient`): multi-scale CNN style acoustic representation.
- Chapter 3 CNN equation and pooling discussion in this thesis.
""",
    )


def rnn_diagram():
    fig, ax = setup("RNN / LSTM：按时间递推聚合语音上下文", (12.3, 5.8))
    heat(ax, 0.05, 0.62, 0.24, 0.14, seed=61, log=True)
    ax.text(0.17, 0.56, "帧级声学序列", ha="center", fontsize=10, color=INK)
    xs = np.linspace(0.17, 0.78, 6)
    for i, x in enumerate(xs):
        box(ax, (x - 0.035, 0.35), (0.07, 0.09), f"h{i+1}", fc="#F9D8D8", ec=RED, fontsize=11, weight="bold")
        ax.text(x, 0.29, f"x{i+1}", ha="center", fontsize=9, color=MUTED)
        arrow(ax, (x, 0.58), (x, 0.45), color="#9AA7B2", lw=1.2, mutation=10)
        if i < len(xs) - 1:
            arrow(ax, (x + 0.04, 0.395), (xs[i + 1] - 0.04, 0.395), color=RED, lw=1.8, mutation=12)
    arrow(ax, (0.78, 0.40), (0.88, 0.40), color=MUTED)
    box(ax, (0.88, 0.35), (0.09, 0.09), "发话级\n表示", fc="#FFF2CC", ec=ORANGE, fontsize=10)
    ax.text(0.50, 0.18, "隐藏状态逐步携带过去上下文；BiLSTM会再加入反向序列的信息。", ha="center", fontsize=10, color=MUTED)
    save(
        fig,
        OUT / "chapter3" / "rnn",
        "rnn_sequence_context",
        """
# Sources
- Mirsamadi et al., 2017 (`mirsamadi2017automatic`): RNN hidden states with local attention pooling.
- Liu et al., 2023 (`liu2023dualrobustness`): BiLSTM as part of a hybrid SER architecture.
""",
    )


def attention_diagram():
    fig, ax = setup("注意力机制：从Q-K相关性到加权语音表示", (12.5, 6.3))
    for i, (x, label, color) in enumerate([(0.12, "Q\n查询", BLUE), (0.12, "K\n键", TEAL), (0.12, "V\n值", ORANGE)]):
        y = 0.70 - i * 0.16
        box(ax, (x - 0.04, y - 0.04), (0.08, 0.08), label, fc="white", ec=color, fontsize=11, weight="bold")
        arrow(ax, (0.17, y), (0.30, 0.54), color=color, lw=1.6, mutation=11)
    box(ax, (0.31, 0.49), (0.13, 0.10), "QK^T / sqrt(d)\n相关性得分", fc="#EEF3F7", ec=MUTED, fontsize=10)
    arrow(ax, (0.44, 0.54), (0.54, 0.54), color=MUTED)
    for i in range(5):
        for j in range(5):
            val = 0.25 + 0.65 * np.exp(-abs(i - j) / 1.2)
            ax.add_patch(Rectangle((0.55 + i * 0.027, 0.47 + j * 0.027), 0.022, 0.022, fc=PURPLE, alpha=val, ec="white", lw=0.3))
    ax.text(0.62, 0.42, "Softmax权重矩阵", ha="center", fontsize=10, color=INK)
    arrow(ax, (0.70, 0.54), (0.78, 0.54), color=MUTED)
    for i, h in enumerate([0.08, 0.14, 0.24, 0.13, 0.06]):
        ax.add_patch(Rectangle((0.80 + i * 0.025, 0.43), 0.018, h, fc=RED, ec=RED, alpha=0.75))
    box(ax, (0.79, 0.67), (0.16, 0.08), "sum alpha_ij V_j", fc="#FFF2CC", ec=ORANGE, fontsize=11, weight="bold")
    ax.text(0.87, 0.36, "关键帧/关键频带\n获得更大权重", ha="center", fontsize=10, color=INK)
    save(
        fig,
        OUT / "chapter3" / "attention",
        "attention_qkv_weighting",
        """
# Sources
- Vaswani et al., 2017 (`vaswani2017attention`): scaled dot-product attention.
- Mirsamadi et al., 2017 (`mirsamadi2017automatic`): attention pooling for emotionally salient speech regions.
""",
    )


def transformer_architecture():
    fig, ax = setup("Transformer：编码器-解码器与语音分类用编码器路径", (13.0, 7.2))
    # Encoder stack
    x0 = 0.18
    ys = [0.20, 0.34, 0.48, 0.62]
    labels = ["输入嵌入\n+位置编码", "多头自注意力\nMulti-Head Self-Attn", "前馈网络\nFFN", "编码器输出\n上下文表示"]
    colors = [BLUE, PURPLE, TEAL, ORANGE]
    for y, label, color in zip(ys, labels, colors):
        box(ax, (x0, y), (0.23, 0.08), label, fc="white", ec=color, fontsize=9.8, radius=0.012, lw=1.7)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow(ax, (x0 + 0.115, y1 + 0.08), (x0 + 0.115, y2), color=MUTED, mutation=10)
    ax.text(x0 + 0.115, 0.76, "Encoder × N", ha="center", fontsize=12, weight="bold", color=INK)
    # Decoder stack
    x1 = 0.58
    dlabels = ["目标输入\n+位置编码", "Masked Self-Attn", "Cross-Attn\n参考编码器输出", "FFN + 线性/Softmax"]
    for y, label, color in zip(ys, dlabels, [BLUE, RED, PURPLE, TEAL]):
        box(ax, (x1, y), (0.24, 0.08), label, fc="white", ec=color, fontsize=9.8, radius=0.012, lw=1.7)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow(ax, (x1 + 0.12, y1 + 0.08), (x1 + 0.12, y2), color=MUTED, mutation=10)
    ax.text(x1 + 0.12, 0.76, "Decoder × N", ha="center", fontsize=12, weight="bold", color=INK)
    arrow(ax, (x0 + 0.23, 0.66), (x1, 0.52), color=PURPLE, lw=2.0, rad=-0.08)
    # Speech classification branch
    arrow(ax, (x0 + 0.115, 0.70), (x0 + 0.115, 0.86), color=ORANGE, mutation=12)
    box(ax, (0.07, 0.86), (0.32, 0.07), "语音情绪识别常用简化：Encoder输出 → Pooling / CLS → 分类头", fc="#FFF2CC", ec=ORANGE, fontsize=9.5)
    ax.add_patch(Rectangle((0.045, 0.095), 0.91, 0.78, fill=False, ec="#CDD6DE", lw=1.0, ls="--"))
    ax.text(0.50, 0.08, "残差连接与层归一化包裹各子层；自注意力负责全局关系，FFN负责逐位置非线性重组。", ha="center", fontsize=9.5, color=MUTED)
    save(
        fig,
        OUT / "chapter3" / "transformer",
        "transformer_full_architecture",
        """
# Sources
- Vaswani et al., 2017 (`vaswani2017attention`), Figure 1: encoder-decoder Transformer with multi-head attention and feed-forward sublayers.
- Chapter 3 Transformer section in this thesis: SER often uses an encoder-centered variant with pooling/classification head.
""",
    )


def main():
    ser_evolution()
    acoustic_features()
    ft_time_frequency()
    ft_vs_stft()
    spectrogram_forms()
    mel_vs_logmel()
    mlp_diagram()
    cnn_diagram()
    rnn_diagram()
    attention_diagram()
    transformer_architecture()


if __name__ == "__main__":
    main()
