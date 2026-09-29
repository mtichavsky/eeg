"""Plot confusion matrices for AllTransformerV4 vs CNNAttn (binary and 4-class)."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ── Data ──────────────────────────────────────────────────────────────────────
bin_cm_v4 = np.array([[2251, 1029], [750, 4374]])
bin_cm_attn = np.array([[2185, 816], [803, 3265]])
bin_labels = ["Healthy", "Pathological"]

four_cm_v4 = np.array(
    [
        [1507, 448, 341, 984],
        [96, 790, 0, 759],
        [93, 1, 1718, 61],
        [19, 187, 1, 1399],
    ]
)
four_cm_attn = np.array(
    [
        [1557, 518, 319, 607],
        [50, 955, 0, 504],
        [137, 2, 1669, 64],
        [2, 106, 0, 579],
    ]
)
four_labels = ["Healthy", "Anxiety", "Depression", "Comorbid"]


def plot_cm(
    ax: plt.Axes,
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    annot_size: int = 10,
    font_size: int = 10,
    show_ylabel: bool = True,
    title_size: float | None = None,
) -> None:
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names if show_ylabel else False,
        linewidths=0.5,
        linecolor="white",
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        annot_kws={"size": annot_size},
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=font_size)
    if show_ylabel:
        ax.set_ylabel("True label", fontsize=font_size)
    ax.set_title(title, fontsize=title_size if title_size is not None else font_size + 1, pad=8)
    ax.tick_params(axis="both", labelsize=font_size - 1)


def add_colorbar(
    fig: plt.Figure,
    axes: list[plt.Axes],
    font_size: int = 10,
    tick_size: int | None = None,
    label: str = "Row-normalised proportion",
) -> None:
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(label, fontsize=font_size)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    cbar.ax.tick_params(labelsize=tick_size if tick_size is not None else font_size - 1)


# ── Figure 1: Binary ──────────────────────────────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.08})
plot_cm(ax1, bin_cm_v4, bin_labels, "AllTransformerV4\n(all channels)", annot_size=22, font_size=21, title_size=17.5)
plot_cm(ax2, bin_cm_attn, bin_labels, "CNN-AttnS\n(in-ear channel)", annot_size=22, font_size=21, show_ylabel=False, title_size=17.5)
for ax in (ax1, ax2):
    ax.xaxis.label.set_size(17.5)
    ax.yaxis.label.set_size(17.5)
    ax.tick_params(axis="both", labelsize=17.5)
add_colorbar(fig1, [ax1, ax2], font_size=17.5, tick_size=19, label="Row-normalized proportion")
fig1.savefig("docs/confusion_matrix_binary.png", bbox_inches="tight", dpi=600)
print("Saved confusion_matrix_binary.png")
plt.close(fig1)

# ── Figure 2: 4-class ─────────────────────────────────────────────────────────
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"wspace": 0.05})
plot_cm(ax3, four_cm_v4, four_labels, "AllTransformerV4\n(all channels)", annot_size=16, font_size=19)
plot_cm(ax4, four_cm_attn, four_labels, "CNNAttnS\n(in-ear channel)", annot_size=16, font_size=19, show_ylabel=False)
for ax in (ax3, ax4):
    ax.tick_params(axis="both", labelsize=15)
plt.setp(ax3.get_yticklabels(), rotation=0)
for ax in (ax3, ax4):
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
add_colorbar(fig2, [ax3, ax4], font_size=19)
fig2.savefig("docs/confusion_matrix_4class.png", bbox_inches="tight", dpi=600)
print("Saved confusion_matrix_4class.png")
plt.close(fig2)
