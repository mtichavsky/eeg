"""Plot confusion matrices for AllTransformerV4 vs CNNAttn (binary and 4-class)."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ── Data ──────────────────────────────────────────────────────────────────────
bin_cm_v4 = np.array([[2480, 818], [820, 4326]])
bin_cm_attn = np.array([[2114, 905], [620, 3470]])
bin_labels = ["Healthy", "Pathological"]

four_cm_v4 = np.array(
    [
        [1855, 647, 330, 466],
        [209, 927, 58, 451],
        [96, 42, 1710, 47],
        [293, 263, 17, 1033],
    ]
)
four_cm_attn = np.array(
    [
        [1565, 480, 350, 624],
        [60, 815, 30, 604],
        [98, 0, 1735, 61],
        [4, 75, 0, 608],
    ]
)
four_labels = ["Healthy", "Anxiety", "Depression", "Comorbid"]


def plot_cm(
    ax: plt.Axes,
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    annot_size: int = 10,
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
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        annot_kws={"size": annot_size},
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    ax.tick_params(axis="both", labelsize=9)


def add_colorbar(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Row-normalised proportion", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))


# ── Figure 1: Binary ──────────────────────────────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
plot_cm(ax1, bin_cm_v4,   bin_labels, "AllTransformerV4\n(all channels)", annot_size=11)
plot_cm(ax2, bin_cm_attn, bin_labels, "CNNAttn\n(in-ear channel)", annot_size=11)
add_colorbar(fig1, [ax1, ax2])
fig1.savefig("docs/confusion_matrix_binary.png", bbox_inches="tight", dpi=600)
print("Saved confusion_matrix_binary.png")
plt.close(fig1)

# ── Figure 2: 4-class ─────────────────────────────────────────────────────────
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5.2))
plot_cm(ax3, four_cm_v4,   four_labels, "AllTransformerV4\n(all channels)", annot_size=9)
plot_cm(ax4, four_cm_attn, four_labels, "CNNAttn\n(in-ear channel)", annot_size=9)
add_colorbar(fig2, [ax3, ax4])
fig2.savefig("docs/confusion_matrix_4class.png", bbox_inches="tight", dpi=600)
print("Saved confusion_matrix_4class.png")
plt.close(fig2)
