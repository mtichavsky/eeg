"""Plot confusion matrices for AllTransformerV4 vs CNNAttn (binary and 4-class)."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ── Data ──────────────────────────────────────────────────────────────────────
bin_labels = ["Healthy", "Pathological"]

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



import sys
new = sys.argv[1] == "new"
bin_cm_v4 = np.array([[2251, 1029], [750, 4374]]) if new else np.array([[2234, 1046], [921, 4203]])
bin_cm_attn = np.array([[2185, 816], [803, 3265]])
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
plot_cm(ax1, bin_cm_v4, bin_labels, "AllTransformerV4\n(all channels)", annot_size=11)
plot_cm(ax2, bin_cm_attn, bin_labels, "CNNAttnS\n(in-ear channel)", annot_size=11)
add_colorbar(fig1, [ax1, ax2])
fig1.savefig(sys.argv[2], bbox_inches="tight", dpi=600)
print("saved", sys.argv[2])
