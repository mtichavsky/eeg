"""Box plot comparing chunk accuracy across architectures, one box per dataset."""

import matplotlib.pyplot as plt
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# Aggregate per-dataset chunk accuracy from results.txt (6 models × 3 datasets)

architectures = [
    "AllTransformerV4",
    "CNN-Attn-All",
    "CNN-Cat-LSTM",
    "DeformerS",
    "LGGNetS",
    "TSceptionS\n(4 s)",
]
categories = ["Spectrogram"] * 3 + ["Raw-EEG"] * 3

# [model_idx] for each dataset
acc = {
    "CANE\n(Anxiety+Depression)": [74.62, 72.75, 71.83, 67.73, 66.96, 62.81],
    "MDD\n(Depression)":          [90.44, 88.44, 85.51, 87.42, 88.41, 84.84],
    "SAD\n(Anxiety)":             [71.60, 70.90, 69.14, 65.61, 57.58, 57.19],
}


dataset_keys = list(acc.keys())
n_datasets = len(dataset_keys)
n_arch = len(architectures)

fig, ax = plt.subplots(figsize=(9, 5.5))

bp = ax.boxplot(
    [acc[k] for k in dataset_keys],
    positions=range(n_datasets),
    widths=0.45,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="", linestyle="none"),
    boxprops=dict(linewidth=1.2),
)

box_colors = ["#AEC6E8", "#98D4A3", "#FFB347"]
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xticks(range(n_datasets))
ax.set_xticklabels(dataset_keys, fontsize=11)
ax.set_ylabel("Chunk Accuracy (%)", fontsize=11)
ax.set_title("Per-Dataset Chunk Accuracy across 8-channel Binary Classification", fontsize=12, fontweight="bold", pad=10)
ax.set_ylim(55, 95)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out = "docs/per_dataset_comparison.png"
plt.savefig(out, dpi=450, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
