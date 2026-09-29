"""Box plot comparing chunk accuracy across architectures, one box per dataset."""

import matplotlib.pyplot as plt

# ── Data ──────────────────────────────────────────────────────────────────────
# Per-dataset chunk accuracy of the binary models (see paper/NEW_NUMBERS.md).
# Order: ATV4, DeformerS (8ch), LGGNetS, TSceptionS, CNN-AttnS, CNN-Attn, DeformerS (in-ear)
acc = {
    "CANE\n(Anxiety+Depression)": [72.24, 66.68, 63.29, 60.66, 66.72, 66.29, 59.90],
    "MDD\n(Depression)": [88.96, 88.10, 86.47, 84.19, 88.97, 88.24, 88.06],
    "SAD\n(Anxiety)": [71.40, 65.58, 62.22, 58.65, 64.64, 61.74, 61.83],
}

FONT_SIZE = 17


dataset_keys = list(acc.keys())
n_datasets = len(dataset_keys)

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
ax.set_xticklabels(dataset_keys, fontsize=FONT_SIZE)
ax.set_ylabel("Chunk Accuracy (%)", fontsize=FONT_SIZE)
ax.tick_params(axis="y", labelsize=FONT_SIZE)
ax.set_ylim(55, 95)
ax.set_yticks(range(55, 96, 5))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out = "docs/per_dataset_comparison.png"
plt.savefig(out, dpi=450, bbox_inches="tight")
print(f"Saved → {out}")
