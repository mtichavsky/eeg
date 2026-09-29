import sys
import matplotlib.pyplot as plt

new = sys.argv[1] == "new"
# Order: ATV4, DeformerS (8ch), LGGNetS, TSceptionS, CNN-AttnS, CNN-Attn, DeformerS (in-ear)
acc = {
    "CANE\n(Anxiety+Depression)": [72.24 if new else 69.01, 66.68, 63.29, 60.66, 66.72, 66.29, 59.90],
    "MDD\n(Depression)": [88.96 if new else 87.95, 88.10, 86.47, 84.19, 88.97, 88.24, 88.06],
    "SAD\n(Anxiety)": [71.40 if new else 68.93, 65.58, 62.22, 58.65, 64.64, 61.74, 61.83],
}
keys = list(acc)
fig, ax = plt.subplots(figsize=(9, 5.5))
bp = ax.boxplot([acc[k] for k in keys], positions=range(3), widths=0.45, patch_artist=True,
    medianprops=dict(color="black", linewidth=2), whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2), flierprops=dict(marker="", linestyle="none"), boxprops=dict(linewidth=1.2))
for patch, color in zip(bp["boxes"], ["#AEC6E8", "#98D4A3", "#FFB347"]):
    patch.set_facecolor(color); patch.set_alpha(0.65)
ax.set_xticks(range(3)); ax.set_xticklabels(keys, fontsize=11)
ax.set_ylabel("Chunk Accuracy (%)", fontsize=11)
ax.set_title("Per-Dataset Chunk Accuracy across Binary Classification Models", fontsize=12, fontweight="bold", pad=10)
ax.set_ylim(55, 95)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(sys.argv[2], dpi=450, bbox_inches="tight")
print("saved", sys.argv[2])
