"""Three-panel explainability figure: channel importance, channel x band grid, band profiles."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── Data ──────────────────────────────────────────────────────────────────────
# Unlike the other scripts in this directory, this one reads the JSON that
# `python main.py explain` writes rather than hardcoding numbers. The results are an 8x5 grid
# plus per-fold confidence intervals, which is more than is reasonable to transcribe by hand,
# and the figure has to be regenerated whenever an ablation is re-run.

CHANNEL_ORDER = ["Fp1", "Fp2", "C3", "Cz", "C4", "T7", "T8", "O2/Oz"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

# Region colours, matched to the pastel palette of plot_per_dataset_comparison.py.
REGION_COLORS = {
    "frontal": "#AEC6E8",
    "central": "#98D4A3",
    "temporal": "#FFB347",
    "occipital": "#C9B3E0",
}
CHANNEL_REGION = {
    "Fp1": "frontal",
    "Fp2": "frontal",
    "C3": "central",
    "Cz": "central",
    "C4": "central",
    "T7": "temporal",
    "T8": "temporal",
    "O2/Oz": "occipital",
}


def load(path: Path) -> dict:
    """
    Read one experiment's result JSON.

    :param Path path: File written by ``main.py explain``.
    :return: Parsed results.
    :rtype: dict
    :raises SystemExit: If the file is missing, with a hint about how to produce it.
    """
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run: python main.py explain <run-dir>")
    return json.loads(path.read_text())


def channel_drops(channels: dict) -> tuple[list[float], list[float], list[float]]:
    """
    Per-channel accuracy drop and its bootstrap confidence interval, in canonical order.

    :param dict channels: Contents of ``channels.json``.
    :return: ``(drops, lower_errors, upper_errors)`` in percentage points, ready for
        ``ax.bar(yerr=...)``.
    :rtype: tuple
    """
    drops, lower, upper = [], [], []
    for name in CHANNEL_ORDER:
        entry = channels["ablations"][f"drop_{name}"]
        interval = entry["drop_ci"]
        drops.append(interval["mean"] * 100)
        lower.append((interval["mean"] - interval["lower"]) * 100)
        upper.append((interval["upper"] - interval["mean"]) * 100)
    return drops, lower, upper


def band_grid(bands: dict) -> np.ndarray:
    """
    Channel-by-band accuracy drops as an 8x5 array in percentage points.

    :param dict bands: Contents of ``bands.json`` for the eight-channel run.
    :return: Array indexed ``[channel, band]``.
    :rtype: numpy.ndarray
    """
    baseline = bands["baseline_mean"]
    grid = np.zeros((len(CHANNEL_ORDER), len(BANDS)))
    for row, channel in enumerate(CHANNEL_ORDER):
        for column, band in enumerate(BANDS):
            entry = bands["ablations"].get(f"band_{band}_{channel}")
            grid[row, column] = (baseline - entry["mean"]) * 100 if entry else np.nan
    return grid


def band_profile(bands: dict, channel: str | None = None) -> list[float]:
    """
    Per-band accuracy drop, either pooled over channels or for one channel.

    :param dict bands: Contents of a ``bands.json``.
    :param str channel: Channel name, or ``None`` for the all-channel marginal.
    :return: Drops in percentage points, one per band.
    :rtype: list[float]
    """
    baseline = bands["baseline_mean"]
    suffix = f"_{channel}" if channel else ""
    return [(baseline - bands["ablations"][f"band_{band}{suffix}"]["mean"]) * 100 for band in BANDS]


def main() -> None:
    """Build the figure and write it to the output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eight-channel",
        type=Path,
        required=True,
        help="explain/ directory of the AllTransformerV4 eight-channel run",
    )
    parser.add_argument(
        "--in-ear",
        type=Path,
        required=True,
        help="explain/ directory of the CNN-AttnS in-ear run",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/explainability.png"))
    args = parser.parse_args()

    channels = load(args.eight_channel / "channels.json")
    bands_8ch = load(args.eight_channel / "bands.json")
    bands_inear = load(args.in_ear / "bands.json")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # ── (a) per-channel importance ────────────────────────────────────────────
    ax = axes[0]
    drops, lower, upper = channel_drops(channels)
    colors = [REGION_COLORS[CHANNEL_REGION[c]] for c in CHANNEL_ORDER]
    ax.bar(
        range(len(CHANNEL_ORDER)),
        drops,
        yerr=[lower, upper],
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        capsize=3,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(CHANNEL_ORDER)))
    ax.set_xticklabels(CHANNEL_ORDER, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Chunk accuracy drop (pp)", fontsize=10)
    ax.set_title("(a) Leave-one-channel-out", fontsize=11, pad=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend below the axes: the tallest bar's position is not known in advance, so an
    # in-axes legend risks covering the result.
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.8)
            for color in REGION_COLORS.values()
        ],
        labels=list(REGION_COLORS),
        fontsize=8,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        columnspacing=1.0,
        handlelength=1.2,
    )

    # ── (b) channel x band grid ───────────────────────────────────────────────
    ax = axes[1]
    sns.heatmap(
        band_grid(bands_8ch),
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 8},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Accuracy drop (pp)"},
        xticklabels=BANDS,
        yticklabels=CHANNEL_ORDER,
    )
    ax.set_title("(b) Channel $\\times$ band occlusion", fontsize=11, pad=8)
    ax.tick_params(labelsize=9)

    # ── (c) in-ear versus the temporal pair it surrogates ─────────────────────
    ax = axes[2]
    positions = np.arange(len(BANDS))
    ax.plot(
        positions,
        band_profile(bands_inear),
        marker="o",
        linewidth=2,
        color="#FF8C00",
        label="CNN-AttnS (in-ear)",
    )
    for channel, style in (("T7", "--"), ("T8", ":")):
        ax.plot(
            positions,
            band_profile(bands_8ch, channel),
            marker="s",
            linestyle=style,
            linewidth=1.6,
            color="#4878A8",
            label=f"AllTransformerV4 ({channel})",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(BANDS, fontsize=9)
    ax.set_ylabel("Chunk accuracy drop (pp)", fontsize=10)
    ax.set_title("(c) In-ear vs. temporal pair", fontsize=11, pad=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=600, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
