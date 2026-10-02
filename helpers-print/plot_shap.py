"""SHAP figures for the paper: band importance, channel x band heatmap, in-ear beeswarm.

Reads what ``python main.py explain`` writes to ``<run>/explain/``:

* ``shap_bands.png``: mean |phi| per band, per dataset, for the in-ear and the 8-channel
  model (error bars: std over folds), with a width-normalised (per-bin) second row,
* ``shap_channel_band.png``: signed mean phi of the 48-player channel x band grid, hatched
  where the Monte Carlo standard error is at least half the value,
* ``shap_beeswarm_inear.png``: per-chunk phi of each band for the in-ear model, coloured by the
  chunk's log power in that band.

Figures whose input JSON/npz is missing are skipped with a message.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

FONT_SIZE = 13
# Reference categorical slots 1-3 (validated all-pairs), one per dataset, fixed order.
DATASET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
# Diverging blue <-> red around a neutral gray.
DIVERGING = LinearSegmentedColormap.from_list("shap", ["#2a78d6", "#f0efec", "#e34948"])
SEQUENTIAL = LinearSegmentedColormap.from_list("power", ["#cde2fb", "#2a78d6", "#0d366b"])
BAND_LABELS = {
    "sub_delta": "<1 Hz",
    "delta": "δ",
    "theta": "θ",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
}


def load(path: Path) -> dict[str, Any] | None:
    """Read a results JSON, or return ``None`` (with a message) when it does not exist."""
    if not path.is_file():
        logger.warning(f"Missing {path}, skipping")
        return None
    return json.loads(path.read_text())


def band_panel(ax: plt.Axes, result: dict[str, Any], key: str, title: str) -> None:
    """Grouped bars: one group per band, one bar per dataset, std over folds as error bars."""
    players = result["players"]
    source = result["importance"] if key == "mean_abs_phi" else result["importance"]["per_bin"]
    datasets = list(result["datasets"])
    width = 0.8 / len(datasets)
    x = np.arange(len(players))
    for i, dataset in enumerate(datasets):
        entry = source["by_dataset"].get(dataset)
        if entry is None:
            continue
        summary = entry["mean_abs_phi"] if key == "mean_abs_phi" else entry
        if summary["mean"] is None:
            continue
        ax.bar(
            x + (i - (len(datasets) - 1) / 2) * width,
            [summary["mean"][p] for p in players],
            width * 0.9,
            yerr=[summary["std"][p] for p in players],
            color=DATASET_COLORS[i],
            error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#52514e"},
            label=result["datasets"][dataset],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([BAND_LABELS.get(p, p) for p in players], fontsize=FONT_SIZE)
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.tick_params(axis="y", labelsize=FONT_SIZE - 2)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_bands(inear: dict[str, Any] | None, eight: dict[str, Any] | None, out: Path) -> None:
    """Figure 1: band importance for both models, total and per bin."""
    runs = [(r, t) for r, t in ((inear, "CNN-AttnS, in-ear"), (eight, "AllTransformerV4")) if r]
    if not runs:
        return
    fig, axes = plt.subplots(2, len(runs), figsize=(6 * len(runs), 7.5), squeeze=False)
    for col, (result, title) in enumerate(runs):
        band_panel(axes[0, col], result, "mean_abs_phi", title)
        band_panel(axes[1, col], result, "per_bin", "")
        axes[0, col].legend(fontsize=FONT_SIZE - 3, frameon=False)
    axes[0, 0].set_ylabel("mean |φ| (logit)", fontsize=FONT_SIZE)
    axes[1, 0].set_ylabel("mean |φ| per bin", fontsize=FONT_SIZE)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


def plot_grid(result: dict[str, Any] | None, out: Path) -> None:
    """Figure 2: signed mean phi of the channel x band grid, uncertain cells hatched."""
    if result is None:
        return
    players = result["players"]
    channels = list(dict.fromkeys(p.split(":")[0] for p in players))
    bands = list(dict.fromkeys(p.split(":")[1] for p in players))
    mean = result["importance"]["all"]["mean_phi"]["mean"]
    values = np.array([[mean[f"{c}:{b}"] for b in bands] for c in channels])
    se_map = result.get("monte_carlo_se", {}).get("mean_phi_se")

    limit = float(np.abs(values).max()) or 1.0
    fig, ax = plt.subplots(figsize=(7, 6.5))
    image = ax.imshow(values, cmap=DIVERGING, norm=TwoSlopeNorm(0.0, -limit, limit), aspect="auto")
    for i, channel in enumerate(channels):
        for j, band in enumerate(bands):
            value = values[i, j]
            ax.text(
                j,
                i,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=FONT_SIZE - 4,
                color="#0b0b0b",
            )
            if se_map is not None and se_map[f"{channel}:{band}"] >= abs(value) / 2:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="#52514e",
                        linewidth=0,
                    )
                )
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([BAND_LABELS.get(b, b) for b in bands], fontsize=FONT_SIZE)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels, fontsize=FONT_SIZE)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("mean φ (toward pathological)", fontsize=FONT_SIZE - 1)
    if se_map is not None:
        ax.legend(
            handles=[
                Patch(facecolor="white", edgecolor="#52514e", hatch="///", label="SE ≥ |φ|/2")
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            frameon=False,
            fontsize=FONT_SIZE - 3,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


def plot_beeswarm(chunks_path: Path, out: Path) -> None:
    """Figure 3: per-chunk phi per band, coloured by that band's log power (rank within band)."""
    if not chunks_path.is_file():
        logger.warning(f"Missing {chunks_path}, skipping")
        return
    data = np.load(chunks_path)
    players, phi, power = list(data["players"]), data["phi"], data["log_power"]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(8, 0.9 * len(players) + 1.5))
    for row, _ in enumerate(players):
        # Colour by rank within the band, so bands with different power ranges share a scale.
        ranks = power[:, row].argsort().argsort() / max(1, len(power) - 1)
        jitter = rng.uniform(-0.3, 0.3, len(phi))
        scatter = ax.scatter(
            phi[:, row], row + jitter, c=ranks, cmap=SEQUENTIAL, s=8, vmin=0, vmax=1, linewidths=0
        )
    ax.axvline(0, color="#52514e", linewidth=0.8)
    ax.set_yticks(range(len(players)))
    ax.set_yticklabels([BAND_LABELS.get(p, p) for p in players], fontsize=FONT_SIZE)
    ax.invert_yaxis()
    ax.set_xlabel("φ (logit, toward pathological)", fontsize=FONT_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    colorbar = fig.colorbar(scatter, ax=ax, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["low", "high"])
    colorbar.set_label("band log power (rank)", fontsize=FONT_SIZE - 1)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--inear-run", type=Path, default=Path("experiments/all_041_inear_ec+eo_f70")
    )
    parser.add_argument("--all-run", type=Path, default=Path("experiments/all_041_all_ec+eo_f70"))
    parser.add_argument("--out-dir", type=Path, default=Path("../thesis-text/obrazky-figures"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_bands(
        load(args.inear_run / "explain" / "shap_bands.json"),
        load(args.all_run / "explain" / "shap_bands.json"),
        args.out_dir / "shap_bands.png",
    )
    plot_grid(
        load(args.all_run / "explain" / "shap_grid.json"), args.out_dir / "shap_channel_band.png"
    )
    plot_beeswarm(
        args.inear_run / "explain" / "shap_bands_chunks.npz",
        args.out_dir / "shap_beeswarm_inear.png",
    )


if __name__ == "__main__":
    main()
