#!/usr/bin/env python3
"""
Plot training and evaluation loss curves for each fold from training logs.

Usage:
    python plot_training_curves.py <log_file_path>

Example:
    python plot_training_curves.py checkpoints_fp1_003_mdd_mqm/training_ec_Fp1_noica_20251114_160354.log
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_log_file(log_path: Path) -> Dict[int, Dict[str, List[Tuple[int, float]]]]:
    """
    Parse training log file and extract loss values per fold.

    :param Path log_path: Path to the training log file
    :return: Dictionary mapping fold number to train/eval losses
    :rtype: Dict[int, Dict[str, List[Tuple[int, float]]]]
    """
    # Pattern for TRAIN CHUNK lines
    train_pattern = re.compile(
        r"TRAIN CHUNK \| Fold (\d+) \| Epoch (\d+)/\d+ \| Loss: ([\d.]+)"
    )

    # Pattern for EVAL CHUNK lines
    eval_pattern = re.compile(
        r"EVAL CHUNK \| Fold (\d+) \| Epoch (\d+)/\d+ \| Loss: ([\d.]+)"
    )

    # Dictionary to store data: fold_num -> {'train': [(epoch, loss)], 'eval': [(epoch, loss)]}
    data: Dict[int, Dict[str, List[Tuple[int, float]]]] = {}

    with open(log_path, "r") as f:
        for line in f:
            # Try matching TRAIN CHUNK
            train_match = train_pattern.search(line)
            if train_match:
                fold_num = int(train_match.group(1))
                epoch = int(train_match.group(2))
                loss = float(train_match.group(3))

                if fold_num not in data:
                    data[fold_num] = {"train": [], "eval": []}

                data[fold_num]["train"].append((epoch, loss))
                continue

            # Try matching EVAL CHUNK
            eval_match = eval_pattern.search(line)
            if eval_match:
                fold_num = int(eval_match.group(1))
                epoch = int(eval_match.group(2))
                loss = float(eval_match.group(3))

                if fold_num not in data:
                    data[fold_num] = {"train": [], "eval": []}

                data[fold_num]["eval"].append((epoch, loss))

    return data


def plot_fold_losses(
    fold_num: int,
    train_data: List[Tuple[int, float]],
    eval_data: List[Tuple[int, float]],
    output_dir: Path,
) -> None:
    """
    Plot training and evaluation loss curves for a single fold.

    :param int fold_num: Fold number
    :param List[Tuple[int, float]] train_data: List of (epoch, loss) tuples for training
    :param List[Tuple[int, float]] eval_data: List of (epoch, loss) tuples for evaluation
    :param Path output_dir: Directory to save the plot
    :return: None
    :rtype: None
    """
    # Extract epochs and losses
    train_epochs, train_losses = zip(*train_data) if train_data else ([], [])
    eval_epochs, eval_losses = zip(*eval_data) if eval_data else ([], [])

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(train_epochs, train_losses, label="Train Loss", linewidth=2, alpha=0.8)

    # Plot evaluation loss
    plt.plot(
        eval_epochs,
        eval_losses,
        label="Eval Loss",
        linewidth=2,
        alpha=0.8,
        marker="o",
        markersize=4,
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Training and Evaluation Loss - Fold {fold_num}", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"fold_{fold_num}_loss_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot for fold {fold_num} to {output_path}")


def plot_all_folds_combined(
    data: Dict[int, Dict[str, List[Tuple[int, float]]]], output_dir: Path
) -> None:
    """
    Plot all folds' loss curves in a single figure with subplots.

    :param Dict[int, Dict[str, List[Tuple[int, float]]]] data: Parsed fold data
    :param Path output_dir: Directory to save the plot
    :return: None
    :rtype: None
    """
    num_folds = len(data)
    if num_folds == 0:
        print("No data to plot!")
        return

    # Determine grid layout (e.g., 2 rows x 5 cols for 10 folds)
    ncols = 5
    nrows = (num_folds + ncols - 1) // ncols  # Ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten() if num_folds > 1 else [axes]

    for idx, (fold_num, fold_data) in enumerate(sorted(data.items())):
        ax = axes[idx]

        train_data = fold_data["train"]
        eval_data = fold_data["eval"]

        # Extract epochs and losses
        train_epochs, train_losses = zip(*train_data) if train_data else ([], [])
        eval_epochs, eval_losses = zip(*eval_data) if eval_data else ([], [])

        # Plot
        ax.plot(train_epochs, train_losses, label="Train", linewidth=1.5, alpha=0.8)
        ax.plot(
            eval_epochs,
            eval_losses,
            label="Eval",
            linewidth=1.5,
            alpha=0.8,
            marker="o",
            markersize=3,
        )

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_title(f"Fold {fold_num}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_folds, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    # Save combined figure
    output_path = output_dir / "all_folds_combined_loss_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved combined plot for all folds to {output_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python plot_training_curves.py <log_file_path>")
        print(
            "Example: python plot_training_curves.py checkpoints_fp1_003_mdd_mqm/training_ec_Fp1_noica_20251114_160354.log"
        )
        sys.exit(1)

    log_path = Path(sys.argv[1])

    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    print(f"Parsing log file: {log_path}")
    data = parse_log_file(log_path)

    if not data:
        print("No training/evaluation loss data found in the log file!")
        sys.exit(1)

    print(f"Found data for {len(data)} folds")

    # Create output directory (same directory as log file)
    output_dir = log_path.parent / "loss_curves"
    output_dir.mkdir(exist_ok=True)

    # Plot individual folds
    print("\nGenerating individual fold plots...")
    for fold_num in sorted(data.keys()):
        fold_data = data[fold_num]
        plot_fold_losses(
            fold_num, fold_data["train"], fold_data["eval"], output_dir
        )

    # Plot all folds combined
    print("\nGenerating combined plot for all folds...")
    plot_all_folds_combined(data, output_dir)

    print(f"\nAll plots saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()