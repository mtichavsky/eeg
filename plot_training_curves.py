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


def parse_log_file(log_path: Path) -> Dict[int, Dict[str, any]]:
    """
    Parse training log file and extract loss values per fold.

    :param Path log_path: Path to the training log file
    :return: Dictionary mapping fold number to train/eval losses and early stopping info
    :rtype: Dict[int, Dict[str, any]]
    """
    # Pattern for TRAIN CHUNK lines
    train_pattern = re.compile(
        r"TRAIN CHUNK \| Fold (\d+) \| Epoch (\d+)/\d+ \| Loss: ([\d.]+)"
    )

    # Pattern for EVAL CHUNK lines
    eval_pattern = re.compile(
        r"EVAL CHUNK \| Fold (\d+) \| Epoch (\d+)/\d+ \| Loss: ([\d.]+)"
    )

    # Pattern for early stopping lines
    early_stop_pattern = re.compile(
        r"Early stopping triggered.*Best score: ([\d.]+) at epoch (\d+)"
    )

    # Dictionary to store data: fold_num -> {'train': [(epoch, loss)], 'eval': [(epoch, loss)], 'best_epoch': int, 'best_loss': float}
    data: Dict[int, Dict[str, any]] = {}
    current_fold = None

    with open(log_path, "r") as f:
        for line in f:
            # Check for fold start to track current fold
            fold_start_match = re.search(r"FOLD (\d+)/\d+", line)
            if fold_start_match:
                current_fold = int(fold_start_match.group(1))
                if current_fold not in data:
                    data[current_fold] = {"train": [], "eval": [], "best_epoch": None, "best_loss": None}
                continue

            # Try matching TRAIN CHUNK
            train_match = train_pattern.search(line)
            if train_match:
                fold_num = int(train_match.group(1))
                epoch = int(train_match.group(2))
                loss = float(train_match.group(3))

                if fold_num not in data:
                    data[fold_num] = {"train": [], "eval": [], "best_epoch": None, "best_loss": None}

                data[fold_num]["train"].append((epoch, loss))
                continue

            # Try matching EVAL CHUNK
            eval_match = eval_pattern.search(line)
            if eval_match:
                fold_num = int(eval_match.group(1))
                epoch = int(eval_match.group(2))
                loss = float(eval_match.group(3))

                if fold_num not in data:
                    data[fold_num] = {"train": [], "eval": [], "best_epoch": None, "best_loss": None}

                data[fold_num]["eval"].append((epoch, loss))
                continue

            # Try matching early stopping
            early_stop_match = early_stop_pattern.search(line)
            if early_stop_match and current_fold is not None:
                _best_score = float(early_stop_match.group(1))
                best_epoch = int(early_stop_match.group(2))

                if current_fold in data:
                    data[current_fold]["best_epoch"] = best_epoch
                    # Find the actual eval loss at the best epoch
                    for epoch, loss in data[current_fold]["eval"]:
                        if epoch == best_epoch:
                            data[current_fold]["best_loss"] = loss
                            break

    return data


def plot_fold_losses(
    fold_num: int,
    train_data: List[Tuple[int, float]],
    eval_data: List[Tuple[int, float]],
    output_dir: Path,
    best_epoch: int = None,
    best_loss: float = None,
) -> None:
    """
    Plot training and evaluation loss curves for a single fold.

    :param int fold_num: Fold number
    :param List[Tuple[int, float]] train_data: List of (epoch, loss) tuples for training
    :param List[Tuple[int, float]] eval_data: List of (epoch, loss) tuples for evaluation
    :param Path output_dir: Directory to save the plot
    :param int best_epoch: Best epoch from early stopping (optional)
    :param float best_loss: Best loss value at early stopping (optional)
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

    # Add vertical line at best model epoch if available
    if best_epoch is not None:
        plt.axvline(
            x=best_epoch,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Best Model (epoch {best_epoch})",
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
    data: Dict[int, Dict[str, any]], output_dir: Path
) -> None:
    """
    Plot all folds' loss curves in a single figure with subplots.

    :param Dict[int, Dict[str, any]] data: Parsed fold data
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
        best_epoch = fold_data.get("best_epoch")
        best_loss = fold_data.get("best_loss")

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

        # Add vertical line at best model epoch if available
        if best_epoch is not None:
            ax.axvline(
                x=best_epoch,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"Best (ep {best_epoch})",
            )

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_title(f"Fold {fold_num}", fontsize=11)
        ax.legend(fontsize=8)
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
            fold_num,
            fold_data["train"],
            fold_data["eval"],
            output_dir,
            fold_data.get("best_epoch"),
            fold_data.get("best_loss"),
        )

    # Plot all folds combined
    print("\nGenerating combined plot for all folds...")
    plot_all_folds_combined(data, output_dir)

    print(f"\nAll plots saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()