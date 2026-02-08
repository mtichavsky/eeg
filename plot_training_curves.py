#!/usr/bin/env python3
"""
Plot training and evaluation loss curves for each fold from training logs.

Usage:
    python plot_training_curves.py <log_file_path>

Example:
    python plot_training_curves.py checkpoints_fp1_003_mdd_mqm/training.log
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


class TrainingCurvesError(Exception):
    """Base exception for training curves generation errors."""

    pass


class LogFileNotFoundError(TrainingCurvesError):
    """Raised when the log file does not exist."""

    pass


class NoTrainingDataError(TrainingCurvesError):
    """Raised when no training data is found in the log file."""

    pass


def _ensure_fold(data: Dict[int, Dict[str, Any]], fold_num: int) -> None:
    """Initialise the data entry for a fold if not already present.

    :param Dict[int, Dict[str, Any]] data: Parsed fold data accumulator.
    :param int fold_num: Fold number to initialise.
    """
    if fold_num not in data:
        data[fold_num] = {"train": [], "eval": [], "best_epoch": None, "best_loss": None}


def _process_json_record(record: dict, data: Dict[int, Dict[str, Any]]) -> None:
    """Extract train/eval data from a single JSON metrics record.

    :param dict record: Parsed JSON object with at least ``phase``, ``fold``, ``epoch``, ``loss``.
    :param Dict[int, Dict[str, Any]] data: Parsed fold data accumulator (mutated in place).
    """
    fold = record["fold"]
    epoch = record["epoch"]
    loss = record["loss"]
    _ensure_fold(data, fold)

    if record["phase"] == "train":
        data[fold]["train"].append((epoch, loss))
    elif record["phase"] == "eval":
        acc = record["chunk"]["acc"]
        data[fold]["eval"].append((epoch, loss, acc))


def parse_log_file(log_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Parse training log file and extract loss values per fold.

    Supports both JSON metrics records (new format) and legacy text lines,
    falling back to regex when a line is not valid JSON.

    :param Path log_path: Path to the training log file
    :return: Dictionary mapping fold number to train/eval losses and early stopping info
    :rtype: Dict[int, Dict[str, any]]
    """
    # Legacy patterns (used as fallback for old log files)
    train_pattern = re.compile(r"TRAIN CHUNK \| Fold (\d+) \| Epoch (\d+)/\d+ \| Loss: ([\d.]+)")
    eval_pattern = re.compile(
        r"EVAL CHUNK \| Fold (\d+) \| Epoch (\d+)/\d+ \| Loss: ([\d.]+) \| Acc: ([\d.]+)"
    )
    early_stop_pattern = re.compile(
        r"Early stopping triggered.*Best score: ([\d.]+) at epoch (\d+)"
    )

    # fold_num -> {'train': [(epoch, loss)], 'eval': [(epoch, loss, acc)],
    #              'best_epoch': int | None, 'best_loss': float | None}
    data: Dict[int, Dict[str, Any]] = {}
    current_fold = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            # --- JSON record (new format) -----------------------------------------
            if line.startswith("{"):
                try:
                    record = json.loads(line)
                    _process_json_record(record, data)
                    continue
                except (json.JSONDecodeError, KeyError):
                    pass  # Fall through to legacy regex

            # --- Legacy text format ------------------------------------------------
            # Track current fold for early-stopping association
            fold_start_match = re.search(r"FOLD (\d+)/\d+", line)
            if fold_start_match:
                current_fold = int(fold_start_match.group(1))
                _ensure_fold(data, current_fold)
                continue

            train_match = train_pattern.search(line)
            if train_match:
                fold_num = int(train_match.group(1))
                _ensure_fold(data, fold_num)
                data[fold_num]["train"].append(
                    (int(train_match.group(2)), float(train_match.group(3)))
                )
                continue

            eval_match = eval_pattern.search(line)
            if eval_match:
                fold_num = int(eval_match.group(1))
                _ensure_fold(data, fold_num)
                data[fold_num]["eval"].append(
                    (
                        int(eval_match.group(2)),
                        float(eval_match.group(3)),
                        float(eval_match.group(4)),
                    )
                )
                continue

            early_stop_match = early_stop_pattern.search(line)
            if early_stop_match and current_fold is not None:
                best_epoch = int(early_stop_match.group(2))
                if current_fold in data:
                    data[current_fold]["best_epoch"] = best_epoch
                    for epoch, loss, acc in data[current_fold]["eval"]:
                        if epoch == best_epoch:
                            data[current_fold]["best_loss"] = loss
                            break

    return data


def plot_axes(
    ax1,
    train_epochs,
    train_losses,
    eval_epochs,
    eval_losses,
    eval_acc,
    best_epoch=None,
    fontsize_label=12,
    fontsize_legend=11,
    linewidth=2,
    markersize=4,
):
    """
    Plot training/evaluation loss and accuracy on dual y-axes.

    :param ax1: Primary matplotlib axis for loss
    :param train_epochs: List of training epoch numbers
    :param train_losses: List of training loss values
    :param eval_epochs: List of evaluation epoch numbers
    :param eval_losses: List of evaluation loss values
    :param eval_acc: List of evaluation accuracy values
    :param best_epoch: Best epoch from early stopping (optional)
    :param int fontsize_label: Font size for axis labels
    :param int fontsize_legend: Font size for legend
    :param float linewidth: Line width for plots
    :param float markersize: Marker size for plots
    :return: Tuple of (ax1, ax2) - primary and secondary axes
    :rtype: Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
    """
    ax1.set_xlabel("Epoch", fontsize=fontsize_label)
    ax1.set_ylabel("Loss", fontsize=fontsize_label)
    ax1.plot(
        train_epochs, train_losses, label="Train Loss", linewidth=linewidth, alpha=0.8, color="blue"
    )
    ax1.plot(
        eval_epochs,
        eval_losses,
        label="Eval Loss",
        linewidth=linewidth,
        alpha=0.8,
        color="orange",
    )
    ax1.grid(True, alpha=0.3)

    # Create secondary y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", fontsize=fontsize_label)
    ax2.plot(
        eval_epochs,
        eval_acc,
        label="Eval Accuracy",
        linewidth=linewidth,
        alpha=0.8,
        color="gray",
    )
    ax2.set_ylim(0.25, 1)

    # Add vertical line at best model epoch if available
    if best_epoch is not None:
        ax1.axvline(
            x=best_epoch,
            color="red",
            linestyle="--",
            linewidth=linewidth,
            alpha=0.7,
            label=f"Best Model (epoch {best_epoch})",
        )

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=fontsize_legend, loc="best")

    return ax1, ax2


def plot_fold_losses(
    fold_num: int,
    train_data: List[Tuple[int, float]],
    eval_data: List[Tuple[int, float]],
    output_dir: Path,
    best_epoch: int | None = None,
    best_loss: float | None = None,
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
    eval_epochs, eval_losses, eval_acc = zip(*eval_data) if eval_data else ([], [], [])

    # Create figure with primary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot using shared function
    plot_axes(
        ax1,
        train_epochs,
        train_losses,
        eval_epochs,
        eval_losses,
        eval_acc,
        best_epoch=best_epoch,
    )

    # Title
    ax1.set_title(f"Training and Evaluation Loss - Fold {fold_num}", fontsize=14)

    # Tight layout and save
    plt.tight_layout()
    output_path = output_dir / f"fold_{fold_num}_loss_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot for fold {fold_num} to {output_path}")


def plot_all_folds_combined(data: Dict[int, Dict[str, Any]], output_dir: Path) -> None:
    """
    Plot all folds' loss curves in a single figure with subplots.

    :param Dict[int, Dict[str, Any]] data: Parsed fold data
    :param Path output_dir: Directory to save the plot
    :return: None
    :rtype: None
    """
    num_folds = len(data)
    if num_folds == 0:
        print("No data to plot!")
        return

    # Determine grid layout (e.g., 2 rows x 5 cols for 10 folds)
    if num_folds >= 10:
        ncols = 5
    else:
        ncols = int(num_folds / 2 + num_folds % 2)
    nrows = (num_folds + ncols - 1) // ncols  # Ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten() if num_folds > 1 else [axes]

    for idx, (fold_num, fold_data) in enumerate(sorted(data.items())):
        ax = axes[idx]

        train_data = fold_data["train"]
        eval_data = fold_data["eval"]
        best_epoch = fold_data.get("best_epoch")

        # Extract epochs, losses, and accuracy
        train_epochs, train_losses = zip(*train_data) if train_data else ([], [])
        eval_epochs, eval_losses, eval_acc = zip(*eval_data) if eval_data else ([], [], [])

        # Plot using shared function with smaller fonts for subplot grid
        plot_axes(
            ax,
            train_epochs,
            train_losses,
            eval_epochs,
            eval_losses,
            eval_acc,
            best_epoch=best_epoch,
            fontsize_label=10,
            fontsize_legend=8,
            linewidth=1.5,
            markersize=3,
        )

        ax.set_title(f"Fold {fold_num}", fontsize=11)

    # Hide unused subplots
    for idx in range(num_folds, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    # Save combined figure
    output_path = output_dir / "all_folds_combined_loss_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved combined plot for all folds to {output_path}")


def generate_training_curves(log_path: Path) -> Path:
    """
    Generate training curve plots from a training log file.

    This function parses the training log, extracts loss and accuracy metrics,
    and generates both individual fold plots and a combined overview plot.

    :param Path log_path: Path to the training log file.
    :return: Path to the output directory containing the plots.
    :rtype: Path
    :raises LogFileNotFoundError: If the log file does not exist.
    :raises NoTrainingDataError: If no training data is found in the log file.
    """
    if not log_path.exists():
        raise LogFileNotFoundError(f"Log file not found: {log_path}")

    print(f"Parsing log file: {log_path}")
    data = parse_log_file(log_path)

    if not data:
        raise NoTrainingDataError("No training/evaluation loss data found in the log file")

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
    return output_dir


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python plot_training_curves.py <log_file_path>")
        print(
            "Example: python plot_training_curves.py "
            "checkpoints_fp1_003_mdd_mqm/training_ec_Fp1_noica_20251114_160354.log"
        )
        sys.exit(1)

    log_path = Path(sys.argv[1])

    try:
        generate_training_curves(log_path)
        print("Done!")
    except LogFileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except NoTrainingDataError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
