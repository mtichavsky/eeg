import argparse
import logging
import random
import string
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader, Subset

from thesis.dataset import (
    CANEDataset,
    FlattenedSpectrogramDataset,
    MDDDataset,
    SpectrogramDataset,
    collate_spectrograms,
)
from thesis.early_stopping import EarlyStopping
from thesis.model import CNN_LSTM_DepCap

RANDOM_SEED = 42
LOG_FORMAT = "[%(asctime)s %(levelname)s %(module)s.%(funcName)s] %(message)s"
LOG_LEVEL = "INFO"
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)

EXPECTED_SPECTROGRAM_SHAPE = (129, 41)  # Expected spectrogram shape from training


class IllegalPathError(Exception):
    pass


def get_unique_checkpoint_dir(checkpoint_dir: Path, suffix_length: int = 3) -> Path:
    """
    Generate a unique checkpoint directory name by appending a random suffix if it already exists.

    :param Path checkpoint_dir: Desired checkpoint directory path.
    :param int suffix_length: Length of random suffix to append (default: 3).
    :return: Unique directory path (either original or with random suffix).
    :rtype: Path
    """
    if not checkpoint_dir.exists():
        return checkpoint_dir

    # Directory exists, add random suffix
    original_name = checkpoint_dir.name
    parent_dir = checkpoint_dir.parent

    # Generate random suffix
    random_suffix = "".join(random.choices(string.ascii_lowercase, k=suffix_length))
    new_dir = parent_dir / f"{original_name}_{random_suffix}"

    logger.warning(
        f"Checkpoint directory '{checkpoint_dir}' already exists. "
        f"Using '{new_dir}' instead to avoid overwriting."
    )

    return new_dir


def setup_logging(
    checkpoint_dir: Path, condition: str, skip_ica: bool, channel: str | None = None
) -> Path:
    """
    Configure logging to output to both console and a timestamped log file.

    :param Path checkpoint_dir: Directory to save log files.
    :param str condition: EEG condition being trained on.
    :param bool skip_ica: Whether ICA was skipped.
    :param str | None channel: Single channel being used (if applicable).
    :return: Path to the log file.
    :rtype: Path
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    channel_suffix = f"_{channel}" if channel else ""
    ica_suffix = "_noica" if skip_ica else ""
    log_filename = f"training_{condition}{channel_suffix}{ica_suffix}_{timestamp}.log"
    log_path = checkpoint_dir / log_filename

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to console and file: {log_path}")

    return log_path


# TODO check and understand, check with paper too, figure out how you'll be writing
# about this in a thesis
# TODO discuss what we care about, basically eval chapter
def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2
) -> dict[str, float | int | np.ndarray]:
    """
    Compute classification metrics from true and predicted labels.

    For binary classification (num_classes=2): computes accuracy, precision, recall, specificity.
    For multi-class (num_classes>2): computes accuracy and per-class precision/recall.

    :param np.ndarray y_true: True labels.
    :param np.ndarray y_pred: Predicted labels.
    :param int num_classes: Number of classes (2 for binary, 3 for ternary).
    :return: Dictionary with metrics.
    :rtype: dict
    """
    acc = np.mean(y_true == y_pred)
    metrics = {"accuracy": acc}

    if num_classes == 2:
        # Binary classification: compute traditional metrics
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics.update(
                {
                    "precision": precision,
                    "recall": recall,
                    "specificity": specificity,
                    "tp": int(tp),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                }
            )
        else:
            # Handle edge case where only one class is predicted
            metrics.update(
                {
                    "precision": 0.0,
                    "recall": 0.0,
                    "specificity": 0.0,
                    "tp": 0,
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                }
            )
    else:
        # Multi-class: compute per-class precision and recall
        all_labels = list(range(num_classes))
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)

        for class_idx in all_labels:
            # Per-class metrics
            tp = cm[class_idx, class_idx]
            fp = cm[:, class_idx].sum() - tp
            fn = cm[class_idx, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            class_names = ["normal", "mdd", "anxious"]
            class_name = (
                class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
            )

            metrics[f"precision_{class_name}"] = precision
            metrics[f"recall_{class_name}"] = recall

        # Store confusion matrix as well
        metrics["confusion_matrix"] = cm

    return metrics


def aggregate_subject_predictions(
    chunk_preds: np.ndarray, chunk_labels: np.ndarray, subjects: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate chunk-level predictions to subject-level using majority voting.

    For each subject, collects all chunk predictions and computes the most frequent
    prediction (mode) as the final subject-level prediction. This is useful for
    EEG analysis where multiple temporal chunks come from the same subject.

    :param np.ndarray chunk_preds: Predictions for each chunk (0/1).
    :param np.ndarray chunk_labels: Ground truth labels for each chunk (0/1).
    :param list[str] subjects: Subject ID for each chunk (e.g., "H S1 EC", "MDD S2 EO").
    :return: Tuple of (subject_predictions, subject_labels, subject_ids) where each array
             contains one value per unique subject.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]

    Example:
        >>> chunk_preds = np.array([0, 1, 1, 0, 0])
        >>> chunk_labels = np.array([0, 0, 0, 1, 1])
        >>> subjects = ["H S1 EC", "H S1 EC", "H S1 EC", "MDD S2 EO", "MDD S2 EO"]
        >>> subj_preds, subj_labels, subj_ids = aggregate_subject_predictions(
        ...     chunk_preds, chunk_labels, subjects
        ... )
        >>> subj_preds  # H S1 EC gets 1 (majority), MDD S2 EO gets 0 (majority)
        array([1, 0])
        >>> subj_labels  # H S1 EC is 0, MDD S2 EO is 1
        array([0, 1])
    """
    subject_preds_dict = {}
    subject_labels_dict = {}

    # Group predictions by subject
    for pred, label, subject in zip(chunk_preds, chunk_labels, subjects):
        if subject not in subject_preds_dict:
            subject_preds_dict[subject] = []
            subject_labels_dict[subject] = label  # All chunks from same subject have same label
        subject_preds_dict[subject].append(pred)

    # Compute majority vote for each subject
    subject_preds = []
    subject_labels = []
    subject_ids = []
    for subject in subject_preds_dict:
        chunk_preds_for_subject = subject_preds_dict[subject]
        majority_pred = np.bincount(chunk_preds_for_subject).argmax()
        subject_preds.append(majority_pred)
        subject_labels.append(subject_labels_dict[subject])
        subject_ids.append(subject)

    return np.array(subject_preds), np.array(subject_labels), np.array(subject_ids)


# TODO
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 2,
) -> dict[str, float | int | np.ndarray]:
    """
    Train model for one epoch at chunk/spectrogram level.

    DataLoader batches files, then collate_spectrograms flattens to individual spectrograms.
    Each batch contains multiple spectrograms from multiple files.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in dataloader:
        # Unpack batch: collate_spectrograms returns (spectrograms_tensor,
        # labels_tensor, subjects_list)
        xb, yb, _ = batch  # Ignore subjects during training
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        # TODO also this calculation
        running_loss += float(loss.item()) * xb.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.detach().cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = classification_metrics(all_labels, all_preds, num_classes=num_classes)
    # TODO corresponding line
    avg_loss = running_loss / len(dataloader.dataset)
    metrics["loss"] = avg_loss
    return metrics


# TODO
def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 2,
) -> dict[str, dict[str, float | int | np.ndarray] | float]:
    """
    Evaluate model for one epoch, computing both chunk-level and subject-level metrics.

    :param nn.Module model: Model to evaluate.
    :param DataLoader dataloader: Validation data loader (expects Subset of SpectrogramDataset).
    :param nn.Module criterion: Loss function.
    :param torch.device device: Device to evaluate on.
    :param int num_classes: Number of classes (2 or 3).
    :return: Dictionary containing chunk-level and subject-level metrics.
    :rtype: dict
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_subjects = []

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch: SpectrogramDataset returns (spectrogram, label, subject)
            xb, yb, subjects = batch
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            running_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(yb.detach().cpu().numpy())
            all_subjects.extend(subjects)  # subjects is a list of strings from the batch

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Chunk-level metrics
    chunk_metrics = classification_metrics(all_labels, all_preds, num_classes=num_classes)
    # TODO maybe double check this line - so that it's normalized properly
    avg_loss = running_loss / len(dataloader.dataset)

    # Subject-level metrics (majority voting)
    subject_preds, subject_labels, subject_ids = aggregate_subject_predictions(
        all_preds, all_labels, all_subjects
    )
    subject_metrics = classification_metrics(subject_labels, subject_preds, num_classes=num_classes)

    # Separate metrics by condition (EC vs EO)
    condition_metrics = {}
    for condition in ["EC", "EO"]:
        # Filter subjects by condition
        condition_mask = [condition in subj_id for subj_id in subject_ids]
        if any(condition_mask):
            cond_preds = subject_preds[condition_mask]
            cond_labels = subject_labels[condition_mask]
            condition_metrics[condition] = classification_metrics(
                cond_labels, cond_preds, num_classes=num_classes
            )

    return {
        "chunk": chunk_metrics,
        "subject": subject_metrics,
        "condition": condition_metrics,
        "loss": avg_loss,
    }


def format_metrics_for_logging(metrics: dict[str, float | int], num_classes: int) -> str:
    """
    Format metrics dictionary for logging.

    :param dict metrics: Metrics dictionary from classification_metrics().
    :param int num_classes: Number of classes (2 or 3).
    :return: Formatted metrics string.
    :rtype: str
    """
    if num_classes == 2:
        # Binary classification: use traditional metrics
        return (
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"Spec: {metrics['specificity']:.4f}"
        )
    else:
        # Multi-class: show per-class precision and recall
        parts = [f"Acc: {metrics['accuracy']:.4f}"]
        for class_name in ["normal", "mdd", "anxious"]:
            if f"precision_{class_name}" in metrics:
                prec = metrics[f"precision_{class_name}"]
                rec = metrics[f"recall_{class_name}"]
                parts.append(f"{class_name}: P={prec:.3f} R={rec:.3f}")
        return " | ".join(parts)


def train_one_fold(
    fold: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int = 2,
    num_epochs: int = 50,
    save_every: int = 10,
    val_every: int = 2,
    patience: int = 15,
    checkpoint_dir: Path = Path("checkpoints"),
) -> dict:
    """
    Train model for one-fold with comprehensive logging, checkpointing, and early stopping.

    :param int fold: Current fold number (for logging and checkpointing).
    :param nn.Module model: Model to train.
    :param DataLoader train_loader: Training data loader.
    :param DataLoader val_loader: Validation data loader.
    :param nn.Module criterion: Loss function.
    :param torch.optim.Optimizer optimizer: Optimizer.
    :param torch.device device: Device to train on.
    :param int num_epochs: Maximum number of epochs to train.
    :param int save_every: Save checkpoint every N epochs.
    :param int val_every: Validate every N epochs.
    :param int patience: Early stopping patience.
    :param Path checkpoint_dir: Directory to save checkpoints.
    :return: Dictionary with fold results.
    :rtype: dict
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStopping(patience=patience, maximize=True)
    best_combined_acc = 0.0
    corr_chunk_acc = 0.0
    corr_subject_acc = 0.0
    best_epoch = 0

    fold_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "chunk_acc": [],
        "subject_acc": [],
        "epochs": [],
    }

    logger.info(f"Starting training for fold {fold + 1}")

    epoch = 0
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, num_classes)

        metrics_str = format_metrics_for_logging(train_metrics, num_classes)
        logger.info(
            f"TRAIN CHUNK | Fold {fold + 1} | Epoch {epoch:03d}/{num_epochs} | "
            f"Loss: {train_metrics['loss']:.4f} | {metrics_str}"
        )

        fold_history["train_loss"].append(train_metrics["loss"])
        fold_history["train_acc"].append(train_metrics["accuracy"])
        fold_history["epochs"].append(epoch)

        # Validate every val_every epochs
        if epoch % val_every == 0 or epoch == num_epochs:
            eval_metrics = eval_epoch(model, val_loader, criterion, device, num_classes)

            # Extract chunk and subject metrics
            chunk_metrics = eval_metrics["chunk"]
            subject_metrics = eval_metrics["subject"]
            condition_metrics = eval_metrics.get("condition", {})

            chunk_metrics_str = format_metrics_for_logging(chunk_metrics, num_classes)
            subject_metrics_str = format_metrics_for_logging(subject_metrics, num_classes)

            logger.info(
                f"EVAL CHUNK | Fold {fold + 1} | Epoch {epoch:03d}/{num_epochs} | "
                f"Loss: {eval_metrics['loss']:.4f} | {chunk_metrics_str}"
            )
            if num_classes == 2 and "confusion_matrix" not in chunk_metrics:
                # Log confusion matrix for binary classification
                logger.info(
                    f"  Confusion: TP={chunk_metrics['tp']}, TN={chunk_metrics['tn']}, "
                    f"FP={chunk_metrics['fp']}, FN={chunk_metrics['fn']}"
                )

            logger.info(
                f"EVAL SUBJECT | Fold {fold + 1} | Epoch {epoch:03d}/{num_epochs} | "
                f"Loss: {eval_metrics['loss']:.4f} | {subject_metrics_str}"
            )
            if num_classes == 2 and "confusion_matrix" not in subject_metrics:
                # Log confusion matrix for binary classification
                logger.info(
                    f"  Confusion: TP={subject_metrics['tp']}, TN={subject_metrics['tn']}, "
                    f"FP={subject_metrics['fp']}, FN={subject_metrics['fn']}"
                )

            # Log condition-specific metrics if available (combined EC vs EO on one line)
            if "EC" in condition_metrics and "EO" in condition_metrics:
                ec_acc = condition_metrics["EC"]["accuracy"]
                eo_acc = condition_metrics["EO"]["accuracy"]
                logger.info(
                    f"EVAL EC vs EO | Fold {fold + 1} | Epoch {epoch:03d}/{num_epochs} | "
                    f"EO chunk acc: {eo_acc:.4f} | EC chunk acc: {ec_acc:.4f}"
                )

            fold_history["val_loss"].append(eval_metrics["loss"])
            fold_history["chunk_acc"].append(chunk_metrics["accuracy"])
            fold_history["subject_acc"].append(subject_metrics["accuracy"])

            # Combined metric: chunk accuracy weighted by subject accuracy
            # This handles small validation sets better (e.g., 6 subjects in 10-fold CV)
            combined_metric = chunk_metrics["accuracy"] * subject_metrics["accuracy"]

            # Save the best model (based on combined metric)
            if combined_metric > best_combined_acc:
                best_combined_acc = combined_metric
                corr_chunk_acc = chunk_metrics["accuracy"]
                corr_subject_acc = subject_metrics["accuracy"]
                best_epoch = epoch
                best_model_path = checkpoint_dir / f"fold_{fold + 1}_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "combined_acc": best_combined_acc,
                        "val_metrics": eval_metrics,
                    },
                    best_model_path,
                )
                logger.info(
                    f"✓ Saved best model: {best_model_path} "
                    f"(combined={combined_metric:.4f}, chunk={chunk_metrics['accuracy']:.4f}, "
                    f"subject={subject_metrics['accuracy']:.4f})"
                )

            # Early stopping check (based on combined metric)
            if early_stopping(chunk_metrics["accuracy"], epoch):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Periodic checkpoint save
        if epoch % save_every == 0:
            checkpoint_path = checkpoint_dir / f"fold_{fold + 1}_epoch_{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    logger.info(
        f"\nFold {fold + 1} training completed. "
        f"Best Combined Val Acc: {best_combined_acc:.4f} at epoch {best_epoch}"
    )

    return {
        "best_eval_combined_acc": best_combined_acc,
        "eval_chunk_acc": corr_chunk_acc,
        "eval_subject_acc": corr_subject_acc,
        "best_epoch": best_epoch,
        "final_epoch": epoch,
        "history": fold_history,
    }


def prepare_mdd_dataset(
    conditions: list[Literal["EC", "EO", "TASK"]],
    skip_ica: bool,
    channel: str | None,
    rng: np.random.RandomState,
) -> tuple:
    """
    Prepare MDD dataset with spectrograms and split subjects by class.

    :param list[str] conditions: EEG conditions to load (e.g., ["EC", "EO"]).
    :param bool skip_ica: Whether to skip ICA preprocessing.
    :param str | None channel: Single channel to use (e.g., "Fp1").
    :param np.random.RandomState rng: Random number generator for shuffling.
    :return: Tuple of (flat_dataset, normal_subjects, depressed_subjects, anxious_subjects).
    :rtype: tuple
    """
    all_flat_datasets = []
    all_subjects = []

    for condition in conditions:
        mdd_dataset = MDDDataset(condition=condition, skip_ica=skip_ica, channel=channel)
        mdd_spec_dataset = SpectrogramDataset(mdd_dataset, fs=MDDDataset.FS)
        mdd_flat_dataset = FlattenedSpectrogramDataset(mdd_spec_dataset)

        spec_shape = mdd_flat_dataset[0][0].shape[1:]
        assert spec_shape == torch.Size(list(EXPECTED_SPECTROGRAM_SHAPE)), (
            f"Expected (129, 41), got {spec_shape}. "
            f"The neural net was designed using this assumption."
        )

        all_flat_datasets.append(mdd_flat_dataset)
        all_subjects.extend(mdd_dataset.get_sorted_subjects())

    # Combine datasets if multiple conditions
    if len(all_flat_datasets) == 1:
        combined_flat_dataset = all_flat_datasets[0]
    else:
        combined_flat_dataset = ConcatDataset(all_flat_datasets)

    mdd_normal, mdd_depressed, mdd_anxious = split_subjects_into_classes(all_subjects, "mdd", rng)
    return combined_flat_dataset, mdd_normal, mdd_depressed, mdd_anxious


def prepare_cane_dataset(
    conditions: list[Literal["ec", "eo"]],
    channel: str | None,
    rng: np.random.RandomState,
    remap_labels: bool = False,
) -> tuple:
    """
    Prepare CANE dataset with spectrograms and split subjects by class.

    :param list[str] conditions: EEG conditions to load (e.g., ["ec", "eo"]).
    :param str | None channel: Single channel to use (e.g., "Fp1").
    :param np.random.RandomState rng: Random number generator for shuffling.
    :param bool remap_labels: If True, remap labels {0->0, 2->1} for binary classification.
    :return: Tuple of (flat_dataset, normal_subjects, depressed_subjects, anxious_subjects).
    :rtype: tuple
    """
    all_flat_datasets = []
    all_subjects = []

    for condition in conditions:
        cane_dataset = CANEDataset(
            condition=condition, channel=channel, skip_extreme_artifacts=True
        )
        cane_spec_dataset = SpectrogramDataset(
            cane_dataset,
            fs=CANEDataset.FS,
            nperseg=CANEDataset.STFT_NPERSEG,
            noverlap=CANEDataset.STFT_NOVERLAP,
        )

        # Apply label remapping if training on CANE alone (binary classification)
        label_mapping = {0: 0, 2: 1} if remap_labels else None
        cane_flat_dataset = FlattenedSpectrogramDataset(
            cane_spec_dataset, label_mapping=label_mapping
        )

        spec_shape = cane_flat_dataset[0][0].shape[1:]
        assert spec_shape == torch.Size(list(EXPECTED_SPECTROGRAM_SHAPE)), (
            f"Expected (129, 41), got {spec_shape}. "
            f"The neural net was designed using this assumption."
        )

        all_flat_datasets.append(cane_flat_dataset)
        all_subjects.extend(cane_dataset.get_sorted_subjects())

    # Combine datasets if multiple conditions
    if len(all_flat_datasets) == 1:
        cane_combined_flat = all_flat_datasets[0]
    else:
        cane_combined_flat = ConcatDataset(all_flat_datasets)

    cane_normal, cane_depressed, cane_anxious = split_subjects_into_classes(
        all_subjects, "cane", rng
    )
    return cane_combined_flat, cane_normal, cane_depressed, cane_anxious


def split_subjects_into_classes(
    subjects: list[str], dataset_label: str, rng: np.random.RandomState
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    normal: list[tuple[str, str]] = []
    depressed: list[tuple[str, str]] = []
    anxious: list[tuple[str, str]] = []
    for subj in subjects:
        if subj.startswith("H "):
            normal.append((dataset_label, subj))
        elif subj.startswith("MDD "):
            depressed.append((dataset_label, subj))
        elif subj.startswith("AX "):
            anxious.append((dataset_label, subj))

    if len(normal) > 0:
        rng.shuffle(normal)
    if len(depressed) > 0:
        rng.shuffle(depressed)
    if len(anxious) > 0:
        rng.shuffle(anxious)
    return normal, depressed, anxious


def create_balanced_folds(
    mdd_normal: list[tuple[str, str]],
    mdd_depressed: list[tuple[str, str]],
    cane_normal: list[tuple[str, str]],
    cane_depressed: list[tuple[str, str]],
    cane_anxious: list[tuple[str, str]],
    n_folds: int,
) -> tuple[Any, Any, Any]:
    """
    Create balanced folds ensuring both datasets appear in each fold.

    Splits each dataset separately into n_folds, then merges corresponding folds.
    Groups subjects by base ID to keep EC/EO recordings together.

    :param list mdd_normal: List of tuples (dataset_label, subject_id) from MDD normal
        subjects.
    :param list mdd_depressed: List of tuples (dataset_label, subject_id) from MDD
        depressed subjects.
    :param list cane_normal: List of tuples (dataset_label, subject_id) from CANE normal
        subjects.
    :param list cane_depressed: List of tuples (dataset_label, subject_id) from CANE
        depressed subjects.
    :param list cane_anxious: List of tuples (dataset_label, subject_id) from CANE anxious
        subjects.
    :param int n_folds: Number of folds to create.
    :return: Tuple of (normal_folds, anxious_folds, mdd_folds).
    :rtype: tuple
    """
    empty_folds: list[Any] = [np.array([]) for _ in range(n_folds)]

    mdd_normal_folds = split_into_folds(mdd_normal, n_folds)
    mdd_mdd_folds = split_into_folds(mdd_depressed, n_folds)
    cane_normal_folds = split_into_folds(cane_normal, n_folds)
    cane_anxious_folds = split_into_folds(cane_anxious, n_folds)
    cane_mdd_folds = split_into_folds(cane_depressed, n_folds) if cane_depressed else empty_folds

    normal_folds: list[Any] = []
    anxious_folds = cane_anxious_folds
    mdd_folds: list[Any] = []
    for mn, cn, mm, cm in zip(mdd_normal_folds, cane_normal_folds, mdd_mdd_folds, cane_mdd_folds):
        normal_folds.append(np.concatenate([mn, cn]))
        if len(cm) > 0:
            mdd_folds.append(np.concatenate([mm, cm]))
        else:
            mdd_folds.append(mm)

    return normal_folds, anxious_folds, mdd_folds


def get_indices_from_concat_dataset(
    concat_dataset: ConcatDataset, subject_list: list[str]
) -> list[int]:
    """
    Get indices for subjects from a ConcatDataset.

    :param ConcatDataset concat_dataset: ConcatDataset containing multiple
        FlattenedSpectrogramDatasets.
    :param list[str] subject_list: List of subject IDs.
    :return: List of indices in the concatenated dataset.
    :rtype: list[int]
    """
    all_indices = []
    offset = 0
    for dataset in concat_dataset.datasets:
        dataset_indices = dataset.get_indices_for_subjects(subject_list)
        # Adjust indices by offset in concatenated dataset
        all_indices.extend([idx + offset for idx in dataset_indices])
        offset += len(dataset)
    return all_indices


def get_datasets_for_fold(
    fold: int,
    normal_folds: Any,
    mdd_folds: Any,
    anxious_folds: Any,
    dataset_type: str,
    mdd_flat_dataset: Any,
    cane_flat_dataset: Any,
    flat_dataset: Any,
) -> tuple[Any, Any]:
    # Determine train/val subjects for this fold from all classes
    val_subjects_with_dataset: list[Any] = []
    train_subjects_with_dataset: list[Any] = []

    # Collect validation subjects from all classes
    val_subjects_with_dataset.extend(normal_folds[fold])
    val_subjects_with_dataset.extend(mdd_folds[fold])
    val_subjects_with_dataset.extend(anxious_folds[fold])

    # Collect training subjects from all other folds
    for i in range(len(normal_folds)):
        if i != fold:
            train_subjects_with_dataset.extend(normal_folds[i])
            train_subjects_with_dataset.extend(mdd_folds[i])
            train_subjects_with_dataset.extend(anxious_folds[i])

    # Log subject distribution
    val_subjects_clean = [f"{ds}:{subj}" for ds, subj in val_subjects_with_dataset]
    train_subjects_clean = [f"{ds}:{subj}" for ds, subj in train_subjects_with_dataset]
    logger.info(f"Train subjects ({len(train_subjects_clean)}): {train_subjects_clean[:100]}...")
    logger.info(f"Val subjects ({len(val_subjects_clean)}): {val_subjects_clean}")

    # Get indices for train/val based on subjects (chunk-level indices)
    # Need to handle combined dataset differently
    if dataset_type == "both":
        # Split by dataset source
        mdd_train_subjects = [subj for ds, subj in train_subjects_with_dataset if ds == "mdd"]
        mdd_val_subjects = [subj for ds, subj in val_subjects_with_dataset if ds == "mdd"]
        cane_train_subjects = [subj for ds, subj in train_subjects_with_dataset if ds == "cane"]
        cane_val_subjects = [subj for ds, subj in val_subjects_with_dataset if ds == "cane"]

        # Handle case where datasets might be ConcatDataset (multiple conditions)
        if isinstance(mdd_flat_dataset, ConcatDataset):
            mdd_train_indices = get_indices_from_concat_dataset(
                mdd_flat_dataset, mdd_train_subjects
            )
            mdd_val_indices = get_indices_from_concat_dataset(mdd_flat_dataset, mdd_val_subjects)
        else:
            mdd_train_indices = mdd_flat_dataset.get_indices_for_subjects(mdd_train_subjects)
            mdd_val_indices = mdd_flat_dataset.get_indices_for_subjects(mdd_val_subjects)

        if isinstance(cane_flat_dataset, ConcatDataset):
            cane_train_indices = get_indices_from_concat_dataset(
                cane_flat_dataset, cane_train_subjects
            )
            cane_val_indices = get_indices_from_concat_dataset(cane_flat_dataset, cane_val_subjects)
        else:
            cane_train_indices = cane_flat_dataset.get_indices_for_subjects(cane_train_subjects)
            cane_val_indices = cane_flat_dataset.get_indices_for_subjects(cane_val_subjects)

        train_dataset = ConcatDataset(
            [
                Subset(mdd_flat_dataset, mdd_train_indices),
                Subset(cane_flat_dataset, cane_train_indices),
            ]
        )
        val_dataset = ConcatDataset(
            [
                Subset(mdd_flat_dataset, mdd_val_indices),
                Subset(cane_flat_dataset, cane_val_indices),
            ]
        )

        logger.info(
            f"Training chunks: MDD={len(mdd_train_indices)}, CANE={len(cane_train_indices)}, "
            f"Total={len(train_dataset)}"
        )
        logger.info(
            f"Validation chunks: MDD={len(mdd_val_indices)}, CANE={len(cane_val_indices)}, "
            f"Total={len(val_dataset)}"
        )
    else:
        # Single dataset: extract just the subject names
        train_subjects = [subj for ds, subj in train_subjects_with_dataset]
        val_subjects = [subj for ds, subj in val_subjects_with_dataset]

        # Handle case where dataset might be ConcatDataset (multiple conditions)
        if isinstance(flat_dataset, ConcatDataset):
            train_indices = get_indices_from_concat_dataset(flat_dataset, train_subjects)
            val_indices = get_indices_from_concat_dataset(flat_dataset, val_subjects)
        else:
            train_indices = flat_dataset.get_indices_for_subjects(train_subjects)
            val_indices = flat_dataset.get_indices_for_subjects(val_subjects)

        logger.info(f"Training chunks: {len(train_indices)}")
        logger.info(f"Validation chunks: {len(val_indices)}")

        # Create Subset datasets (reusing precomputed spectrograms!)
        train_dataset = Subset(flat_dataset, train_indices)
        val_dataset = Subset(flat_dataset, val_indices)

    return train_dataset, val_dataset


def split_into_folds(subjects: list[tuple[str, str]], n_folds: int) -> list[list[tuple[str, str]]]:
    mapping: dict[str, list[tuple[str, str]]] = {}
    for dataset, subject in subjects:
        subject_parts = subject.split()
        base = " ".join(subject_parts[:-1])
        if base not in mapping:
            mapping[base] = [(dataset, subject)]
        else:
            mapping[base].append((dataset, subject))

    # Use unique base subjects only (no duplicates)
    base_subjects = list(mapping.keys())
    folds = np.array_split(base_subjects, n_folds)
    folds_with_conditions = [[s for base in split for s in mapping[base]] for split in folds]
    return folds_with_conditions


def train_cross_validation(
    dataset_type: Literal["mdd", "cane", "both"] = "mdd",
    condition: str = "EC",
    n_folds: int = 10,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    val_every: int = 2,
    save_every: int = 10,
    patience: int = 15,
    dropout: float = 0.5,
    weight_decay: float = 0.0,
    checkpoint_dir: Path = Path("checkpoints"),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    skip_ica: bool = False,
    channel: str | None = None,
) -> dict:
    """
    Train model using n-fold cross-validation with comprehensive logging and checkpointing.

    :param str dataset_type: Dataset to use ("mdd", "cane", or "both").
    :param str condition: EEG condition to use ("EC", "EO", "TASK", or "EC+EO").
    :param int n_folds: Number of cross-validation folds.
    :param int batch_size: Batch size for training.
    :param int num_epochs: Maximum number of epochs per fold.
    :param float learning_rate: Learning rate for optimizer.
    :param int val_every: Validate every N epochs.
    :param int save_every: Save checkpoint every N epochs.
    :param int patience: Early stopping patience.
    :param float dropout: Dropout.
    :param float weight_decay: Weight decay (L2 penalty).
    :param Path checkpoint_dir: Directory to save checkpoints.
    :param torch.device device: Device to train on.
    :param bool skip_ica: If True, skip ICA artifact removal during preprocessing.
    :param str | None channel: Single channel to use (e.g., "Fp1"). If None, uses all channels.
    :return: Dictionary with cross-validation results. Subject accuracy corresponds to the
           best combined accuracy model.
    :rtype: dict
    """
    # Parse condition argument to handle EC+EO
    condition_upper = condition.upper()
    if "+" in condition_upper:
        conditions = condition_upper.split("+")
    else:
        conditions = [condition_upper]

    logger.info(f"{'=' * 80}")
    logger.info(f"Starting {n_folds}-Fold Cross-Validation Training")
    logger.info(f"Dataset: {dataset_type}, Condition(s): {conditions}")
    logger.info(f"Batch Size: {batch_size}, LR: {learning_rate}")
    logger.info(f"Device: {device}")
    logger.info(f"Skip ICA: {skip_ica}")
    logger.info(f"{'=' * 80}")

    cv_results = {
        "fold_best_eval_combined_acc": [],
        "fold_eval_chunk_acc": [],
        "fold_eval_subject_acc": [],
        "fold_best_epoch": [],
        "fold_final_epoch": [],
    }

    rng = np.random.RandomState(RANDOM_SEED)
    if "TASK" in conditions and dataset_type == "cane":
        raise ValueError("condition TASK not possible for CANE dataset")

    mdd_flat_dataset, cane_flat_dataset, flat_dataset = None, None, None
    num_classes = 2
    if dataset_type == "mdd":
        flat_dataset, normal, depressed, anxious = prepare_mdd_dataset(
            conditions, skip_ica, channel, rng
        )
        logger.info(f"MDD dataset: {len(normal)} normal, {len(depressed)} MDD subjects")
    elif dataset_type == "cane":
        # Convert to lowercase for CANE
        cane_conditions = [c.lower() for c in conditions]
        flat_dataset, normal, depressed, anxious = prepare_cane_dataset(
            cane_conditions, channel, rng, remap_labels=True
        )
        logger.info(f"CANE dataset: {len(normal)} normal, {len(anxious)} anxious subjects")
    elif dataset_type == "both":
        mdd_flat_dataset, mdd_normal, mdd_depressed, mdd_anxious = prepare_mdd_dataset(
            conditions, skip_ica, channel, rng
        )
        cane_conditions = [c.lower() for c in conditions]
        cane_flat_dataset, cane_normal, cane_depressed, cane_anxious = prepare_cane_dataset(
            cane_conditions, channel, rng
        )
        num_classes = 3

    # Create folds for each class separately (stratified)
    if dataset_type == "both":
        # For combined dataset, stratify each dataset separately then merge corresponding folds
        normal_folds, anxious_folds, mdd_folds = create_balanced_folds(
            mdd_normal, mdd_depressed, cane_normal, cane_depressed, cane_anxious, n_folds
        )
        logger.info(
            f"Created {n_folds} balanced folds with subjects from both MDD and CANE datasets"
        )
    else:
        empty_folds = [[] for _ in range(n_folds)]
        normal_folds = split_into_folds(normal, n_folds)
        mdd_folds = split_into_folds(depressed, n_folds) if depressed else empty_folds
        anxious_folds = split_into_folds(anxious, n_folds) if anxious else empty_folds

    spec_shape = EXPECTED_SPECTROGRAM_SHAPE
    for fold in range(n_folds):
        logger.info(f"{'=' * 80}")
        logger.info(f"FOLD {fold + 1}/{n_folds}")
        logger.info(f"{'=' * 80}")

        train_dataset, val_dataset = get_datasets_for_fold(
            fold,
            normal_folds,
            mdd_folds,
            anxious_folds,
            dataset_type,
            mdd_flat_dataset,
            cane_flat_dataset,
            flat_dataset,
        )

        # Create DataLoaders
        # Small dataset fits in RAM → extra workers add overhead, not speed
        # pin_memory speeds up CPU→GPU transfers, enabling Direct Memory Access
        pin_memory = True if device == "cuda" else False
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=collate_spectrograms,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=collate_spectrograms,
        )

        # Initialize model for this fold
        model = CNN_LSTM_DepCap(
            input_shape=spec_shape,
            in_channels=1,
            rnn_type="LSTM",
            rnn_hidden=100,
            dropout=dropout,
            num_classes=num_classes,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Train this fold
        fold_result = train_one_fold(
            fold=fold,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            num_epochs=num_epochs,
            save_every=save_every,
            val_every=val_every,
            patience=patience,
            checkpoint_dir=checkpoint_dir,
        )

        cv_results["fold_best_eval_combined_acc"].append(fold_result["best_eval_combined_acc"])
        cv_results["fold_eval_chunk_acc"].append(fold_result["eval_chunk_acc"])
        cv_results["fold_eval_subject_acc"].append(fold_result["eval_subject_acc"])
        cv_results["fold_best_epoch"].append(fold_result["best_epoch"])
        cv_results["fold_final_epoch"].append(fold_result["final_epoch"])

    # Print final cross-validation results
    logger.info(f"{'=' * 80}")
    logger.info(f"{n_folds}-FOLD CROSS-VALIDATION RESULTS")
    logger.info(f"{'=' * 80}")

    logger.info("\nPer-fold best validation accuracy:")
    for i, combined_acc in enumerate(cv_results["fold_best_eval_combined_acc"]):
        logger.info(
            f"  Fold {i + 1}: {combined_acc:.4f} (epoch {cv_results['fold_best_epoch'][i]})"
        )

    write_results(logger.info, cv_results)
    logger.info(f"{'=' * 80}\n")

    return cv_results


def write_results(writer: Callable[[str], Any], cv_results: dict[str, list[float]]) -> None:
    combined_mean_acc = np.mean(cv_results["fold_best_eval_combined_acc"])
    combined_std_acc = np.std(cv_results["fold_best_eval_combined_acc"])
    chunk_mean_acc = np.mean(cv_results["fold_eval_chunk_acc"])
    chunk_std_acc = np.std(cv_results["fold_eval_chunk_acc"])
    subject_mean_acc = np.mean(cv_results["fold_eval_subject_acc"])
    subject_std_acc = np.std(cv_results["fold_eval_subject_acc"])

    writer(
        f"COMBINED Accuracy Mean: {combined_mean_acc:.4f} ± {combined_std_acc:.4f}, "
        f"Min: {np.min(cv_results['fold_best_eval_combined_acc']):.4f}, "
        f"Max: {np.max(cv_results['fold_best_eval_combined_acc']):.4f}\n"
    )
    writer(
        f"CHUNK Accuracy Mean: {chunk_mean_acc:.4f} ± {chunk_std_acc:.4f}, "
        f"Min: {np.min(cv_results['fold_eval_chunk_acc']):.4f}, "
        f"Max: {np.max(cv_results['fold_eval_chunk_acc']):.4f}\n"
    )
    writer(
        f"SUBJECT Accuracy Mean: {subject_mean_acc:.4f} ± {subject_std_acc:.4f}, "
        f"Min: {np.min(cv_results['fold_eval_subject_acc']):.4f}, "
        f"Max: {np.max(cv_results['fold_eval_subject_acc']):.4f}\n"
    )


def add_preprocessing_args(parser: argparse.ArgumentParser) -> None:
    """
    Add shared preprocessing arguments to a parser.

    :param argparse.ArgumentParser parser: Parser or subparser to add arguments to.
    """
    preproc_group = parser.add_argument_group("Preprocessing options")
    preproc_group.add_argument(
        "--condition",
        type=str,
        default="EC",
        help="EEG condition to use (case-insensitive). Options: EC, EO, TASK (MDD only), "
        "or EC+EO to train on both eyes closed and eyes open together. "
        "When using EC+EO, subject EC and EO recordings stay in the same fold but are "
        "evaluated as separate recordings. "
        "Examples: --condition ec, --condition ec+eo",
    )
    preproc_group.add_argument(
        "--skip-ica",
        action="store_true",
        help="Skip ICA artifact removal (faster but less clean data)",
    )
    preproc_group.add_argument(
        "--channel",
        type=str,
        choices=list(
            set(MDDDataset.CHANNEL_MAPPING.values()) | set(CANEDataset.CHANNEL_MAPPING.values())
        ),
        default="Fp1",
        help="Single channel to use (e.g., --channel Fp1). "
        "When a single channel is selected, ICA is automatically skipped.",
    )


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EEG Classification Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the model using cross-validation")
    add_preprocessing_args(train_parser)
    train_parser.add_argument(
        "--dataset",
        type=str,
        default="mdd",
        choices=["mdd", "cane", "both"],
        help="Dataset to train on: 'mdd' (2 classes: normal, mdd), "
        "'cane' (2-3 classes: normal, anxious[, mdd]), "
        "or 'both' (3 classes: normal, mdd, anxious)",
    )
    train_parser.add_argument(
        "--n-folds", type=int, default=10, help="Number of cross-validation folds"
    )
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum number of epochs per fold"
    )
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )
    train_parser.add_argument("--val-every", type=int, default=2, help="Validate every N epochs")
    train_parser.add_argument(
        "--save-every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    train_parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    train_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to train on",
    )

    run_parser = subparsers.add_parser("run", help="Run inference on a single EDF file")
    add_preprocessing_args(run_parser)
    run_parser.add_argument("model_path", type=str, help="Path to trained model checkpoint (.pth)")
    run_parser.add_argument("edf_file", type=str, help="Path to EDF file to classify")
    run_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on",
    )

    return parser


def train(args: argparse.Namespace) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_dir = get_unique_checkpoint_dir(Path(args.checkpoint_dir))

    # Setup logging to both console and file
    log_path = setup_logging(checkpoint_dir, args.condition, args.skip_ica, args.channel)

    logger.info("Starting EEG Classification Training")
    logger.info("Hyperparameters:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Condition: {args.condition}")
    logger.info(f"  Channel: {args.channel}")
    logger.info(f"  N-Fold CV: {args.n_folds}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.lr}")
    logger.info(f"  Dropout: {args.dropout}")
    logger.info(f"  Weight decay (L2 penalty): {args.weight_decay}")
    logger.info(f"  Max Epochs: {args.epochs}")
    logger.info(f"  Validation Every: {args.val_every} epochs")
    logger.info(f"  Save Checkpoint Every: {args.save_every} epochs")
    logger.info(f"  Early Stopping Patience: {args.patience} epochs")
    logger.info(f"  Skip ICA: {args.skip_ica}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Checkpoint Directory: {checkpoint_dir}")
    logger.info(f"  Log File: {log_path}")

    # Run cross-validation training
    results = train_cross_validation(
        dataset_type=args.dataset,
        condition=args.condition,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        val_every=args.val_every,
        save_every=args.save_every,
        patience=args.patience,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        checkpoint_dir=checkpoint_dir,
        device=device,
        skip_ica=args.skip_ica,
        channel=args.channel,
    )

    # Save final results to file
    results_file = checkpoint_dir / "cv_results.txt"
    with open(results_file, "w") as f:
        f.write(f"{args.n_folds}-Fold Cross-Validation Results\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Condition: {args.condition}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Dropout: {args.dropout}\n")
        f.write(f"Weight Decay (L2 regularization): {args.weight_decay}\n")
        f.write(f"Skip ICA: {args.skip_ica}\n\n")
        f.write("Per-fold best validation accuracy:\n")
        for i in range(len(results["fold_best_eval_combined_acc"])):
            combined_acc = results["fold_best_eval_combined_acc"][i]
            chunk_acc = results["fold_eval_chunk_acc"][i]
            subject_acc = results["fold_eval_subject_acc"][i]
            f.write(
                f"  Fold {i + 1}: combined={combined_acc:.4f}, chunk={chunk_acc:.4f}, "
                f"subject={subject_acc:.4f} (epoch {results['fold_best_epoch'][i]}, "
                f"stopped at epoch {results['fold_final_epoch'][i]})\n"
            )
        write_results(f.write, results)
    logger.info(f"\nResults saved to {results_file}")
    logger.info("Training completed!")


def run(args: argparse.Namespace) -> None:
    """
    Run inference on a single EDF file using a trained model.

    :param argparse.Namespace args: Command-line arguments.
    """
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_path = Path(args.model_path)
    edf_file = Path(args.edf_file)

    # Validate paths
    if not model_path.exists():
        raise IllegalPathError(f"Model checkpoint not found: {model_path}")

    if not edf_file.exists():
        raise IllegalPathError(f"EDF file not found: {edf_file}")

    logger.info("Running Inference")
    logger.info(f"{'=' * 80}")
    logger.info(f"Model: {model_path}")
    logger.info(f"EDF File: {edf_file}")
    logger.info(f"Channel: {args.channel}")
    logger.info(f"Skip ICA: {args.skip_ica}")
    logger.info(f"Device: {device}")
    logger.info(f"{'=' * 80}")

    # TODO - maybe I should remove drophout here no? I HAVE TO make sure dropout is
    # not applied in eval nor here
    # Load model checkpoint
    logger.info("Loading model checkpoint...")
    model = CNN_LSTM_DepCap(
        input_shape=EXPECTED_SPECTROGRAM_SHAPE,
        in_channels=1,
        rnn_type="LSTM",
        rnn_hidden=100,
        dropout=0,
        num_classes=2,
    )

    # Note: weights_only=False is required to load optimizer state and other training info
    saved = torch.load(model_path, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)
    model.eval()  # This among other things deactivates dropout
    logger.info("Model loaded successfully")

    # Preprocess EDF file
    logger.info("Preprocessing EDF file...")
    chunks = MDDDataset.load_and_preprocess_mdd_raw_file(
        edf_file, args.channel, skip_ica=args.skip_ica
    )
    logger.info(f"Extracted {len(chunks)} chunks from EDF file")
    if len(chunks) == 0:
        raise RuntimeError("No valid chunks extracted from EDF file")
    spectograms = SpectrogramDataset.convert_to_spectrograms(
        chunks, nperseg=256, fs=250, noverlap=192, window="hamming"
    )

    logger.info("\nRunning inference on each chunk:")
    logger.info(f"{'Chunk':<8} {'Prediction':<12} {'Class':<10}")
    logger.info("-" * 50)

    chunk_predictions = []

    # TODO reviewed code till here
    with torch.no_grad():
        for chunk_idx, spec in enumerate(spectograms):
            # Add batch dimension (spec already has shape (1, H, W))
            spec_tensor = spec.unsqueeze(0)  # (1, 1, H, W)
            spec_tensor = spec_tensor.to(device)

            # Run inference
            logits = model(spec_tensor)
            pred = logits.argmax(dim=1).item()

            chunk_predictions.append(pred)
            class_name = "Healthy" if pred == 0 else "MDD"
            logger.info(f"{chunk_idx + 1:<8} {pred:<12} {class_name:<10}")

    # Aggregate predictions using majority voting
    if len(chunk_predictions) == 0:
        logger.error("No valid predictions generated")
        return

    chunk_predictions = np.array(chunk_predictions)

    # Count predictions
    healthy_count = np.sum(chunk_predictions == 0)
    mdd_count = np.sum(chunk_predictions == 1)

    # Majority vote
    final_prediction = np.bincount(chunk_predictions).argmax()
    final_class = "Healthy" if final_prediction == 0 else "MDD"

    # Print final results
    logger.info("-" * 50)
    logger.info("FINAL RESULT (Majority Voting)")
    logger.info("-" * 50)
    logger.info(f"Total chunks analyzed: {len(chunk_predictions)}")
    healthy_pct = healthy_count / len(chunk_predictions) * 100
    logger.info(f"Healthy predictions: {healthy_count} ({healthy_pct:.1f}%)")
    logger.info(f"MDD predictions: {mdd_count} ({mdd_count / len(chunk_predictions) * 100:.1f}%)")
    logger.info(f"Final Prediction: {final_class} (Class {final_prediction})")


def main() -> None:
    parser = get_arg_parser()
    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "run":
        run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
