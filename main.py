import argparse
import logging
import random
import string
from collections.abc import Callable
from dataclasses import replace as dataclasses_replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from plot_training_curves import (
    TrainingCurvesError,
    generate_training_curves,
)
from thesis.augmentation import EEGAugmentation
from thesis.cli import get_arg_parser
from thesis.data_preparation import (
    EXPECTED_SPECTROGRAM_SHAPE,
    create_balanced_folds,
    determine_num_classes,
    get_datasets_for_fold,
    prepare_cane_dataset,
    prepare_idun_dataset,
    prepare_mdd_dataset,
    prepare_sad_dataset,
    split_into_folds,
)
from thesis.dataset import (
    collate_spectrograms,
)
from thesis.model import RAW_EEG_MODELS
from thesis.early_stopping import EarlyStopping
from thesis.inference import preprocess_and_infer
from thesis.json_logging import log_metrics_json
from thesis.loss import FocalLoss
from thesis.metrics import (
    aggregate_subject_predictions,
    classification_metrics,
    compute_per_dataset_metrics,
    extract_classification_metrics,
    write_results,
)
from thesis.model_factory import (
    DeformerConfig,
    LGGNetConfig,
    _RAW_EEG_DEFAULT_CONFIGS,
    create_model,
)

RANDOM_SEED = 42
LOG_FORMAT = "[%(asctime)s %(levelname)s %(module)s.%(funcName)s] %(message)s"
LOG_LEVEL = "INFO"
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


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


def setup_logging(checkpoint_dir: Path, condition: str, channel: str | None = None) -> Path:
    """
    Configure logging to output to both console and a timestamped log file.

    :param Path checkpoint_dir: Directory to save log files.
    :param str condition: EEG condition being trained on.
    :param str | None channel: Single channel being used (if applicable).
    :return: Path to the log file.
    :rtype: Path
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    channel_suffix = f"_{channel}" if channel else ""
    log_filename = f"training_{condition}{channel_suffix}_{timestamp}.log"
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
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to console and file: {log_path}")

    return log_path


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
    subject_dataset_map: dict[str, str] | None = None,
) -> dict[str, dict[str, float | int | np.ndarray] | float]:
    """
    Evaluate model for one epoch, computing both chunk-level and subject-level metrics.

    :param nn.Module model: Model to evaluate.
    :param DataLoader dataloader: Validation data loader (expects Subset of SpectrogramDataset).
    :param nn.Module criterion: Loss function.
    :param torch.device device: Device to evaluate on.
    :param int num_classes: Number of classes (2 or 3).
    :param dict[str, str] | None subject_dataset_map: Optional mapping from subject ID to dataset
        label. When provided, per-dataset chunk accuracy is computed.
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

    # Per-dataset metrics (when mapping is available)
    per_dataset: dict[str, dict[str, float | int]] | None = None
    if subject_dataset_map is not None:
        per_dataset = compute_per_dataset_metrics(
            all_preds, all_labels, all_subjects, subject_dataset_map
        )

    return {
        "chunk": chunk_metrics,
        "subject": subject_metrics,
        "condition": condition_metrics,
        "loss": avg_loss,
        "per_dataset": per_dataset,
    }


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
    val_subject_dataset_map: dict[str, str] | None = None,
    log_file: Path | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
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
    :param dict[str, str] | None val_subject_dataset_map: Optional mapping from validation subject
        IDs to dataset labels for per-dataset metrics.
    :param Path | None log_file: Path to the log file for JSON metrics output.
    :param torch.optim.lr_scheduler.LRScheduler | None scheduler: Optional LR scheduler stepped
        once per epoch. Pass ``None`` (default) to keep a constant learning rate.
    :return: Dictionary with fold results.
    :rtype: dict
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStopping(patience=patience, maximize=True)
    best_chunk_acc = 0.0
    corr_combined_acc = 0.0
    best_epoch = 0
    best_train_acc: float = 0.0

    best_chunk_metrics: dict[str, float] = {}
    best_subject_metrics: dict[str, float] = {}
    best_chunk_confusion_matrix: np.ndarray = np.zeros((num_classes, num_classes), dtype=int)
    best_per_dataset_metrics: dict[str, dict[str, float | int]] | None = None

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
    if scheduler is not None:
        logger.info(f"LR schedule: {scheduler.__class__.__name__} over {num_epochs} epochs")

    epoch = 0
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, num_classes)

        log_metrics_json(
            "train",
            fold + 1,
            epoch,
            num_epochs,
            train_metrics["loss"],
            train_metrics,
            num_classes=num_classes,
            log_file=log_file,
        )

        fold_history["train_loss"].append(train_metrics["loss"])
        fold_history["train_acc"].append(train_metrics["accuracy"])
        fold_history["epochs"].append(epoch)

        if scheduler is not None:
            scheduler.step()

        # Validate every val_every epochs
        if epoch % val_every == 0 or epoch == num_epochs:
            eval_metrics = eval_epoch(
                model, val_loader, criterion, device, num_classes, val_subject_dataset_map
            )

            # Extract chunk and subject metrics
            chunk_metrics = eval_metrics["chunk"]
            subject_metrics = eval_metrics["subject"]
            condition_metrics = eval_metrics.get("condition", {})

            log_metrics_json(
                "eval",
                fold + 1,
                epoch,
                num_epochs,
                eval_metrics["loss"],
                chunk_metrics,
                subject_metrics=subject_metrics,
                condition_metrics=condition_metrics if condition_metrics else None,
                num_classes=num_classes,
                log_file=log_file,
            )

            fold_history["val_loss"].append(eval_metrics["loss"])
            fold_history["chunk_acc"].append(chunk_metrics["accuracy"])
            fold_history["subject_acc"].append(subject_metrics["accuracy"])

            # Save the best model (based on chunk accuracy - PRIMARY METRIC)
            if chunk_metrics["accuracy"] > best_chunk_acc:
                best_chunk_acc = chunk_metrics["accuracy"]
                corr_combined_acc = chunk_metrics["accuracy"] * subject_metrics["accuracy"]
                best_epoch = epoch
                best_train_acc = train_metrics["accuracy"]

                # Capture metrics including accuracy
                best_chunk_metrics = extract_classification_metrics(chunk_metrics, num_classes)
                best_subject_metrics = extract_classification_metrics(subject_metrics, num_classes)
                best_chunk_confusion_matrix = chunk_metrics["confusion_matrix"]
                best_per_dataset_metrics = eval_metrics.get("per_dataset")

                best_model_path = checkpoint_dir / f"fold_{fold + 1}_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "chunk_acc": best_chunk_acc,
                        "val_metrics": eval_metrics,
                    },
                    best_model_path,
                )
                logger.info(
                    f"✓ Saved best model: {best_model_path} "
                    f"(chunk={chunk_metrics['accuracy']:.4f}, "
                    f"subject={subject_metrics['accuracy']:.4f}, "
                    f"combined={corr_combined_acc:.4f})"
                )

            # Early stopping check
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
        f"Best Chunk Val Acc: {best_chunk_acc:.4f} at epoch {best_epoch}"
    )

    return {
        "eval_combined_acc": corr_combined_acc,
        "best_epoch": best_epoch,
        "final_epoch": epoch,
        "train_acc_at_best": best_train_acc,
        "history": fold_history,
        "chunk_metrics": best_chunk_metrics,
        "subject_metrics": best_subject_metrics,
        "chunk_confusion_matrix": best_chunk_confusion_matrix,
        "per_dataset_metrics": best_per_dataset_metrics,
    }


def compute_class_weights(
    dataset: Dataset, num_classes: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """
    Compute class weights for weighted loss based on chunk-level class distribution.

    Pass ``device="cpu"`` (default) when weights are only used for
    :class:`~torch.utils.data.WeightedRandomSampler` (CPU-only).
    Pass the model device (e.g. ``"cuda"``) when weights will be fed to
    ``nn.CrossEntropyLoss(weight=class_weights)``, which requires the tensor
    to live on the same device as model outputs.

    :param Dataset dataset: Dataset to compute weights for (FlattenedSpectrogramDataset).
    :param int num_classes: Number of classes (2 or 4).
    :param device: Device to place weights tensor on. Defaults to ``"cpu"``.
    :type device: torch.device | str
    :return: Class weights tensor of shape (num_classes,).
    :rtype: torch.Tensor
    """
    class_counts = torch.zeros(num_classes, dtype=torch.float)

    # Count chunks for each class
    for idx in range(len(dataset)):
        _, label, _ = dataset[idx]
        class_counts[label] += 1

    # Calculate weights using sklearn's balanced formula:
    # weight[c] = total_samples / (num_classes * class_counts[c])
    total_samples = class_counts.sum()
    weights = total_samples / (num_classes * class_counts)

    return weights.to(device)


def train_cross_validation(
    dataset_type: Literal["mdd", "cane", "sad", "all"] = "mdd",
    condition: str = "EC",
    class_mode: str = "2",
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
    channel: str | None = None,
    model_name: str = "CNN_LSTM_DepCap",
    skip_artifact_removal: bool = False,
    pretrained_checkpoint: str | None = None,
    freeze_cnn: bool = False,
    freeze_lstm: bool = False,
    augmentation: Callable | None = None,
    test_mode: bool = False,
    log_file: Path | None = None,
    focal_loss: bool = False,
    focal_gamma: float = 2.0,
    cosine_lr: bool = True,
    raw_eeg_config: DeformerConfig | LGGNetConfig | None = None,
    chunk_duration: float = 10.0,
) -> dict:
    """
    Train model using n-fold cross-validation with comprehensive logging and checkpointing.

    :param str dataset_type: Dataset to use ("mdd", "cane", "sad", or "all").
    :param str condition: EEG condition to use ("EC", "EO", "TASK", or "EC+EO").
    :param str class_mode: Classification mode ("2", or "4").
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
    :param str | None channel: Single channel to use (e.g., "Fp1"). If None, uses all channels.
    :param str model_name: Model architecture to use ("CNN_LSTM_DepCap" or "Smaller").
    :param bool skip_artifact_removal: If True, skip artifact interpolation/clipping in CANE.
    :param str | None pretrained_checkpoint: Path to pretrained model checkpoint for transfer
        learning. Must use same model architecture as pretrained model.
    :param bool freeze_cnn: If True, freeze CNN layers (conv1, conv2) during training.
        Can be used alone or with freeze_lstm.
    :param bool freeze_lstm: If True, freeze LSTM layer during training.
        Can be used alone or with freeze_cnn.
    :param Callable | None augmentation: Optional augmentation to apply to raw EEG during training.
    :param bool test_mode: If True, load only one file per class for debugging.
    :param bool cosine_lr: If True (default), use cosine annealing LR schedule (T_max=num_epochs,
        eta_min=1e-6). Set to False for constant LR.
    :param Path | None log_file: Path to the log file for JSON metrics output.
    :return: Dictionary with cross-validation results. Subject accuracy corresponds to the
           best chunk accuracy model (primary metric).
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
    logger.info(f"{'=' * 80}")

    cv_results: dict[str, list] = {
        "fold_eval_combined_acc": [],
        "fold_best_epoch": [],
        "fold_final_epoch": [],
        "fold_train_acc": [],
        "fold_chunk_metrics": [],
        "fold_subject_metrics": [],
        "fold_chunk_confusion_matrices": [],
        "fold_per_dataset_metrics": [],
    }

    rng = np.random.RandomState(RANDOM_SEED)
    if "TASK" in conditions and dataset_type == "cane":
        raise ValueError("condition TASK not possible for CANE dataset")

    # Determine num_classes and label_mapping based on class_mode
    num_classes, label_mapping = determine_num_classes(class_mode)
    logger.info(f"Classification mode: {num_classes}-class")
    if label_mapping:
        logger.info(f"Label mapping: {label_mapping} (collapsing to binary)")

    # Detect whether model needs raw EEG (Deformer) or spectrograms (CNN-LSTM family)
    use_raw_eeg = model_name in RAW_EEG_MODELS
    if use_raw_eeg:
        logger.info(f"Model {model_name} uses raw EEG pipeline (no STFT)")

    mdd_flat_dataset, cane_flat_dataset, sad_flat_dataset, flat_dataset = (
        None,
        None,
        None,
        None,
    )
    if dataset_type == "mdd":
        flat_dataset, subject_classes = prepare_mdd_dataset(
            conditions,
            channel,
            rng,
            augmentation=augmentation,
            test_mode=test_mode,
            num_classes=num_classes,
            label_mapping=label_mapping,
            use_raw_eeg=use_raw_eeg,
            chunk_duration=chunk_duration,
        )
        normal, anxiety, depression, anxiety_depression = (
            subject_classes.normal,
            subject_classes.anxiety,
            subject_classes.depression,
            subject_classes.anxiety_depression,
        )
        logger.info(f"MDD dataset: {len(normal)} normal, {len(depression)} depression subjects")
    elif dataset_type == "cane":
        # Convert to lowercase for CANE/IDUN
        cane_conditions = [c.lower() for c in conditions]
        if channel == "in-ear":
            logger.info("Using IDUN real in-ear dataset instead of CANE synthetic derivation")
            flat_dataset, subject_classes = prepare_idun_dataset(
                cane_conditions,
                channel,
                rng,
                label_mapping=label_mapping,
                augmentation=augmentation,
                test_mode=test_mode,
                use_raw_eeg=use_raw_eeg,
                chunk_duration=chunk_duration,
            )
        else:
            flat_dataset, subject_classes = prepare_cane_dataset(
                cane_conditions,
                channel,
                rng,
                label_mapping=label_mapping,
                skip_artifact_removal=skip_artifact_removal,
                augmentation=augmentation,
                test_mode=test_mode,
                use_raw_eeg=use_raw_eeg,
                chunk_duration=chunk_duration,
            )
        normal, anxiety, depression, anxiety_depression = (
            subject_classes.normal,
            subject_classes.anxiety,
            subject_classes.depression,
            subject_classes.anxiety_depression,
        )
        dataset_name = "IDUN" if channel == "in-ear" else "CANE"
        logger.info(
            f"{dataset_name} dataset: {len(normal)} normal, {len(anxiety)} anxiety, "
            f"{len(depression)} depression, {len(anxiety_depression)} anxiety+depression subjects"
        )
    elif dataset_type == "sad":
        flat_dataset, subject_classes = prepare_sad_dataset(
            conditions,
            channel,
            rng,
            augmentation=augmentation,
            test_mode=test_mode,
            num_classes=num_classes,
            label_mapping=label_mapping,
            use_raw_eeg=use_raw_eeg,
            chunk_duration=chunk_duration,
        )
        normal, anxiety, depression, anxiety_depression = (
            subject_classes.normal,
            subject_classes.anxiety,
            subject_classes.depression,
            subject_classes.anxiety_depression,
        )
        logger.info(f"SAD dataset: {len(normal)} normal, {len(anxiety)} anxiety subjects")
    elif dataset_type == "all":
        # Load MDD dataset
        # T3=T7 and T4=T8 for these purposes, otherwise I couldn't combine the datasets
        if channel in ["T7", "T8"]:
            channel = {"T7": "T3", "T8": "T4"}[channel]
        mdd_flat_dataset, mdd_subject_classes = prepare_mdd_dataset(
            conditions,
            channel,
            rng,
            augmentation=augmentation,
            test_mode=test_mode,
            num_classes=num_classes,
            label_mapping=label_mapping,
            use_raw_eeg=use_raw_eeg,
            chunk_duration=chunk_duration,
        )
        mdd_normal, mdd_anxiety, mdd_depression, mdd_anxiety_depression = (
            mdd_subject_classes.normal,
            mdd_subject_classes.anxiety,
            mdd_subject_classes.depression,
            mdd_subject_classes.anxiety_depression,
        )
        if channel in ["T3", "T4"]:
            channel = {"T3": "T7", "T4": "T8"}[channel]

        # Load CANE dataset (or IDUN real in-ear when channel == "in-ear")
        cane_conditions = [c.lower() for c in conditions]
        if channel == "in-ear":
            logger.info("Using IDUN real in-ear dataset instead of CANE synthetic derivation")
            cane_flat_dataset, cane_subject_classes = prepare_idun_dataset(
                cane_conditions,
                channel,
                rng,
                label_mapping=label_mapping,
                augmentation=augmentation,
                test_mode=test_mode,
                use_raw_eeg=use_raw_eeg,
                chunk_duration=chunk_duration,
            )
        else:
            cane_flat_dataset, cane_subject_classes = prepare_cane_dataset(
                cane_conditions,
                channel,
                rng,
                label_mapping=label_mapping,
                skip_artifact_removal=skip_artifact_removal,
                augmentation=augmentation,
                test_mode=test_mode,
                use_raw_eeg=use_raw_eeg,
                chunk_duration=chunk_duration,
            )
        cane_normal, cane_anxiety, cane_depression, cane_anxiety_depression = (
            cane_subject_classes.normal,
            cane_subject_classes.anxiety,
            cane_subject_classes.depression,
            cane_subject_classes.anxiety_depression,
        )

        # Load SAD dataset
        sad_flat_dataset, sad_subject_classes = prepare_sad_dataset(
            conditions,
            channel,
            rng,
            augmentation=augmentation,
            test_mode=test_mode,
            num_classes=num_classes,
            label_mapping=label_mapping,
            use_raw_eeg=use_raw_eeg,
            chunk_duration=chunk_duration,
        )
        sad_normal, sad_anxiety, sad_depression, sad_anxiety_depression = (
            sad_subject_classes.normal,
            sad_subject_classes.anxiety,
            sad_subject_classes.depression,
            sad_subject_classes.anxiety_depression,
        )

    # Create folds for each class separately (stratified)
    if dataset_type == "all":
        # For combined dataset, stratify each dataset separately then merge corresponding folds
        normal_folds, anxiety_folds, depression_folds, anxiety_depression_folds = (
            create_balanced_folds(
                [mdd_normal, cane_normal, sad_normal],
                [mdd_anxiety, cane_anxiety, sad_anxiety],
                [mdd_depression, cane_depression, sad_depression],
                [mdd_anxiety_depression, cane_anxiety_depression, sad_anxiety_depression],
                n_folds,
            )
        )
        logger.info(
            f"Created {n_folds} balanced folds with subjects from MDD, CANE, and SAD datasets"
        )
    else:
        # Single dataset: some classes may be empty
        empty_folds = [[] for _ in range(n_folds)]
        normal_folds = split_into_folds(normal, n_folds)
        anxiety_folds = split_into_folds(anxiety, n_folds) if anxiety else empty_folds
        depression_folds = split_into_folds(depression, n_folds) if depression else empty_folds
        anxiety_depression_folds = (
            split_into_folds(anxiety_depression, n_folds) if anxiety_depression else empty_folds
        )

    spec_shape = EXPECTED_SPECTROGRAM_SHAPE
    for fold in range(n_folds):
        logger.info(f"{'=' * 80}")
        logger.info(f"FOLD {fold + 1}/{n_folds}")
        logger.info(f"{'=' * 80}")

        train_dataset, val_dataset, val_subject_dataset_map = get_datasets_for_fold(
            fold,
            normal_folds,
            anxiety_folds,
            depression_folds,
            anxiety_depression_folds,
            dataset_type,
            mdd_flat_dataset,
            cane_flat_dataset,
            sad_flat_dataset,
            flat_dataset,
        )

        # Compute class weights for balanced loss (CPU only used for sampler, otherwise use `device` as value)
        class_weights = compute_class_weights(train_dataset, num_classes, device="cpu")
        logger.info(f"Fold {fold + 1} | Class weights | {class_weights}")

        # Build per-sample weights for WeightedRandomSampler
        sample_weights: list[float] = []
        for idx in range(len(train_dataset)):
            _, label, _ = train_dataset[idx]
            sample_weights.append(class_weights[label].item())
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

        # Create DataLoaders
        # num_workers>0 overlaps CPU data loading/STFT computation with GPU training,
        # preventing GPU from sitting idle while spectrograms are computed on CPU.
        # pin_memory speeds up CPU→GPU transfers, enabling Direct Memory Access
        pin_memory = device.type == "cuda"
        num_workers = 4
        # Raw EEG models return uniform-shape tensors so default collate works;
        # spectrogram models use collate_spectrograms for shape validation.
        fold_collate_fn = None if use_raw_eeg else collate_spectrograms
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            # To roll back to pure random sampling: replace sampler= with shuffle=True,
            # recompute class_weights with device=device (needs GPU for the loss),
            # and pass them to: nn.CrossEntropyLoss(weight=class_weights)
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=fold_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=fold_collate_fn,
        )

        model = create_model(
            model_name=model_name,
            spec_shape=spec_shape,
            dropout=dropout,
            num_classes=num_classes,
            device=device,
            in_channels=8 if channel == "all" else 1,
            pretrained_checkpoint=pretrained_checkpoint,
            freeze_cnn=freeze_cnn,
            freeze_lstm=freeze_lstm,
            raw_eeg_config=raw_eeg_config,
        )

        if focal_loss:
            criterion: nn.Module = FocalLoss(alpha=class_weights.to(device), gamma=focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss()
        # Only optimize trainable parameters (important when CNN layers are frozen)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        fold_scheduler: torch.optim.lr_scheduler.LRScheduler | None
        if cosine_lr:
            fold_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=1e-6
            )
        else:
            fold_scheduler = None

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
            val_subject_dataset_map=val_subject_dataset_map,
            log_file=log_file,
            scheduler=fold_scheduler,
        )

        cv_results["fold_eval_combined_acc"].append(fold_result["eval_combined_acc"])
        cv_results["fold_best_epoch"].append(fold_result["best_epoch"])
        cv_results["fold_final_epoch"].append(fold_result["final_epoch"])
        cv_results["fold_train_acc"].append(fold_result["train_acc_at_best"])
        cv_results["fold_chunk_metrics"].append(fold_result["chunk_metrics"])
        cv_results["fold_subject_metrics"].append(fold_result["subject_metrics"])
        cv_results["fold_chunk_confusion_matrices"].append(fold_result["chunk_confusion_matrix"])
        if fold_result["per_dataset_metrics"] is not None:
            cv_results["fold_per_dataset_metrics"].append(fold_result["per_dataset_metrics"])

    # Print final cross-validation results
    logger.info(f"{'=' * 80}")
    logger.info(f"{n_folds}-FOLD CROSS-VALIDATION RESULTS")
    logger.info(f"{'=' * 80}")

    logger.info("\nPer-fold best validation accuracy (chunk):")
    for i, chunk_metrics in enumerate(cv_results["fold_chunk_metrics"]):
        chunk_acc = chunk_metrics["accuracy"]
        logger.info(f"  Fold {i + 1}: {chunk_acc:.4f} (epoch {cv_results['fold_best_epoch'][i]})")

    write_results(logger.info, cv_results)
    logger.info(f"{'=' * 80}\n")

    return cv_results


def train(args: argparse.Namespace) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_dir = get_unique_checkpoint_dir(Path(args.checkpoint_dir))

    # Setup logging to both console and file
    log_path = setup_logging(checkpoint_dir, args.condition, args.channel)

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
    logger.info(f"  Skip Artifact Removal: {args.skip_artifact_removal}")
    if args.augment_data is not None:
        augmentation: EEGAugmentation | None = EEGAugmentation(p_aug=args.augment_data)
        logger.info(f"  Data Augmentation: enabled (p={args.augment_data})")
    else:
        augmentation = None
        logger.info("  Data Augmentation: disabled")
    logger.info(f"  Pretrained Checkpoint: {args.pretrained_checkpoint}")
    logger.info(f"  Freeze CNN: {args.freeze_cnn}")
    logger.info(f"  Freeze LSTM: {args.freeze_lstm}")
    logger.info(f"  Class mode: {args.class_mode}")
    logger.info(f"  Test mode: {args.test_mode}")
    logger.info(f"  Cosine LR: {args.cosine_lr}")
    logger.info(
        f"  Focal Loss: {args.focal_loss}"
        + (f" (gamma={args.focal_gamma})" if args.focal_loss else "")
    )
    logger.info(f"  Device: {device}")
    logger.info(f"  Checkpoint Directory: {checkpoint_dir}")
    logger.info(f"  Log File: {log_path}")

    # Build Deformer config: start from the model-appropriate defaults, then apply
    # any CLI overrides (None means "keep model default").
    _base_cfg = _RAW_EEG_DEFAULT_CONFIGS.get(args.model, DeformerConfig)()
    _overrides: dict = {"num_time": int(args.chunk_duration * 250)}
    if args.deformer_depth is not None:
        _overrides["depth"] = args.deformer_depth
    if args.deformer_heads is not None:
        _overrides["heads"] = args.deformer_heads
    if args.deformer_num_kernel is not None:
        _overrides["num_kernel"] = args.deformer_num_kernel
    if args.deformer_mlp_dim is not None:
        _overrides["mlp_dim"] = args.deformer_mlp_dim
    if args.deformer_dim_head is not None:
        _overrides["dim_head"] = args.deformer_dim_head
    if args.deformer_temporal_kernel is not None:
        _overrides["temporal_kernel"] = args.deformer_temporal_kernel
    deformer_config = dataclasses_replace(_base_cfg, **_overrides)
    logger.info(f"Deformer config: {deformer_config} (chunk_duration={args.chunk_duration}s)")

    # Build LGGNet config: start from the model-appropriate defaults, then apply CLI overrides.
    logger.info(f"  LGGNet num_T: {args.lggnet_num_t}")
    logger.info(f"  LGGNet out_graph: {args.lggnet_out_graph}")
    logger.info(f"  LGGNet pool: {args.lggnet_pool}")
    logger.info(f"  LGGNet pool_step_rate: {args.lggnet_pool_step_rate}")
    logger.info(f"  LGGNet graph_type: {args.lggnet_graph_type}")
    _lgg_base = _RAW_EEG_DEFAULT_CONFIGS.get(args.model, LGGNetConfig)()
    _lgg_overrides: dict = {"num_time": int(args.chunk_duration * 250), "sampling_rate": 250}
    if args.lggnet_num_t is not None:
        _lgg_overrides["num_T"] = args.lggnet_num_t
    if args.lggnet_out_graph is not None:
        _lgg_overrides["out_graph"] = args.lggnet_out_graph
    if args.lggnet_pool is not None:
        _lgg_overrides["pool"] = args.lggnet_pool
    if args.lggnet_pool_step_rate is not None:
        _lgg_overrides["pool_step_rate"] = args.lggnet_pool_step_rate
    if args.lggnet_graph_type is not None:
        _lgg_overrides["graph_type"] = args.lggnet_graph_type
    lggnet_config = dataclasses_replace(_lgg_base, **_lgg_overrides)
    logger.info(f"LGGNet config: {lggnet_config} (chunk_duration={args.chunk_duration}s)")

    raw_eeg_config: DeformerConfig | LGGNetConfig = (
        lggnet_config if args.model in {"LGGNet", "LGGNetS"} else deformer_config
    )

    # Run cross-validation training
    results = train_cross_validation(
        dataset_type=args.dataset,
        condition=args.condition,
        class_mode=args.class_mode,
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
        channel=args.channel,
        model_name=args.model,
        skip_artifact_removal=args.skip_artifact_removal,
        pretrained_checkpoint=args.pretrained_checkpoint,
        freeze_cnn=args.freeze_cnn,
        freeze_lstm=args.freeze_lstm,
        augmentation=augmentation,
        test_mode=args.test_mode,
        log_file=log_path,
        focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        cosine_lr=args.cosine_lr,
        raw_eeg_config=raw_eeg_config,
        chunk_duration=args.chunk_duration,
    )

    # Save final results to file
    results_file = checkpoint_dir / "results.txt"
    with open(results_file, "w") as f:
        f.write(f"{args.n_folds}-Fold Cross-Validation Results\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Condition: {args.condition}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Dropout: {args.dropout}\n")
        f.write(f"Weight Decay (L2 regularization): {args.weight_decay}\n")
        f.write(f"Cosine LR: {args.cosine_lr}\n")
        f.write(f"Skip Artifact Removal: {args.skip_artifact_removal}\n")
        if args.augment_data is not None:
            f.write(f"Data Augmentation: enabled (p={args.augment_data})\n")
        else:
            f.write("Data Augmentation: disabled\n")
        f.write(f"Pretrained Checkpoint: {args.pretrained_checkpoint}\n")
        f.write(f"Freeze CNN: {args.freeze_cnn}\n")
        f.write(f"Freeze LSTM: {args.freeze_lstm}\n")
        f.write(f"LGGNet num_T: {args.lggnet_num_t}\n")
        f.write(f"LGGNet out_graph: {args.lggnet_out_graph}\n")
        f.write(f"LGGNet pool: {args.lggnet_pool}\n")
        f.write(f"LGGNet pool_step_rate: {args.lggnet_pool_step_rate}\n")
        f.write(f"LGGNet graph_type: {args.lggnet_graph_type}\n\n")
        f.write("Per-fold best validation accuracy:\n")
        for i in range(len(results["fold_chunk_metrics"])):
            chunk_acc = results["fold_chunk_metrics"][i]["accuracy"]
            subject_acc = results["fold_subject_metrics"][i]["accuracy"]
            combined_acc = results["fold_eval_combined_acc"][i]
            train_acc = results["fold_train_acc"][i] if results.get("fold_train_acc") else None
            train_str = f", train={train_acc:.4f}" if train_acc is not None else ""
            f.write(
                f"  Fold {i + 1}: chunk={chunk_acc:.4f}, subject={subject_acc:.4f}, "
                f"combined={combined_acc:.4f}{train_str} (epoch {results['fold_best_epoch'][i]}, "
                f"stopped at epoch {results['fold_final_epoch'][i]})\n"
            )
        write_results(f.write, results)
    logger.info(f"\nResults saved to {results_file}")

    # Generate training curves automatically
    logger.info("Generating training curves...")
    try:
        curves_dir = generate_training_curves(log_path)
        logger.info(f"Training curves saved to {curves_dir}")
    except TrainingCurvesError as e:
        logger.warning(f"Failed to generate training curves: {e}")

    logger.info("Training completed!")


def run(args: argparse.Namespace) -> None:
    """
    Run inference on a single EEG file (.edf or .csv) using a trained model.

    Sampling rate is auto-detected from the file when not provided.

    :param argparse.Namespace args: Command-line arguments.
    """
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_path = Path(args.model_path)
    input_file = Path(args.file)

    # Validate paths
    if not model_path.exists():
        raise IllegalPathError(f"Model checkpoint not found: {model_path}")

    if not input_file.exists():
        raise IllegalPathError(f"Input file not found: {input_file}")

    num_classes = int(args.class_mode)
    in_channels = 8 if args.channel == "all" else 1

    logger.info("Running Inference")
    logger.info(f"{'=' * 80}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Channel: {args.channel}")
    logger.info(f"Classes: {num_classes}")
    logger.info(f"Device: {device}")
    logger.info(f"{'=' * 80}")

    # Load model via create_model with pretrained_checkpoint (handles weight loading)
    logger.info("Loading model checkpoint...")
    model = create_model(
        model_name=args.model,
        spec_shape=EXPECTED_SPECTROGRAM_SHAPE,
        dropout=0,
        num_classes=num_classes,
        device=device,
        in_channels=in_channels,
        pretrained_checkpoint=str(model_path),
    )
    model.eval()
    logger.info("Model loaded successfully")

    # Run shared inference pipeline — sampling rate auto-detected from file if not provided
    file_format = input_file.suffix.lower()
    result = preprocess_and_infer(
        file_path=input_file,
        model=model,
        device=device,
        channel=args.channel,
        file_format=file_format,
        num_classes=num_classes,
        sampling_rate=args.fs,
    )

    # Log per-chunk results
    logger.info(f"{'Chunk':<8} {'Prediction':<12} {'Class':<10}")
    logger.info("-" * 50)
    for cr in result.chunk_results:
        logger.info(f"{cr.chunk_index + 1:<8} {cr.predicted_class:<12} {cr.class_name:<10}")

    # Log final results
    logger.info("-" * 50)
    logger.info("FINAL RESULT (Majority Voting)")
    logger.info("-" * 50)
    logger.info(f"Total chunks analyzed: {result.total_chunks}")
    for name, count in result.class_distribution.items():
        pct = result.class_percentages[name]
        logger.info(f"{name} predictions: {count} ({pct:.1f}%)")
    logger.info(f"Final Prediction: {result.final_class_name} (Class {result.final_prediction})")


def main() -> None:
    parser = get_arg_parser()
    args = parser.parse_args()

    # Validate transfer learning arguments
    if args.command == "train" and args.freeze_cnn and not args.pretrained_checkpoint:
        parser.error("--freeze-cnn requires --pretrained-checkpoint")
    if args.command == "train" and args.freeze_lstm and not args.pretrained_checkpoint:
        parser.error("--freeze-lstm requires --pretrained-checkpoint")

    if args.command == "train":
        train(args)
    elif args.command == "run":
        run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
