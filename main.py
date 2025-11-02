import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset

from thesis.dataset import (
    MDDDataset,
    SpectrogramDataset,
    collate_spectrograms,
)
from thesis.early_stopping import EarlyStopping
from thesis.model import CNN_LSTM_DepCap

LOG_FORMAT = "[%(asctime)s %(levelname)s %(module)s.%(funcName)s] %(message)s"
LOG_LEVEL = "INFO"
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# TODO check this for training too
EXPECTED_SPECTOGRAM_SHAPE = (129, 41)  # Expected spectrogram shape from training


class IllegalPathError(Exception):
    pass


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


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """
    Compute classification metrics from true and predicted labels.

    :param np.ndarray y_true: True labels (0/1).
    :param np.ndarray y_pred: Predicted labels (0/1).
    :return: Dictionary with accuracy, precision, recall, specificity, and confusion matrix values.
    :rtype: dict
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    # TODO idk what the following mean
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def aggregate_subject_predictions(
    chunk_preds: np.ndarray, chunk_labels: np.ndarray, subjects: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate chunk-level predictions to subject-level using majority voting.

    For each subject, collects all chunk predictions and computes the most frequent
    prediction (mode) as the final subject-level prediction. This is useful for
    EEG analysis where multiple temporal chunks come from the same subject.

    :param np.ndarray chunk_preds: Predictions for each chunk (0/1).
    :param np.ndarray chunk_labels: Ground truth labels for each chunk (0/1).
    :param list[str] subjects: Subject ID for each chunk (e.g., "H S1", "MDD S2").
    :return: Tuple of (subject_predictions, subject_labels) where each array contains
             one value per unique subject.
    :rtype: tuple[np.ndarray, np.ndarray]

    Example:
        >>> chunk_preds = np.array([0, 1, 1, 0, 0])
        >>> chunk_labels = np.array([0, 0, 0, 1, 1])
        >>> subjects = ["H S1", "H S1", "H S1", "MDD S2", "MDD S2"]
        >>> subj_preds, subj_labels = aggregate_subject_predictions(
        ...     chunk_preds, chunk_labels, subjects
        ... )
        >>> subj_preds  # H S1 gets 1 (majority), MDD S2 gets 0 (majority)
        array([1, 0])
        >>> subj_labels  # H S1 is 0, MDD S2 is 1
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
    for subject in subject_preds_dict:
        chunk_preds_for_subject = subject_preds_dict[subject]
        majority_pred = np.bincount(chunk_preds_for_subject).argmax()
        subject_preds.append(majority_pred)
        subject_labels.append(subject_labels_dict[subject])

    return np.array(subject_preds), np.array(subject_labels)


# -------------------------
# Training loop
# -------------------------
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | int]:
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
        running_loss += float(loss.item()) * xb.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.detach().cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = classification_metrics(all_labels, all_preds)
    avg_loss = running_loss / len(dataloader.dataset)
    metrics["loss"] = avg_loss
    return metrics


def eval_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> dict[str, dict[str, float | int] | float]:
    """
    Evaluate model for one epoch, computing both chunk-level and subject-level metrics.

    :param nn.Module model: Model to evaluate.
    :param DataLoader dataloader: Validation data loader (expects Subset of SpectrogramDataset).
    :param nn.Module criterion: Loss function.
    :param torch.device device: Device to evaluate on.
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
    chunk_metrics = classification_metrics(all_labels, all_preds)
    avg_loss = running_loss / len(dataloader.dataset)

    # Subject-level metrics (majority voting)
    subject_preds, subject_labels = aggregate_subject_predictions(
        all_preds, all_labels, all_subjects
    )
    subject_metrics = classification_metrics(subject_labels, subject_preds)

    return {
        "chunk": chunk_metrics,
        "subject": subject_metrics,
        "loss": avg_loss,
    }


def train_one_fold(
    fold: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
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
    best_val_acc = 0.0
    best_epoch = 0

    fold_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epochs": [],
    }

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Starting training for fold {fold + 1}")
    logger.info(f"{'=' * 80}")

    # TODO if it doesn't run, just raise an exception
    epoch = 0  # Initialize epoch in case loop doesn't run
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        logger.info(
            f"Fold {fold + 1} | Epoch {epoch:03d}/{num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Prec: {train_metrics['precision']:.4f} | "
            f"Recall: {train_metrics['recall']:.4f} | "
            f"Spec: {train_metrics['specificity']:.4f}"
        )

        fold_history["train_loss"].append(train_metrics["loss"])
        fold_history["train_acc"].append(train_metrics["accuracy"])
        fold_history["epochs"].append(epoch)

        # Validate every val_every epochs
        if epoch % val_every == 0 or epoch == num_epochs:
            val_metrics = eval_epoch(model, val_loader, criterion, device)

            # Extract chunk and subject metrics
            chunk_metrics = val_metrics["chunk"]
            subject_metrics = val_metrics["subject"]

            logger.info(
                f"Fold {fold + 1} | Epoch {epoch:03d}/{num_epochs} | "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )
            logger.info(
                f"  Chunk-level  -> Acc: {chunk_metrics['accuracy']:.4f} | "
                f"Prec: {chunk_metrics['precision']:.4f} | "
                f"Recall: {chunk_metrics['recall']:.4f} | "
                f"Spec: {chunk_metrics['specificity']:.4f} | "
                f"Confusion: TP={chunk_metrics['tp']}, TN={chunk_metrics['tn']}, "
                f"FP={chunk_metrics['fp']}, FN={chunk_metrics['fn']}"
            )
            logger.info(
                f"  Subject-level -> Acc: {subject_metrics['accuracy']:.4f} | "
                f"Prec: {subject_metrics['precision']:.4f} | "
                f"Recall: {subject_metrics['recall']:.4f} | "
                f"Spec: {subject_metrics['specificity']:.4f} | "
                f"Confusion: TP={subject_metrics['tp']}, TN={subject_metrics['tn']}, "
                f"FP={subject_metrics['fp']}, FN={subject_metrics['fn']}"
            )

            fold_history["val_loss"].append(val_metrics["loss"])
            fold_history["val_acc"].append(
                subject_metrics["accuracy"]
            )  # Use subject-level for tracking

            # Combined metric: chunk accuracy weighted by subject accuracy
            # This handles small validation sets better (e.g., 6 subjects in 10-fold CV)
            combined_metric = chunk_metrics["accuracy"] * subject_metrics["accuracy"]

            # Save best model (based on combined metric)
            if combined_metric > best_val_acc:
                best_val_acc = combined_metric
                best_epoch = epoch
                best_model_path = checkpoint_dir / f"fold_{fold + 1}_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": best_val_acc,
                        "val_metrics": val_metrics,
                    },
                    best_model_path,
                )
                logger.info(
                    f"✓ Saved best model: {best_model_path} "
                    f"(combined={combined_metric:.4f}, chunk={chunk_metrics['accuracy']:.4f}, "
                    f"subject={subject_metrics['accuracy']:.4f})"
                )

            # Early stopping check (based on combined metric)
            if early_stopping(combined_metric, epoch):
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
        f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}"
    )

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_epoch": epoch,
        "history": fold_history,
    }


def train_cross_validation(
    condition: Literal["EC", "EO", "TASK"] = "EC",
    n_folds: int = 10,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    val_every: int = 2,
    save_every: int = 10,
    patience: int = 15,
    checkpoint_dir: Path = Path("checkpoints"),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    skip_ica: bool = False,
    channel: str | None = None,
) -> dict:
    """
    Train model using 10-fold cross-validation with comprehensive logging and checkpointing.

    :param str condition: EEG condition to use ("EC", "EO", "TASK", or None for all).
    :param int n_folds: Number of cross-validation folds.
    :param int batch_size: Batch size for training.
    :param int num_epochs: Maximum number of epochs per fold.
    :param float learning_rate: Learning rate for optimizer.
    :param int val_every: Validate every N epochs.
    :param int save_every: Save checkpoint every N epochs.
    :param int patience: Early stopping patience.
    :param Path checkpoint_dir: Directory to save checkpoints.
    :param torch.device device: Device to train on.
    :param bool skip_ica: If True, skip ICA artifact removal during preprocessing.
    :param str | None channel: Single channel to use (e.g., "Fp1"). If None, uses all channels.
    :return: Dictionary with cross-validation results.
    :rtype: dict
    """
    logger.info(f"{'=' * 80}")
    logger.info(f"Starting {n_folds}-Fold Cross-Validation Training")
    logger.info(f"Condition: {condition}, Batch Size: {batch_size}, LR: {learning_rate}")
    logger.info(f"Device: {device}")
    logger.info(f"Skip ICA: {skip_ica}")
    logger.info(f"{'=' * 80}")

    cv_results = {
        "fold_best_acc": [],
        "fold_best_epoch": [],
        "fold_final_epoch": [],
    }

    logger.info(
        "Step 1: Loading full dataset and converting to spectrograms (one-time preprocessing)..."
    )
    full_mdd_dataset = MDDDataset(
        condition=condition, preload=False, skip_ica=skip_ica, channel=channel
    )
    all_subjects = full_mdd_dataset.get_subjects()

    logger.info(f"Found {len(all_subjects)} subjects: {all_subjects}")

    full_spec_dataset = SpectrogramDataset(full_mdd_dataset)
    logger.info(f"Spectrograms created: {len(full_spec_dataset)} samples")

    # Verify spectrogram shape
    # SpectrogramDataset returns (list[spectrograms], label, subject), so [0][0][0]
    # gets first spectrogram
    spec_shape = full_spec_dataset[0][0][0].shape[1:]  # (H, W) without channel dim
    assert spec_shape == torch.Size([129, 41]), (
        f"Expected (129, 41), got {spec_shape}. The neural net was designed using this assumption."
    )
    logger.info(f"Spectrogram shape: {spec_shape}")

    logger.info(f"Step 2: Creating {n_folds}-fold cross-validation splits...")

    # Split subjects by class (stratified)
    healthy_subjects = [s for s in all_subjects if s.startswith("H ")]
    mdd_subjects = [s for s in all_subjects if s.startswith("MDD ")]

    logger.info(f"Healthy subjects: {len(healthy_subjects)}, MDD subjects: {len(mdd_subjects)}")

    # Shuffle subjects
    rng = np.random.RandomState(42)
    rng.shuffle(healthy_subjects)
    rng.shuffle(mdd_subjects)

    # Create folds for each class separately (stratified)
    healthy_folds = np.array_split(healthy_subjects, n_folds)
    mdd_folds = np.array_split(mdd_subjects, n_folds)

    for fold in range(n_folds):
        logger.info(f"{'=' * 80}")
        logger.info(f"FOLD {fold + 1}/{n_folds}")
        logger.info(f"{'=' * 80}")

        # Determine train/val subjects for this fold
        val_subjects = list(healthy_folds[fold]) + list(mdd_folds[fold])
        train_subjects = []
        for i in range(n_folds):
            if i != fold:
                train_subjects.extend(healthy_folds[i])
                train_subjects.extend(mdd_folds[i])

        # Convert numpy strings to regular strings for cleaner logging
        train_subjects_clean = [str(s) for s in train_subjects]
        val_subjects_clean = [str(s) for s in val_subjects]

        logger.info(f"Train subjects ({len(train_subjects)}): {train_subjects_clean}")
        logger.info(f"Val subjects ({len(val_subjects)}): {val_subjects_clean}")

        # Get indices for train/val based on subjects
        train_indices = full_spec_dataset.get_indices_for_subjects(train_subjects)
        val_indices = full_spec_dataset.get_indices_for_subjects(val_subjects)

        logger.info(f"Training samples: {len(train_indices)}")
        logger.info(f"Validation samples: {len(val_indices)}")

        # Create Subset datasets (reusing precomputed spectrograms!)
        train_spec_dataset = Subset(full_spec_dataset, train_indices)
        val_spec_dataset = Subset(full_spec_dataset, val_indices)

        # Create DataLoaders with custom collate function for spectrograms
        # batch_size refers to number of FILES, which will be flattened into individual spectrograms
        train_loader = DataLoader(
            train_spec_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_spectrograms,
        )
        val_loader = DataLoader(
            val_spec_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_spectrograms,
        )

        # Initialize model for this fold
        model = CNN_LSTM_DepCap(
            input_shape=spec_shape,
            in_channels=1,
            rnn_type="LSTM",
            rnn_hidden=100,
            dropout=0.2,
            num_classes=2,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train this fold
        fold_result = train_one_fold(
            fold=fold,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            save_every=save_every,
            val_every=val_every,
            patience=patience,
            checkpoint_dir=checkpoint_dir,
        )

        cv_results["fold_best_acc"].append(fold_result["best_val_acc"])
        cv_results["fold_best_epoch"].append(fold_result["best_epoch"])
        cv_results["fold_final_epoch"].append(fold_result["final_epoch"])

    # Print final cross-validation results
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{n_folds}-FOLD CROSS-VALIDATION RESULTS")
    logger.info(f"{'=' * 80}")

    mean_acc = np.mean(cv_results["fold_best_acc"])
    std_acc = np.std(cv_results["fold_best_acc"])

    logger.info("\nPer-fold best validation accuracy:")
    for i, acc in enumerate(cv_results["fold_best_acc"]):
        logger.info(f"  Fold {i + 1}: {acc:.4f} (epoch {cv_results['fold_best_epoch'][i]})")

    logger.info(f"\nMean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"Min Accuracy: {np.min(cv_results['fold_best_acc']):.4f}")
    logger.info(f"Max Accuracy: {np.max(cv_results['fold_best_acc']):.4f}")
    logger.info(f"{'=' * 80}\n")

    return cv_results


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
        choices=["EC", "EO", "TASK"],
        help="EEG condition to use",
    )
    preproc_group.add_argument(
        "--skip-ica",
        action="store_true",
        help="Skip ICA artifact removal (faster but less clean data)",
    )
    preproc_group.add_argument(
        "--channel",
        type=str,
        choices=list(MDDDataset.CHANNEL_MAPPING.values()),
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
        "--n-folds", type=int, default=10, help="Number of cross-validation folds"
    )
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of epochs per fold"
    )
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--val-every", type=int, default=2, help="Validate every N epochs")
    train_parser.add_argument(
        "--save-every", type=int, default=10, help="Save checkpoint every N epochs"
    )
    train_parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
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

    checkpoint_dir = Path(args.checkpoint_dir)

    # Setup logging to both console and file
    log_path = setup_logging(checkpoint_dir, args.condition, args.skip_ica, args.channel)

    logger.info("Starting MDD EEG Classification Training")
    logger.info("Hyperparameters:")
    logger.info(f"  Condition: {args.condition}")
    logger.info(f"  Channel: {args.channel}")
    logger.info(f"  N-Fold CV: {args.n_folds}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.lr}")
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
        condition=args.condition,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        val_every=args.val_every,
        save_every=args.save_every,
        patience=args.patience,
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
        f.write(f"Condition: {args.condition}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Skip ICA: {args.skip_ica}\n\n")
        f.write("Per-fold best validation accuracy:\n")
        for i, acc in enumerate(results["fold_best_acc"]):
            f.write(
                f"  Fold {i + 1}: {acc:.4f} (epoch {results['fold_best_epoch'][i]}, "
                f"stopped at epoch {results['fold_final_epoch'][i]})\n"
            )
        f.write(
            f"\nMean Accuracy: {np.mean(results['fold_best_acc']):.4f} ± {np.std(results['fold_best_acc']):.4f}\n"
        )
        f.write(f"Min Accuracy: {np.min(results['fold_best_acc']):.4f}\n")
        f.write(f"Max Accuracy: {np.max(results['fold_best_acc']):.4f}\n")

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

    # Load model checkpoint
    logger.info("Loading model checkpoint...")
    model = CNN_LSTM_DepCap(
        input_shape=EXPECTED_SPECTOGRAM_SHAPE,
        in_channels=1,
        rnn_type="LSTM",
        rnn_hidden=100,
        dropout=0.2,
        num_classes=2,
    )

    saved = torch.load(model_path, weights_only=False)  # TODO overriding some security check here
    model.load_state_dict(saved["model_state_dict"])
    model.to(device)
    model.eval()
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
    logger.info(
        f"Healthy predictions: {healthy_count} ({healthy_count / len(chunk_predictions) * 100:.1f}%)"
    )
    logger.info(f"MDD predictions: {mdd_count} ({mdd_count / len(chunk_predictions) * 100:.1f}%)")
    logger.info(f"Final Prediction: {final_class} (Class {final_prediction})")


def main() -> None:
    parser = get_arg_parser()
    args = parser.parse_args()

    # TODO channel still doesnt work
    if args.command == "train":
        train(args)
    elif args.command == "run":
        run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
