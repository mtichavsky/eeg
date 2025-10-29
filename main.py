import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset

from thesis.dataset import MDDDataset, SpectrogramDataset, collate_variable_length_eeg
from thesis.model import CNN_LSTM_DepCap

LOG_FORMAT = "[%(asctime)s %(levelname)s %(module)s.%(funcName)s] %(message)s"
LOG_LEVEL = "INFO"
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.

    :param int patience: Number of epochs with no improvement after which training stops.
    :param float min_delta: Minimum change to qualify as an improvement (default: 0.0).
    :param bool maximize: Whether to maximize the metric (True) or minimize it (False).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, maximize: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.maximize = maximize
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        :param float score: Current validation metric score.
        :param int epoch: Current epoch number.
        :return: True if training should stop, False otherwise.
        :rtype: bool
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.maximize:
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"Early stopping triggered. Best score: {self.best_score:.4f} "
                    f"at epoch {self.best_epoch}"
                )
                return True

        return False


# -------------------------
# Metrics
# -------------------------
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    y_true, y_pred : 1d arrays of ints (0/1). Returns dict with accuracy, precision, recall (sensitivity), specificity
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
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


# -------------------------
# Training loop
# -------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * xb.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.detach().cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = classification_metrics(all_labels, all_preds)
    avg_loss = running_loss / len(dataloader.dataset)
    metrics["loss"] = avg_loss
    return metrics


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            running_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.detach().cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = classification_metrics(all_labels, all_preds)
    avg_loss = running_loss / len(dataloader.dataset)
    metrics["loss"] = avg_loss
    return metrics


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
    Train model for one fold with comprehensive logging, checkpointing, and early stopping.

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

            logger.info(
                f"Fold {fold + 1} | Epoch {epoch:03d}/{num_epochs} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Prec: {val_metrics['precision']:.4f} | "
                f"Recall: {val_metrics['recall']:.4f} | "
                f"Spec: {val_metrics['specificity']:.4f} | "
                f"Confusion: TP={val_metrics['tp']}, TN={val_metrics['tn']}, "
                f"FP={val_metrics['fp']}, FN={val_metrics['fn']}"
            )

            fold_history["val_loss"].append(val_metrics["loss"])
            fold_history["val_acc"].append(val_metrics["accuracy"])

            # Save best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
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
                logger.info(f"✓ Saved best model: {best_model_path} (acc={best_val_acc:.4f})")

            # Early stopping check
            if early_stopping(val_metrics["accuracy"], epoch):
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
    condition: str = "EC",
    n_folds: int = 10,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    val_every: int = 2,
    save_every: int = 10,
    patience: int = 15,
    checkpoint_dir: Path = Path("checkpoints"),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
    :return: Dictionary with cross-validation results.
    :rtype: dict
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Starting {n_folds}-Fold Cross-Validation Training")
    logger.info(f"Condition: {condition}, Batch Size: {batch_size}, LR: {learning_rate}")
    logger.info(f"Device: {device}")
    logger.info(f"{'=' * 80}\n")

    cv_results = {
        "fold_best_acc": [],
        "fold_best_epoch": [],
        "fold_final_epoch": [],
    }

    logger.info("Step 1: Loading full dataset and converting to spectrograms (one-time preprocessing)...")
    full_mdd_dataset = MDDDataset(condition=condition, preload=False)
    all_subjects = full_mdd_dataset.get_subjects()

    logger.info(f"Found {len(all_subjects)} subjects: {all_subjects}")
    full_loader = DataLoader(
        full_mdd_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_variable_length_eeg,
    )

    full_spec_dataset = SpectrogramDataset(full_loader)
    logger.info(f"Spectrograms created: {len(full_spec_dataset)} samples")

    # Verify spectrogram shape
    spec_shape = full_spec_dataset[0][0].shape[1:]  # (H, W) without channel dim
    assert spec_shape == torch.Size([129, 41]), f"Expected (129, 41), got {spec_shape}"
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
        logger.info(f"\n{'=' * 80}")
        logger.info(f"FOLD {fold + 1}/{n_folds}")
        logger.info(f"{'=' * 80}")

        # Determine train/val subjects for this fold
        val_subjects = list(healthy_folds[fold]) + list(mdd_folds[fold])
        train_subjects = []
        for i in range(n_folds):
            if i != fold:
                train_subjects.extend(healthy_folds[i])
                train_subjects.extend(mdd_folds[i])

        logger.info(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        logger.info(f"Val subjects ({len(val_subjects)}): {val_subjects}")

        # Get indices for train/val based on subjects
        train_indices = full_spec_dataset.get_indices_for_subjects(train_subjects)
        val_indices = full_spec_dataset.get_indices_for_subjects(val_subjects)

        logger.info(f"Training samples: {len(train_indices)}")
        logger.info(f"Validation samples: {len(val_indices)}")

        # Create Subset datasets (reusing precomputed spectrograms!)
        train_spec_dataset = Subset(full_spec_dataset, train_indices)
        val_spec_dataset = Subset(full_spec_dataset, val_indices)

        # Create DataLoaders
        train_loader = DataLoader(
            train_spec_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_spec_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
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


if __name__ == "__main__":
    # -------------------------
    # Hyperparameters
    # -------------------------
    CONDITION = "EC"  # "EC", "EO", "TASK", or None for all conditions
    N_FOLDS = 10
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 50
    VAL_EVERY = 2  # Validate every N epochs
    SAVE_EVERY = 10  # Save checkpoint every N epochs
    PATIENCE = 15  # Early stopping patience
    CHECKPOINT_DIR = Path("checkpoints")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Starting MDD EEG Classification Training")
    logger.info("Hyperparameters:")
    logger.info(f"  Condition: {CONDITION}")
    logger.info(f"  N-Fold CV: {N_FOLDS}")
    logger.info(f"  Batch Size: {BATCH_SIZE}")
    logger.info(f"  Learning Rate: {LR}")
    logger.info(f"  Max Epochs: {EPOCHS}")
    logger.info(f"  Validation Every: {VAL_EVERY} epochs")
    logger.info(f"  Save Checkpoint Every: {SAVE_EVERY} epochs")
    logger.info(f"  Early Stopping Patience: {PATIENCE} epochs")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Checkpoint Directory: {CHECKPOINT_DIR}")

    # Run 10-fold cross-validation training
    results = train_cross_validation(
        condition=CONDITION,
        n_folds=N_FOLDS,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        learning_rate=LR,
        val_every=VAL_EVERY,
        save_every=SAVE_EVERY,
        patience=PATIENCE,
        checkpoint_dir=CHECKPOINT_DIR,
        device=DEVICE,
    )

    # Save final results to file
    results_file = CHECKPOINT_DIR / "cv_results.txt"
    with open(results_file, "w") as f:
        f.write(f"{N_FOLDS}-Fold Cross-Validation Results\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Condition: {CONDITION}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LR}\n\n")
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
