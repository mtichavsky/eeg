"""
Example training script using 10-fold cross-validation
Replicates the approach from EEG MDD papers
"""

import numpy as np
import torch
import torch.nn as nn

from thesis.dataset import create_cross_validation_splits


def train_with_cross_validation(
    model_class,
    condition="EC",
    n_folds=10,
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train a model using 10-fold cross-validation.

    Args:
        model_class: Class to instantiate for each fold (e.g., YourModel)
        condition: EEG condition to use
        n_folds: Number of folds (10 for paper replication)
        batch_size: Batch size
        num_epochs: Epochs per fold
        learning_rate: Learning rate
        device: Device to train on

    Returns:
        dict: Results for each fold
    """
    # Storage for results
    fold_results = {
        "val_accuracy": [],
        "val_loss": [],
        "best_epoch": [],
    }

    # Create cross-validation splits
    cv_splits = create_cross_validation_splits(
        condition=condition,
        n_folds=n_folds,
        batch_size=batch_size,
        random_seed=42,
    )

    # Iterate over folds
    for fold, (train_loader, val_loader, train_dataset, val_dataset) in enumerate(cv_splits):
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'=' * 70}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        # Initialize model, optimizer, loss for this fold
        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Track best validation accuracy for this fold
        best_val_acc = 0.0
        best_epoch = 0

        # Training loop for this fold
        for epoch in range(num_epochs):
            # ==================== TRAINING ====================
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_idx, batch in enumerate(train_loader):
                # Get data
                eeg = batch["eeg"].to(device)  # (batch_size, 6, 2500)
                labels = batch["label"].to(device)  # (batch_size,)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(eeg)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Statistics
                train_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

            train_accuracy = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # ==================== VALIDATION ====================
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch["eeg"].to(device)
                    labels = batch["label"].to(device)

                    outputs = model(eeg)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    predictions = outputs.argmax(dim=1)
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

            val_accuracy = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)

            # Track best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_epoch = epoch + 1
                # Optionally save model: torch.save(model.state_dict(), f'fold_{fold}_best.pt')

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:3d}/{num_epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
                )

        # Store results for this fold
        fold_results["val_accuracy"].append(best_val_acc)
        fold_results["val_loss"].append(avg_val_loss)
        fold_results["best_epoch"].append(best_epoch)

        print(
            f"\nFold {fold + 1} completed - Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}"
        )

    # ==================== FINAL RESULTS ====================
    print(f"\n{'=' * 70}")
    print("10-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'=' * 70}")

    mean_acc = np.mean(fold_results["val_accuracy"])
    std_acc = np.std(fold_results["val_accuracy"])

    print("\nPer-fold accuracy:")
    for i, acc in enumerate(fold_results["val_accuracy"]):
        print(f"  Fold {i + 1}: {acc:.4f}")

    print(f"\nMean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Min Accuracy: {np.min(fold_results['val_accuracy']):.4f}")
    print(f"Max Accuracy: {np.max(fold_results['val_accuracy']):.4f}")

    return fold_results


# Example usage
if __name__ == "__main__":
    # Define your model (replace with actual model)
    class SimpleEEGModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(6, 32, kernel_size=50)
            self.pool = nn.MaxPool1d(4)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=25)
            self.fc1 = nn.Linear(64 * 608, 128)  # Adjust based on your architecture
            self.fc2 = nn.Linear(128, 2)  # Binary classification: H vs MDD
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            # x shape: (batch_size, 6, 2500)
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # Run 10-fold cross-validation
    results = train_with_cross_validation(
        model_class=SimpleEEGModel,
        condition="EC",
        n_folds=10,
        batch_size=32,
        num_epochs=50,
        learning_rate=0.001,
    )
