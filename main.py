import os
import logging
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

from thesis.dataset import get_preprocessed_chunks, SFREQ
from thesis.model import CNN_LSTM_DepCap
import pandas as pd
from scipy.stats import zscore
import numpy as np
from scipy.signal import stft

import torch.nn.functional as F

LOG_FORMAT = "[%(asctime)s %(levelname)s %(module)s.%(funcName)s] %(message)s"
LOG_LEVEL = "INFO"
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# -------------------------
# Dataset
# -------------------------
class SpectrogramDataset(Dataset):
    """
    Dataset that can either:
      - load precomputed spectrogram numpy arrays (as .npy) with labels, or
      - compute spectrograms from raw_eeg segments (numpy arrays).
    Expected data structure for precomputed:
      root/
        images/
          sample_000.npy  # shape (C,H,W) or (H,W)
        labels.csv  # lines: filename,label (0/1)
    Or you can pass `raw_segments` and `labels` lists to this class.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        csv_labels: Optional[str] = None,
        raw_segments: Optional[List[np.ndarray]] = None,
        raw_labels: Optional[List[int]] = None,
        transform=None,
        fs: int = 256,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        to_rgb: bool = False,
        target_size=(254, 342),
    ):
        super().__init__()
        self.transform = transform
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.to_rgb = to_rgb
        self.target_size = target_size

        if root is not None and csv_labels is not None:
            import pandas as pd

            df = pd.read_csv(csv_labels, header=None)
            # df: filename,label
            self.files = [os.path.join(root, str(fn)) for fn in df[0].astype(str).tolist()]
            self.labels = df[1].astype(int).tolist()
            self.mode = "precomp"
        elif raw_segments is not None and raw_labels is not None:
            assert len(raw_segments) == len(raw_labels)
            self.raw_segments = raw_segments
            self.labels = raw_labels
            self.mode = "raw"
        else:
            raise ValueError("Provide either (root + csv_labels) or (raw_segments + raw_labels)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        if self.mode == "precomp":
            arr = np.load(self.files[idx])
            # ensure shape (C,H,W)
            if arr.ndim == 2:
                arr = arr[None, ...]
        else:
            seg = self.raw_segments[idx]
            arr = eeg_to_spectrogram(
                seg,
                fs=self.fs,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                to_rgb=self.to_rgb,
                target_size=self.target_size,
            )
        # to tensor
        x = torch.from_numpy(arr).float()
        if self.transform:
            x = self.transform(x)
        return x, label


# -------------------------
# Model
# -------------------------


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


if __name__ == "__main__":
    # Hyperparams
    BATCH = 128
    LR = 1e-4
    EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO batch
    # logger.info(f"Starting training script with BATCH={BATCH}, LR={LR}, EPOCHS={EPOCHS}")
    logger.info(f"Using device: {DEVICE}")

    # jeden ten spektogram v jejich implementaci ma 254x342
    # muj return STFT ma 129, 21 shape - je ale divne, ze to nemaji 3d (pro vic diod)

    # Replace with your data loading strategy:
    # Option A: precomputed spectrogram .npy files with labels.csv
    # dataset = SpectrogramDataset(root="data/images", csv_labels="data/labels.csv", to_rgb=False)
    # Option B: raw EEG windows
    # raw_segments = [...]  # list of np.ndarray (1D EEG windows)
    # raw_labels   = [...]  # list of ints 0/1
    # dataset = SpectrogramDataset(raw_segments=raw_segments, raw_labels=raw_labels, fs=256, nperseg=256, to_rgb=False)

    # Example: small dummy dataset for sanity-check (remove in real use)
    # create dummy data: 100 samples of white noise
    # dummy_X = [np.random.randn(256*2) for _ in range(200)]
    # dummy_y = [0 if i<100 else 1 for i in range(200)]
    # dataset = SpectrogramDataset(raw_segments=dummy_X, raw_labels=dummy_y, fs=128, nperseg=128, to_rgb=False, target_size=(254,342))
    # # train / val split
    # from sklearn.model_selection import train_test_split
    # idx = list(range(len(dataset)))
    # train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=dataset.labels, random_state=42)
    # train_ds = torch.utils.data.Subset(dataset, train_idx)
    # val_ds = torch.utils.data.Subset(dataset, val_idx)
    # train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    # val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    MDD_CHUNK_SHAPE=(129, 41)
    # model
    logger.info(f"Creating CNN_LSTM_DepCap model with input_shape={MDD_CHUNK_SHAPE}")
    model = CNN_LSTM_DepCap(
        input_shape=MDD_CHUNK_SHAPE, in_channels=1, rnn_type="LSTM", rnn_hidden=100, dropout=0.2, num_classes=2
    )
    model = model.to(DEVICE)
    logger.info(f"Model moved to device: {DEVICE}")

    logger.info("Loading preprocessed chunks...")
    chunks = get_preprocessed_chunks()
    logger.info(f"Loaded {len(chunks)} chunks")

    processed_count = 0
    skipped_count = 0

    for idx, chunk in enumerate(chunks):
        norm = pd.DataFrame(zscore(chunk, axis=0))  # hope this is correct axis
        channel = chunk["Fp1"]

        if len(channel) < 256: # Otherwise the STFT might not work
            logger.debug(f"Skipping chunk {idx} due to insufficient length ({len(channel)} < 256 samples)")
            skipped_count += 1
            continue

        f, t, Zxx = stft(channel, fs=SFREQ, nperseg=256, noverlap=192, window='hamming')
        Zxx_mag = np.log1p(np.abs(Zxx))
        x = torch.tensor(Zxx_mag, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        y = model.forward(x)

        probabilities = F.softmax(y, dim=1)
        logger.info(f"Chunk {idx}: Probabilities: {probabilities.detach().numpy()}")
        processed_count += 1

    logger.info(f"Processing complete: {processed_count} chunks processed, {skipped_count} chunks skipped")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # best_val_acc = 0.0
    # for epoch in range(1, EPOCHS+1):
    #     train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    #     val_metrics = eval_epoch(model, val_loader, criterion, DEVICE)
    #     logger.info(f"Epoch {epoch:02d} | Train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
    #                 f"Val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f}")
    #     if val_metrics['accuracy'] > best_val_acc:
    #         best_val_acc = val_metrics['accuracy']
    #         torch.save(model.state_dict(), "best_depcap_model.pth")
    #         logger.info("Saved best model.")
