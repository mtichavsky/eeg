"""
Post-hoc harness for explaining trained cross-validation runs.

Reconstructs the exact cross-validation folds a training run used and reloads each fold's
best checkpoint, so every fold's model is explained on the subjects it never saw in training.

Two safeguards matter enough to be non-optional:

* **Fold reconstruction is verified, not assumed.** A single ``RandomState`` is threaded
  through every dataset-preparation call in a fixed order, so fold membership depends on which
  datasets are loaded and in what order. :func:`verify_baseline` re-runs the *unablated*
  evaluation (through the same :func:`thesis.evaluation.eval_epoch` the training loop uses)
  and checks it against the metrics stored inside the checkpoint. If those disagree, the folds
  were rebuilt wrongly and every downstream number is meaningless.
* **Checkpoints are loaded strictly.** ``model.load_state_dict(..., strict=False)`` still
  raises a ``RuntimeError`` immediately if a parameter with a matching name has the wrong
  *shape* — PyTorch does this unconditionally, regardless of ``strict``. What ``strict=False``
  does silently ignore is a mismatch in parameter *names*: a checkpoint written before a
  refactor that renamed, added, or removed a layer would load with those layers left at their
  random initialisation, no error raised, no name in the loaded checkpoint's keys to notice
  it by. That is exactly how a stale checkpoint produces plausible-looking garbage.
  :func:`load_fold_model` defaults to ``strict=True`` so a name mismatch raises too.
"""

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from thesis.data_preparation import CVFolds, build_cv_folds
from thesis.dataset import collate_spectrograms, num_model_channels
from thesis.evaluation import eval_epoch
from thesis.model_factory import create_model
from thesis.stft import FREQ_CUTOFF_HZ, spectrogram_shape

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration of a completed training run, recovered from its ``results.txt``."""

    model: str
    dataset: Literal["mdd", "cane", "sad", "all"]
    condition: str
    channel: str
    class_mode: str
    n_folds: int
    batch_size: int
    dropout: float
    chunk_duration: float
    skip_artifact_removal: bool
    checkpoint_dir: Path
    test_mode: bool = False
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        """
        Number of output classes.

        :return: 2 or 4.
        :rtype: int
        """
        return int(self.class_mode)

    @property
    def conditions(self) -> list[str]:
        """
        Conditions as the fold builder expects them, upper-cased and split on ``+``.

        :return: e.g. ``["EC", "EO"]``.
        :rtype: list[str]
        """
        return self.condition.upper().split("+")

    @property
    def freq_cutoff_hz(self) -> float:
        """
        Spectrogram frequency cutoff the run was trained with.

        :return: The recorded ``freq_cutoff``; runs predating the flag used 70 Hz.
        :rtype: float
        """
        return float(self.raw.get("freq_cutoff", FREQ_CUTOFF_HZ))

    @property
    def spec_shape(self) -> tuple[int, int]:
        """
        Spectrogram ``(F, T)`` shape the run's models expect.

        :return: e.g. ``(72, 41)`` at 70 Hz, ``(31, 41)`` at 30 Hz.
        :rtype: tuple[int, int]
        """
        return spectrogram_shape(self.freq_cutoff_hz)

    @property
    def in_channels(self) -> int:
        """
        Number of model input channels.

        :return: 8 for ``--channel all``, else 1.
        :rtype: int
        """
        return num_model_channels(self.channel)


def _coerce(value: str) -> str | bool | None:
    """
    Convert the string forms ``results.txt`` writes back into Python values.

    :param str value: Raw value text.
    :return: ``None`` for ``"None"``, a bool for ``"True"``/``"False"``, else the string.
    :rtype: str | bool | None
    """
    if value == "None":
        return None
    if value in {"True", "False"}:
        return value == "True"
    return value


def read_run_config(checkpoint_dir: Path) -> RunConfig:
    """
    Recover a run's configuration from the ``Configuration:`` block of its ``results.txt``.

    Checkpoints store no configuration of their own — not the model name, class count, or
    channel spec — so this file is the only record of how a run was set up.

    :param Path checkpoint_dir: Directory holding ``results.txt`` and ``fold_N_best.pth``.
    :return: The parsed configuration.
    :rtype: RunConfig
    :raises FileNotFoundError: If ``results.txt`` is missing.
    :raises ValueError: If the configuration block is absent or incomplete.
    """
    results = checkpoint_dir / "results.txt"
    if not results.is_file():
        raise FileNotFoundError(f"No results.txt in {checkpoint_dir}")

    cfg: dict[str, str] = {}
    in_block = False
    for line in results.read_text().splitlines():
        if line.startswith("Configuration:"):
            in_block = True
            continue
        if in_block:
            match = re.match(r"^  (\w+): (.*)$", line)
            if not match:
                break
            cfg[match.group(1)] = match.group(2)

    if not cfg:
        raise ValueError(f"No Configuration block found in {results}")
    missing = {"model", "dataset", "condition", "channel", "class_mode", "n_folds"} - cfg.keys()
    if missing:
        raise ValueError(f"{results} is missing configuration keys: {sorted(missing)}")

    return RunConfig(
        model=cfg["model"],
        dataset=cast(Literal["mdd", "cane", "sad", "all"], cfg["dataset"]),
        condition=cfg["condition"],
        channel=cfg["channel"],
        class_mode=cfg["class_mode"],
        n_folds=int(cfg["n_folds"]),
        batch_size=int(cfg.get("batch_size", 64)),
        dropout=float(cfg.get("dropout", 0.5)),
        chunk_duration=float(cfg.get("chunk_duration", 10.0)),
        skip_artifact_removal=bool(_coerce(cfg.get("skip_artifact_removal", "False"))),
        checkpoint_dir=checkpoint_dir,
        test_mode=bool(_coerce(cfg.get("test_mode", "False"))),
        raw=cfg,
    )


def load_fold_model(
    config: RunConfig, fold: int, device: torch.device, strict: bool = True
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Load one fold's best checkpoint into a freshly built model.

    :param RunConfig config: The run's configuration.
    :param int fold: Zero-indexed fold number; checkpoints are named one-indexed.
    :param torch.device device: Device to place the model on.
    :param bool strict: Reject any state-dict key-name mismatch. Leave this on: a shape
        mismatch on a matching key (e.g. a ``proj`` layer sized for a different spectrogram
        cutoff) already raises a ``RuntimeError`` unconditionally, even with ``strict=False``
        below — but a checkpoint whose parameter *names* no longer match the current model
        code (say, after a refactor renamed or dropped a layer) would otherwise load with
        those layers silently left at their random initialisation. This flag turns that into
        a ``RuntimeError`` too.
    :return: ``(model in eval mode, checkpoint dict)``.
    :rtype: tuple[torch.nn.Module, dict]
    :raises FileNotFoundError: If the checkpoint file is absent.
    :raises RuntimeError: If the state dict has a shape mismatch on any key (raised directly by
        ``load_state_dict``, regardless of ``strict``), or if ``strict`` and the state dict has
        missing/unexpected key names.
    """
    path = config.checkpoint_dir / f"fold_{fold + 1}_best.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {path}")

    model = create_model(
        model_name=config.model,
        spec_shape=config.spec_shape,
        dropout=config.dropout,
        num_classes=config.num_classes,
        device=device,
        in_channels=config.in_channels,
    )

    checkpoint = torch.load(path, weights_only=False, map_location=device)
    recorded_cutoff = float(checkpoint.get("freq_cutoff_hz", FREQ_CUTOFF_HZ))
    if recorded_cutoff != config.freq_cutoff_hz:
        raise RuntimeError(
            f"{path} was trained with freq_cutoff={recorded_cutoff}, but results.txt says "
            f"{config.freq_cutoff_hz}"
        )
    # A shape mismatch on a matching key (e.g. a differently-sized proj layer) raises here
    # directly from load_state_dict regardless of strict; only a genuine key-name mismatch
    # reaches the check below.
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            f"{path} does not match {config.model}. "
            f"Missing: {sorted(incompatible.missing_keys)}. "
            f"Unexpected: {sorted(incompatible.unexpected_keys)}. "
            "A checkpoint saved by a different model architecture is the usual cause."
        )

    model.eval()
    return model, checkpoint


@dataclass
class FoldContext:
    """
    One fold's validation data plus the model trained without it.

    ``val_datasets`` and ``train_datasets`` give the dataset key (``mdd``/``cane``/``sad``) of
    every index of the validation and training sets. They are derived from the fold's dataset
    structure, not from subject IDs, which are not unique across datasets (``"H S1 EC"`` exists
    in several).
    """

    fold: int
    model: nn.Module
    loader: DataLoader
    subject_dataset_map: dict[str, str]
    checkpoint: dict[str, Any]
    train_dataset: Dataset
    val_datasets: list[str]
    train_datasets: list[str]

    @property
    def stored_metrics(self) -> dict[str, Any] | None:
        """
        Validation metrics recorded when this checkpoint was saved.

        :return: The stored ``val_metrics`` dict, or ``None`` for older checkpoints.
        :rtype: dict | None
        """
        return cast(dict[str, Any] | None, self.checkpoint.get("val_metrics"))


def index_datasets(dataset: Dataset, folds: CVFolds) -> list[str]:
    """
    Dataset key of every index of a fold's train or validation set.

    For ``--dataset all`` the fold set is a ``ConcatDataset`` of one ``Subset`` per source
    dataset, and each subset's backing dataset identifies its source. Single-dataset runs are
    all one key.

    :param torch.utils.data.Dataset dataset: A train or validation set from
        :meth:`CVFolds.datasets_for_fold`.
    :param CVFolds folds: The folds it came from.
    :return: One key per index.
    :rtype: list[str]
    :raises ValueError: If a part of a combined set cannot be matched to a source dataset.
    """
    size = len(dataset)  # type: ignore[arg-type]
    if folds.dataset_type != "all":
        return [folds.dataset_type] * size

    sources = {
        id(folds.mdd_flat_dataset): "mdd",
        id(folds.cane_flat_dataset): "cane",
        id(folds.sad_flat_dataset): "sad",
    }
    if not isinstance(dataset, ConcatDataset):
        raise ValueError(f"Expected a ConcatDataset for --dataset all, got {type(dataset)}")
    keys: list[str] = []
    for part in dataset.datasets:
        backing = part.dataset if isinstance(part, Subset) else part
        if id(backing) not in sources:
            raise ValueError("Cannot match a fold subset to MDD, CANE or SAD")
        keys.extend([sources[id(backing)]] * len(part))  # type: ignore[arg-type]
    return keys


def iter_folds(
    config: RunConfig,
    device: torch.device,
    folds: CVFolds | None = None,
    fold_numbers: list[int] | None = None,
) -> Iterator[FoldContext]:
    """
    Yield each fold's validation loader together with the model trained on the other folds.

    :param RunConfig config: The run to reproduce.
    :param torch.device device: Device to evaluate on.
    :param CVFolds folds: Prebuilt folds from :func:`build_folds`; built here when omitted.
        Pass it in when iterating more than once, since dataset preparation is the slow step.
    :param fold_numbers: Zero-indexed folds to visit; all of them when omitted. Folds whose
        checkpoint is missing are skipped with a warning rather than aborting the study.
    :yield: One :class:`FoldContext` per fold.
    """
    folds = folds if folds is not None else build_folds(config)
    wanted = fold_numbers if fold_numbers is not None else list(range(config.n_folds))

    for fold in wanted:
        try:
            model, checkpoint = load_fold_model(config, fold, device)
        except FileNotFoundError as exc:
            logger.warning(f"Skipping fold {fold + 1}: {exc}")
            continue

        train_dataset, val_dataset, subject_dataset_map = folds.datasets_for_fold(fold)
        collate = None if folds.use_raw_eeg else collate_spectrograms
        loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
        )
        yield FoldContext(
            fold=fold,
            model=model,
            loader=loader,
            subject_dataset_map=subject_dataset_map,
            checkpoint=checkpoint,
            train_dataset=train_dataset,
            val_datasets=index_datasets(val_dataset, folds),
            train_datasets=index_datasets(train_dataset, folds),
        )


def build_folds(config: RunConfig) -> CVFolds:
    """
    Rebuild the folds a run used.

    :param RunConfig config: The run to reproduce.
    :return: The reconstructed folds.
    :rtype: CVFolds
    """
    return build_cv_folds(
        config.dataset,
        config.conditions,
        config.class_mode,
        config.n_folds,
        config.channel,
        config.model,
        skip_artifact_removal=config.skip_artifact_removal,
        augmentation=None,  # never augment during evaluation
        test_mode=config.test_mode,
        chunk_duration=config.chunk_duration,
        freq_cutoff_hz=config.freq_cutoff_hz,
    )


def evaluate(context: FoldContext, config: RunConfig, device: torch.device) -> dict[str, Any]:
    """
    Score one fold with the training loop's own evaluation function.

    :param FoldContext context: The fold to score.
    :param RunConfig config: The run's configuration.
    :param torch.device device: Device to evaluate on.
    :return: The :func:`thesis.evaluation.eval_epoch` result dict.
    :rtype: dict
    """
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        return eval_epoch(
            context.model,
            context.loader,
            criterion,
            device,
            num_classes=config.num_classes,
            subject_dataset_map=context.subject_dataset_map,
        )


def verify_baseline(
    context: FoldContext, config: RunConfig, device: torch.device, max_flips: float = 1.5
) -> tuple[bool, float]:
    """
    Check that the reconstructed fold reproduces the checkpoint's own stored metrics.

    This is the gate for the whole analysis. Fold membership depends on a shared, stateful
    random generator replayed across dataset-preparation calls; if the replay diverges, the
    validation set is simply a different set of subjects and every number computed from it is
    void. Run this before trusting any result.

    :param FoldContext context: The fold to check.
    :param RunConfig config: The run's configuration.
    :param torch.device device: Device to evaluate on.
    :param float max_flips: Allowed difference in chunk accuracy, in chunks. The default of
        1.5 tolerates one borderline chunk whose prediction flips because of hardware numerics
        (CPU versus GPU, TF32). A wrongly rebuilt fold holds different subjects and differs by
        far more.
    :return: ``(matched, absolute_difference)``. ``matched`` is ``False`` with a difference of
        ``nan`` when the checkpoint stores no metrics to compare against.
    :rtype: tuple[bool, float]
    """
    stored = context.stored_metrics
    if stored is None:
        logger.warning(f"Fold {context.fold + 1}: checkpoint stores no val_metrics to verify")
        return False, float("nan")

    recomputed = evaluate(context, config, device)
    expected = float(stored["chunk"]["accuracy"])
    actual = float(recomputed["chunk"]["accuracy"])
    difference = abs(expected - actual)
    n_chunks = len(context.loader.dataset)  # type: ignore[arg-type]
    matched = difference <= max_flips / n_chunks

    level = logger.info if matched else logger.error
    level(
        f"Fold {context.fold + 1} baseline: stored={expected:.6f} recomputed={actual:.6f} "
        f"diff={difference:.2e} {'OK' if matched else 'MISMATCH'}"
    )
    return matched, difference


def write_explain_results(payload: dict[str, Any], destination: Path) -> Path:
    """
    Write an experiment's results as JSON.

    :param dict payload: Serialisable results.
    :param Path destination: Output file; parent directories are created.
    :return: The path written.
    :rtype: Path
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info(f"Wrote {destination}")
    return destination
