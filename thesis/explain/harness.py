"""
Post-hoc evaluation harness for ablation experiments.

Reconstructs the exact cross-validation folds a training run used, reloads each fold's best
checkpoint, and re-scores the validation set under an ablation. Scoring goes through the same
:func:`main.eval_epoch` the training loop uses, so an ablated accuracy is directly comparable
with the numbers reported in the paper.

Two safeguards matter enough to be non-optional:

* **Fold reconstruction is verified, not assumed.** A single ``RandomState`` is threaded
  through every dataset-preparation call in a fixed order, so fold membership depends on which
  datasets are loaded and in what order. :func:`verify_baseline` re-runs the *unablated*
  evaluation and checks it against the metrics stored inside the checkpoint. If those disagree,
  the folds were rebuilt wrongly and every downstream number is meaningless.
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
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from thesis.data_preparation import CVFolds, build_cv_folds
from thesis.dataset import collate_spectrograms, num_model_channels
from thesis.model_factory import create_model
from thesis.stft import EXPECTED_SPECTROGRAM_SHAPE

logger = logging.getLogger(__name__)

#: Metrics whose fold-wise values an ablation study compares against baseline.
TRACKED_METRICS = ("accuracy", "sensitivity", "specificity")


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
        raw=cfg,
    )


class AblatedModel(nn.Module):
    """
    Wraps a model so an ablation applies inside its forward pass.

    Keeping the ablation here rather than in the evaluation loop means
    :func:`main.eval_epoch` runs unmodified, so ablated and baseline numbers are produced by
    exactly the same code.

    :param torch.nn.Module model: The trained model to wrap.
    :param input_transform: Optional callable applied to each input batch, e.g. band occlusion
        or the hemisphere mirror.
    :param torch.Tensor channel_mask: Optional boolean channel mask forwarded to the wrapped
        model, which drops those channels' tokens. Only ``AllTransformerV4`` accepts one.
    """

    def __init__(
        self,
        model: nn.Module,
        input_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        channel_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_transform = input_transform
        self.channel_mask = channel_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the ablation, then delegate.

        :param torch.Tensor x: Input batch.
        :return: Class logits.
        :rtype: torch.Tensor
        """
        if self.input_transform is not None:
            x = self.input_transform(x)
        if self.channel_mask is not None:
            return cast(torch.Tensor, self.model(x, channel_mask=self.channel_mask))
        return cast(torch.Tensor, self.model(x))


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
        spec_shape=EXPECTED_SPECTROGRAM_SHAPE,
        dropout=config.dropout,
        num_classes=config.num_classes,
        device=device,
        in_channels=num_model_channels(config.channel),
    )

    checkpoint = torch.load(path, weights_only=False, map_location=device)
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
    """One fold's validation data plus the model trained without it."""

    fold: int
    model: nn.Module
    loader: DataLoader
    subject_dataset_map: dict[str, str]
    checkpoint: dict[str, Any]
    train_loader: DataLoader

    @property
    def stored_metrics(self) -> dict[str, Any] | None:
        """
        Validation metrics recorded when this checkpoint was saved.

        :return: The stored ``val_metrics`` dict, or ``None`` for older checkpoints.
        :rtype: dict | None
        """
        return cast(dict[str, Any] | None, self.checkpoint.get("val_metrics"))


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
        Pass it in when iterating more than once — dataset preparation is the slow step.
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

        train_dataset, val_dataset, subject_dataset_map = folds.datasets_for_fold(
            fold, config.dataset
        )
        collate = None if folds.use_raw_eeg else collate_spectrograms
        loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
        )
        # Shuffled so that a truncated pass over it still gives an unbiased occlusion
        # reference; see reference_spectrogram.
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        yield FoldContext(fold, model, loader, subject_dataset_map, checkpoint, train_loader)


def reference_spectrogram(context: FoldContext, max_samples: int = 2048) -> torch.Tensor:
    """
    Mean spectrogram over this fold's *training* data, for use as the occlusion baseline.

    Training-fold only: computing it over the validation set would leak the evaluation data
    into the ablation's reference value.

    :param FoldContext context: The fold whose training data to average.
    :param int max_samples: Stop after this many chunks. The mean of a few thousand
        spectrograms is stable well below the differences ablation measures, and the loader is
        shuffled so a truncated pass stays unbiased.
    :return: Mean spectrogram of shape ``(C, F, T)``.
    :rtype: torch.Tensor
    """
    from thesis.explain.masks import spectrogram_reference

    batches: list[torch.Tensor] = []
    seen = 0
    for inputs, _, _ in context.train_loader:
        batches.append(inputs)
        seen += inputs.shape[0]
        if seen >= max_samples:
            break
    return spectrogram_reference(batches)


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
        test_mode=False,
        chunk_duration=config.chunk_duration,
    )


def evaluate(
    context: FoldContext,
    config: RunConfig,
    device: torch.device,
    input_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    channel_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """
    Score one fold under an ablation, using the training loop's own evaluation function.

    :param FoldContext context: The fold to score.
    :param RunConfig config: The run's configuration.
    :param torch.device device: Device to evaluate on.
    :param input_transform: Optional input ablation.
    :param torch.Tensor channel_mask: Optional channel mask.
    :return: The :func:`main.eval_epoch` result dict.
    :rtype: dict
    """
    # Imported here: main imports this package's siblings, so a module-level import would
    # create a cycle.
    from main import eval_epoch

    model: nn.Module = context.model
    if input_transform is not None or channel_mask is not None:
        model = AblatedModel(context.model, input_transform, channel_mask).to(device)
        model.eval()

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        return eval_epoch(
            model,
            context.loader,
            criterion,
            device,
            num_classes=config.num_classes,
            subject_dataset_map=context.subject_dataset_map,
        )


def verify_baseline(
    context: FoldContext, config: RunConfig, device: torch.device, tolerance: float = 1e-4
) -> tuple[bool, float]:
    """
    Check that the reconstructed fold reproduces the checkpoint's own stored metrics.

    This is the gate for the whole analysis. Fold membership depends on a shared, stateful
    random generator replayed across dataset-preparation calls; if the replay diverges, the
    validation set is simply a different set of subjects and every ablation number computed
    from it is void. Run this before trusting any result.

    :param FoldContext context: The fold to check.
    :param RunConfig config: The run's configuration.
    :param torch.device device: Device to evaluate on.
    :param float tolerance: Allowed absolute difference in chunk accuracy.
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
    matched = difference <= tolerance

    level = logger.info if matched else logger.error
    level(
        f"Fold {context.fold + 1} baseline: stored={expected:.6f} recomputed={actual:.6f} "
        f"diff={difference:.2e} {'OK' if matched else 'MISMATCH'}"
    )
    return matched, difference


def extract(metrics: dict[str, Any]) -> dict[str, float]:
    """
    Flatten an :func:`main.eval_epoch` result into the scalars ablation studies compare.

    :param dict metrics: An ``eval_epoch`` result.
    :return: ``chunk_*`` and ``subject_*`` scalars, plus ``loss``. Keys absent for the current
        class count (sensitivity and specificity are binary-only) are omitted.
    :rtype: dict[str, float]
    """
    flat: dict[str, float] = {"loss": float(metrics["loss"])}
    for level in ("chunk", "subject"):
        for name in TRACKED_METRICS:
            value = metrics[level].get(name)
            if value is not None:
                flat[f"{level}_{name}"] = float(value)
    for key, value in metrics["chunk"].items():
        if key.startswith("recall_"):
            flat[f"chunk_{key}"] = float(value)
    return flat


def write_results(payload: dict[str, Any], destination: Path) -> Path:
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
