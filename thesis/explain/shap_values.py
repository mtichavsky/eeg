"""
Grouped SHAP values for spectrogram models.

The ``shap`` library does all of the Shapley arithmetic. This module supplies what is specific
to EEG spectrograms:

* a **value function** (:func:`make_value_fn`) that turns an on/off vector over players into
  composite spectrograms (present players from the explained chunk, absent ones from a
  background chunk) and returns the model output averaged over the background,
* the **explained point / masker trick** (:func:`explain_chunk`): the library explains the
  on/off vector, not the spectrogram. The explained point is all ones (the real chunk) and the
  masker's background is all zeros (every player replaced by background),
* the **efficiency check**, computed independently of the library's ``base_values``,
* the **background sampler** (:func:`sample_background`), which draws label-stratified
  training-fold chunks from the explained chunk's own dataset.

The name is deliberately not ``shap.py``: that would shadow the library.
"""

import bisect
import logging
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import shap
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset, Subset

from thesis.explain.groups import PlayerSet

logger = logging.getLogger(__name__)

#: Maximum tolerated ``|sum(phi) - (f(x) - E f(b))|`` for exact enumeration. The Shapley values
#: are an exact linear combination of the same float32 model outputs, so anything larger means
#: the masker or the value function is miswired, not numerical noise.
EXACT_EFFICIENCY_TOLERANCE = 1e-4

#: Games with at most this many players are enumerated exactly (2^M coalitions); larger ones
#: use antithetic permutation sampling.
MAX_EXACT_PLAYERS = 10

TargetFn = Callable[[torch.Tensor], torch.Tensor]
ValueFn = Callable[[np.ndarray], np.ndarray]


def binary_margin(logits: torch.Tensor) -> torch.Tensor:
    """
    Log-odds of the pathological class, ``logit_1 - logit_0``.

    :param torch.Tensor logits: ``(N, 2)`` model output.
    :return: ``(N,)`` margins; positive means the model leans pathological.
    :rtype: torch.Tensor
    """
    return logits[:, 1] - logits[:, 0]


def make_value_fn(
    model: nn.Module,
    x: torch.Tensor,
    background: torch.Tensor,
    players: PlayerSet,
    target_fn: TargetFn = binary_margin,
    max_batch: int = 1024,
) -> ValueFn:
    """
    Build the coalition value function ``v(S) = mean_k target(model(where(S, x, b_k)))``.

    Interventional: every absent player is filled from the *same* background chunk ``b_k``,
    then the output is averaged over the ``K`` background chunks. Composites are built inside
    the batch loop, so at most ``max_batch`` spectrograms exist at once.

    :param torch.nn.Module model: Model in eval mode, on the same device as ``x``.
    :param torch.Tensor x: The explained chunk, ``(C, F, T)``.
    :param torch.Tensor background: ``(K, C, F, T)`` background chunks, same device as ``x``.
    :param PlayerSet players: The game's players; masks must match ``x``'s shape.
    :param target_fn: Maps ``(N, n_classes)`` logits to ``(N,)`` scalars.
    :param int max_batch: Maximum number of composite spectrograms per forward pass.
    :return: A function from ``(S, M)`` 0/1 arrays to ``(S,)`` values.
    :rtype: Callable
    :raises ValueError: If the shapes of ``x``, ``background`` and the masks disagree.
    """
    if tuple(players.masks.shape[1:]) != tuple(x.shape):
        raise ValueError(f"Player masks {tuple(players.masks.shape[1:])} vs chunk {tuple(x.shape)}")
    if tuple(background.shape[1:]) != tuple(x.shape):
        raise ValueError(f"Background {tuple(background.shape[1:])} vs chunk {tuple(x.shape)}")

    device = x.device
    flat_masks = players.flat.to(device)  # (M, P); a no-op when players are on the device
    k = background.shape[0]
    coalitions_per_batch = max(1, max_batch // k)

    def value(switches: np.ndarray) -> np.ndarray:
        s = torch.as_tensor(np.asarray(switches), dtype=torch.float32, device=device)
        if s.ndim == 1:
            s = s.unsqueeze(0)
        on = (s @ flat_masks).view(-1, *x.shape) > 0.5  # (S, C, F, T)
        values: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in on.split(coalitions_per_batch):
                composite = torch.where(batch[:, None], x, background[None])  # (s, K, C, F, T)
                out = target_fn(model(composite.flatten(0, 1)))
                values.append(out.view(len(batch), k).mean(dim=1))
        return torch.cat(values).double().cpu().numpy()

    return value


@dataclass
class ChunkAttribution:
    """
    Shapley values for one chunk.

    :ivar numpy.ndarray phi: ``(M,)`` Shapley value per player.
    :ivar float fx: ``v(all players)``, the target on the real chunk.
    :ivar float fbase: ``v(no players)``, the mean target over the background.
    :ivar float residual: ``sum(phi) - (fx - fbase)``; zero up to float error.
    :ivar int n_evals: Number of coalitions the library evaluated.
    """

    phi: np.ndarray
    fx: float
    fbase: float
    residual: float
    n_evals: int


def explain_chunk(
    model: nn.Module,
    x: torch.Tensor,
    background: torch.Tensor,
    players: PlayerSet,
    *,
    n_permutations: int | None = None,
    seed: int = 42,
    target_fn: TargetFn = binary_margin,
    max_batch: int = 1024,
    estimator: Literal["auto", "exact", "permutation"] = "auto",
) -> ChunkAttribution:
    """
    Compute grouped Shapley values for one chunk with the ``shap`` library.

    Games with at most :data:`MAX_EXACT_PLAYERS` players use ``shap.explainers.Exact`` (all
    ``2^M`` coalitions); larger ones use ``shap.explainers.Permutation``, whose antithetic
    forward-and-backward walk keeps efficiency exact per permutation.

    :param torch.nn.Module model: Model in eval mode.
    :param torch.Tensor x: The explained chunk, ``(C, F, T)``.
    :param torch.Tensor background: ``(K, C, F, T)`` background chunks.
    :param PlayerSet players: The game's players.
    :param n_permutations: Permutations for the sampled estimator. Required when the game has
        more than :data:`MAX_EXACT_PLAYERS` players, ignored otherwise. Each permutation costs
        ``2 * M + 1`` coalitions in the library's accounting.
    :param int seed: Seed for the permutation sampler.
    :param target_fn: Maps logits to the explained scalar.
    :param int max_batch: Maximum composite spectrograms per forward pass.
    :param str estimator: ``auto`` picks by player count; ``exact`` or ``permutation`` force
        one (used to validate the permutation configuration against exact values).
    :return: The chunk's Shapley values and efficiency residual.
    :rtype: ChunkAttribution
    :raises ValueError: If a sampled game has no ``n_permutations``, or if an exact game's
        efficiency residual exceeds :data:`EXACT_EFFICIENCY_TOLERANCE`.
    """
    m = len(players)
    value = make_value_fn(model, x, background, players, target_fn, max_batch)

    n_evals = 0

    def counted(switches: np.ndarray) -> np.ndarray:
        nonlocal n_evals
        n_evals += len(switches)
        return value(switches)

    masker = shap.maskers.Independent(np.zeros((1, m)), max_samples=1)
    point = np.ones((1, m))
    exact = m <= MAX_EXACT_PLAYERS if estimator == "auto" else estimator == "exact"
    if exact:
        result = shap.explainers.Exact(counted, masker)(point, silent=True)
    else:
        if n_permutations is None:
            raise ValueError(f"{m} players need n_permutations for the Permutation explainer")
        # The library runs max_evals // (2M + 1) permutations, each walked both ways.
        max_evals = n_permutations * (2 * m + 1)
        explainer = shap.explainers.Permutation(counted, masker, seed=seed)
        result = explainer(point, max_evals=max_evals, silent=True)

    phi = np.asarray(result.values[0], dtype=float)
    # Independent of result.base_values, so a misconfigured masker cannot hide here.
    fx = float(value(np.ones((1, m)))[0])
    fbase = float(value(np.zeros((1, m)))[0])
    residual = float(phi.sum() - (fx - fbase))
    if exact and abs(residual) > EXACT_EFFICIENCY_TOLERANCE:
        raise ValueError(
            f"Efficiency violated: sum(phi)={phi.sum():.6f}, f(x)-E f(b)={fx - fbase:.6f}, "
            f"residual={residual:.2e}"
        )
    return ChunkAttribution(phi=phi, fx=fx, fbase=fbase, residual=residual, n_evals=n_evals)


# ── Background sampling ───────────────────────────────────────────────────────


def person_of(subject_id: str) -> str:
    """
    Strip the recording condition from a subject ID.

    Dataset subject IDs are per recording (``"H S12 EC"``); the same person's eyes-open
    recording is a separate ID. Background diversity and subject-level averaging both want the
    person.

    :param str subject_id: Subject ID as the datasets emit it.
    :return: The ID without a trailing ``EC``/``EO``/``TASK`` token.
    :rtype: str
    """
    head, _, tail = subject_id.rpartition(" ")
    return head if head and tail.upper() in {"EC", "EO", "TASK"} else subject_id


def _recording_key(dataset: Dataset, index: int) -> Hashable:
    """
    Identify the recording a chunk comes from, without loading it.

    Walks ``Subset`` and ``ConcatDataset`` wrappers down to a flattened dataset, whose
    ``index`` maps each chunk to ``(file_idx, chunk_idx)``. Chunks of one recording share a
    key, which lets the sampler read one chunk per recording instead of recomputing a whole
    file's spectrograms for every random chunk.

    :param torch.utils.data.Dataset dataset: The dataset indexed by ``index``.
    :param int index: Chunk index.
    :return: A key shared by all chunks of the same recording; the chunk index itself when the
        dataset exposes no recording structure.
    :rtype: Hashable
    """
    while True:
        if isinstance(dataset, Subset):
            index = int(dataset.indices[index])
            dataset = dataset.dataset
        elif isinstance(dataset, ConcatDataset):
            which = bisect.bisect_right(dataset.cumulative_sizes, index)
            if which > 0:
                index -= dataset.cumulative_sizes[which - 1]
            dataset = dataset.datasets[which]
        else:
            break
    chunk_index = getattr(dataset, "index", None)
    if isinstance(chunk_index, list):
        return (id(dataset), chunk_index[index][0])
    return (id(dataset), "chunk", index)


def _recordings_for(
    dataset: Dataset, index_datasets: Sequence[str], dataset_key: str | None
) -> dict[Hashable, list[int]]:
    """
    Group the indices of one dataset (or all of them) by recording.

    :param torch.utils.data.Dataset dataset: A fold's training dataset.
    :param index_datasets: Dataset key for each index of ``dataset``.
    :param dataset_key: Restrict to this dataset, or ``None`` for all of them.
    :return: Recording key -> chunk indices, in index order.
    :rtype: dict
    :raises ValueError: If no chunk matches ``dataset_key``.
    """
    recordings: dict[Hashable, list[int]] = {}
    for i, name in enumerate(index_datasets):
        if dataset_key is None or name == dataset_key:
            recordings.setdefault(_recording_key(dataset, i), []).append(i)
    if not recordings:
        raise ValueError(f"No training chunks for dataset {dataset_key!r}")
    return recordings


@dataclass
class Background:
    """
    Background chunks plus where they came from.

    :ivar torch.Tensor chunks: ``(K, C, F, T)``.
    :ivar list[int] indices: Training-set index of each chunk; ``[]`` for a mean baseline.
    :ivar list[str] subjects: Subject ID of each chunk.
    :ivar list[int] labels: Label of each chunk.
    :ivar list[str] datasets: Dataset key of each chunk.
    """

    chunks: torch.Tensor
    indices: list[int]
    subjects: list[str]
    labels: list[int]
    datasets: list[str]


def _split_evenly(total: int, n: int) -> list[int]:
    """
    Split ``total`` into ``n`` near-equal non-negative integers.

    :param int total: Amount to split.
    :param int n: Number of parts.
    :return: Parts, larger ones first.
    :rtype: list[int]
    """
    return [total // n + (1 if i < total % n else 0) for i in range(n)]


def sample_background(
    dataset: Dataset,
    index_datasets: Sequence[str],
    k: int,
    seed: int,
    dataset_key: str | None,
    n_labels: int = 2,
) -> Background:
    """
    Draw ``k`` training chunks, stratified by label and spread over as many people as possible.

    With ``dataset_key`` set (the headline configuration), only chunks of that dataset are
    candidates, so an absent player is filled from a recording with the same hardware and
    preprocessing as the explained chunk and dataset identity stays out of phi. ``None`` gives
    a pooled background stratified by dataset x label.

    Recordings are visited in a seeded random order, one random chunk each, until every
    stratum holds enough distinct people; strata are then filled one chunk per person, topping
    up with further chunks only when a stratum has fewer people than its quota.

    :param torch.utils.data.Dataset dataset: A fold's **training** dataset, yielding
        ``(spectrogram, label, subject_id)``. Never pass validation data.
    :param index_datasets: Dataset key for each index of ``dataset``.
    :param int k: Number of chunks to draw.
    :param int seed: Seed for the draw.
    :param dataset_key: Restrict to this dataset, or ``None`` for a pooled background.
    :param int n_labels: Labels expected per dataset; scanning stops early once each has
        enough people.
    :return: The drawn chunks and their provenance.
    :rtype: Background
    :raises ValueError: If ``k`` is not positive or no chunk matches ``dataset_key``.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(index_datasets) != len(dataset):  # type: ignore[arg-type]
        raise ValueError("index_datasets must have one entry per dataset index")

    rng = np.random.default_rng(seed)
    recordings = _recordings_for(dataset, index_datasets, dataset_key)
    keys = list(recordings)
    order = rng.permutation(len(keys))

    n_datasets = 1 if dataset_key is not None else len(set(index_datasets))
    expected_strata = n_labels * n_datasets
    # Stratum -> recordings seen, as (key, first chunk index, person, label, dataset, tensor).
    seen: dict[tuple[str, int], list[tuple[Hashable, int, str, int, str, torch.Tensor]]] = {}
    people: dict[tuple[str, int], set[str]] = {}

    def enough() -> bool:
        if len(seen) < expected_strata:
            return False
        quotas = _split_evenly(k, len(seen))
        return all(len(people[s]) >= q for s, q in zip(sorted(seen), quotas))

    for position in order:
        key = keys[position]
        index = int(rng.choice(recordings[key]))
        chunk, label, subject = dataset[index]
        stratum = (index_datasets[index], int(label))
        seen.setdefault(stratum, []).append(
            (key, index, subject, int(label), index_datasets[index], chunk)
        )
        people.setdefault(stratum, set()).add(person_of(subject))
        if enough():
            break

    strata = sorted(seen)
    quotas = _split_evenly(k, len(strata))
    chosen: list[tuple[int, str, int, str, torch.Tensor]] = []
    for stratum, quota in zip(strata, quotas):
        picked: list[tuple[int, str, int, str, torch.Tensor]] = []
        used_people: set[str] = set()
        # One chunk per person first.
        for _, index, subject, label, name, chunk in seen[stratum]:
            if len(picked) == quota:
                break
            if person_of(subject) not in used_people:
                used_people.add(person_of(subject))
                picked.append((index, subject, label, name, chunk))
        # Fewer people than the quota: top up with other chunks of the recordings seen.
        taken = {p[0] for p in picked}
        pool = [i for key, *_ in seen[stratum] for i in recordings[key] if i not in taken]
        rng.shuffle(pool)
        for index in pool[: quota - len(picked)]:
            chunk, label, subject = dataset[index]
            picked.append((index, subject, int(label), index_datasets[index], chunk))
        if len(picked) < quota:
            logger.warning(
                f"Background stratum {stratum}: only {len(picked)} chunks for a quota of {quota}"
            )
        chosen.extend(picked)

    return Background(
        chunks=torch.stack([c[4] for c in chosen]),
        indices=[c[0] for c in chosen],
        subjects=[c[1] for c in chosen],
        labels=[c[2] for c in chosen],
        datasets=[c[3] for c in chosen],
    )


def mean_background(
    dataset: Dataset,
    index_datasets: Sequence[str],
    dataset_key: str | None,
    seed: int,
    max_samples: int = 2048,
) -> Background:
    """
    Mean training spectrogram of one dataset, as a single-chunk (``K = 1``) background.

    Whole recordings are added in a seeded random order until ``max_samples`` chunks are
    averaged. The mean of a few thousand spectrograms is stable well below the effects SHAP
    measures.

    :param torch.utils.data.Dataset dataset: A fold's training dataset.
    :param index_datasets: Dataset key for each index of ``dataset``.
    :param dataset_key: Restrict to this dataset, or ``None`` for all of them.
    :param int seed: Seed for the recording order.
    :param int max_samples: Stop after averaging this many chunks.
    :return: A background whose ``chunks`` is the ``(1, C, F, T)`` mean.
    :rtype: Background
    :raises ValueError: If no chunk matches ``dataset_key``.
    """
    recordings = _recordings_for(dataset, index_datasets, dataset_key)
    keys = list(recordings)
    rng = np.random.default_rng(seed)

    total: torch.Tensor | None = None
    count = 0
    for position in rng.permutation(len(keys)):
        for index in recordings[keys[position]]:
            chunk = dataset[index][0].to(torch.float64)
            total = chunk if total is None else total + chunk
            count += 1
        if count >= max_samples:
            break
    assert total is not None  # _recordings_for guarantees at least one chunk
    name = dataset_key if dataset_key is not None else "pooled"
    return Background(
        chunks=(total / count).to(torch.float32).unsqueeze(0),
        indices=[],
        subjects=[f"mean of {count} chunks"],
        labels=[-1],
        datasets=[name],
    )


def reinitialise(model: nn.Module, seed: int) -> None:
    """
    Re-draw every parameter from its layer's default initialiser, in place.

    The model-randomisation sanity check of Adebayo et al. (2018): attributions of a network
    with random weights must differ clearly from those of the trained one. Layers with a
    ``reset_parameters`` (or ``MultiheadAttention``'s ``_reset_parameters``) use it; bare
    parameters owned directly by a model (e.g. learned channel/time embeddings) are drawn from
    ``N(0, 0.02^2)``.

    :param torch.nn.Module model: Model to reset.
    :param int seed: Seed, so the control is reproducible.
    """
    torch.manual_seed(seed)
    reset: set[int] = set()
    for module in model.modules():
        reset_fn: Any = getattr(module, "reset_parameters", None) or getattr(
            module, "_reset_parameters", None
        )
        if callable(reset_fn):
            reset_fn()
            reset.update(id(p) for p in module.parameters(recurse=False))
    with torch.no_grad():
        for parameter in model.parameters():
            if id(parameter) not in reset:
                parameter.normal_(0.0, 0.02)
