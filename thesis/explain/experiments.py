"""
The SHAP experiments: grouped Shapley values over every fold of a trained run.

:func:`run_shap` explains each fold's validation chunks with the model trained without them,
so every attribution is out-of-fold. Results are aggregated with the **subject** as the unit
(average over a subject's chunks, then over subjects within a fold, then over folds), so
subjects with many chunks do not dominate. Nothing here plots; reporting is a separate step
(``helpers-print/plot_shap.py``), so the expensive evaluation runs once and can be re-analysed.
"""

import itertools
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from thesis.data_preparation import CVFolds
from thesis.explain.groups import Game, PlayerSet, build_players
from thesis.explain.harness import (
    FoldContext,
    RunConfig,
    build_folds,
    iter_folds,
    verify_baseline,
)
from thesis.explain.shap_values import (
    MAX_EXACT_PLAYERS,
    Background,
    explain_chunk,
    mean_background,
    person_of,
    reinitialise,
    sample_background,
)
from thesis.explain.stats import (
    benjamini_hochberg,
    nadeau_bengio_corrected_t,
    paired_wilcoxon,
    spearman,
)
from thesis.model import RAW_EEG_MODELS
from thesis.version import get_git_commit

logger = logging.getLogger(__name__)

#: Experiment names accepted by ``main.py explain --experiment``, mapped to their games.
EXPERIMENT_GAMES: dict[str, list[Game]] = {
    "shap-bands": ["bands"],
    "shap-channels": ["channels"],
    "shap-grid": ["grid"],
    "shap-gamma": ["gamma"],
    "shap-all": ["bands", "channels", "grid"],
}

#: Largest ratio of the random-weights model's mean ``|f(x) - E f(b)|`` to the trained model's
#: for the model-randomisation test to pass. Spearman rho between the two profiles is reported
#: but not tested: it stays high (~0.8) because the profile order follows input spectral power,
#: even though random-weights attributions are two orders of magnitude smaller.
RANDOMISATION_GAP_RATIO = 0.1

CLASS_NAMES = {0: "healthy", 1: "pathological"}


@dataclass
class ShapOptions:
    """
    Settings of one SHAP run.

    :ivar str game: ``bands``, ``channels`` or ``grid``.
    :ivar int background_size: Background chunks per dataset (``K``) for the sample baseline.
    :ivar int max_chunks_per_fold: Validation chunks explained per fold, subsampled evenly
        across subjects when the fold has more.
    :ivar str baseline: ``sample`` (K training chunks of the same dataset) or ``mean`` (the
        dataset's mean training spectrogram, ``K = 1``).
    :ivar int grid_permutations: Permutations per chunk for games too large to enumerate.
    :ivar int grid_se_seeds: Independent permutation runs used to estimate the Monte Carlo
        standard error of sampled games.
    :ivar int grid_se_chunks: Chunks per fold re-run for that estimate.
    :ivar int seed: Base seed for background draws and permutations.
    :ivar bool random_init: Explain a randomly re-initialised model (sanity control).
    :ivar int max_batch: Maximum composite spectrograms per forward pass.
    :ivar folds: Zero-indexed folds to run, or ``None`` for all.
    """

    game: Game
    background_size: int = 32
    max_chunks_per_fold: int = 400
    baseline: Literal["sample", "mean"] = "sample"
    grid_permutations: int = 64
    grid_se_seeds: int = 3
    grid_se_chunks: int = 50
    seed: int = 42
    random_init: bool = False
    max_batch: int = 1024
    folds: list[int] | None = None

    @property
    def partial(self) -> bool:
        """
        Whether only some folds are explained (a quick check, never a reportable result).

        :return: ``True`` when ``folds`` is set.
        :rtype: bool
        """
        return self.folds is not None

    @property
    def stem(self) -> str:
        """
        File stem of this run's outputs.

        :return: See :func:`output_stem`.
        :rtype: str
        """
        return output_stem(self.game, self.baseline, self.random_init, self.partial)


def output_stem(
    game: Game, baseline: str = "sample", random_init: bool = False, partial: bool = False
) -> str:
    """
    File stem of a run's outputs: ``shap_<game>[_mean][_random][_partial]``.

    :param str game: The game.
    :param str baseline: ``sample`` or ``mean``.
    :param bool random_init: Whether the model was re-initialised.
    :param bool partial: Whether only some folds were explained.
    :return: The stem, e.g. ``shap_bands_mean``.
    :rtype: str
    """
    stem = f"shap_{game}"
    if baseline == "mean":
        stem += "_mean"
    if random_init:
        stem += "_random"
    if partial:
        stem += "_partial"
    return stem


def check_explainable(config: RunConfig) -> None:
    """
    Reject runs the SHAP games do not support, before any data is loaded.

    :param RunConfig config: The trained run.
    :raises NotImplementedError: For a four-class run.
    :raises ValueError: For a raw-EEG model, which has no spectrogram to group.
    """
    if config.num_classes != 2:
        raise NotImplementedError("SHAP is implemented for binary runs only")
    if config.model in RAW_EEG_MODELS:
        raise ValueError(f"{config.model} consumes raw EEG; SHAP games need spectrograms")


def dataset_names(config: RunConfig) -> dict[str, str]:
    """
    Public dataset keys and display names for a run.

    For the in-ear model the ``cane`` label is really IDUN (real in-ear recordings) while MDD
    and SAD are synthetic T8-T7 derivations; the keys and names say so.

    :param RunConfig config: The run.
    :return: Internal key -> ``(public key, display name)`` flattened as public key -> name.
    :rtype: dict[str, str]
    """
    if config.channel == "in-ear":
        return {
            "mdd": "MDD (synthetic T8-T7)",
            "idun": "IDUN (real in-ear)",
            "sad": "SAD (synthetic T8-T7)",
        }
    return {"mdd": "MDD", "cane": "CANE", "sad": "SAD"}


def _public_key(config: RunConfig, key: str) -> str:
    """
    Map an internal dataset key to the one written to results.

    :param RunConfig config: The run.
    :param str key: ``mdd``, ``cane`` or ``sad``.
    :return: ``idun`` for CANE-slot data of an in-ear run, else ``key``.
    :rtype: str
    """
    return "idun" if config.channel == "in-ear" and key == "cane" else key


def choose_chunks(
    persons: list[str], strata: list[str], max_chunks: int
) -> tuple[list[int], dict[str, Any]]:
    """
    Pick at most ``max_chunks`` chunks, spread evenly across people.

    Every person gets the same cap, ``max_chunks // n_people``, taken at evenly spaced
    positions through their recording. When there are more people than chunks allowed, one
    chunk each is taken from people evenly spaced through the list ordered by stratum, so the
    dataset x label mix is kept.

    :param persons: Person key of every candidate chunk, e.g. ``"mdd:H S3"``.
    :param strata: Stratum of every candidate chunk, e.g. ``"mdd/1"``.
    :param int max_chunks: Upper bound on the number chosen.
    :return: Chosen indices (sorted) and a report of the stratum mix before and after.
    :rtype: tuple[list[int], dict]
    """
    by_person: dict[str, list[int]] = {}
    for i, person in enumerate(persons):
        by_person.setdefault(person, []).append(i)

    if len(persons) <= max_chunks:
        chosen = list(range(len(persons)))
    elif len(by_person) <= max_chunks:
        cap = max_chunks // len(by_person)
        chosen = []
        for indices in by_person.values():
            positions = np.linspace(0, len(indices) - 1, min(cap, len(indices)))
            chosen.extend(indices[int(round(p))] for p in positions)
    else:
        ordered = sorted(by_person, key=lambda p: (strata[by_person[p][0]], p))
        picks = np.linspace(0, len(ordered) - 1, max_chunks)
        chosen = [
            by_person[ordered[int(round(p))]][len(by_person[ordered[int(round(p))]]) // 2]
            for p in picks
        ]
    chosen = sorted(set(chosen))

    def mix(indices: list[int]) -> dict[str, float]:
        counts: dict[str, int] = {}
        for i in indices:
            counts[strata[i]] = counts.get(strata[i], 0) + 1
        return {k: counts[k] / max(1, len(indices)) for k in sorted(counts)}

    return chosen, {"before": mix(list(range(len(persons)))), "after": mix(chosen)}


@dataclass
class _Chunks:
    """Validation chunks selected for explanation in one fold."""

    x: torch.Tensor
    labels: list[int]
    subjects: list[str]
    datasets: list[str]
    mix: dict[str, Any]


def _select_chunks(context: FoldContext, max_chunks: int) -> _Chunks:
    """
    Read a fold's validation set and pick the chunks to explain.

    :param FoldContext context: The fold.
    :param int max_chunks: Upper bound on the chunks chosen.
    :return: The chosen chunks.
    :rtype: _Chunks
    """
    xs: list[torch.Tensor] = []
    labels: list[int] = []
    subjects: list[str] = []
    for inputs, batch_labels, batch_subjects in context.loader:
        xs.append(inputs)
        labels.extend(int(y) for y in batch_labels)
        subjects.extend(batch_subjects)
    x = torch.cat(xs)
    datasets = context.val_datasets
    persons = [f"{d}:{person_of(s)}" for d, s in zip(datasets, subjects)]
    strata = [f"{d}/{y}" for d, y in zip(datasets, labels)]
    chosen, mix = choose_chunks(persons, strata, max_chunks)
    logger.info(
        f"Fold {context.fold + 1}: explaining {len(chosen)} of {len(persons)} chunks; "
        f"stratum mix before {mix['before']}, after {mix['after']}"
    )
    return _Chunks(
        x=x[chosen],
        labels=[labels[i] for i in chosen],
        subjects=[subjects[i] for i in chosen],
        datasets=[datasets[i] for i in chosen],
        mix=mix,
    )


def _player_log_power(x: torch.Tensor, players: PlayerSet) -> np.ndarray:
    """
    Mean spectrogram value (log power) inside each player's mask, for beeswarm plots.

    :param torch.Tensor x: One chunk, ``(C, F, T)``.
    :param PlayerSet players: The game's players.
    :return: ``(M,)`` means.
    :rtype: numpy.ndarray
    """
    flat = players.flat.to(x.device)
    return ((flat @ x.flatten().float()) / flat.sum(dim=1)).cpu().numpy()


# ── Aggregation ───────────────────────────────────────────────────────────────


def fold_means(
    values: np.ndarray, folds: np.ndarray, persons: np.ndarray, select: np.ndarray
) -> dict[int, np.ndarray]:
    """
    Subject-weighted mean per fold: chunks -> person mean -> mean over persons.

    :param numpy.ndarray values: ``(N, M)`` per-chunk values.
    :param numpy.ndarray folds: ``(N,)`` fold of each chunk.
    :param numpy.ndarray persons: ``(N,)`` person key of each chunk.
    :param numpy.ndarray select: ``(N,)`` boolean filter.
    :return: Fold -> ``(M,)`` mean, only for folds with at least one selected chunk.
    :rtype: dict[int, numpy.ndarray]
    """
    result: dict[int, np.ndarray] = {}
    for fold in sorted(set(folds[select].tolist())):
        in_fold = select & (folds == fold)
        person_means = [
            values[in_fold & (persons == p)].mean(axis=0) for p in sorted(set(persons[in_fold]))
        ]
        result[int(fold)] = np.mean(person_means, axis=0)
    return result


def _chunk_weights(folds: np.ndarray, persons: np.ndarray) -> np.ndarray:
    """
    Weight of each chunk in the cross-fold subject-weighted mean.

    :param numpy.ndarray folds: ``(N,)`` fold of each chunk.
    :param numpy.ndarray persons: ``(N,)`` person key of each chunk.
    :return: ``(N,)`` weights summing to one.
    :rtype: numpy.ndarray
    """
    weights = np.zeros(len(folds))
    fold_ids = sorted(set(folds.tolist()))
    for fold in fold_ids:
        in_fold = folds == fold
        fold_persons = sorted(set(persons[in_fold]))
        for p in fold_persons:
            member = in_fold & (persons == p)
            weights[member] = 1.0 / (len(fold_ids) * len(fold_persons) * member.sum())
    return weights


def summarise(per_fold: dict[int, np.ndarray], names: list[str]) -> dict[str, Any]:
    """
    Mean and standard deviation over folds, keyed by player.

    :param per_fold: Fold -> ``(M,)`` values.
    :param names: Player names.
    :return: ``mean``, ``std`` (ddof 1; 0 for a single fold), ``per_fold`` (one-indexed fold
        keys) and ``n_folds``.
    :rtype: dict
    """
    if not per_fold:
        return {"mean": None, "std": None, "per_fold": {}, "n_folds": 0}
    stacked = np.stack(list(per_fold.values()))
    std = stacked.std(axis=0, ddof=1) if len(per_fold) > 1 else np.zeros(stacked.shape[1])
    return {
        "mean": dict(zip(names, stacked.mean(axis=0).tolist())),
        "std": dict(zip(names, std.tolist())),
        "per_fold": {str(f + 1): dict(zip(names, v.tolist())) for f, v in per_fold.items()},
        "n_folds": len(per_fold),
    }


def _importance(
    phi: np.ndarray, folds: np.ndarray, persons: np.ndarray, select: np.ndarray, names: list[str]
) -> dict[str, Any]:
    """
    Global importance (mean |phi|) and net direction (mean phi) for a subset of chunks.

    :param numpy.ndarray phi: ``(N, M)`` Shapley values.
    :param numpy.ndarray folds: ``(N,)`` folds.
    :param numpy.ndarray persons: ``(N,)`` person keys.
    :param numpy.ndarray select: ``(N,)`` filter.
    :param names: Player names.
    :return: ``mean_abs_phi`` and ``mean_phi`` summaries.
    :rtype: dict
    """
    return {
        "mean_abs_phi": summarise(fold_means(np.abs(phi), folds, persons, select), names),
        "mean_phi": summarise(fold_means(phi, folds, persons, select), names),
        "n_chunks": int(select.sum()),
    }


def _dataset_differences(
    phi: np.ndarray,
    folds: np.ndarray,
    persons: np.ndarray,
    datasets: np.ndarray,
    names: list[str],
    n_folds_total: int,
) -> dict[str, Any]:
    """
    Paired per-fold tests of whether two datasets rely on a player to a different extent.

    For each dataset pair and player, the fold-level mean |phi| of the two datasets are paired
    by fold. Wilcoxon and the Nadeau-Bengio corrected t are reported, each with
    Benjamini-Hochberg correction over the whole family (all pairs x players of this game).

    :param numpy.ndarray phi: ``(N, M)`` Shapley values.
    :param numpy.ndarray folds: ``(N,)`` folds.
    :param numpy.ndarray persons: ``(N,)`` person keys.
    :param numpy.ndarray datasets: ``(N,)`` public dataset keys.
    :param names: Player names.
    :param int n_folds_total: The run's fold count, for the test fraction.
    :return: ``tests`` keyed ``"<a>_vs_<b>"`` then player, or a ``skipped`` reason.
    :rtype: dict
    """
    per_dataset = {
        d: fold_means(np.abs(phi), folds, persons, datasets == d) for d in sorted(set(datasets))
    }
    tests: list[tuple[str, str, dict[str, float], dict[str, float]]] = []
    for a, b in itertools.combinations(sorted(per_dataset), 2):
        common = sorted(set(per_dataset[a]) & set(per_dataset[b]))
        if len(common) < 2:
            continue
        va = np.stack([per_dataset[a][f] for f in common])
        vb = np.stack([per_dataset[b][f] for f in common])
        for m, name in enumerate(names):
            wil = paired_wilcoxon(va[:, m].tolist(), vb[:, m].tolist())
            nb = nadeau_bengio_corrected_t(
                va[:, m].tolist(), vb[:, m].tolist(), 1.0 / n_folds_total
            )
            tests.append((f"{a}_vs_{b}", name, wil, nb))
    if not tests:
        return {"skipped": "fewer than two folds with both datasets"}

    q_wil = benjamini_hochberg([t[2]["p_value"] for t in tests])
    q_nb = benjamini_hochberg([t[3]["p_value"] for t in tests])
    out: dict[str, dict[str, Any]] = {}
    for i, (pair, name, wil, nb) in enumerate(tests):
        out.setdefault(pair, {})[name] = {
            "difference_a_minus_b": wil["mean_difference"],
            "wilcoxon_p": wil["p_value"],
            "wilcoxon_q": q_wil.q_values[i],
            "nadeau_bengio_t": nb["t_statistic"],
            "nadeau_bengio_p": nb["p_value"],
            "nadeau_bengio_q": q_nb.q_values[i],
        }
    return {"tests": out, "family_size": len(tests)}


# ── The runner ────────────────────────────────────────────────────────────────


def run_shap(
    config: RunConfig, device: torch.device, options: ShapOptions, folds: CVFolds | None = None
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Explain every requested fold's validation chunks with grouped SHAP.

    Per fold: verify the fold rebuild against the checkpoint (skip the fold on a mismatch),
    draw one background per dataset from the fold's training data, explain the selected
    validation chunks, and record one row per chunk.

    :param RunConfig config: The trained run.
    :param torch.device device: Device for the forward passes.
    :param ShapOptions options: What to run.
    :param CVFolds folds: Prebuilt folds from :func:`build_folds`; built here when omitted.
        Pass them in when running several games, since dataset preparation is the slow step.
    :return: ``(summary, per-chunk arrays)``; the summary is JSON-serialisable and the arrays
        are for ``numpy.savez``.
    :rtype: tuple[dict, dict[str, numpy.ndarray]]
    :raises NotImplementedError: For a four-class run.
    :raises ValueError: For a raw-EEG model, which has no spectrogram to group.
    """
    check_explainable(config)

    players = build_players(options.game, config.in_channels, *config.spec_shape).to(device)
    names = players.names
    sampled = len(players) > MAX_EXACT_PLAYERS
    k = 1 if options.baseline == "mean" else options.background_size
    logger.info(
        f"SHAP {options.game}: {len(players)} players "
        f"({'permutation, %d perms' % options.grid_permutations if sampled else 'exact'}), "
        f"{options.baseline} baseline, K={k}"
    )

    rows: dict[str, list[Any]] = {
        key: []
        for key in (
            "fold", "subject", "person", "dataset", "label", "pred",
            "fx", "fbase", "residual", "phi", "log_power",
        )
    }  # fmt: skip
    folds_info: dict[str, Any] = {}
    skipped: list[int] = []
    mc_var: dict[int, np.ndarray] = {}
    first_evals: int | None = None

    if folds is None:
        folds = build_folds(config)
    for context in iter_folds(config, device, folds, options.folds):
        fold = context.fold
        matched, difference = verify_baseline(context, config, device)
        if not matched:
            logger.error(f"Fold {fold + 1}: fold rebuild does not match the checkpoint; skipped")
            skipped.append(fold + 1)
            continue

        model = context.model
        if options.random_init:
            reinitialise(model, options.seed + fold)
            model.eval()

        chunks = _select_chunks(context, options.max_chunks_per_fold)
        backgrounds: dict[str, Background] = {}
        for name in sorted(set(chunks.datasets)):
            if options.baseline == "mean":
                backgrounds[name] = mean_background(
                    context.train_dataset, context.train_datasets, name, options.seed + fold
                )
            else:
                backgrounds[name] = sample_background(
                    context.train_dataset, context.train_datasets, k, options.seed + fold, name
                )
        on_device = {name: bg.chunks.to(device) for name, bg in backgrounds.items()}

        fold_phi: list[np.ndarray] = []
        for i in range(len(chunks.labels)):
            x = chunks.x[i].to(device)
            attribution = explain_chunk(
                model,
                x,
                on_device[chunks.datasets[i]],
                players,
                n_permutations=options.grid_permutations if sampled else None,
                seed=options.seed + 100_000 * fold + i,
                max_batch=options.max_batch,
            )
            if first_evals is None:
                first_evals = attribution.n_evals
                logger.info(f"First chunk: {attribution.n_evals} coalitions evaluated")
            rows["fold"].append(fold)
            rows["subject"].append(chunks.subjects[i])
            rows["person"].append(f"{chunks.datasets[i]}:{person_of(chunks.subjects[i])}")
            rows["dataset"].append(_public_key(config, chunks.datasets[i]))
            rows["label"].append(chunks.labels[i])
            rows["pred"].append(int(attribution.fx > 0))
            rows["fx"].append(attribution.fx)
            rows["fbase"].append(attribution.fbase)
            rows["residual"].append(attribution.residual)
            rows["phi"].append(attribution.phi)
            rows["log_power"].append(_player_log_power(x, players))
            fold_phi.append(attribution.phi)

        if sampled and options.grid_se_seeds > 1:
            mc_var[fold] = _monte_carlo_variance(
                model, chunks, on_device, players, options, fold, fold_phi, device
            )

        folds_info[str(fold + 1)] = {
            "baseline_check_difference": difference,
            "n_chunks": len(chunks.labels),
            "stratum_mix": chunks.mix,
            "background": {
                _public_key(config, name): {
                    "size": len(bg.subjects),
                    "distinct_people": len({person_of(s) for s in bg.subjects}),
                    "label_counts": {
                        str(label): bg.labels.count(label) for label in sorted(set(bg.labels))
                    },
                }
                for name, bg in backgrounds.items()
            },
        }
        logger.info(f"Fold {fold + 1}: explained {len(chunks.labels)} chunks")

    if not rows["fold"]:
        raise RuntimeError("No fold passed the baseline verification gate; nothing explained")

    arrays = _to_arrays(rows, names)
    payload = _summary(
        config, options, players, arrays, k, sampled, first_evals, folds_info, skipped
    )
    if mc_var:
        payload["monte_carlo_se"] = _monte_carlo_se(arrays, mc_var, names)
    return payload, arrays


def _monte_carlo_variance(
    model: torch.nn.Module,
    chunks: _Chunks,
    backgrounds: dict[str, torch.Tensor],
    players: PlayerSet,
    options: ShapOptions,
    fold: int,
    fold_phi: list[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """
    Per-player Monte Carlo variance of one chunk's permutation estimate.

    The first ``grid_se_chunks`` chunks are re-explained with ``grid_se_seeds - 1`` further
    seeds; together with the main estimate that gives ``grid_se_seeds`` independent estimates
    per chunk. The library returns no standard error of its own.

    :return: ``(M,)`` variance (ddof 1), averaged over the re-run chunks.
    :rtype: numpy.ndarray
    :raises ValueError: If there is no chunk to re-run or fewer than two seeds.
    """
    n = min(options.grid_se_chunks, len(fold_phi))
    if n < 1 or options.grid_se_seeds < 2:
        raise ValueError("The Monte Carlo error needs at least one chunk and two seeds")
    variances = []
    for i in range(n):
        estimates = [fold_phi[i]]
        for repeat in range(1, options.grid_se_seeds):
            estimates.append(
                explain_chunk(
                    model,
                    chunks.x[i].to(device),
                    backgrounds[chunks.datasets[i]],
                    players,
                    n_permutations=options.grid_permutations,
                    seed=options.seed + 100_000 * fold + i + 7_919 * repeat,
                    max_batch=options.max_batch,
                ).phi
            )
        variances.append(np.var(np.stack(estimates), axis=0, ddof=1))
    logger.info(
        f"Fold {fold + 1}: Monte Carlo variance from {n} chunks x {options.grid_se_seeds} seeds"
    )
    return np.asarray(np.mean(variances, axis=0), dtype=float)


def _monte_carlo_se(
    arrays: dict[str, np.ndarray], mc_var: dict[int, np.ndarray], names: list[str]
) -> dict[str, Any]:
    """
    Monte Carlo standard error of the cross-fold mean phi of a sampled game.

    The cross-fold subject-weighted mean is linear in the chunk estimates, so with independent
    sampling noise its variance is ``sum_i w_i^2 var_i``, taking ``var_i`` as the fold's
    average per-chunk variance.

    :return: ``mean_phi_se`` (per player), ``per_chunk_se`` (per player, for context).
    :rtype: dict
    """
    folds, persons = arrays["fold"], arrays["person"]
    weights = _chunk_weights(folds, persons)
    variance = np.zeros(len(names))
    for fold, var in mc_var.items():
        variance += (weights[folds == fold] ** 2).sum() * var
    per_chunk = np.sqrt(np.mean(np.stack(list(mc_var.values())), axis=0))
    return {
        "mean_phi_se": dict(zip(names, np.sqrt(variance).tolist())),
        "per_chunk_se": dict(zip(names, per_chunk.tolist())),
        "folds": [f + 1 for f in sorted(mc_var)],
    }


def _to_arrays(rows: dict[str, list[Any]], names: list[str]) -> dict[str, np.ndarray]:
    """
    Convert collected rows to arrays for aggregation and ``numpy.savez``.

    :return: One array per column, plus ``players``.
    :rtype: dict[str, numpy.ndarray]
    """
    arrays = {
        "fold": np.asarray(rows["fold"], dtype=int),
        "subject": np.asarray(rows["subject"], dtype=str),
        "person": np.asarray(rows["person"], dtype=str),
        "dataset": np.asarray(rows["dataset"], dtype=str),
        "label": np.asarray(rows["label"], dtype=int),
        "pred": np.asarray(rows["pred"], dtype=int),
        "fx": np.asarray(rows["fx"], dtype=float),
        "fbase": np.asarray(rows["fbase"], dtype=float),
        "residual": np.asarray(rows["residual"], dtype=float),
        "phi": np.stack(rows["phi"]).astype(float),
        "log_power": np.stack(rows["log_power"]).astype(float),
        "players": np.asarray(names, dtype=str),
    }
    return arrays


def _summary(
    config: RunConfig,
    options: ShapOptions,
    players: PlayerSet,
    arrays: dict[str, np.ndarray],
    k: int,
    sampled: bool,
    first_evals: int | None,
    folds_info: dict[str, Any],
    skipped: list[int],
) -> dict[str, Any]:
    """
    Aggregate per-chunk rows into the JSON summary.

    :return: The JSON-serialisable summary.
    :rtype: dict
    """
    names = players.names
    phi, folds, persons = arrays["phi"], arrays["fold"], arrays["person"]
    datasets, labels = arrays["dataset"], arrays["label"]
    everything = np.ones(len(folds), dtype=bool)
    correct = arrays["pred"] == labels

    importance: dict[str, Any] = {
        "all": _importance(phi, folds, persons, everything, names),
        "by_dataset": {
            d: _importance(phi, folds, persons, datasets == d, names) for d in sorted(set(datasets))
        },
        "by_class": {
            CLASS_NAMES[c]: _importance(phi, folds, persons, labels == c, names)
            for c in sorted(set(labels.tolist()))
        },
        "by_outcome": {
            "correct": _importance(phi, folds, persons, correct, names),
            "incorrect": _importance(phi, folds, persons, ~correct, names),
        },
    }
    if options.game in ("bands", "gamma"):
        per_bin = np.abs(phi) / np.asarray(players.n_bins, dtype=float)
        importance["per_bin"] = {
            "all": summarise(fold_means(per_bin, folds, persons, everything), names),
            "by_dataset": {
                d: summarise(fold_means(per_bin, folds, persons, datasets == d), names)
                for d in sorted(set(datasets))
            },
        }

    gap = (arrays["fx"] - arrays["fbase"])[:, None]
    gap_summary = summarise(fold_means(np.abs(gap), folds, persons, everything), ["abs_gap"])
    residual = float(np.abs(arrays["residual"]).max())
    if sampled and residual > 1e-3:
        logger.warning(f"Permutation efficiency residual {residual:.2e} is larger than expected")
    logger.info(f"Max efficiency residual: {residual:.2e}")

    return {
        "experiment": f"shap-{options.game}",
        "game": options.game,
        "run_dir": str(config.checkpoint_dir),
        "model": config.model,
        "channel": config.channel,
        "dataset": config.dataset,
        "condition": config.condition,
        "freq_cutoff_hz": config.freq_cutoff_hz,
        "target": "logit_1 - logit_0 (log-odds of pathological)",
        "players": names,
        "player_n_bins": players.n_bins,
        "datasets": dataset_names(config),
        "estimator": "permutation" if sampled else "exact",
        "grid_permutations": options.grid_permutations if sampled else None,
        "coalitions_first_chunk": first_evals,
        "baseline": options.baseline,
        "background": "same-dataset, label-stratified, training fold only",
        "background_size": k,
        "random_init": options.random_init,
        "max_chunks_per_fold": options.max_chunks_per_fold,
        "partial": options.partial,
        "n_chunks_per_fold": {f: info["n_chunks"] for f, info in folds_info.items()},
        "seed": options.seed,
        "git_commit": get_git_commit(),
        "max_efficiency_residual": residual,
        "skipped_folds": skipped,
        "folds": folds_info,
        "importance": importance,
        "efficiency_gap": gap_summary,
        "dataset_differences": _dataset_differences(
            phi, folds, persons, datasets, names, config.n_folds
        ),
    }


# ── Comparisons between runs ──────────────────────────────────────────────────


def _profile_agreement(reference: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """
    Spearman agreement of two runs' ``mean_abs_phi`` profiles, per fold and on the fold mean.

    :param dict reference: Summary of the reference run.
    :param dict other: Summary of the compared run.
    :return: ``per_fold`` rho, ``mean_rho`` over folds, and ``rho_of_means``.
    :rtype: dict
    """
    ref = reference["importance"]["all"]["mean_abs_phi"]
    oth = other["importance"]["all"]["mean_abs_phi"]
    names = reference["players"]
    per_fold: dict[str, float] = {}
    for fold in sorted(set(ref["per_fold"]) & set(oth["per_fold"]), key=int):
        a = [ref["per_fold"][fold][n] for n in names]
        b = [oth["per_fold"][fold][n] for n in names]
        per_fold[fold] = spearman(a, b)["rho"]
    rhos = [r for r in per_fold.values() if not math.isnan(r)]
    return {
        "per_fold": per_fold,
        "mean_rho": float(np.mean(rhos)) if rhos else float("nan"),
        "rho_of_means": spearman([ref["mean"][n] for n in names], [oth["mean"][n] for n in names])[
            "rho"
        ],
    }


def attach_comparisons(payload: dict[str, Any], output_dir: Path, options: ShapOptions) -> None:
    """
    Compare a derived run with its reference run, when the reference has already been written.

    * ``--baseline mean`` is compared with the sample-baseline run of the same game
      (``baseline_agreement``), the robustness line for the paper.
    * ``--random-init`` is compared with the trained model's run under the same baseline
      (``randomisation_check``). It passes when the random model's mean ``|f(x) - E f(b)|`` is
      below :data:`RANDOMISATION_GAP_RATIO` times the trained model's; rho is informational.

    :param dict payload: The derived run's summary; updated in place.
    :param Path output_dir: Where the reference run's JSON lives.
    :param ShapOptions options: The derived run's options. Partial runs are never compared.
    """
    if options.partial:
        return
    if options.baseline == "mean" and not options.random_init:
        reference_path = output_dir / f"{output_stem(options.game)}.json"
        if reference_path.is_file():
            reference = json.loads(reference_path.read_text())
            payload["baseline_agreement"] = {
                "reference": reference_path.name,
                **_profile_agreement(reference, payload),
            }
        else:
            logger.warning(f"No {reference_path.name}; baseline agreement not computed")

    if options.random_init:
        reference_path = output_dir / f"{output_stem(options.game, options.baseline)}.json"
        if not reference_path.is_file():
            logger.warning(f"No {reference_path.name}; randomisation check not computed")
            return
        trained = json.loads(reference_path.read_text())
        agreement = _profile_agreement(trained, payload)
        trained_gap = trained["efficiency_gap"]["mean"]["abs_gap"]
        random_gap = payload["efficiency_gap"]["mean"]["abs_gap"]
        passed = random_gap < RANDOMISATION_GAP_RATIO * trained_gap
        payload["randomisation_check"] = {
            "reference": reference_path.name,
            **agreement,
            "gap_ratio_threshold": RANDOMISATION_GAP_RATIO,
            "trained_mean_abs_gap": trained_gap,
            "random_mean_abs_gap": random_gap,
            "passed": bool(passed),
        }
        level = logger.info if passed else logger.warning
        level(
            f"Randomisation check {'PASSED' if passed else 'FAILED'}: "
            f"gap trained={trained_gap:.3f} random={random_gap:.3f} "
            f"(mean rho={agreement['mean_rho']:.3f}, informational)"
        )
