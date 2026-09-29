"""
The explainability experiments themselves.

Each ``run_*`` function evaluates one family of ablations across every fold and returns a
JSON-serialisable result: per-fold raw numbers, plus the paired statistics that turn a
difference into a claim. Nothing here prints or plots; reporting is a separate step, so the
expensive evaluation runs once and can be re-analysed freely.

Every comparison is paired within a fold — an ablation is always measured against the *same*
fold's unablated score, never against a pooled mean.
"""

import logging
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch

from thesis.dataset import CANONICAL_CHANNEL_ORDER
from thesis.explain import attribution as attr
from thesis.explain.harness import (
    FoldContext,
    RunConfig,
    build_folds,
    evaluate,
    extract,
    iter_folds,
    reference_spectrogram,
)
from thesis.explain.masks import (
    CHANNEL_REGIONS,
    EEG_BANDS,
    band_bins,
    channel_mask,
    matched_control_windows,
    mirror_channels,
    occlude,
)
from thesis.explain.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    nadeau_bengio_corrected_t,
    paired_wilcoxon,
)

logger = logging.getLogger(__name__)

#: Metric the headline ablation tables and figures are built on.
PRIMARY_METRIC = "chunk_accuracy"


def _summarise(
    per_fold: dict[str, list[dict[str, float]]],
    baseline: list[dict[str, float]],
    metric: str,
    test_fraction: float,
) -> dict[str, Any]:
    """
    Turn per-fold ablation scores into paired statistics with FDR correction.

    :param per_fold: Ablation name -> per-fold metric dicts, fold order preserved.
    :param baseline: Per-fold unablated metric dicts, same fold order.
    :param str metric: Key within those dicts to test, e.g. ``chunk_accuracy``.
    :param float test_fraction: Held-out fraction per fold, for the Nadeau-Bengio correction.
    :return: Per-ablation statistics plus the FDR-adjusted q-values across the family.
    :rtype: dict
    """
    base_values = [fold[metric] for fold in baseline]
    names = list(per_fold)
    stats: dict[str, Any] = {}

    for name in names:
        values = [fold[metric] for fold in per_fold[name]]
        wilcoxon = paired_wilcoxon(values, base_values)
        stats[name] = {
            "per_fold": values,
            "mean": float(np.mean(values)),
            "wilcoxon": wilcoxon,
            "corrected_t": nadeau_bengio_corrected_t(values, base_values, test_fraction),
            "drop_ci": bootstrap_ci([b - a for a, b in zip(values, base_values)]),
        }

    corrected = benjamini_hochberg([stats[n]["wilcoxon"]["p_value"] for n in names])
    for name, q_value, is_rejected in zip(names, corrected.q_values, corrected.rejected):
        stats[name]["q_value"] = float(q_value)
        stats[name]["significant"] = bool(is_rejected)

    return {
        "metric": metric,
        "baseline_per_fold": base_values,
        "baseline_mean": float(np.mean(base_values)),
        "ablations": stats,
    }


def _collect(
    config: RunConfig,
    device: torch.device,
    ablations: Callable[[FoldContext], dict[str, dict[str, Any]]],
    verify: bool = True,
) -> tuple[dict[str, list[dict[str, float]]], list[dict[str, float]], list[dict[str, Any]]]:
    """
    Evaluate a set of ablations on every fold.

    :param RunConfig config: The run to analyse.
    :param torch.device device: Device to evaluate on.
    :param ablations: Given a fold, returns a mapping from ablation name to the keyword
        arguments for :func:`~thesis.explain.harness.evaluate` (``input_transform`` and/or
        ``channel_mask``). Called per fold so an ablation can depend on fold-specific data,
        such as the training-set occlusion reference.
    :param bool verify: Check each fold's unablated score against the checkpoint's stored
        metrics before using it.
    :return: ``(per-ablation per-fold metrics, per-fold baseline metrics, verification log)``.
    :rtype: tuple
    """
    folds = build_folds(config)
    per_fold: dict[str, list[dict[str, float]]] = {}
    baseline: list[dict[str, float]] = []
    verification: list[dict[str, Any]] = []

    for context in iter_folds(config, device, folds=folds):
        unablated = evaluate(context, config, device)
        baseline.append(extract(unablated))

        if verify:
            stored = context.stored_metrics
            expected = float(stored["chunk"]["accuracy"]) if stored else float("nan")
            actual = float(unablated["chunk"]["accuracy"])
            matched = bool(stored) and abs(expected - actual) <= 1e-4
            verification.append(
                {
                    "fold": context.fold + 1,
                    "stored_chunk_accuracy": expected,
                    "recomputed_chunk_accuracy": actual,
                    "matched": matched,
                }
            )
            if not matched:
                logger.error(
                    f"Fold {context.fold + 1} baseline mismatch "
                    f"(stored {expected:.6f} vs recomputed {actual:.6f}). "
                    "Fold reconstruction is wrong; results from this fold are not usable."
                )

        for name, kwargs in ablations(context).items():
            metrics = evaluate(context, config, device, **kwargs)
            per_fold.setdefault(name, []).append(extract(metrics))
        logger.info(f"Fold {context.fold + 1}: evaluated {len(per_fold)} ablations")

    return per_fold, baseline, verification


def run_channel_ablation(
    config: RunConfig, device: torch.device, metric: str = PRIMARY_METRIC
) -> dict[str, Any]:
    """
    Leave-one-channel-out, leave-one-channel-in, and grouped-region ablation (E1).

    Channels are removed by dropping their tokens from the sequence, so the channel takes no
    part in attention or the mean pool and its ``chan_embedding`` leaves with it. Region groups
    are included because neighbouring electrodes are spatially redundant: a flat single-channel
    ranking does not license the conclusion that no channel matters, whereas a large drop for a
    whole region does.

    :param RunConfig config: The run to analyse; must be an eight-channel spectrogram run.
    :param torch.device device: Device to evaluate on.
    :param str metric: Metric to test.
    :return: Statistics for every ablation, plus the fold-verification log.
    :rtype: dict
    :raises ValueError: If the run is not multi-channel.
    """
    n_channels = len(CANONICAL_CHANNEL_ORDER)
    if config.channel != "all":
        raise ValueError(f"Channel ablation needs an eight-channel run, got {config.channel!r}")

    def ablations(_context: FoldContext) -> dict[str, dict[str, Any]]:
        specs: dict[str, dict[str, Any]] = {}
        for index, name in enumerate(CANONICAL_CHANNEL_ORDER):
            specs[f"drop_{name}"] = {"channel_mask": channel_mask(drop=[index])}
            specs[f"only_{name}"] = {"channel_mask": channel_mask(keep=[index])}
        for region, indices in CHANNEL_REGIONS.items():
            specs[f"drop_region_{region}"] = {
                "channel_mask": channel_mask(drop=list(indices), n_channels=n_channels)
            }
            specs[f"only_region_{region}"] = {
                "channel_mask": channel_mask(keep=list(indices), n_channels=n_channels)
            }
        return specs

    per_fold, baseline, verification = _collect(config, device, ablations)
    summary = _summarise(per_fold, baseline, metric, 1.0 / config.n_folds)
    summary["verification"] = verification
    summary["channel_order"] = CANONICAL_CHANNEL_ORDER
    return summary


def run_channel_occlusion(
    config: RunConfig, device: torch.device, metric: str = PRIMARY_METRIC
) -> dict[str, Any]:
    """
    Leave-one-channel-out by input occlusion rather than token dropping (E1 robustness check).

    Overwrites a channel's whole spectrogram with the training-set mean, leaving the token in
    the sequence. Reported alongside token dropping: agreement between the two rankings means
    the result is a property of the channel, not of the removal mechanism.

    :param RunConfig config: The run to analyse.
    :param torch.device device: Device to evaluate on.
    :param str metric: Metric to test.
    :return: Statistics for every channel occlusion.
    :rtype: dict
    """

    def ablations(context: FoldContext) -> dict[str, dict[str, Any]]:
        reference = reference_spectrogram(context)
        return {
            f"occlude_{name}": {
                "input_transform": _occluder(reference, channels=[index], bins=None)
            }
            for index, name in enumerate(CANONICAL_CHANNEL_ORDER)
        }

    per_fold, baseline, verification = _collect(config, device, ablations)
    summary = _summarise(per_fold, baseline, metric, 1.0 / config.n_folds)
    summary["verification"] = verification
    return summary


def _occluder(
    reference: torch.Tensor, channels: list[int] | None, bins: list[int] | None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build an input transform that occludes a channel/band rectangle.

    :param torch.Tensor reference: Mean spectrogram to write into the occluded region.
    :param channels: Channel indices, or ``None`` for all.
    :param bins: Frequency-bin indices, or ``None`` for all.
    :return: A callable suitable as ``input_transform``.
    :rtype: Callable
    """

    def transform(x: torch.Tensor) -> torch.Tensor:
        return occlude(x, reference, channels=channels, bins=bins)

    return transform


def run_band_ablation(
    config: RunConfig,
    device: torch.device,
    metric: str = PRIMARY_METRIC,
    n_control_draws: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Frequency-band occlusion, per channel and pooled over channels (E3).

    Produces the channel-by-band grid. Each band is also compared against width-matched control
    windows placed elsewhere in the spectrum, because bands differ enormously in width — delta
    spans 3 bins and gamma 41 — so a raw drop partly measures how much input was destroyed
    rather than how informative the band was. Gamma is too wide for any disjoint control to
    exist in a 72-bin spectrum, so its control list comes back empty and its drop must be
    interpreted per-bin instead.

    :param RunConfig config: The run to analyse; works for both eight-channel and in-ear runs.
    :param torch.device device: Device to evaluate on.
    :param str metric: Metric to test.
    :param int n_control_draws: Width-matched control windows to sample per band.
    :param int seed: Seed for control-window placement.
    :return: Statistics for every band ablation, with control windows recorded.
    :rtype: dict
    """
    n_channels = len(CANONICAL_CHANNEL_ORDER) if config.channel == "all" else 1
    rng = np.random.default_rng(seed)
    controls = {
        band: matched_control_windows(
            len(band_bins(band)), n_control_draws, rng, exclude=band_bins(band)
        )
        for band in EEG_BANDS
    }

    def ablations(context: FoldContext) -> dict[str, dict[str, Any]]:
        reference = reference_spectrogram(context)
        specs: dict[str, dict[str, Any]] = {}
        for band in EEG_BANDS:
            bins = band_bins(band)
            specs[f"band_{band}"] = {"input_transform": _occluder(reference, None, bins)}
            for draw, window in enumerate(controls[band]):
                specs[f"control_{band}_{draw}"] = {
                    "input_transform": _occluder(reference, None, window)
                }
            if n_channels > 1:
                for index, name in enumerate(CANONICAL_CHANNEL_ORDER):
                    specs[f"band_{band}_{name}"] = {
                        "input_transform": _occluder(reference, [index], bins)
                    }
        return specs

    per_fold, baseline, verification = _collect(config, device, ablations)
    summary = _summarise(per_fold, baseline, metric, 1.0 / config.n_folds)
    summary["verification"] = verification
    summary["band_bins"] = {band: band_bins(band) for band in EEG_BANDS}
    summary["control_windows"] = controls
    summary["n_channels"] = n_channels
    return summary


def run_hemisphere_mirror(
    config: RunConfig, device: torch.device, metric: str = PRIMARY_METRIC
) -> dict[str, Any]:
    """
    Left-right electrode swap (E2).

    Swapping Fp1/Fp2, C3/C4 and T7/T8 leaves the montage's pooled power spectrum identical and
    destroys only interhemispheric asymmetry. A substantial drop is evidence the model uses
    lateralisation, the family that frontal alpha asymmetry belongs to; no drop is evidence it
    uses bilateral power alone, which is an equally reportable result.

    :param RunConfig config: The run to analyse; must be an eight-channel run.
    :param torch.device device: Device to evaluate on.
    :param str metric: Metric to test.
    :return: Statistics for the mirrored evaluation.
    :rtype: dict
    :raises ValueError: If the run is not multi-channel.
    """
    if config.channel != "all":
        raise ValueError(f"Hemisphere mirror needs an eight-channel run, got {config.channel!r}")

    def ablations(_context: FoldContext) -> dict[str, dict[str, Any]]:
        return {"hemisphere_mirror": {"input_transform": mirror_channels}}

    per_fold, baseline, verification = _collect(config, device, ablations)
    summary = _summarise(per_fold, baseline, metric, 1.0 / config.n_folds)
    summary["verification"] = verification
    return summary


def run_attribution(
    config: RunConfig,
    device: torch.device,
    n_batches: int = 8,
    n_steps: int = 32,
) -> dict[str, Any]:
    """
    Integrated Gradients and attention rollout, reduced to per-channel and per-band scores (E4).

    Both are corroborating evidence only. Attention weights are not an explanation on their own
    and gradient attribution answers a different question than "what happens to accuracy if
    this channel is gone"; the result worth reporting is whether the three rankings agree.

    :param RunConfig config: The run to analyse.
    :param torch.device device: Device to evaluate on.
    :param int n_batches: Validation batches to attribute per fold. Attribution is far more
        expensive than a forward pass, and per-channel means stabilise quickly.
    :param int n_steps: Integrated Gradients interpolation steps.
    :return: Per-fold channel and band scores for each method, plus the worst observed
        completeness error as a correctness check on the integration.
    :rtype: dict
    """
    n_channels = len(CANONICAL_CHANNEL_ORDER) if config.channel == "all" else 1
    folds = build_folds(config)
    supports_rollout = hasattr(config, "model") and config.model == "AllTransformerV4"

    ig_channels: list[list[float]] = []
    ig_bands: list[dict[str, float]] = []
    rollout_channels: list[list[float]] = []
    worst_completeness = 0.0

    for context in iter_folds(config, device, folds=folds):
        reference = reference_spectrogram(context).to(device)
        fold_ig: list[torch.Tensor] = []
        fold_rollout: list[torch.Tensor] = []

        for batch_index, (inputs, labels, _) in enumerate(context.loader):
            if batch_index >= n_batches:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)

            attributions = attr.integrated_gradients(
                context.model, inputs, reference, labels, n_steps=n_steps
            )
            error = attr.completeness_error(context.model, inputs, reference, attributions, labels)
            worst_completeness = max(worst_completeness, float(error.max()))
            # Magnitude: positive and negative evidence both count as "used".
            fold_ig.append(attributions.abs().detach().cpu())

            if supports_rollout:
                with attr.capture_attention(context.model) as captured:
                    with torch.no_grad():
                        context.model(inputs)
                influence = attr.rollout_token_influence(attr.attention_rollout(captured))
                fold_rollout.append(attr.tokens_to_channels(influence, n_channels).cpu())

        stacked = torch.cat(fold_ig)  # (N, C, F, T)
        ig_channels.append(stacked.sum(dim=(2, 3)).mean(dim=0).tolist())
        total = stacked.sum()
        ig_bands.append(
            {band: float(stacked[:, :, band_bins(band), :].sum() / total) for band in EEG_BANDS}
        )
        if fold_rollout:
            rollout_channels.append(torch.cat(fold_rollout).mean(dim=0).tolist())
        logger.info(f"Fold {context.fold + 1}: attribution done")

    return {
        "channel_order": CANONICAL_CHANNEL_ORDER[:n_channels],
        "integrated_gradients": {
            "per_fold_channel_scores": ig_channels,
            "mean_channel_scores": np.mean(ig_channels, axis=0).tolist() if ig_channels else [],
            "per_fold_band_fractions": ig_bands,
            "mean_band_fractions": {
                band: float(np.mean([fold[band] for fold in ig_bands])) for band in EEG_BANDS
            }
            if ig_bands
            else {},
            "max_completeness_error": worst_completeness,
            "n_steps": n_steps,
        },
        "attention_rollout": {
            "per_fold_channel_scores": rollout_channels,
            "mean_channel_scores": np.mean(rollout_channels, axis=0).tolist()
            if rollout_channels
            else [],
            "available": supports_rollout,
        },
    }


def run_site_probe(config: RunConfig, device: torch.device, seed: int = 42) -> dict[str, Any]:
    """
    Can the source dataset be read off the trained representation? (E8, confound audit)

    Fits a multinomial logistic regression on the frozen pooled embedding to predict which
    cohort a chunk came from, under the same subject-independent folds. The same probe is fitted
    on a randomly-initialised backbone as a control: that establishes how much site identity is
    trivially present in the spectrograms themselves, so the gap between the two is the part the
    *pathology objective* actively encoded.

    This addresses the paper's own limitation about cross-site heterogeneity with a measurement
    rather than a caveat. High trained-probe accuracy does not by itself invalidate the
    classifier — cohorts differ in prevalence too — but it bounds how much of the combined-corpus
    accuracy could ride on site identity.

    :param RunConfig config: The run to analyse.
    :param torch.device device: Device to evaluate on.
    :param int seed: Seed for the control model's initialisation and the probe's solver.
    :return: Per-fold probe accuracy for the trained and control backbones, with the
        majority-class baseline for reference.
    :rtype: dict
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    from thesis.explain.harness import load_fold_model

    folds = build_folds(config)
    trained_scores: list[float] = []
    control_scores: list[float] = []
    majority: list[float] = []

    for context in iter_folds(config, device, folds=folds):
        control, _ = load_fold_model(config, context.fold, device)
        _reinitialise(control, seed + context.fold)

        embeddings, control_embeddings, sites, subjects = _collect_embeddings(
            context, control, device
        )
        if len(set(sites)) < 2:
            logger.warning(f"Fold {context.fold + 1}: only one cohort present, skipping probe")
            continue

        # Split the fold's validation set in two to fit and score the probe. Grouping by
        # subject stops the probe memorising a subject instead of a cohort, and stratifying by
        # cohort keeps both classes on each side — a plain positional split would not, because
        # the validation loader is unshuffled and so emits chunks grouped by cohort.
        try:
            splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
            fit_index, score_index = next(splitter.split(embeddings, sites, groups=subjects))
        except ValueError as exc:
            logger.warning(f"Fold {context.fold + 1}: cannot split for the probe ({exc})")
            continue

        for name, features, store in (
            ("trained", embeddings, trained_scores),
            ("control", control_embeddings, control_scores),
        ):
            probe = LogisticRegression(max_iter=2000, random_state=seed)
            probe.fit(features[fit_index], [sites[i] for i in fit_index])
            store.append(float(probe.score(features[score_index], [sites[i] for i in score_index])))
            logger.info(f"Fold {context.fold + 1} {name} site probe: {store[-1]:.4f}")

        held_out = [sites[i] for i in score_index]
        majority.append(max(held_out.count(s) for s in set(held_out)) / len(held_out))

    return {
        "trained_backbone": {
            "per_fold": trained_scores,
            "mean": float(np.mean(trained_scores)) if trained_scores else float("nan"),
        },
        "random_backbone": {
            "per_fold": control_scores,
            "mean": float(np.mean(control_scores)) if control_scores else float("nan"),
        },
        "majority_baseline": {
            "per_fold": majority,
            "mean": float(np.mean(majority)) if majority else float("nan"),
        },
    }


def _reinitialise(model: torch.nn.Module, seed: int) -> None:
    """
    Re-randomise every parameter of a model in place.

    :param torch.nn.Module model: Model to reset.
    :param int seed: Seed, so the control is reproducible.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.empty(parameter.shape, device="cpu")
            # Match the scale of the parameter being replaced, so the control backbone is a
            # plausible untrained network rather than a degenerate one.
            std = float(parameter.std()) if parameter.numel() > 1 else 0.02
            noise.normal_(0.0, max(std, 1e-4), generator=generator)
            parameter.copy_(noise.to(parameter.device))


def _collect_embeddings(
    context: FoldContext, control: torch.nn.Module, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Pooled embeddings from the trained and control backbones, with cohort and subject labels.

    :param FoldContext context: The fold to read.
    :param torch.nn.Module control: A randomly-initialised copy of the same architecture.
    :param torch.device device: Device to evaluate on.
    :return: ``(trained_embeddings, control_embeddings, site_labels, subject_ids)``.
    :rtype: tuple
    :raises AttributeError: If the model exposes no ``encode`` method.
    """
    trained_rows: list[torch.Tensor] = []
    control_rows: list[torch.Tensor] = []
    sites: list[str] = []
    subject_ids: list[str] = []

    with torch.no_grad():
        for inputs, _, subjects in context.loader:
            inputs = inputs.to(device)
            # encode() is defined on AllTransformerV4, not on nn.Module.
            trained_rows.append(cast(Any, context.model).encode(inputs).cpu())
            control_rows.append(cast(Any, control).encode(inputs).cpu())
            sites.extend(context.subject_dataset_map.get(s, "unknown") for s in subjects)
            subject_ids.extend(subjects)

    return (
        torch.cat(trained_rows).numpy(),
        torch.cat(control_rows).numpy(),
        sites,
        subject_ids,
    )


def run_sanity_checks(
    config: RunConfig, device: torch.device, metric: str = PRIMARY_METRIC, seed: int = 42
) -> dict[str, Any]:
    """
    Channel ablation on a randomly-initialised model (E7).

    An attribution method that returns the same answer for an untrained network explains
    nothing (Adebayo et al., 2018). Re-running leave-one-channel-out against random weights must
    produce a flat, non-significant ranking; if instead it reproduces the trained model's
    ranking, the "importance" is an artifact of the architecture or the data, not of anything
    the model learned.

    The companion shuffled-label control needs a retrained model and is therefore not run here.

    :param RunConfig config: The run to analyse; must be an eight-channel run.
    :param torch.device device: Device to evaluate on.
    :param str metric: Metric to test.
    :param int seed: Seed for the random initialisation.
    :return: Channel-ablation statistics for the randomised model, to be compared against the
        trained model's ranking.
    :rtype: dict
    :raises ValueError: If the run is not multi-channel.
    """
    if config.channel != "all":
        raise ValueError(f"Sanity checks need an eight-channel run, got {config.channel!r}")

    def ablations(context: FoldContext) -> dict[str, dict[str, Any]]:
        _reinitialise(context.model, seed + context.fold)
        return {
            f"drop_{name}": {"channel_mask": channel_mask(drop=[index])}
            for index, name in enumerate(CANONICAL_CHANNEL_ORDER)
        }

    # Verification is meaningless here: the weights are deliberately not the trained ones.
    per_fold, baseline, _ = _collect(config, device, ablations, verify=False)
    summary = _summarise(per_fold, baseline, metric, 1.0 / config.n_folds)
    summary["note"] = (
        "Randomly-initialised weights. A flat, non-significant ranking is the expected "
        "(and required) outcome; agreement with the trained ranking would invalidate it."
    )
    summary["channel_order"] = CANONICAL_CHANNEL_ORDER
    return summary
