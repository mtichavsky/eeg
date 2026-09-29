"""
Explainability analyses for the spectrogram models.

The package answers three questions asked of the trained models:

* which electrodes carry the signal (channel ablation, :mod:`thesis.explain.masks`),
* which frequency bands carry it (band occlusion, same module),
* and whether the evidence is neurophysiological or an artifact of the data collection
  (attribution in :mod:`thesis.explain.attribution`, site probing in
  :mod:`thesis.explain.harness`).

Every experiment is scored with the same :func:`main.eval_epoch` the training loop uses, so
ablation results are directly comparable with the accuracies reported in the paper.
"""

from thesis.explain.masks import (
    CHANNEL_REGIONS,
    EEG_BANDS,
    HEMISPHERE_MIRROR,
    band_bins,
    channel_mask,
    matched_control_windows,
    mirror_channels,
    occlude,
    spectrogram_reference,
)
from thesis.explain.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    nadeau_bengio_corrected_t,
    paired_wilcoxon,
    spearman,
)

__all__ = [
    "CHANNEL_REGIONS",
    "EEG_BANDS",
    "HEMISPHERE_MIRROR",
    "band_bins",
    "benjamini_hochberg",
    "bootstrap_ci",
    "channel_mask",
    "matched_control_windows",
    "mirror_channels",
    "nadeau_bengio_corrected_t",
    "occlude",
    "paired_wilcoxon",
    "spearman",
    "spectrogram_reference",
]
