"""
Grouped SHAP explanations for the spectrogram models.

Answers which frequency bands, which electrodes, and which electrode x band cells drive a
trained model's predictions, out-of-fold under the same 10-fold subject-independent protocol
as every other number in the paper:

* :mod:`thesis.explain.harness` rebuilds a run's folds, reloads each fold's checkpoint and
  verifies the rebuild against the metrics stored in the checkpoint,
* :mod:`thesis.explain.groups` defines the players (bands, channels, channel x band),
* :mod:`thesis.explain.shap_values` wraps the model as a coalition value function for the
  ``shap`` library and samples same-dataset training-fold backgrounds,
* :mod:`thesis.explain.experiments` runs the games over every fold and aggregates the results,
* :mod:`thesis.explain.stats` holds the paired fold-level statistics.

Driven by ``python main.py explain <run-dir> --experiment shap-...``. The ``shap`` library is a
regular dependency, pinned in ``pyproject.toml``.
"""

from thesis.explain.groups import EEG_BANDS, PlayerSet, band_bins, build_players
from thesis.explain.stats import (
    benjamini_hochberg,
    nadeau_bengio_corrected_t,
    paired_wilcoxon,
    spearman,
)

__all__ = [
    "EEG_BANDS",
    "PlayerSet",
    "band_bins",
    "benjamini_hochberg",
    "build_players",
    "nadeau_bengio_corrected_t",
    "paired_wilcoxon",
    "spearman",
]
