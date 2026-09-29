"""
Masking primitives for channel and frequency-band ablation.

Two distinct ablation mechanisms live here, and the difference matters when reporting:

* **Token dropping** (:func:`channel_mask`, consumed by ``AllTransformerV4.forward``) removes
  a channel's tokens from the sequence entirely. The channel then contributes to neither
  self-attention nor the mean pool, and its ``chan_embedding`` leaves with it. This is the
  primary channel-ablation mechanism.
* **Input occlusion** (:func:`occlude`) overwrites part of the spectrogram with a reference
  value. It is the only option for frequency bands, and serves as a robustness check for
  channels.

Occlusion never writes zeros. Inputs are ``log1p|STFT|``, where zero means "no power at all" —
an input far outside the data distribution. Overwriting with the training-set mean
(:func:`spectrogram_reference`) keeps the input plausible, so an accuracy drop measures the
loss of *that band's information* rather than the shock of an out-of-distribution input.
"""

from collections.abc import Sequence

import numpy as np
import torch

from thesis.dataset import CANONICAL_CHANNEL_ORDER
from thesis.stft import FREQ_BIN_WIDTH_HZ, NUM_FREQ_BINS

#: Canonical EEG frequency bands as ``(low_inclusive, high_exclusive)`` in Hz. The upper edge
#: of gamma is the preprocessing band-pass cutoff, above which the spectrogram is cropped.
EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 70.0),
}

#: Scalp regions as indices into :data:`thesis.dataset.CANONICAL_CHANNEL_ORDER`. Grouped
#: ablation is necessary because neighbouring electrodes are spatially redundant: dropping one
#: of a correlated pair can look harmless even when the pair is jointly essential.
CHANNEL_REGIONS: dict[str, tuple[int, ...]] = {
    "frontal": (0, 1),  # Fp1, Fp2
    "central": (2, 3, 4),  # C3, Cz, C4
    "temporal": (5, 6),  # T7, T8
    "occipital": (7,),  # O2/Oz
}

#: Left/right electrode swap: Fp1<->Fp2, C3<->C4, T7<->T8. Cz and O2/Oz are midline and stay
#: put. Applying this permutation destroys interhemispheric asymmetry while leaving the total
#: power spectrum of the montage untouched.
HEMISPHERE_MIRROR: tuple[int, ...] = (1, 0, 4, 3, 2, 6, 5, 7)


def band_bins(band: str, num_freq_bins: int = NUM_FREQ_BINS) -> list[int]:
    """
    Frequency-bin indices belonging to a named band.

    Bin ``i`` is centred at ``i * FREQ_BIN_WIDTH_HZ`` (0.9766 Hz), so band membership is tested
    on centre frequencies. Bands have very unequal bin counts as a result — delta spans 3 bins
    and gamma 41 — so raw occlusion drops are **not** comparable across bands without the
    width-matched control from :func:`matched_control_windows`.

    :param str band: Band name, a key of :data:`EEG_BANDS`.
    :param int num_freq_bins: Number of frequency bins in the spectrogram.
    :return: Sorted bin indices.
    :rtype: list[int]
    :raises KeyError: If the band name is unknown.
    """
    low, high = EEG_BANDS[band]
    return [i for i in range(num_freq_bins) if low <= i * FREQ_BIN_WIDTH_HZ < high]


def channel_mask(
    keep: Sequence[int] | None = None,
    drop: Sequence[int] | None = None,
    n_channels: int = len(CANONICAL_CHANNEL_ORDER),
) -> torch.Tensor:
    """
    Build a boolean channel mask for ``AllTransformerV4.forward``.

    Exactly one of ``keep`` or ``drop`` must be given.

    :param keep: Channel indices to retain (leave-one-in style).
    :param drop: Channel indices to remove (leave-one-out style).
    :param int n_channels: Total channel count.
    :return: Boolean tensor of shape ``(n_channels,)``, ``True`` where the channel survives.
    :rtype: torch.Tensor
    :raises ValueError: If neither or both of ``keep``/``drop`` are given, if an index is out
        of range, or if the mask would remove every channel.
    """
    if (keep is None) == (drop is None):
        raise ValueError("Pass exactly one of keep= or drop=")

    indices = keep if keep is not None else drop
    assert indices is not None  # narrowed by the check above
    for index in indices:
        if not 0 <= index < n_channels:
            raise ValueError(f"Channel index {index} out of range for {n_channels} channels")

    mask = torch.zeros(n_channels, dtype=torch.bool)
    mask[list(indices)] = True
    if drop is not None:
        mask = ~mask
    if not bool(mask.any()):
        raise ValueError("Mask would remove every channel")
    return mask


def spectrogram_reference(
    batches: Sequence[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """
    Per-(channel, frequency, time) mean spectrogram, used as the occlusion baseline.

    Computed over training-fold data only. Using the mean rather than zero keeps an occluded
    input inside the data distribution; see the module docstring.

    :param batches: One tensor of shape ``(N, C, F, T)``, or a sequence of such batches.
    :return: Mean spectrogram of shape ``(C, F, T)``.
    :rtype: torch.Tensor
    :raises ValueError: If no samples are supplied.
    """
    if isinstance(batches, torch.Tensor):
        batches = [batches]

    total: torch.Tensor | None = None
    count = 0
    for batch in batches:
        summed = batch.sum(dim=0, dtype=torch.float64)
        total = summed if total is None else total + summed
        count += batch.shape[0]

    if total is None or count == 0:
        raise ValueError("Cannot compute a reference spectrogram from zero samples")
    return (total / count).to(torch.float32)


def occlude(
    x: torch.Tensor,
    reference: torch.Tensor,
    channels: Sequence[int] | None = None,
    bins: Sequence[int] | None = None,
) -> torch.Tensor:
    """
    Replace a (channel x frequency-bin) rectangle of the input with the reference value.

    ``channels=None`` occludes the band in every channel; ``bins=None`` occludes the whole
    spectrum of the named channels.

    :param torch.Tensor x: Input batch of shape ``(N, C, F, T)``.
    :param torch.Tensor reference: Reference spectrogram of shape ``(C, F, T)``, from
        :func:`spectrogram_reference`.
    :param channels: Channel indices to occlude, or ``None`` for all.
    :param bins: Frequency-bin indices to occlude, or ``None`` for all.
    :return: A new tensor; ``x`` is not modified.
    :rtype: torch.Tensor
    :raises ValueError: If the reference shape does not match the input.
    """
    if reference.shape != x.shape[1:]:
        raise ValueError(
            f"Reference shape {tuple(reference.shape)} does not match input {tuple(x.shape[1:])}"
        )

    channel_idx = list(range(x.shape[1])) if channels is None else list(channels)
    bin_idx = list(range(x.shape[2])) if bins is None else list(bins)
    if not channel_idx or not bin_idx:
        return x.clone()

    out = x.clone()
    reference = reference.to(device=x.device, dtype=x.dtype)
    # Outer-product indexing so the two index lists select a rectangle, not a diagonal.
    rows = torch.tensor(channel_idx, device=x.device).view(-1, 1)
    cols = torch.tensor(bin_idx, device=x.device).view(1, -1)
    out[:, rows, cols, :] = reference[rows, cols, :]
    return out


def mirror_channels(
    x: torch.Tensor, permutation: Sequence[int] = HEMISPHERE_MIRROR
) -> torch.Tensor:
    """
    Swap left and right electrodes, leaving midline channels in place.

    Probes whether a model relies on interhemispheric asymmetry — frontal alpha asymmetry is
    the canonical depression marker. The mirrored montage has an identical pooled power
    spectrum, so any accuracy change is attributable to lateralisation alone.

    :param torch.Tensor x: Input batch of shape ``(N, C, F, T)``.
    :param permutation: New channel order; defaults to :data:`HEMISPHERE_MIRROR`.
    :return: Channel-permuted copy of ``x``.
    :rtype: torch.Tensor
    :raises ValueError: If the permutation length does not match the channel count.
    """
    if len(permutation) != x.shape[1]:
        raise ValueError(
            f"Permutation of length {len(permutation)} does not match {x.shape[1]} channels"
        )
    return x[:, list(permutation), :, :]


def matched_control_windows(
    width: int,
    n_draws: int,
    rng: np.random.Generator,
    exclude: Sequence[int] = (),
    num_freq_bins: int = NUM_FREQ_BINS,
) -> list[list[int]]:
    """
    Sample contiguous frequency windows of a given width, avoiding the band under test.

    Occluding gamma removes 41 bins and delta only 3, so a larger drop for gamma may reflect
    nothing but the amount of input destroyed. Comparing each band against occlusions of the
    same width placed elsewhere in the spectrum controls for that.

    :param int width: Number of bins per window, i.e. the width of the band being controlled.
    :param int n_draws: How many windows to sample.
    :param numpy.random.Generator rng: Seeded random generator.
    :param exclude: Bin indices the control windows must not overlap (the band itself).
    :param int num_freq_bins: Number of frequency bins in the spectrogram.
    :return: Up to ``n_draws`` windows, each a list of contiguous bin indices. Fewer are
        returned when the spectrum cannot accommodate that many disjoint-from-``exclude``
        placements; empty when the band is too wide for any control to exist.
    :rtype: list[list[int]]
    :raises ValueError: If ``width`` is not positive.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if width > num_freq_bins:
        return []

    excluded = set(exclude)
    # Bin 0 is DC, which the 1 Hz high-pass has already emptied; never start a control there.
    candidates = [
        start
        for start in range(1, num_freq_bins - width + 1)
        if not excluded.intersection(range(start, start + width))
    ]
    if not candidates:
        return []

    chosen = rng.choice(len(candidates), size=min(n_draws, len(candidates)), replace=False)
    return [list(range(candidates[i], candidates[i] + width)) for i in sorted(chosen)]
