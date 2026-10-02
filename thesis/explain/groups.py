"""
Player groups for grouped SHAP on spectrogram inputs.

A *player* is a set of input pixels that is switched on (taken from the explained chunk) or
off (taken from a background chunk) as one unit. Four games are defined over a
``(C, F, T)`` spectrogram:

* **bands**: one player per EEG frequency band, spanning every channel and time frame,
* **channels**: one player per electrode, spanning its whole ``(F, T)`` plane,
* **grid**: one player per (channel, band) cell,
* **gamma**: the band game with gamma split into 30-45 and 45-70 Hz.

The players of every game **partition** the input: each pixel belongs to exactly one player.
Shapley efficiency (``sum(phi) == f(x) - E f(b)``) only accounts for the whole prediction when
nothing is left out, which is why the bins below delta get a player of their own
(``sub_delta``), even though they carry little beyond band-pass roll-off.
"""

from dataclasses import dataclass
from functools import cached_property, partial
from typing import Literal

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

#: Name of the player holding every bin that no band in :data:`EEG_BANDS` covers.
SUB_DELTA = "sub_delta"

#: Where the ``gamma`` game splits the gamma band: 30-45 Hz is below the 50 Hz notch, 45-70 Hz
#: contains it, and EMG dominates both, so the split separates "low" from "notch-affected" gamma.
GAMMA_SPLIT_HZ = 45.0

Game = Literal["bands", "channels", "grid", "gamma"]


def band_bins(band: str, num_freq_bins: int = NUM_FREQ_BINS) -> list[int]:
    """
    Frequency-bin indices belonging to a named band.

    Bin ``i`` is centred at ``i * FREQ_BIN_WIDTH_HZ`` (0.9766 Hz), so band membership is tested
    on centre frequencies. Bands have very unequal bin counts as a result (delta spans 3 bins,
    gamma 41), which is why per-bin importance is reported next to the per-band totals.

    :param str band: Band name, a key of :data:`EEG_BANDS`.
    :param int num_freq_bins: Number of frequency bins in the spectrogram.
    :return: Sorted bin indices.
    :rtype: list[int]
    :raises KeyError: If the band name is unknown.
    """
    low, high = EEG_BANDS[band]
    return [i for i in range(num_freq_bins) if low <= i * FREQ_BIN_WIDTH_HZ < high]


def band_partition(
    num_freq_bins: int = NUM_FREQ_BINS, split_gamma: bool = False
) -> dict[str, list[int]]:
    """
    Split the frequency axis into ``sub_delta`` plus the canonical bands.

    Bands with no bins (gamma under a 30 Hz cutoff) are omitted: an empty player is a dummy
    whose Shapley value is zero by definition, so it would only add cost.

    :param int num_freq_bins: Number of frequency bins in the spectrogram.
    :param bool split_gamma: Replace ``gamma`` by ``gamma_low`` and ``gamma_high`` at
        :data:`GAMMA_SPLIT_HZ`.
    :return: Player name -> bin indices, in ascending frequency order, covering every bin once.
    :rtype: dict[str, list[int]]
    """
    bands = {name: band_bins(name, num_freq_bins) for name in EEG_BANDS}
    covered = {i for bins in bands.values() for i in bins}
    partition = {SUB_DELTA: [i for i in range(num_freq_bins) if i not in covered]}
    partition.update(bands)
    if split_gamma:
        gamma = partition.pop("gamma")
        split = sum(i * FREQ_BIN_WIDTH_HZ < GAMMA_SPLIT_HZ for i in gamma)
        partition["gamma_low"], partition["gamma_high"] = gamma[:split], gamma[split:]
    return {name: bins for name, bins in partition.items() if bins}


@dataclass(frozen=True)
class PlayerSet:
    """
    The players of one game.

    :ivar list[str] names: Player names, in the order of ``masks``.
    :ivar torch.Tensor masks: Boolean ``(M, C, F, T)``; ``True`` where a player's pixels live.
    :ivar list[int] n_bins: Number of frequency bins each player spans (per channel), for
        width-normalised importance.
    """

    names: list[str]
    masks: torch.Tensor
    n_bins: list[int]

    def __post_init__(self) -> None:
        if self.masks.dtype != torch.bool:
            raise ValueError(f"Player masks must be bool, got {self.masks.dtype}")
        if self.masks.shape[0] != len(self.names):
            raise ValueError(f"{self.masks.shape[0]} masks for {len(self.names)} player names")
        coverage = self.masks.sum(dim=0)
        if not bool((coverage == 1).all()):
            raise ValueError(
                "Player masks must partition the input: "
                f"{int((coverage == 0).sum())} pixels uncovered, "
                f"{int((coverage > 1).sum())} covered more than once"
            )

    def __len__(self) -> int:
        return len(self.names)

    @cached_property
    def flat(self) -> torch.Tensor:
        """
        Masks as a float ``(M, C*F*T)`` matrix on the masks' device, built once per set.

        :return: 1.0 where a player's pixels live.
        :rtype: torch.Tensor
        """
        return self.masks.flatten(1).to(torch.float32)

    def to(self, device: torch.device) -> "PlayerSet":
        """
        Copy of this player set with its masks on ``device``.

        :param torch.device device: Target device.
        :return: The moved player set.
        :rtype: PlayerSet
        """
        return PlayerSet(self.names, self.masks.to(device), self.n_bins)


def _channel_names(n_channels: int) -> list[str]:
    """
    Electrode names for a model's input channels.

    :param int n_channels: 8 for the full montage, 1 for a single derived channel.
    :return: Channel names.
    :rtype: list[str]
    :raises ValueError: For any other channel count.
    """
    if n_channels == len(CANONICAL_CHANNEL_ORDER):
        return list(CANONICAL_CHANNEL_ORDER)
    if n_channels == 1:
        return ["single"]
    raise ValueError(f"Unsupported channel count {n_channels}")


def band_players(
    n_channels: int, freq_bins: int, time_frames: int, split_gamma: bool = False
) -> PlayerSet:
    """
    One player per frequency band, across all channels and time frames.

    :param int n_channels: Number of input channels.
    :param int freq_bins: Number of frequency bins.
    :param int time_frames: Number of STFT time frames.
    :param bool split_gamma: Split gamma into ``gamma_low`` / ``gamma_high`` (the ``gamma`` game).
    :return: Players ``sub_delta, delta, theta, alpha, beta, gamma`` (fewer below 70 Hz).
    :rtype: PlayerSet
    """
    partition = band_partition(freq_bins, split_gamma)
    masks = torch.zeros(len(partition), n_channels, freq_bins, time_frames, dtype=torch.bool)
    for i, bins in enumerate(partition.values()):
        masks[i, :, bins, :] = True
    return PlayerSet(list(partition), masks, [len(b) for b in partition.values()])


def channel_players(n_channels: int, freq_bins: int, time_frames: int) -> PlayerSet:
    """
    One player per electrode, covering its whole ``(F, T)`` plane.

    :param int n_channels: Number of input channels; must be greater than one.
    :param int freq_bins: Number of frequency bins.
    :param int time_frames: Number of STFT time frames.
    :return: One player per channel, in :data:`CANONICAL_CHANNEL_ORDER`.
    :rtype: PlayerSet
    :raises ValueError: For a single-channel input, where the game is trivial.
    """
    if n_channels < 2:
        raise ValueError("The channel game needs a multi-channel model")
    masks = torch.zeros(n_channels, n_channels, freq_bins, time_frames, dtype=torch.bool)
    for c in range(n_channels):
        masks[c, c] = True
    return PlayerSet(_channel_names(n_channels), masks, [freq_bins] * n_channels)


def grid_players(n_channels: int, freq_bins: int, time_frames: int) -> PlayerSet:
    """
    One player per (channel, band) cell, channel-major.

    :param int n_channels: Number of input channels; must be greater than one.
    :param int freq_bins: Number of frequency bins.
    :param int time_frames: Number of STFT time frames.
    :return: Players named ``"<channel>:<band>"``, 48 for eight channels at 70 Hz.
    :rtype: PlayerSet
    :raises ValueError: For a single-channel input, where the grid equals the band game.
    """
    if n_channels < 2:
        raise ValueError("The channel x band game needs a multi-channel model")
    partition = band_partition(freq_bins)
    names: list[str] = []
    n_bins: list[int] = []
    masks = torch.zeros(
        n_channels * len(partition), n_channels, freq_bins, time_frames, dtype=torch.bool
    )
    for c, channel in enumerate(_channel_names(n_channels)):
        for b, (band, bins) in enumerate(partition.items()):
            masks[c * len(partition) + b, c, bins, :] = True
            names.append(f"{channel}:{band}")
            n_bins.append(len(bins))
    return PlayerSet(names, masks, n_bins)


def build_players(game: Game, n_channels: int, freq_bins: int, time_frames: int) -> PlayerSet:
    """
    Build the players of a named game.

    :param str game: ``bands``, ``channels``, ``grid`` or ``gamma``.
    :param int n_channels: Number of input channels.
    :param int freq_bins: Number of frequency bins.
    :param int time_frames: Number of STFT time frames.
    :return: The game's players.
    :rtype: PlayerSet
    :raises ValueError: For an unknown game.
    """
    builders = {
        "bands": band_players,
        "channels": channel_players,
        "grid": grid_players,
        "gamma": partial(band_players, split_gamma=True),
    }
    if game not in builders:
        raise ValueError(f"Unknown game {game!r}")
    return builders[game](n_channels, freq_bins, time_frames)
