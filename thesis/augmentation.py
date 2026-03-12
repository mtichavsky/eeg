"""EEG data augmentation transforms for training."""

import numpy as np
import torch
from scipy.interpolate import CubicSpline


class EEGAugmentation:
    """
    EEG augmentation pipeline applied to raw EEG before STFT conversion.

    Applies augmentations independently with probability p_aug.
    Only includes transforms with clear justification for resting-state EEG.
    """

    def __init__(
        self,
        p_aug: float = 0.5,
        use_gaussian_noise: bool = True,
        noise_std: float = 0.03,
        use_mag_warp: bool = True,
        mag_warp_sigma: float = 0.1,
        mag_warp_knots: int = 4,
    ) -> None:
        """
        Initialize augmentation pipeline.

        :param float p_aug: Probability of applying each augmentation (default 0.5).
        :param bool use_gaussian_noise: Add Gaussian noise to simulate electrode noise.
        :param float noise_std: Std of additive Gaussian noise relative to signal std.
        :param bool use_mag_warp: Smooth amplitude modulation simulating within-window drift.
        :param float mag_warp_sigma: Sigma for magnitude warping curve (default 0.1).
        :param int mag_warp_knots: Cubic spline knots for warping curve (default 4).
        """
        self.p_aug = p_aug
        self.use_gaussian_noise = use_gaussian_noise
        self.noise_std = noise_std
        self.use_mag_warp = use_mag_warp
        self.mag_warp_sigma = mag_warp_sigma
        self.mag_warp_knots = mag_warp_knots

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to a single EEG chunk.

        :param np.ndarray x: EEG chunk of shape (samples,) or (channels, samples).
        :return: Augmented EEG chunk with same shape.
        :rtype: np.ndarray
        """
        is_1d = x.ndim == 1
        if is_1d:
            x = x.reshape(1, -1)

        if self.use_gaussian_noise and np.random.random() < self.p_aug:
            x = self._gaussian_noise(x)

        if self.use_mag_warp and np.random.random() < self.p_aug:
            x = self._mag_warp(x)

        return x.squeeze() if is_1d else x

    def _gaussian_noise(self, x: np.ndarray) -> np.ndarray:
        """Add Gaussian noise scaled to per-channel signal std."""
        std = np.std(x, axis=-1, keepdims=True)
        noise = np.random.normal(0, self.noise_std * std, x.shape)
        return x + noise

    def _mag_warp(self, x: np.ndarray) -> np.ndarray:
        """
        Magnitude Warping: multiply by smooth random curve centered at 1.

        Uses cubic spline interpolation to create smooth warping curves.
        """
        n_samples = x.shape[1]
        knot_xs = np.linspace(0, n_samples - 1, self.mag_warp_knots + 2)
        knot_ys = np.random.normal(1.0, self.mag_warp_sigma, len(knot_xs))
        cs = CubicSpline(knot_xs, knot_ys)
        warp_curve = cs(np.arange(n_samples))
        return x * warp_curve


class SpecAugment:
    """
    SpecAugment-style masking applied to EEG spectrograms.

    Randomly zeros out contiguous time and/or frequency strips. Forces the model
    to use distributed time-frequency features rather than relying on a single
    spectrogram region.

    Applied to spectrogram tensors of shape (C, F, T) after STFT conversion.
    """

    def __init__(
        self,
        p_aug: float = 0.5,
        max_time_mask: int = 5,
        max_freq_mask: int = 10,
        n_time_masks: int = 1,
        n_freq_masks: int = 1,
    ) -> None:
        """
        Initialize SpecAugment.

        :param float p_aug: Probability of applying each mask type (default 0.5).
        :param int max_time_mask: Max width of time mask in frames (default 5, ~1s at 41 frames).
        :param int max_freq_mask: Max height of frequency mask in bins (default 10, ~5 Hz at 129 bins).
        :param int n_time_masks: Number of time masks to apply (default 1).
        :param int n_freq_masks: Number of frequency masks to apply (default 1).
        """
        self.p_aug = p_aug
        self.max_time_mask = max_time_mask
        self.max_freq_mask = max_freq_mask
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time and frequency masking.

        :param torch.Tensor x: Spectrogram of shape (C, F, T).
        :return: Masked spectrogram of same shape.
        :rtype: torch.Tensor
        """
        x = x.clone()
        _, n_freq, n_time = x.shape

        if np.random.random() < self.p_aug:
            for _ in range(self.n_time_masks):
                width = np.random.randint(1, self.max_time_mask + 1)
                start = np.random.randint(0, max(1, n_time - width))
                x[:, :, start : start + width] = 0.0

        if np.random.random() < self.p_aug:
            for _ in range(self.n_freq_masks):
                height = np.random.randint(1, self.max_freq_mask + 1)
                start = np.random.randint(0, max(1, n_freq - height))
                x[:, start : start + height, :] = 0.0

        return x
