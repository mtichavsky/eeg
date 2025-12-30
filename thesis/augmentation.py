"""EEG data augmentation transforms for training."""

import numpy as np
from scipy.interpolate import CubicSpline


class EEGAugmentation:
    """
    Composable EEG augmentation pipeline.

    Applies augmentations to raw EEG chunks before spectrogram conversion.
    Each augmentation is applied independently with probability p_aug.
    """

    def __init__(
        self,
        p_aug: float = 0.5,
        use_ft_surrogate: bool = True,
        use_mag_warp: bool = True,
        use_time_reverse: bool = True,
        use_scaling: bool = True,
        scaling_range: tuple[float, float] = (0.8, 1.2),
        mag_warp_sigma: float = 0.2,
        mag_warp_knots: int = 4,
    ):
        """
        Initialize augmentation pipeline.

        :param float p_aug: Probability of applying each augmentation (default 0.5).
        :param bool use_ft_surrogate: Enable FT Surrogate augmentation.
        :param bool use_mag_warp: Enable Magnitude Warping augmentation.
        :param bool use_time_reverse: Enable Time Reversal augmentation.
        :param bool use_scaling: Enable Scaling augmentation.
        :param tuple scaling_range: Min/max scaling factor (default 0.8-1.2).
        :param float mag_warp_sigma: Sigma for magnitude warping curve (default 0.2).
        :param int mag_warp_knots: Number of knots for cubic spline in MagWarp.
        """
        self.p_aug = p_aug
        self.use_ft_surrogate = use_ft_surrogate
        self.use_mag_warp = use_mag_warp
        self.use_time_reverse = use_time_reverse
        self.use_scaling = use_scaling
        self.scaling_range = scaling_range
        self.mag_warp_sigma = mag_warp_sigma
        self.mag_warp_knots = mag_warp_knots

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to a single EEG chunk.

        :param np.ndarray x: EEG chunk of shape (samples,) or (channels, samples).
        :return: Augmented EEG chunk with same shape.
        :rtype: np.ndarray
        """
        # Handle both 1D and 2D inputs
        is_1d = x.ndim == 1
        if is_1d:
            x = x.reshape(1, -1)

        # Apply each augmentation with probability p_aug
        if self.use_ft_surrogate and np.random.random() < self.p_aug:
            x = self._ft_surrogate(x)

        if self.use_mag_warp and np.random.random() < self.p_aug:
            x = self._mag_warp(x)

        if self.use_time_reverse and np.random.random() < self.p_aug:
            x = self._time_reverse(x)

        if self.use_scaling and np.random.random() < self.p_aug:
            x = self._scaling(x)

        return x.squeeze() if is_1d else x

    def _ft_surrogate(self, x: np.ndarray) -> np.ndarray:
        """
        Fourier Transform Surrogate: randomize phases, preserve magnitude.

        Creates signals with identical power spectrum but different time-domain structure.
        """
        result = np.zeros_like(x)
        for ch in range(x.shape[0]):
            # FFT
            fft = np.fft.rfft(x[ch])
            # Randomize phases (keep Nyquist real if present)
            phases = np.random.uniform(0, 2 * np.pi, len(fft))
            phases[0] = 0  # DC component stays real
            if len(x[ch]) % 2 == 0:
                phases[-1] = 0  # Nyquist stays real for even-length signals
            # Apply random phases to magnitude
            fft_surrogate = np.abs(fft) * np.exp(1j * phases)
            # IFFT
            result[ch] = np.fft.irfft(fft_surrogate, n=x.shape[1])
        return result

    def _mag_warp(self, x: np.ndarray) -> np.ndarray:
        """
        Magnitude Warping: multiply by smooth random curve centered at 1.

        Uses cubic spline interpolation to create smooth warping curves.
        """
        n_samples = x.shape[1]
        # Generate random knot points
        knot_xs = np.linspace(0, n_samples - 1, self.mag_warp_knots + 2)
        knot_ys = np.random.normal(1.0, self.mag_warp_sigma, len(knot_xs))
        # Cubic spline interpolation
        cs = CubicSpline(knot_xs, knot_ys)
        warp_curve = cs(np.arange(n_samples))
        # Apply warping
        return x * warp_curve

    def _time_reverse(self, x: np.ndarray) -> np.ndarray:
        """Time Reversal: flip signal along time axis."""
        return np.flip(x, axis=-1).copy()  # .copy() to ensure contiguous array

    def _scaling(self, x: np.ndarray) -> np.ndarray:
        """Scaling: multiply by random scalar."""
        scale = np.random.uniform(self.scaling_range[0], self.scaling_range[1])
        return x * scale
