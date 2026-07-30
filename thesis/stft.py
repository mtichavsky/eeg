"""
Canonical spectrogram configuration shared by training and inference.

Every dataset is resampled to :data:`MODEL_FS` **before** the STFT, so that a given row of a
spectrogram always means the same frequency regardless of the source dataset's native sampling
rate. The bin width is ``fs / nperseg``, so without the resample a 500 Hz dataset and a 250 Hz
one would put the same frequency on different rows.

Rows above :data:`FREQ_CUTOFF_HZ` are dropped: the preprocessing band-pass is 1-70 Hz, so those
bins carry only filter roll-off and numerical noise.
"""

import numpy as np
from scipy.signal import resample, stft

#: EEG chunk duration in seconds.
CHUNK_DURATION_SEC = 10

#: Sampling rate every dataset is resampled to before the STFT (and the rate raw-EEG models
#: such as Deformer expect). Use for ``target_samples = int(chunk_duration * MODEL_FS)``.
MODEL_FS: int = 250

#: Samples per STFT window. Determines the number of frequency bins (``nperseg // 2 + 1``).
STFT_NPERSEG: int = 256

#: Samples shared between consecutive STFT windows. Determines the hop, hence the frame count.
STFT_NOVERLAP: int = 192

#: Window function applied before each FFT.
STFT_WINDOW: str = "hamming"

#: Highest frequency kept. Matches the 1-70 Hz preprocessing band-pass upper edge.
FREQ_CUTOFF_HZ: float = 70.0

#: Hop between consecutive STFT windows, in samples.
STFT_HOP: int = STFT_NPERSEG - STFT_NOVERLAP

#: Width of one frequency bin in Hz (``MODEL_FS / STFT_NPERSEG``) = 0.9766 Hz.
FREQ_BIN_WIDTH_HZ: float = MODEL_FS / STFT_NPERSEG

#: Number of frequency bins kept, i.e. bins whose centre frequency is <= FREQ_CUTOFF_HZ.
#: Bin ``i`` is centred at ``i * FREQ_BIN_WIDTH_HZ``, so the last kept bin is 69.34 Hz.
NUM_FREQ_BINS: int = int(FREQ_CUTOFF_HZ / FREQ_BIN_WIDTH_HZ) + 1

#: Number of samples in a chunk once resampled to MODEL_FS.
CHUNK_SAMPLES: int = int(CHUNK_DURATION_SEC * MODEL_FS)


def stft_time_frames(num_samples: int) -> int:
    """
    Number of STFT time frames scipy produces for a signal of the given length.

    Mirrors :func:`scipy.signal.stft` with its defaults (``boundary="zeros"``, ``padded=True``):
    the signal is extended by ``nperseg // 2`` on each side, then zero-padded until a whole
    number of hops fits.

    :param int num_samples: Length of the input signal in samples.
    :return: Number of time frames (columns) in the resulting spectrogram.
    :rtype: int
    """
    padded = num_samples + 2 * (STFT_NPERSEG // 2)
    remainder = (padded - STFT_NPERSEG) % STFT_HOP
    if remainder:
        padded += STFT_HOP - remainder
    return 1 + (padded - STFT_NPERSEG) // STFT_HOP


#: Number of time frames in a spectrogram of one canonical chunk.
NUM_TIME_FRAMES: int = stft_time_frames(CHUNK_SAMPLES)

#: Spectrogram shape (frequency bins, time frames) every spectrogram model is built for.
EXPECTED_SPECTROGRAM_SHAPE: tuple[int, int] = (NUM_FREQ_BINS, NUM_TIME_FRAMES)


def resample_to_length(signal: np.ndarray, target_samples: int, axis: int = -1) -> np.ndarray:
    """
    Resample a signal along ``axis`` to an exact number of samples.

    Uses FFT-based :func:`scipy.signal.resample`. Safe against aliasing for EEG here because the
    signal has already been band-passed to 1-70 Hz, well below the 125 Hz Nyquist of
    :data:`MODEL_FS`. Returns the input untouched when it already has the target length.

    :param np.ndarray signal: Input signal.
    :param int target_samples: Desired number of samples along ``axis``.
    :param int axis: Axis to resample along (default: last).
    :return: Signal with ``target_samples`` samples along ``axis``.
    :rtype: np.ndarray
    """
    if signal.shape[axis] == target_samples:
        return signal

    return np.asarray(resample(signal, target_samples, axis=axis))


def resample_to_model_fs(signal: np.ndarray, source_fs: float) -> np.ndarray:
    """
    Resample a 1-D signal from its native rate to :data:`MODEL_FS`.

    Returns the input untouched when it is already at :data:`MODEL_FS`.

    :param np.ndarray signal: 1-D EEG signal at ``source_fs``.
    :param float source_fs: Native sampling rate of ``signal`` in Hz.
    :return: Signal resampled to :data:`MODEL_FS`.
    :rtype: np.ndarray
    """
    if round(source_fs) == MODEL_FS:
        return signal

    return resample_to_length(signal, int(round(len(signal) * MODEL_FS / source_fs)))


def compute_log_spectrogram(signal: np.ndarray, source_fs: float) -> np.ndarray:
    """
    Convert a 1-D EEG signal to a cropped log-magnitude spectrogram.

    Pipeline: resample to :data:`MODEL_FS` → STFT with the canonical parameters →
    ``log1p(abs(...))`` → drop bins above :data:`FREQ_CUTOFF_HZ`.

    :param np.ndarray signal: 1-D EEG signal at ``source_fs``.
    :param float source_fs: Native sampling rate of ``signal`` in Hz.
    :return: Log-magnitude spectrogram of shape ``(NUM_FREQ_BINS, time_frames)``.
    :rtype: np.ndarray
    """
    resampled = resample_to_model_fs(signal, source_fs)

    freqs, _, Zxx = stft(
        resampled,
        fs=MODEL_FS,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        window=STFT_WINDOW,
    )

    return np.asarray(np.log1p(np.abs(Zxx))[freqs <= FREQ_CUTOFF_HZ])
