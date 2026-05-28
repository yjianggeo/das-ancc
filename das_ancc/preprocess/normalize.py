import numpy as np
from scipy.ndimage import uniform_filter1d


def rms_normalize(data: np.ndarray, win_len: int = 128) -> np.ndarray:
    """Running RMS (root-mean-square) normalization to suppress spikes."""
    data_squared = data ** 2
    rms_squared = uniform_filter1d(data_squared, size=win_len, mode="reflect")
    rms = np.sqrt(np.where(rms_squared > 0, rms_squared, 1.0))
    return data / rms


def onebit_normalize(data: np.ndarray) -> np.ndarray:
    return np.sign(data).astype(data.dtype)
