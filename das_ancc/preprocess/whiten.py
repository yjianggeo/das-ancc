import numpy as np
from scipy.ndimage import uniform_filter1d


def spectral_whiten(data: np.ndarray, fs: float,
                    freq_range: tuple[float, float],
                    smooth_len: int = 1) -> np.ndarray:
    """
    Divide the spectrum by its smoothed amplitude within freq_range,
    zero amplitude outside freq_range.
    """
    n = len(data)
    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)

    amp = np.abs(fft)
    smoothed = uniform_filter1d(amp, size=smooth_len, mode="reflect")
    smoothed = np.where(smoothed > 0, smoothed, 1.0)

    fmin, fmax = freq_range
    mask = (freqs >= fmin) & (freqs <= fmax)
    whitened_fft = np.where(mask, fft / smoothed, 0.0)
    return np.fft.irfft(whitened_fft, n=n)
