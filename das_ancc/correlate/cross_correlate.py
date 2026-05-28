import numpy as np


def fft_correlate(a: np.ndarray, b: np.ndarray,
                  max_lag_samples: int) -> np.ndarray:
    """
    FFT cross-correlation of a and b.
    Returns symmetric NCF with shape (2*max_lag_samples + 1,).
    Index max_lag_samples corresponds to zero lag.
    Positive lags: b leads a.
    """
    n = len(a)
    n_fft = 2 * n
    fa = np.fft.rfft(a, n=n_fft)
    fb = np.fft.rfft(b, n=n_fft)
    corr = np.fft.irfft(fb * np.conj(fa), n=n_fft)
    # Rearrange: negative lags live at indices [n_fft - max_lag : n_fft]
    neg = corr[n_fft - max_lag_samples:]
    pos = corr[:max_lag_samples + 1]
    return np.concatenate([neg, pos])
