import numpy as np
import pytest
from das_ancc.correlate.cross_correlate import fft_correlate


def test_fft_correlate_known_delay():
    """NCF peak should appear at lag = delay_samples."""
    fs = 100.0
    n = 10_000
    delay_samples = 20          # 0.2 s at 100 Hz
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * 1.0 * t)
    a = signal
    b = np.roll(signal, delay_samples)

    max_lag = 50
    ncf = fft_correlate(a, b, max_lag_samples=max_lag)

    assert len(ncf) == 2 * max_lag + 1
    peak_lag = np.argmax(ncf) - max_lag   # shift to signed lag
    assert peak_lag == pytest.approx(delay_samples, abs=1)


def test_fft_correlate_zero_lag_for_identical():
    data = np.random.default_rng(7).standard_normal(1000)
    ncf = fft_correlate(data, data, max_lag_samples=100)
    peak_lag = np.argmax(ncf) - 100
    assert peak_lag == 0
