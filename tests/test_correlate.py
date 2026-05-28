import numpy as np
import pytest
from das_ancc.correlate.cross_correlate import fft_correlate
from das_ancc.correlate.component_projection import (
    inter_station_azimuth, projection_weights
)


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


def test_inter_station_azimuth_east():
    az = inter_station_azimuth(0, 0, 100, 0)  # due East
    assert az == pytest.approx(90.0, abs=0.1)


def test_inter_station_azimuth_north():
    az = inter_station_azimuth(0, 0, 0, 100)  # due North
    assert az == pytest.approx(0.0, abs=0.1)


def test_projection_weights_parallel_fiber():
    """
    Fiber aligned with ray (both East = 90 deg): full ZZ, zero TT.
    """
    w_zz, w_tt = projection_weights(
        azimuth_i_deg=90.0,
        azimuth_j_deg=90.0,
        ray_azimuth_deg=90.0,
    )
    assert w_zz == pytest.approx(1.0, abs=1e-9)
    assert w_tt == pytest.approx(0.0, abs=1e-9)


def test_projection_weights_perpendicular_fiber():
    """
    Fiber perpendicular to ray: zero ZZ, full TT.
    """
    w_zz, w_tt = projection_weights(
        azimuth_i_deg=180.0,   # North fiber
        azimuth_j_deg=180.0,
        ray_azimuth_deg=90.0,  # East ray
    )
    assert w_zz == pytest.approx(0.0, abs=1e-9)
    assert w_tt == pytest.approx(1.0, abs=1e-9)
