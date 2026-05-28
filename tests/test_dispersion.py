import numpy as np
import pytest
from das_ancc.dispersion.ftan import ftan_phase_velocity
from das_ancc.dispersion.picking import auto_pick


def make_synthetic_ncf(dist_m: float, velocity: float,
                        period: float, fs: float = 100.0,
                        max_lag: float = 30.0) -> np.ndarray:
    """Sinusoidal NCF with known phase velocity."""
    n = int(2 * max_lag * fs) + 1
    lags = (np.arange(n) - n // 2) / fs
    f0 = 1.0 / period
    ncf = np.cos(2 * np.pi * f0 * (lags - dist_m / velocity))
    return ncf


def test_ftan_peak_at_correct_velocity():
    dist = 500.0      # m
    velocity = 400.0  # m/s
    period = 1.0      # s
    fs = 100.0
    ncf = make_synthetic_ncf(dist, velocity, period, fs)
    periods = np.array([period])
    velocities = np.linspace(200, 800, 100)
    image = ftan_phase_velocity(ncf, dt=1.0/fs, dist=dist,
                                 periods=periods, velocities=velocities)
    peak_velocity = velocities[np.argmax(image[0])]
    assert abs(peak_velocity - velocity) / velocity < 0.05  # <5% error


def test_auto_pick_shape():
    periods = np.linspace(0.5, 3.0, 20)
    velocities = np.linspace(200, 800, 60)
    image = np.random.default_rng(3).random((len(periods), len(velocities)))
    picks = auto_pick(image, periods, velocities)
    assert picks.shape == (len(periods), 2)
    assert np.all(picks[:, 0] == periods)
