import numpy as np
import pytest
from das_ancc.stack.stacking import linear_stack, phase_weighted_stack, stack_ncfs


def test_linear_stack_mean():
    rng = np.random.default_rng(0)
    ncfs = rng.standard_normal((10, 201))
    out = linear_stack(ncfs)
    np.testing.assert_allclose(out, ncfs.mean(axis=0))


def test_pws_coherent_signal_enhanced():
    """PWS should give higher SNR than linear stack on coherent+noisy data."""
    rng = np.random.default_rng(1)
    n_seg, n_lag = 50, 201
    signal = np.zeros(n_lag)
    signal[100] = 1.0   # impulse at zero lag
    noisy = np.tile(signal, (n_seg, 1)) + rng.standard_normal((n_seg, n_lag)) * 0.5
    lin = linear_stack(noisy)
    pws = phase_weighted_stack(noisy, power=2)
    assert pws[100] / np.std(pws) > lin[100] / np.std(lin)


def test_stack_ncfs_below_min_returns_none():
    ncfs = [np.zeros(201)] * 3
    result = stack_ncfs(ncfs, method="linear", min_segments=10)
    assert result is None


def test_stack_ncfs_linear():
    rng = np.random.default_rng(2)
    ncfs = [rng.standard_normal(201) for _ in range(15)]
    result = stack_ncfs(ncfs, method="linear", min_segments=10)
    assert result is not None
    assert result.shape == (201,)
