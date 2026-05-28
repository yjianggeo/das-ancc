import numpy as np
import pandas as pd
import pytest
from obspy import Stream, Trace, UTCDateTime

FS = 100.0        # Hz
N_CH = 5
DURATION = 3600   # seconds


@pytest.fixture
def synthetic_stream():
    """White-noise DAS Stream, N_CH channels, DURATION seconds."""
    rng = np.random.default_rng(42)
    traces = []
    t0 = UTCDateTime("2024-01-01T00:00:00")
    for ch in range(N_CH):
        data = rng.standard_normal(int(DURATION * FS)).astype(np.float32)
        tr = Trace(data=data)
        tr.stats.sampling_rate = FS
        tr.stats.starttime = t0
        tr.stats.network = "DAS"
        tr.stats.station = f"CH{ch:03d}"
        tr.stats.channel = "HHZ"
        traces.append(tr)
    return Stream(traces)


@pytest.fixture
def synthetic_geometry():
    """Straight E-W cable, 10 m channel spacing."""
    channels = list(range(N_CH))
    return pd.DataFrame({
        "channel":       channels,
        "x_m":           [ch * 10.0 for ch in channels],
        "y_m":           [0.0] * N_CH,
        "azimuth_deg":   [90.0] * N_CH,   # East-West fiber
        "gauge_length_m":[10.0] * N_CH,
    })


@pytest.fixture
def plane_wave_stream():
    """
    Two-channel stream with a known inter-channel delay.
    Channel 0 leads channel 1 by 20 samples (0.2 s at 100 Hz).
    Dominant frequency: 1 Hz.  Distance: 100 m.  Expected velocity: 500 m/s.
    """
    fs = 100.0
    n = 10_000
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * 1.0 * t).astype(np.float32)
    t0 = UTCDateTime("2024-01-01")
    traces = []
    for delay_samples in [0, 20]:
        data = np.roll(signal, delay_samples)
        tr = Trace(data=data)
        tr.stats.sampling_rate = fs
        tr.stats.starttime = t0
        tr.stats.network = "DAS"
        tr.stats.station = f"CH{delay_samples:03d}"
        traces.append(tr)
    return Stream(traces)
