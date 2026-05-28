import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from das_ancc.preprocess.normalize import rms_normalize, onebit_normalize
from das_ancc.preprocess.whiten import spectral_whiten
from das_ancc.preprocess.pipeline import preprocess_stream


def test_rms_normalize_suppresses_spike():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(10_000).astype(np.float64)
    data[5000] = 1000.0   # spike
    out = rms_normalize(data, win_len=100)
    assert np.max(np.abs(out)) < 10.0


def test_onebit_values():
    data = np.array([-3.0, -1.0, 0.5, 2.0])
    out = onebit_normalize(data)
    np.testing.assert_array_equal(out, [-1.0, -1.0, 1.0, 1.0])


def test_spectral_whiten_flat_spectrum():
    rng = np.random.default_rng(1)
    fs = 100.0
    data = rng.standard_normal(4096).astype(np.float64)
    out = spectral_whiten(data, fs, freq_range=(0.5, 10.0))
    freqs = np.fft.rfftfreq(len(out), 1.0 / fs)
    amp = np.abs(np.fft.rfft(out))
    in_band = amp[(freqs >= 0.5) & (freqs <= 10.0)]
    # coefficient of variation < 0.3 means roughly flat
    assert in_band.std() / in_band.mean() < 0.3


def test_preprocess_stream_preserves_channel_count(synthetic_stream):
    out = preprocess_stream(synthetic_stream,
                            bandpass=(0.1, 5.0),
                            normalization='rms',
                            whiten=True)
    assert len(out) == len(synthetic_stream)
    assert out[0].stats.sampling_rate == synthetic_stream[0].stats.sampling_rate
