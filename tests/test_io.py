import numpy as np
import pandas as pd
import pytest
from obspy import read, Stream
from das_ancc.io.das_metadata import load_geometry, REQUIRED_COLUMNS
from das_ancc.io.sac_reader import read_sac_segments
from das_ancc.io.writer import (
    save_ncf, load_ncfs,
    save_stacked_ncf, load_stacked_ncf,
    save_dispersion, load_dispersion,
)


def test_load_geometry(tmp_path, synthetic_geometry):
    csv_path = tmp_path / "geometry.csv"
    synthetic_geometry.to_csv(csv_path, index=False)
    geom = load_geometry(str(csv_path))
    assert set(REQUIRED_COLUMNS).issubset(geom.columns)
    assert len(geom) == 5
    assert geom.loc[2, "x_m"] == pytest.approx(20.0)


def test_load_geometry_missing_column(tmp_path):
    bad = pd.DataFrame({"channel": [0], "x_m": [0.0]})
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        load_geometry(str(path))


def test_read_sac_segments(tmp_path, synthetic_stream):
    # Write synthetic stream as SAC files
    for tr in synthetic_stream:
        fname = tmp_path / f"{tr.stats.station}.sac"
        tr.write(str(fname), format="SAC")

    segments = read_sac_segments(str(tmp_path), segment_length=1800)
    assert len(segments) == 2          # 3600s / 1800s = 2 segments
    seg_id, st = segments[0]
    assert isinstance(seg_id, str)
    assert len(st) == 5               # N_CH channels
    duration = st[0].stats.endtime - st[0].stats.starttime
    assert abs(duration - 1800) < 2   # within 2 seconds tolerance


def test_ncf_roundtrip(tmp_path):
    ncf = np.random.randn(201).astype(np.float32)
    save_ncf(str(tmp_path), seg_id="2024-01-01T00:00:00",
             ch_i=0, ch_j=1, component="ZZ", ncf=ncf)
    loaded = load_ncfs(str(tmp_path), ch_i=0, ch_j=1, component="ZZ")
    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0], ncf)


def test_stacked_ncf_roundtrip(tmp_path):
    ncf = np.random.randn(201).astype(np.float32)
    save_stacked_ncf(str(tmp_path), ch_i=0, ch_j=1, component="ZZ", ncf=ncf)
    loaded = load_stacked_ncf(str(tmp_path), ch_i=0, ch_j=1, component="ZZ")
    np.testing.assert_allclose(loaded, ncf)


def test_dispersion_roundtrip(tmp_path):
    picks = np.array([[0.5, 400.0], [1.0, 350.0], [2.0, 300.0]])
    save_dispersion(str(tmp_path), ch_i=0, ch_j=1, component="ZZ", picks=picks)
    loaded = load_dispersion(str(tmp_path), ch_i=0, ch_j=1, component="ZZ")
    np.testing.assert_allclose(loaded, picks)
