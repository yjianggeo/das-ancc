"""Integration test: runs the full pipeline on synthetic SAC data."""
import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path
from obspy import Stream, Trace, UTCDateTime


def make_synthetic_sac_dir(tmp_path, n_channels=5, duration=7200, fs=100.0):
    sac_dir = tmp_path / "raw"
    sac_dir.mkdir()
    rng = np.random.default_rng(42)
    t0 = UTCDateTime("2024-01-01")
    for ch in range(n_channels):
        data = rng.standard_normal(int(duration * fs)).astype(np.float32)
        tr = Trace(data=data)
        tr.stats.sampling_rate = fs
        tr.stats.starttime = t0
        tr.stats.network = "DAS"
        tr.stats.station = f"CH{ch:03d}"
        tr.stats.channel = "HHZ"
        st = Stream([tr])
        st.write(str(sac_dir / f"CH{ch:03d}.sac"), format="SAC")
    return sac_dir


def make_geometry_csv(tmp_path, n_channels=5):
    geom_path = tmp_path / "geometry.csv"
    pd.DataFrame({
        "channel":        list(range(n_channels)),
        "x_m":            [ch * 50.0 for ch in range(n_channels)],
        "y_m":            [0.0] * n_channels,
        "azimuth_deg":    [90.0] * n_channels,
        "gauge_length_m": [10.0] * n_channels,
    }).to_csv(geom_path, index=False)
    return geom_path


def make_config(tmp_path, sac_dir, geom_path):
    cfg = {
        "data": {
            "sac_dir":       str(sac_dir),
            "output_dir":    str(tmp_path / "processed"),
            "geometry_file": str(geom_path),
            "components":    ["ZZ"],
        },
        "preprocess": {
            "bandpass":        [0.5, 5.0],
            "normalization":   "rms",
            "whiten":          True,
            "segment_length":  3600,
        },
        "correlate": {
            "max_lag":       20,
            "min_distance":   10,
            "max_distance":  1000,
        },
        "stack": {"method": "linear", "min_segments": 1},
        "dispersion": {"period_range": [0.5, 3.0], "velocity_range": [50, 800]},
        "imaging":    {"grid_spacing": 50, "regularization": 0.1},
        "parallel":   {"n_workers": 1},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return cfg_path


def test_full_pipeline(tmp_path):
    n_ch = 5
    sac_dir   = make_synthetic_sac_dir(tmp_path, n_channels=n_ch)
    geom_path = make_geometry_csv(tmp_path, n_channels=n_ch)
    cfg_path  = make_config(tmp_path, sac_dir, geom_path)

    from das_ancc.config import load_config
    from das_ancc.io.das_metadata import load_geometry
    from das_ancc.io.sac_reader import read_sac_segments
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.run_pipeline import (
        stage_preprocess, stage_correlate,
        stage_stack, stage_dispersion, stage_imaging,
    )

    cfg = load_config(str(cfg_path))
    geom = load_geometry(str(geom_path))
    output_dir = cfg.data["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    segments = read_sac_segments(cfg.data["sac_dir"], cfg.preprocess["segment_length"])
    assert len(segments) >= 1

    preprocessed = stage_preprocess(cfg, segments, geom)
    stage_correlate(cfg, preprocessed, geom, output_dir)
    stage_stack(cfg, geom, output_dir)
    stage_dispersion(cfg, geom, output_dir)
    stage_imaging(cfg, geom, output_dir)

    # Verify output files exist
    assert (Path(output_dir) / "ncf.h5").exists()
    assert (Path(output_dir) / "stacked.h5").exists()
    assert (Path(output_dir) / "dispersion.h5").exists()
    assert (Path(output_dir) / "imaging.h5").exists()
