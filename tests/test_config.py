import yaml
import pytest
from das_ancc.config import load_config, Config

def test_load_config(tmp_path):
    data = {
        'data': {'sac_dir': 'data/raw', 'output_dir': 'data/processed',
                 'geometry_file': 'data/cable_geometry.csv', 'components': ['ZZ', 'TT']},
        'preprocess': {'bandpass': [0.1, 5.0], 'normalization': 'rms',
                       'whiten': True, 'segment_length': 3600},
        'correlate': {'max_lag': 60, 'min_distance': 10, 'max_distance': 2000},
        'stack': {'method': 'linear', 'min_segments': 10},
        'dispersion': {'period_range': [0.2, 5.0], 'velocity_range': [100, 1500]},
        'imaging': {'grid_spacing': 50, 'regularization': 0.1},
        'parallel': {'n_workers': 4},
    }
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml.dump(data))
    cfg = load_config(str(cfg_path))
    assert isinstance(cfg, Config)
    assert cfg.preprocess['bandpass'] == [0.1, 5.0]
    assert cfg.parallel['n_workers'] == 4
