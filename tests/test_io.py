import pandas as pd
import pytest
from das_ancc.io.das_metadata import load_geometry, REQUIRED_COLUMNS


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
