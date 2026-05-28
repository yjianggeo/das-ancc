import pandas as pd

REQUIRED_COLUMNS = ["channel", "x_m", "y_m", "azimuth_deg", "gauge_length_m"]


def load_geometry(geometry_file: str) -> pd.DataFrame:
    df = pd.read_csv(geometry_file)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in geometry file: {missing}")
    return df.reset_index(drop=True)
