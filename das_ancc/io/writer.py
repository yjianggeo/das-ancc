from pathlib import Path
import h5py
import numpy as np


def _open(output_dir: str, filename: str, mode: str = "a") -> h5py.File:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return h5py.File(Path(output_dir) / filename, mode)


def save_ncf(output_dir: str, seg_id: str, ch_i: int, ch_j: int,
             component: str, ncf: np.ndarray) -> None:
    key = f"ncf/{seg_id}/{ch_i}_{ch_j}_{component}"
    with _open(output_dir, "ncf.h5") as f:
        if key in f:
            del f[key]
        f.create_dataset(key, data=ncf)


def load_ncfs(output_dir: str, ch_i: int, ch_j: int,
              component: str) -> list[np.ndarray]:
    path = Path(output_dir) / "ncf.h5"
    if not path.exists():
        return []
    results = []
    with h5py.File(path, "r") as f:
        for seg_id in f.get("ncf", {}).keys():
            key = f"ncf/{seg_id}/{ch_i}_{ch_j}_{component}"
            if key in f:
                results.append(f[key][:])
    return results


def save_stacked_ncf(output_dir: str, ch_i: int, ch_j: int,
                     component: str, ncf: np.ndarray) -> None:
    key = f"stacked/{ch_i}_{ch_j}_{component}"
    with _open(output_dir, "stacked.h5") as f:
        if key in f:
            del f[key]
        f.create_dataset(key, data=ncf)


def load_stacked_ncf(output_dir: str, ch_i: int, ch_j: int,
                     component: str) -> np.ndarray | None:
    path = Path(output_dir) / "stacked.h5"
    if not path.exists():
        return None
    key = f"stacked/{ch_i}_{ch_j}_{component}"
    with h5py.File(path, "r") as f:
        return f[key][:] if key in f else None


def save_dispersion(output_dir: str, ch_i: int, ch_j: int,
                    component: str, picks: np.ndarray) -> None:
    key = f"dispersion/{component}/{ch_i}_{ch_j}"
    with _open(output_dir, "dispersion.h5") as f:
        if key in f:
            del f[key]
        f.create_dataset(key, data=picks)


def load_dispersion(output_dir: str, ch_i: int, ch_j: int,
                    component: str) -> np.ndarray | None:
    path = Path(output_dir) / "dispersion.h5"
    if not path.exists():
        return None
    key = f"dispersion/{component}/{ch_i}_{ch_j}"
    with h5py.File(path, "r") as f:
        return f[key][:] if key in f else None


def save_velocity_map(output_dir: str, period: float, component: str,
                      velocity_map: np.ndarray) -> None:
    key = f"imaging/{component}/{period:.3f}"
    with _open(output_dir, "imaging.h5") as f:
        if key in f:
            del f[key]
        f.create_dataset(key, data=velocity_map)
