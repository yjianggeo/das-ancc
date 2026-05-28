import numpy as np
from scipy.signal import hilbert


def linear_stack(ncfs: np.ndarray) -> np.ndarray:
    """Mean stack. ncfs shape: (n_segments, n_lags)."""
    return ncfs.mean(axis=0)


def phase_weighted_stack(ncfs: np.ndarray, power: int = 2) -> np.ndarray:
    """Phase-weighted stack (Schimmel & Paulssen 1997)."""
    analytic = hilbert(ncfs, axis=1)
    unit_phase = analytic / (np.abs(analytic) + 1e-12)
    phase_weight = np.abs(unit_phase.mean(axis=0)) ** power
    return ncfs.mean(axis=0) * phase_weight


def stack_ncfs(ncfs: list[np.ndarray], method: str = "linear",
               min_segments: int = 1) -> np.ndarray | None:
    if len(ncfs) < min_segments:
        return None
    arr = np.stack(ncfs, axis=0)
    if method == "linear":
        return linear_stack(arr)
    if method == "pws":
        return phase_weighted_stack(arr)
    raise ValueError(f"Unknown stacking method: {method!r}")
