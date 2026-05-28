import numpy as np


def gaussian_tomography(measurements: list[dict],
                         x_grid: np.ndarray,
                         y_grid: np.ndarray,
                         sigma: float,
                         n_samples: int = 50) -> np.ndarray:
    """
    Gaussian-kernel weighted-average straight-ray tomography.

    Each measurement is a dict with keys: x1, y1, x2, y2, velocity.
    Points are sampled along each ray; velocity is smeared onto the grid
    with a Gaussian kernel of width sigma (m).

    Returns velocity map of shape (len(y_grid), len(x_grid)).
    NaN where no rays contribute.
    """
    XX, YY = np.meshgrid(x_grid, y_grid)
    vel_sum = np.zeros_like(XX)
    wgt_sum = np.zeros_like(XX)

    for m in measurements:
        xs = np.linspace(m["x1"], m["x2"], n_samples)
        ys = np.linspace(m["y1"], m["y2"], n_samples)
        v = m["velocity"]
        for xr, yr in zip(xs, ys):
            w = np.exp(-((XX - xr) ** 2 + (YY - yr) ** 2) / (2 * sigma ** 2))
            vel_sum += w * v
            wgt_sum += w

    result = np.full_like(XX, np.nan)
    mask = wgt_sum > 0
    result[mask] = vel_sum[mask] / wgt_sum[mask]
    return result
