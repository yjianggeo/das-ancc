import numpy as np
from scipy.signal import savgol_filter


def auto_pick(image: np.ndarray, periods: np.ndarray,
              velocities: np.ndarray,
              smooth_window: int = 5) -> np.ndarray:
    """
    Pick phase-velocity dispersion curve from FTAN energy image.
    Returns array of shape (n_periods, 2): columns [period, velocity].
    """
    picks = np.zeros((len(periods), 2))
    picks[:, 0] = periods
    for i in range(len(periods)):
        picks[i, 1] = velocities[np.argmax(image[i])]

    if len(periods) >= smooth_window and smooth_window > 2:
        picks[:, 1] = savgol_filter(picks[:, 1],
                                     window_length=smooth_window,
                                     polyorder=2)
    return picks
