import numpy as np


def ftan_phase_velocity(ncf: np.ndarray, dt: float, dist: float,
                         periods: np.ndarray, velocities: np.ndarray,
                         alpha: float = 25.0) -> np.ndarray:
    """
    Phase-velocity FTAN energy image.

    ncf: symmetric NCF, center sample (index n//2) = zero lag.
    dt:  sampling interval (s).
    dist: inter-station distance (m).
    periods: 1-D array of target periods (s).
    velocities: 1-D array of trial phase velocities (m/s).
    alpha: Gaussian filter width parameter.

    Returns: image array of shape (len(periods), len(velocities)).
    """
    n = len(ncf)
    freqs = np.fft.rfftfreq(n, dt)
    ncf_fft = np.fft.rfft(ncf)

    image = np.zeros((len(periods), len(velocities)))

    for i, T in enumerate(periods):
        f0 = 1.0 / T
        gauss = np.exp(-alpha * ((freqs - f0) / f0) ** 2)
        filtered = np.fft.irfft(ncf_fft * gauss, n=n)
        # Positive-lag envelope
        pos = np.abs(filtered[n // 2:])
        pos_times = np.arange(len(pos)) * dt

        for j, v in enumerate(velocities):
            t_arr = dist / v
            idx = int(round(t_arr / dt))
            if 0 <= idx < len(pos):
                image[i, j] = pos[idx]

    return image
