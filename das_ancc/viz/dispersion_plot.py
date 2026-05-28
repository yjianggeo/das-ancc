import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_dispersion_image(image: np.ndarray,
                           periods: np.ndarray,
                           velocities: np.ndarray,
                           picks: np.ndarray | None = None,
                           title: str = "",
                           output_path: str | None = None) -> plt.Figure:
    """
    FTAN dispersion energy image with optional picks overlay.
    image shape: (n_periods, n_velocities)
    picks shape: (n_periods, 2) columns [period, velocity]
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(periods, velocities, image.T,
                        shading="auto", cmap="hot_r")
    plt.colorbar(im, ax=ax, label="Amplitude")
    if picks is not None:
        ax.plot(picks[:, 0], picks[:, 1], "b-o", ms=3, label="picks")
        ax.legend()
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Phase Velocity (m/s)")
    ax.set_title(title or "Dispersion Image")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
