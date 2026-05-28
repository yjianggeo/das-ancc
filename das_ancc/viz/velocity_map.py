import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_velocity_map(velocity_map: np.ndarray,
                       x_grid: np.ndarray,
                       y_grid: np.ndarray,
                       period: float,
                       component: str,
                       output_path: str | None = None) -> plt.Figure:
    """2D lateral phase velocity map for one period."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.pcolormesh(x_grid, y_grid, velocity_map,
                        shading="auto", cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Phase Velocity (m/s)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"{component} Phase Velocity -- T={period:.1f} s")
    ax.set_aspect("equal")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
