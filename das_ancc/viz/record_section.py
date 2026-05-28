import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_record_section(ncfs: dict[tuple, np.ndarray],
                         distances: dict[tuple, float],
                         dt: float,
                         max_lag: float,
                         component: str = "ZZ",
                         output_path: str | None = None) -> plt.Figure:
    """
    Plot NCF record section sorted by inter-station distance.
    ncfs:      {(ch_i, ch_j): ncf_array}
    distances: {(ch_i, ch_j): distance_m}
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    pairs = sorted(ncfs.keys(), key=lambda k: distances.get(k, 0))
    n = len(next(iter(ncfs.values())))
    lags = (np.arange(n) - n // 2) * dt

    for pair in pairs:
        ncf = ncfs[pair]
        dist_km = distances.get(pair, 0) / 1000.0
        norm = np.max(np.abs(ncf)) or 1.0
        ax.plot(lags, ncf / norm + dist_km, "k-", lw=0.5, alpha=0.7)

    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Distance (km)")
    ax.set_title(f"NCF Record Section -- {component}")
    ax.set_xlim([-max_lag, max_lag])
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
