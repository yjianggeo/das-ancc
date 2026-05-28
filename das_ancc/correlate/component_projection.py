import numpy as np


def inter_station_azimuth(x1: float, y1: float,
                           x2: float, y2: float) -> float:
    """
    Azimuth from (x1,y1) to (x2,y2), degrees clockwise from North.
    x = Easting, y = Northing.
    """
    dx = x2 - x1
    dy = y2 - y1
    return float(np.degrees(np.arctan2(dx, dy)) % 360)


def projection_weights(azimuth_i_deg: float, azimuth_j_deg: float,
                        ray_azimuth_deg: float) -> tuple[float, float]:
    """
    ZZ (Rayleigh) and TT (Love) projection weights for a DAS channel pair.

    phi = angle between fiber and ray propagation direction.
    w_ZZ = cos(phi_i) * cos(phi_j)   -- maximised when fiber parallel to ray
    w_TT = sin(phi_i) * sin(phi_j)   -- maximised when fiber perpendicular to ray
    """
    phi_i = np.radians(azimuth_i_deg - ray_azimuth_deg)
    phi_j = np.radians(azimuth_j_deg - ray_azimuth_deg)
    return float(np.cos(phi_i) * np.cos(phi_j)), float(np.sin(phi_i) * np.sin(phi_j))
