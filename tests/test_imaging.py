import numpy as np
import pytest
from das_ancc.imaging.tomography import gaussian_tomography


def test_single_ray_recovers_velocity():
    """A single ray through the grid centre should produce that velocity there."""
    measurements = [{"x1": 0.0, "y1": 0.0,
                     "x2": 1000.0, "y2": 0.0,
                     "velocity": 500.0}]
    x_grid = np.linspace(0, 1000, 21)
    y_grid = np.linspace(-100, 100, 5)
    sigma = 50.0
    vmap = gaussian_tomography(measurements, x_grid, y_grid, sigma)
    # Centre column (x=500, y=0) should be close to 500 m/s
    cx = len(x_grid) // 2
    cy = len(y_grid) // 2
    assert vmap[cy, cx] == pytest.approx(500.0, rel=0.01)


def test_two_rays_weighted_average():
    """Two identical rays with different velocities should give their mean."""
    measurements = [
        {"x1": 0.0, "y1": 0.0, "x2": 1000.0, "y2": 0.0, "velocity": 400.0},
        {"x1": 0.0, "y1": 0.0, "x2": 1000.0, "y2": 0.0, "velocity": 600.0},
    ]
    x_grid = np.linspace(0, 1000, 11)
    y_grid = np.array([0.0])
    sigma = 100.0
    vmap = gaussian_tomography(measurements, x_grid, y_grid, sigma)
    assert vmap[0, 5] == pytest.approx(500.0, rel=0.01)
