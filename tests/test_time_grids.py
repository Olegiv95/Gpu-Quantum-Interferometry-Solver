import numpy as np

from Example_03_two_level_animation import build_time_grid as build_two_level_grid
from Example_04_four_level_animation import build_time_grid as build_four_level_grid


def test_two_level_grid_has_one_more_sample_than_steps():
    periods = 3
    steps_per_period = 20
    grid = build_two_level_grid(1.25, periods, steps_per_period)
    assert len(grid) == periods * steps_per_period + 1
    assert grid[0] == 0.0


def test_four_level_grid_has_one_more_sample_than_steps():
    periods = 4
    steps_per_period = 25
    _w, _period, grid = build_four_level_grid(1.0, 500.0, periods, steps_per_period)
    assert len(grid) == periods * steps_per_period + 1
    assert np.isclose(grid[0], 0.0)
