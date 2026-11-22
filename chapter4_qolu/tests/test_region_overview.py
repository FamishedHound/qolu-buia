"""Figure 4.1: the region grid, the window's place in it, and the check against the run."""

from types import SimpleNamespace

import numpy as np
import pytest

from chapter4_qolu.figures.region_overview import (
    band_grid,
    region_grid,
    region_with_window,
    window_offset,
)
from chapter4_qolu.window_scan import BoundingBoxKm

BOX = BoundingBoxKm(east_min=100, east_max=110, north_min=200, north_max=208)


def _stack(codes: np.ndarray) -> np.ndarray:
    """One-hot (10, rows, cols) cover stack of band indices; -1 = sea."""
    stack = np.zeros((10,) + codes.shape, dtype=np.float32)
    for band in range(10):
        stack[band][codes == band] = 1.0
    return stack


def test_band_grid_marks_sea_and_keeps_classes():
    codes = np.array([[0, 2], [-1, 9]])
    assert band_grid(_stack(codes)).tolist() == [[0, 2], [-1, 9]]


def test_window_offset_counts_rows_from_the_north_edge():
    assert window_offset(BOX, (103, 206), size_km=2) == (2, 3)


def test_window_offset_rejects_a_window_outside_the_region():
    with pytest.raises(ValueError):
        window_offset(BOX, (109, 206), size_km=2)


def test_region_grid_cuts_the_box_north_up():
    codes = np.arange(12 * 14).reshape(12, 14) % 10
    raster = SimpleNamespace(
        fractions=_stack(codes),
        origin_xy=(98_000.0, 210_000.0),
        pixel_size_m=1000.0,
        height=12,
        width=14,
        path=SimpleNamespace(name="test.img"),
    )
    region = region_grid(raster, BOX)
    assert region.shape == (8, 10)
    assert np.array_equal(region, codes[2:10, 2:12])


def test_region_with_window_refuses_a_window_that_differs_from_the_run(tmp_path):
    region = np.zeros((8, 10), dtype=np.int64)
    wrong = np.ones((2, 2), dtype=np.int64)
    with pytest.raises(ValueError):
        region_with_window(region, wrong, wrong, tmp_path, "t", (103, 206), box=BOX, size_km=2)


def test_region_with_window_writes_the_class_shares(tmp_path):
    region = np.zeros((8, 10), dtype=np.int64)
    window = region[2:4, 3:5].copy()
    result = region_with_window(region, window, window, tmp_path, "t", (103, 206), box=BOX, size_km=2)
    csv = tmp_path / "region_overview_t.csv"
    assert csv in result.paths
    assert "region" in csv.read_text(encoding="utf-8")
