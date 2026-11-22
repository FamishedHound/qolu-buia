"""Small synthetic landscapes shared by the baseline test modules.

Not collected by pytest (the file name does not match ``test_*``); it exists so
that the four baseline test modules build identical environments without
importing each other.
"""

from __future__ import annotations

import numpy as np

from chapter4_qolu.config import SCENARIOS, EnvConfig
from chapter4_qolu.env.parcel_grid import ParcelGrid
from chapter4_qolu.env.urban_env import UrbanLandUseEnv
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES, AggregateClass, band_index

__all__ = ["make_env", "make_grid"]


def make_grid(size: int = 12) -> ParcelGrid:
    """Grassland with one urban corner and a block of mountain / heath / bog."""
    composition = np.zeros((N_AGGREGATE_CLASSES, size, size), dtype=np.float32)
    composition[band_index(AggregateClass.SEMI_NATURAL_GRASSLAND)] = 1.0
    composition[:, 0, 0] = 0.0
    composition[band_index(AggregateClass.BUILT_UP), 0, 0] = 1.0
    composition[:, size - 4 :, :3] = 0.0
    composition[band_index(AggregateClass.MOUNTAIN_HEATH_BOG), size - 4 :, :3] = 1.0
    composition[:, :2, size - 2 :] = 0.0
    composition[band_index(AggregateClass.BROADLEAF_WOODLAND), :2, size - 2 :] = 1.0
    return ParcelGrid(composition)


def make_env(scenario: int = 1, **overrides: object) -> UrbanLandUseEnv:
    """A 12 x 12 environment small enough for fast unit tests."""
    settings: dict[str, object] = {
        "grid_size": 12,
        "samples_per_iteration": 8,
        "max_conversions_per_iteration": 4,
        "iterations": 6,
        "observation_window": 5,
        "buia_observation_window": 5,
    }
    settings.update(overrides)
    size = int(settings["grid_size"])  # type: ignore[arg-type]
    return UrbanLandUseEnv(
        make_grid(size),
        config=EnvConfig(**settings),  # type: ignore[arg-type]
        weights=SCENARIOS[scenario],
    )
