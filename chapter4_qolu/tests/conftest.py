"""Shared pytest fixtures for the Chapter 4 QOLU test-suite.

The only place in the code base that is allowed to know where the LCM fixture
raster comes from is this fixture module (plus user-supplied configuration).
Resolution order for the LCM fixture raster:

1. ``CHAPTER4_LCM_TIF`` environment variable (explicit override);
2. ``chapter4_qolu/data_fixtures/*.tif`` (user drops a raster in the repo);
3. otherwise the depending tests are skipped with an explanatory message.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = _REPO_ROOT / "chapter4_qolu" / "data_fixtures"


@pytest.fixture(scope="session")
def lcm_tif_path() -> Path:
    """Return a path to an LCM percentage-aggregate-class GeoTIFF, or skip."""
    override = os.environ.get("CHAPTER4_LCM_TIF")
    if override:
        path = Path(override)
        if not path.is_file():
            pytest.skip(f"CHAPTER4_LCM_TIF points at a missing file: {path}")
        return path

    if _FIXTURE_DIR.is_dir():
        candidates = sorted(_FIXTURE_DIR.glob("*.tif"))
        if candidates:
            return candidates[0]

    pytest.skip(
        "No LCM raster available. Set CHAPTER4_LCM_TIF, or place a "
        "*.tif in chapter4_qolu/data_fixtures/."
    )


@pytest.fixture()
def rng() -> np.random.Generator:
    """Deterministic random generator for tests."""
    return np.random.default_rng(20250101)
