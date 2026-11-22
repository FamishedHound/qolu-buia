"""Top-down stakeholder objectives used by the QOLU agents.

The paper evaluates three stakeholder preferences (Section 4, "Stakeholders
Preferences and Scenarios"):

    (i)   maximise the distance of developed parcels to a set of protected land
          uses (bog, mountain, heath) aggregated into a single land type;
    (ii)  minimise the loss of woodland and arable land caused by development;
    (iii) minimise the distance of newly developed parcels to existing urban or
          suburban land.

The result tables label these three columns "Land Use", "Woodland" and "Urban",
which is the naming adopted here.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["TopDownObjective"]


class TopDownObjective(Enum):
    """The three top-down (QOLU) stakeholder objectives of the paper."""

    LAND_USE_SEPARATION = "land_use"
    WOODLAND_PRESERVATION = "woodland"
    URBAN_PROXIMITY = "urban"

    @property
    def column_name(self) -> str:
        """Name used for this objective in the paper's result tables."""
        return str(self.value)
