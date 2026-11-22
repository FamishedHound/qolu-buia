"""UKCEH Land Cover Map 2015 aggregate-class taxonomy.

The paper's environment is built on the UKCEH LCM2015 *1 km percentage cover,
aggregate class* product.  That product has ten bands, one per aggregate class,
each holding the percentage of the 1 km cell covered by that class (Table 1 of
the LCM2015 dataset documentation, Rowland et al. 2017).

Band order in the raster is the aggregate class number, so band ``i`` (0-based)
holds aggregate class ``i + 1``.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, IntEnum

__all__ = [
    "CLASS_LABELS",
    "GROUP_CLASSES",
    "N_AGGREGATE_CLASSES",
    "AggregateClass",
    "LandCoverGroup",
    "band_index",
    "class_of_band",
    "group_indices",
]


class AggregateClass(IntEnum):
    """The ten LCM2015 aggregate classes (values are the published codes)."""

    BROADLEAF_WOODLAND = 1
    CONIFEROUS_WOODLAND = 2
    ARABLE = 3
    IMPROVED_GRASSLAND = 4
    SEMI_NATURAL_GRASSLAND = 5
    MOUNTAIN_HEATH_BOG = 6
    SALTWATER = 7
    FRESHWATER = 8
    COASTAL = 9
    BUILT_UP = 10


N_AGGREGATE_CLASSES: int = len(AggregateClass)

CLASS_LABELS: Mapping[AggregateClass, str] = {
    AggregateClass.BROADLEAF_WOODLAND: "Broadleaf woodland",
    AggregateClass.CONIFEROUS_WOODLAND: "Coniferous woodland",
    AggregateClass.ARABLE: "Arable",
    AggregateClass.IMPROVED_GRASSLAND: "Improved grassland",
    AggregateClass.SEMI_NATURAL_GRASSLAND: "Semi-natural grassland",
    AggregateClass.MOUNTAIN_HEATH_BOG: "Mountain, heath, bog",
    AggregateClass.SALTWATER: "Saltwater",
    AggregateClass.FRESHWATER: "Freshwater",
    AggregateClass.COASTAL: "Coastal",
    AggregateClass.BUILT_UP: "Built-up areas and gardens",
}


class LandCoverGroup(Enum):
    """Semantic groupings of aggregate classes used by the paper's objectives.

    ``ARABLE`` and ``AGRICULTURAL`` both exist on purpose: the paper's
    experimental settings talk about "woodland and arable land", while the
    methodology section talks about "woodlands and agricultural land".  The
    reward configuration therefore has to be able to express either reading.
    """

    WOODLAND = "woodland"
    ARABLE = "arable"
    AGRICULTURAL = "agricultural"
    SEMI_NATURAL_GRASSLAND = "semi_natural_grassland"
    MOUNTAIN_HEATH_BOG = "mountain_heath_bog"
    SALTWATER = "saltwater"
    FRESHWATER = "freshwater"
    COASTAL = "coastal"
    URBAN = "urban"


GROUP_CLASSES: Mapping[LandCoverGroup, tuple[AggregateClass, ...]] = {
    LandCoverGroup.WOODLAND: (
        AggregateClass.BROADLEAF_WOODLAND,
        AggregateClass.CONIFEROUS_WOODLAND,
    ),
    LandCoverGroup.ARABLE: (AggregateClass.ARABLE,),
    LandCoverGroup.AGRICULTURAL: (
        AggregateClass.ARABLE,
        AggregateClass.IMPROVED_GRASSLAND,
    ),
    LandCoverGroup.SEMI_NATURAL_GRASSLAND: (AggregateClass.SEMI_NATURAL_GRASSLAND,),
    LandCoverGroup.MOUNTAIN_HEATH_BOG: (AggregateClass.MOUNTAIN_HEATH_BOG,),
    LandCoverGroup.SALTWATER: (AggregateClass.SALTWATER,),
    LandCoverGroup.FRESHWATER: (AggregateClass.FRESHWATER,),
    LandCoverGroup.COASTAL: (AggregateClass.COASTAL,),
    LandCoverGroup.URBAN: (AggregateClass.BUILT_UP,),
}


def band_index(land_class: AggregateClass) -> int:
    """Return the 0-based raster band index of an aggregate class."""
    return int(land_class) - 1


def class_of_band(index: int) -> AggregateClass:
    """Inverse of :func:`band_index`."""
    if not 0 <= index < N_AGGREGATE_CLASSES:
        raise ValueError(f"band index out of range: {index}")
    return AggregateClass(index + 1)


def group_indices(group: LandCoverGroup) -> tuple[int, ...]:
    """Return the 0-based band indices belonging to ``group``."""
    try:
        classes = GROUP_CLASSES[group]
    except KeyError as error:  # pragma: no cover - defensive
        raise KeyError(f"unknown land cover group: {group!r}") from error
    return tuple(band_index(land_class) for land_class in classes)
