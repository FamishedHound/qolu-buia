"""The paper's own eight land classes, and how the ten we have map onto them.

The published land-preference legend
(``thesis/overleaf/imgs/edit/colours_legend_pie.pdf``) names eight classes -
Arable Land, Grassland, Heather Grassland, Littoral Sediment, Freshwater,
Urban, Woodland, Suburban - which is a regrouping of the UKCEH LCM2015 **21
target classes**.  This package loads the **10 aggregate classes**, so seven
of the eight can be reproduced exactly and one cannot:

======================  ====================================================
paper class             aggregate classes it is built from
======================  ====================================================
Arable Land             Arable
Grassland               Improved grassland + semi-natural grassland
Heather Grassland       Mountain, heath, bog
Littoral Sediment       Coastal
Freshwater              Freshwater
Urban                   Built-up areas and gardens
Woodland                Broadleaf + coniferous woodland
**Suburban**            *not separable* - the aggregate product merges
                        suburban and urban into one "built-up areas and
                        gardens" class, which only the 21 target classes split
======================  ====================================================

Saltwater is excluded: it is sea, not land cover, and the land mask already
drops it.

Design note (Suburban): rather than invent a split, the class is kept in the
taxonomy with **no** aggregate class behind it, so every figure that uses this
mapping shows an empty Suburban slice and its legend says why.  A built-up /
suburban threshold would be a free parameter in a figure whose job is to
describe the *data*.

The colours are the published ones, read straight out of the legend PDF's
content stream (six RGB fills and two greys), so a regenerated figure is the
same colour as the published one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from chapter4_qolu.land_cover import AggregateClass, band_index

__all__ = [
    "NOT_SEPARABLE_NOTE",
    "PAPER_CLASSES",
    "PaperClass",
    "aggregate_to_paper",
    "paper_class_colours",
    "paper_class_names",
]


@dataclass(frozen=True)
class PaperClass:
    """One class of the published legend."""

    name: str
    colour: str
    #: The aggregate classes it is built from; empty when it is not separable.
    aggregates: tuple[AggregateClass, ...]

    @property
    def separable(self) -> bool:
        return bool(self.aggregates)

    @property
    def band_indices(self) -> tuple[int, ...]:
        return tuple(band_index(item) for item in self.aggregates)


#: The eight classes in the published legend's order, with its exact colours.
PAPER_CLASSES: tuple[PaperClass, ...] = (
    PaperClass("Arable Land", "#d29b38", (AggregateClass.ARABLE,)),
    PaperClass(
        "Grassland",
        "#9be44d",
        (AggregateClass.IMPROVED_GRASSLAND, AggregateClass.SEMI_NATURAL_GRASSLAND),
    ),
    PaperClass("Heather Grassland", "#7854c7", (AggregateClass.MOUNTAIN_HEATH_BOG,)),
    PaperClass("Littoral Sediment", "#4e8b57", (AggregateClass.COASTAL,)),
    PaperClass("Freshwater", "#3184e0", (AggregateClass.FRESHWATER,)),
    PaperClass("Urban", "#df6458", (AggregateClass.BUILT_UP,)),
    PaperClass(
        "Woodland",
        "#6d6d6d",
        (AggregateClass.BROADLEAF_WOODLAND, AggregateClass.CONIFEROUS_WOODLAND),
    ),
    PaperClass("Suburban", "#a0a0a0", ()),
)

NOT_SEPARABLE_NOTE = (
    "Suburban is empty: the LCM2015 *aggregate* product merges suburban and "
    "urban into one 'built-up areas and gardens' class, and only the 21 "
    "*target* classes - a product not available here - split them."
)


def paper_class_names() -> tuple[str, ...]:
    return tuple(item.name for item in PAPER_CLASSES)


def paper_class_colours() -> dict[str, str]:
    return {item.name: item.colour for item in PAPER_CLASSES}


def aggregate_to_paper() -> Mapping[int, str]:
    """``band index -> paper class name``; saltwater and no-data are absent."""
    mapping: dict[int, str] = {}
    for item in PAPER_CLASSES:
        for index in item.band_indices:
            mapping[index] = item.name
    return mapping
