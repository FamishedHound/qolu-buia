"""The LCM2015 aggregate-class colour table used by the land-cover maps.

The UKCEH *1 km dominant aggregate class* product ships a ``*.img.clr`` palette
next to the raster - one ``code R G B`` line per class, code 0 being sea /
outside the mapped area.  :func:`read_colour_table` reads it, and
:data:`AGGREGATE_CLASS_COLOURS` is the same table transcribed so that a figure
can be drawn on a machine that does not have the raster.

Note the two indexings.  The ``.clr`` file is keyed by the **class code**
(1-10), while :mod:`chapter4_qolu.land_cover` and every grid in this package
are keyed by the **band index** (0-9 = code - 1), with ``-1`` for no data.  The
constant below uses the band index, i.e. the same key the stored grids use.

One taxonomy difference worth recording here: the paper's own land-preference
legend uses **eight** classes (Arable Land, Grassland, Heather Grassland,
Littoral Sediment, Freshwater, Urban, Woodland, Suburban), which is a
regrouping of the LCM2015 *21 target classes*, not of the 10 aggregate classes
this package loads.  The maps are therefore drawn in the aggregate-class
taxonomy, and the difference is stated wherever the figures are described.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from chapter4_qolu.land_cover import CLASS_LABELS, AggregateClass

__all__ = [
    "AGGREGATE_CLASS_COLOURS",
    "CLASS_LEGEND",
    "NO_DATA_COLOUR",
    "colours_for_raster",
    "read_colour_table",
]

#: Sea / outside the mapped area - code 0 of the product's own palette.
NO_DATA_COLOUR = "#f0f0f0"

#: Band index (0-9) -> hex colour, transcribed from
#: ``lcm2015_gb_1km_dominant_aggregate_class.img.clr``.
AGGREGATE_CLASS_COLOURS: Mapping[int, str] = {
    0: "#ff0000",  # 1  broadleaf woodland
    1: "#006600",  # 2  coniferous woodland
    2: "#732600",  # 3  arable
    3: "#00ff00",  # 4  improved grassland
    4: "#998100",  # 5  semi-natural grassland
    5: "#801a80",  # 6  mountain, heath, bog
    6: "#000080",  # 7  saltwater
    7: "#0000ff",  # 8  freshwater
    8: "#ffff80",  # 9  coastal
    9: "#808080",  # 10 built-up areas and gardens
}

#: Band index -> human-readable class name, for a shared figure legend.
CLASS_LEGEND: Mapping[int, str] = {
    int(land_class) - 1: label for land_class, label in CLASS_LABELS.items()
}


def _hex(red: int, green: int, blue: int) -> str:
    return f"#{int(red):02x}{int(green):02x}{int(blue):02x}"


def read_colour_table(path: Path | str) -> dict[int, str]:
    """Read a ``*.clr`` palette into ``{class code: hex}`` (code 0 included)."""
    table: dict[int, str] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            code = int(parts[0])
            red, green, blue = (int(value) for value in parts[1:4])
        except ValueError:
            continue
        table[code] = _hex(red, green, blue)
    return table


def colours_for_raster(raster_path: Path | str | None) -> dict[int, str]:
    """Band-index colours, from the raster's own ``.clr`` when there is one.

    Falls back to :data:`AGGREGATE_CLASS_COLOURS`, which is that same file
    transcribed, so a figure drawn without the raster looks identical to one
    drawn with it.
    """
    colours = dict(AGGREGATE_CLASS_COLOURS)
    if raster_path is None:
        return colours
    candidate = Path(str(raster_path) + ".clr")
    if not candidate.is_file():
        candidate = Path(raster_path).with_suffix(".clr")
    if not candidate.is_file():
        return colours
    table = read_colour_table(candidate)
    for land_class in AggregateClass:
        code = int(land_class)
        if code in table:
            colours[code - 1] = table[code]
    return colours
