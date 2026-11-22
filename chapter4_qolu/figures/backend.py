"""Plot backend and output contract shared by every figure.

Two rules, and everything else follows from them:

1. **The data always lands on disk.**  Every figure writes a CSV of the numbers
   it draws, with the figure's own name and next to it.  A thesis chapter
   should be able to quote a figure without this package, and a machine
   without matplotlib should still produce something usable.
2. **The plot is optional.**  PDF and PNG are written when matplotlib is
   importable and silently skipped when it is not, so no caller needs an
   import guard.

Design note (file names): a figure is named for what it shows and what it shows
it of - ``land_cover_scenario1_seed0``, ``curves_scenario2``,
``table_bars_scenario1_minmax``, ``scan_map_se_20260924`` - with no timestamp,
no hash and no counter, so regenerating a figure overwrites exactly the file it
replaces and a chapter's ``\\includegraphics`` never has to change.
"""

from __future__ import annotations

import csv
import importlib.util
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "FIGURE_SUFFIXES",
    "FigureResult",
    "matplotlib_available",
    "new_figure",
    "save_figure",
    "write_rows",
]

#: Vector first, raster second: the thesis uses the PDF, slides use the PNG.
FIGURE_SUFFIXES: tuple[str, ...] = (".pdf", ".png")
#: Design note: 200 dpi for the PNG - readable in a slide, still a small file.
_PNG_DPI = 200


def matplotlib_available() -> bool:
    """Is matplotlib importable in this interpreter?

    Checked through :mod:`importlib.util` rather than by importing, so asking
    the question costs nothing when the answer is no.
    """
    return importlib.util.find_spec("matplotlib") is not None


@dataclass(frozen=True)
class FigureResult:
    """One finished figure: what it is called, where it is, what made it."""

    name: str
    paths: tuple[Path, ...]
    #: What the figure shows, one line, for ``figures_index.md``.
    caption: str
    #: The campaign / scan directory the numbers came from.
    source: str
    #: The command that regenerates it.
    command: str
    notes: tuple[str, ...] = field(default=())

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "paths": [str(path) for path in self.paths],
            "caption": self.caption,
            "source": self.source,
            "command": self.command,
            "notes": list(self.notes),
        }


def write_rows(
    out_dir: Path,
    name: str,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Write a figure's own numbers as ``<name>.csv`` and return the path."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def new_figure(*args: Any, **kwargs: Any) -> Any:
    """``matplotlib.pyplot.subplots`` with the Agg backend forced.

    Agg because a campaign is normally drawn on a machine with no display, and
    a figure module that opened a window would hang a batch job.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt.subplots(*args, **kwargs)


def save_figure(figure: Any, out_dir: Path, name: str) -> tuple[Path, ...]:
    """Save one matplotlib figure as PDF and PNG, then close it."""
    import matplotlib.pyplot as plt

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in FIGURE_SUFFIXES:
        path = directory / f"{name}{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=_PNG_DPI)
        written.append(path)
    plt.close(figure)
    return tuple(written)
