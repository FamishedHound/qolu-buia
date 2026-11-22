"""Figures for Chapter 4: maps, curves, bar charts and the window-scan map.

Four figure families, all built from files a campaign has already written, so
no run has to be repeated to redraw anything:

(a) :mod:`~chapter4_qolu.figures.maps`      before / after land-cover maps per method
(b) :mod:`~chapter4_qolu.figures.curves`    per-iteration series per method, with seed bands
(c) :mod:`~chapter4_qolu.figures.bars`      the published-style tables as grouped bars
(d) :mod:`~chapter4_qolu.figures.scan_map`  the searched windows, coloured by score

Every figure writes **its data** as a CSV next to itself, whether or not
matplotlib is importable, and the PDF and PNG only when it is.  That is not a
fallback so much as the point: the numbers behind a figure in a thesis chapter
should be readable without this package, and a machine without matplotlib
should still produce something usable.

``python -m chapter4_qolu.make_figures --campaign <dir> [--out <dir>]`` builds
the campaign families; ``--scan <dir>`` builds the scan map.
"""

from __future__ import annotations

from chapter4_qolu.figures.backend import FigureResult, matplotlib_available

__all__ = ["FigureResult", "matplotlib_available"]
