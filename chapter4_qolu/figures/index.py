"""``figures_index.md``: what each figure is, where it came from, how to redraw it.

A figure without its provenance is hard to reuse: a reader has to be able to
say which campaign produced it and what command would produce it again.  Every figure family therefore
returns a :class:`~chapter4_qolu.figures.backend.FigureResult` carrying its
source directory and its generation command, and this module turns the list
into one document.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from chapter4_qolu.figures.backend import FigureResult, matplotlib_available

__all__ = ["INDEX_NAME", "render_figures_index", "write_figures_index"]

INDEX_NAME = "figures_index.md"


def render_figures_index(results: Sequence[FigureResult]) -> str:
    """The whole ``figures_index.md`` document."""
    lines = [
        "# Chapter 4 figures",
        "",
        (
            "Every figure below was generated from files already on disk - "
            "per-run JSON records and a campaign's `summary.json` - so none of "
            "them needs a run to be repeated.  Each writes its own numbers as "
            "a `.csv` of the same name"
            + (
                ", plus a `.pdf` and a `.png`."
                if matplotlib_available()
                else "; matplotlib is not importable in the interpreter that "
                "generated this index, so only the `.csv` files were written."
            )
        ),
        "",
    ]
    if not results:
        lines.extend(["No figures were generated.", ""])
        return "\n".join(lines)
    for result in results:
        lines.extend(
            [
                f"## `{result.name}`",
                "",
                result.caption,
                "",
                "* **files**: "
                + ", ".join(f"`{Path(path).name}`" for path in result.paths),
                f"* **source**: `{result.source}`",
                f"* **command**: `{result.command}`",
            ]
        )
        for note in result.notes:
            lines.append(f"* **note**: {note}")
        lines.append("")
    return "\n".join(lines)


def write_figures_index(out_dir: Path, results: Sequence[FigureResult]) -> Path:
    """Write ``figures_index.md`` into ``out_dir`` and return its path."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / INDEX_NAME
    path.write_text(render_figures_index(results), encoding="utf-8")
    return path
