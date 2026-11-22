"""Result-file layout and resume logic for a four-scenario campaign.

One campaign writes one JSON record per *run*, where a run is one
(configuration, scenario, method, seed) cell::

    results/campaign_<tag>/<grid>x<grid>_f<fraction>_cap<cap>/
        scenario<k>/<method>_seed<N>.json

The directory name encodes the three scale settings every method in a cell has
to share (``EnvConfig.grid_size``, ``EnvConfig.conversion_fraction`` and
``EnvConfig.max_conversions_per_iteration``), so two configurations can never
overwrite each other and ``--resume`` can tell from the path alone which runs
are already on disk.

The extent may be square (``50``, the thesis's 50 x 50 grid) or rectangular
(``210x240``, the whole South East England bounding box at 1 km), which is why
the grid token of ``grid:fraction:cap`` accepts both spellings.  The label
reads ``<rows>x<cols>``; for a square extent the two are equal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CAMPAIGN_PREFIX",
    "NO_CAP",
    "ScaleConfig",
    "campaign_directory",
    "is_complete",
    "parse_scale_config",
    "run_json_path",
    "scenario_directory",
    "should_run",
]

#: Prefix of a campaign's results folder, so campaigns sort together.
CAMPAIGN_PREFIX = "campaign_"
#: Spelling of "no per-iteration conversion cap" on the command line.
NO_CAP = "none"
#: ``50`` or ``210x240`` - a square extent or an explicit ``rows x cols`` one.
_GRID_TOKEN = re.compile(r"\A(\d+)(?:[xX](\d+))?\Z")


def _format_fraction(value: float) -> str:
    """Render a conversion fraction as a short, stable directory token.

    Design note: four decimals are enough for any fraction worth sweeping, and
    trailing zeros are stripped down to a single decimal, so 1.0 -> ``1.0`` and
    0.25 -> ``0.25``.  The label is used as a directory name and as a JSON key,
    so it has to be deterministic across platforms.
    """
    text = f"{float(value):.4f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text


@dataclass(frozen=True)
class ScaleConfig:
    """The environment-scale settings a campaign configuration fixes.

    ``grid_size`` is the number of **rows**; ``grid_width`` is the number of
    columns and defaults to ``grid_size``, i.e. a square extent.
    """

    grid_size: int
    conversion_fraction: float
    max_conversions: int | None
    #: Columns.  ``None`` means "square"; a whole-region extent sets it
    #: explicitly.
    grid_width: int | None = None

    def __post_init__(self) -> None:
        if self.grid_size < 1:
            raise ValueError("grid_size must be positive")
        if self.grid_width is not None and self.grid_width < 1:
            raise ValueError("grid_width must be positive")
        if not 0.0 < self.conversion_fraction <= 1.0:
            raise ValueError("conversion_fraction must lie in (0, 1]")
        if self.max_conversions is not None and self.max_conversions < 0:
            raise ValueError("max_conversions must be non-negative or None")

    @property
    def grid_height(self) -> int:
        return int(self.grid_size)

    @property
    def width(self) -> int:
        return int(self.grid_size if self.grid_width is None else self.grid_width)

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, columns)`` of the parcel grid."""
        return self.grid_height, self.width

    @property
    def n_cells(self) -> int:
        return self.grid_height * self.width

    @property
    def label(self) -> str:
        """Directory token, e.g. ``50x50_f1.0_cap10`` or ``210x240_f1.0_cap10``."""
        cap = NO_CAP if self.max_conversions is None else str(self.max_conversions)
        fraction = _format_fraction(self.conversion_fraction)
        return f"{self.grid_height}x{self.width}_f{fraction}_cap{cap}"

    @property
    def description(self) -> str:
        """Human-readable form used in table captions."""
        cap = (
            "no cap" if self.max_conversions is None else f"cap {self.max_conversions}"
        )
        return (
            f"{self.grid_height}x{self.width} grid, conversion fraction "
            f"{_format_fraction(self.conversion_fraction)}, {cap}"
        )

    def as_record(self) -> dict[str, object]:
        """JSON-serialisable form stored in every per-run record."""
        return {
            "label": self.label,
            "description": self.description,
            "grid_size": self.grid_size,
            "grid_height": self.grid_height,
            "grid_width": self.width,
            "conversion_fraction": self.conversion_fraction,
            "max_conversions_per_iteration": self.max_conversions,
        }


def parse_scale_config(text: str) -> ScaleConfig:
    """Parse a ``grid:fraction:cap`` triple such as ``50:1.0:10``.

    The grid token is either a single number (a square extent, ``50``) or
    ``rows x columns`` (``210x240``, the whole South East England box at 1 km).
    """
    parts = str(text).split(":")
    if len(parts) != 3:
        raise ValueError(
            f"configuration {text!r} must be 'grid:fraction:cap', e.g. '50:1.0:10'"
        )
    grid_text, fraction_text, cap_text = (part.strip() for part in parts)
    match = _GRID_TOKEN.match(grid_text)
    if not match:
        raise ValueError(
            f"grid extent {grid_text!r} must be a number or 'rows x cols', "
            "e.g. '50' or '210x240'"
        )
    grid_size = int(match.group(1))
    grid_width = int(match.group(2)) if match.group(2) else None
    try:
        conversion_fraction = float(fraction_text)
    except ValueError as error:
        message = f"conversion fraction {fraction_text!r} is not a number"
        raise ValueError(message) from error
    if cap_text.lower() == NO_CAP:
        max_conversions: int | None = None
    else:
        try:
            max_conversions = int(cap_text)
        except ValueError as error:
            raise ValueError(
                f"conversion cap {cap_text!r} must be an integer or '{NO_CAP}'"
            ) from error
    return ScaleConfig(
        grid_size=grid_size,
        conversion_fraction=conversion_fraction,
        max_conversions=max_conversions,
        grid_width=grid_width,
    )


def campaign_directory(results_dir: Path, tag: str) -> Path:
    """Folder holding every record and document of one campaign."""
    cleaned = str(tag).strip()
    if not cleaned:
        raise ValueError("campaign tag must not be empty")
    return Path(results_dir) / f"{CAMPAIGN_PREFIX}{cleaned}"


def scenario_directory(campaign_dir: Path, scale: ScaleConfig, scenario: int) -> Path:
    """Folder holding one (configuration, scenario) cell."""
    return Path(campaign_dir) / scale.label / f"scenario{int(scenario)}"


def run_json_path(
    campaign_dir: Path,
    scale: ScaleConfig,
    scenario: int,
    method: str,
    seed: int,
) -> Path:
    """Path of one run's JSON record."""
    directory = scenario_directory(campaign_dir, scale, scenario)
    return directory / f"{method}_seed{int(seed)}.json"


def is_complete(path: Path) -> bool:
    """``True`` when ``path`` holds a parsable, finished per-run record.

    Design note: "finished" means the JSON parses *and* carries
    ``final_metrics``, which is written last.  Checking existence alone would
    make ``--resume`` skip a run whose process was killed part-way through
    writing its record.
    """
    path = Path(path)
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    return isinstance(record, dict) and "final_metrics" in record


def should_run(path: Path, resume: bool) -> bool:
    """Whether a run has to be executed, honouring ``--resume``."""
    return not (resume and is_complete(path))
