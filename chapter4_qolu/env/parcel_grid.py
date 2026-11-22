"""Immutable parcel-grid state.

Each state ``s`` of the POSG "describes a set of 1 km^2 parcels with unique
attributes (e.g. current land usage, geographic constraints)".  Here a parcel is
one cell of a ``(n_classes, height, width)`` array of land cover *fractions*,
taken straight from the UKCEH 1 km percentage-cover product.

All mutating operations return a new :class:`ParcelGrid`; the underlying arrays
are marked read-only so accidental in-place edits fail loudly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from chapter4_qolu.land_cover import (
    N_AGGREGATE_CLASSES,
    AggregateClass,
    LandCoverGroup,
    band_index,
    group_indices,
)

__all__ = ["Conversion", "ParcelGrid", "ParcelIndex"]

_TOLERANCE = 1e-4


@dataclass(frozen=True, order=True)
class ParcelIndex:
    """Row/column position of a 1 km^2 parcel in the grid."""

    row: int
    col: int

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield self.row
        yield self.col


@dataclass(frozen=True)
class Conversion:
    """Record of one parcel converted to urban land during an iteration."""

    parcel: ParcelIndex
    #: Cover fraction removed from every aggregate class, shape ``(n_classes,)``.
    lost_cover: np.ndarray
    #: Total cover fraction added to the built-up class.
    urban_gain: float


@dataclass(frozen=True)
class ParcelGrid:
    """Land cover composition of a rectangular block of 1 km^2 parcels."""

    composition: np.ndarray

    #: Value used by :meth:`dominant_class` for parcels without any cover.
    NO_DATA: ClassVar[int] = -1

    def __post_init__(self) -> None:
        array = np.asarray(self.composition, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(
                f"composition must be (n_classes, height, width), got shape {array.shape}"
            )
        if array.shape[0] != N_AGGREGATE_CLASSES:
            raise ValueError(
                f"composition must have {N_AGGREGATE_CLASSES} class bands, "
                f"got {array.shape[0]}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("composition contains non-finite values")
        if float(array.min()) < -_TOLERANCE or float(array.max()) > 1.0 + _TOLERANCE:
            raise ValueError("composition values must be cover fractions in [0, 1]")
        totals = array.sum(axis=0)
        if float(totals.max(initial=0.0)) > 1.0 + _TOLERANCE:
            raise ValueError("per-parcel cover fractions must not sum above 1")
        array = np.clip(array, 0.0, 1.0)
        array.setflags(write=False)
        object.__setattr__(self, "composition", array)

    # ------------------------------------------------------------------
    # shape helpers
    # ------------------------------------------------------------------
    @property
    def n_classes(self) -> int:
        return int(self.composition.shape[0])

    @property
    def height(self) -> int:
        return int(self.composition.shape[1])

    @property
    def width(self) -> int:
        return int(self.composition.shape[2])

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def n_parcels(self) -> int:
        return self.height * self.width

    def contains(self, parcel: ParcelIndex) -> bool:
        return 0 <= parcel.row < self.height and 0 <= parcel.col < self.width

    # ------------------------------------------------------------------
    # cover queries
    # ------------------------------------------------------------------
    def class_fraction(self, land_class: AggregateClass) -> np.ndarray:
        return np.asarray(self.composition[band_index(land_class)], dtype=np.float32)

    def group_fraction(self, group: LandCoverGroup) -> np.ndarray:
        """Total cover fraction of ``group`` in every parcel."""
        indices = list(group_indices(group))
        return np.asarray(self.composition[indices].sum(axis=0), dtype=np.float32)

    def group_mask(self, group: LandCoverGroup, threshold: float = 0.5) -> np.ndarray:
        """Parcels whose ``group`` cover reaches ``threshold``."""
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must lie in (0, 1]")
        return self.group_fraction(group) >= threshold

    def total_cover(self) -> np.ndarray:
        return np.asarray(self.composition.sum(axis=0), dtype=np.float32)

    def land_mask(self) -> np.ndarray:
        """Parcels that carry any land cover at all (outside coverage = False)."""
        return self.total_cover() > _TOLERANCE

    def urban_fraction(self) -> np.ndarray:
        return self.group_fraction(LandCoverGroup.URBAN)

    def convertible_mask(self, urban_saturation: float = 0.99) -> np.ndarray:
        """Parcels that may still be proposed for conversion to urban land."""
        if not 0.0 < urban_saturation <= 1.0:
            raise ValueError("urban_saturation must lie in (0, 1]")
        return self.land_mask() & (self.urban_fraction() < urban_saturation)

    def dominant_class(self) -> np.ndarray:
        """Band index of the most common class per parcel (``NO_DATA`` if empty)."""
        dominant = np.argmax(self.composition, axis=0).astype(np.int64)
        return np.where(self.land_mask(), dominant, self.NO_DATA)

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    def patch(self, parcel: ParcelIndex, window: int) -> np.ndarray:
        """Zero-padded ``(n_classes, window, window)`` patch centred on ``parcel``."""
        if window < 1 or window % 2 == 0:
            raise ValueError("window must be a positive odd number")
        if not self.contains(parcel):
            raise IndexError(f"parcel {parcel} is outside the grid {self.shape}")
        radius = window // 2
        patch = np.zeros((self.n_classes, window, window), dtype=np.float32)
        row_start = max(parcel.row - radius, 0)
        row_end = min(parcel.row + radius + 1, self.height)
        col_start = max(parcel.col - radius, 0)
        col_end = min(parcel.col + radius + 1, self.width)
        patch[
            :,
            row_start - (parcel.row - radius) : row_end - (parcel.row - radius),
            col_start - (parcel.col - radius) : col_end - (parcel.col - radius),
        ] = self.composition[:, row_start:row_end, col_start:col_end]
        return patch

    # ------------------------------------------------------------------
    # transitions
    # ------------------------------------------------------------------
    def convert_to_urban(
        self,
        parcels: Iterable[ParcelIndex],
        fraction: float = 1.0,
        urban_saturation: float = 0.99,
    ) -> tuple[ParcelGrid, tuple[Conversion, ...]]:
        """Apply a synchronised batch of land conversions.

        Duplicate parcels are converted once (the decision buffer samples *with
        replacement*), and parcels that are already urban or carry no data are
        skipped.  Returns the new grid plus a record per applied conversion.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must lie in (0, 1]")

        unique: list[ParcelIndex] = []
        seen: set[ParcelIndex] = set()
        for parcel in parcels:
            if not self.contains(parcel):
                raise IndexError(f"parcel {parcel} is outside the grid {self.shape}")
            if parcel not in seen:
                seen.add(parcel)
                unique.append(parcel)

        convertible = self.convertible_mask(urban_saturation)
        urban_band = band_index(AggregateClass.BUILT_UP)
        updated = np.array(self.composition, dtype=np.float32, copy=True)
        conversions: list[Conversion] = []

        for parcel in unique:
            if not convertible[parcel.row, parcel.col]:
                continue
            cover = updated[:, parcel.row, parcel.col]
            lost = cover * fraction
            lost[urban_band] = 0.0
            gained = float(lost.sum())
            if gained <= 0.0:
                continue
            updated[:, parcel.row, parcel.col] = cover - lost
            updated[urban_band, parcel.row, parcel.col] += gained
            conversions.append(
                Conversion(parcel=parcel, lost_cover=lost.copy(), urban_gain=gained)
            )

        return ParcelGrid(updated), tuple(conversions)

    def with_composition(self, composition: np.ndarray) -> ParcelGrid:
        """Return a new grid with the given composition (never mutates ``self``)."""
        return ParcelGrid(np.array(composition, dtype=np.float32, copy=True))

    @classmethod
    def from_fractions(cls, fractions: np.ndarray) -> ParcelGrid:
        return cls(np.array(fractions, dtype=np.float32, copy=True))

    @classmethod
    def empty(cls, height: int, width: int) -> ParcelGrid:
        return cls(np.zeros((N_AGGREGATE_CLASSES, height, width), dtype=np.float32))


def parcels_from_mask(mask: np.ndarray) -> tuple[ParcelIndex, ...]:
    """All parcels where ``mask`` is true, in row-major order."""
    rows, cols = np.nonzero(mask)
    return tuple(ParcelIndex(int(row), int(col)) for row, col in zip(rows, cols))


def as_parcels(items: Sequence[tuple[int, int]]) -> tuple[ParcelIndex, ...]:
    """Convenience converter for ``(row, col)`` tuples."""
    return tuple(ParcelIndex(int(row), int(col)) for row, col in items)
