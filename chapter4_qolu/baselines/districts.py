"""District lattice and node / parcel features for the LToS baseline.

Mapping (A) of ``docs/baselines_lto_s_and_mitm_research.md``: tile the parcel
grid into districts, put one LToS agent on each, and connect them by rook
(4-neighbour) adjacency *with a self-loop*, which is the neighbourhood ``N_i``
Yi et al. (2021) define (``d_ii`` is included).  On the published 50 x 50 grid
with the default ``districts_per_side = 10`` this gives 10 x 10 districts of
5 x 5 parcels.

Slot layout.  ``N_i`` has a different size for interior, edge and corner
districts, but a network needs a fixed-width input, so the sharing vectors
``w_i^out`` / ``w_i^in`` always have :data:`N_SLOTS` entries in the fixed order
``(self, north, south, west, east)``.  Missing neighbours are masked out and
never receive weight.

Design note: this fixed five-slot layout - Yi et al. describe ``w_i^out`` as a
vector over ``N_i`` without fixing a representation, and a lattice makes the
ordering canonical.  The alternative (ragged per-agent vectors) would rule out
the batched graph attention that DGN needs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from chapter4_qolu.env.attractiveness import AttractivenessField
from chapter4_qolu.env.parcel_grid import ParcelGrid, ParcelIndex
from chapter4_qolu.land_cover import N_AGGREGATE_CLASSES

__all__ = [
    "NODE_FEATURE_DIM",
    "N_SLOTS",
    "PARCEL_FEATURE_DIM",
    "REVERSE_SLOT",
    "DistrictLattice",
    "incoming_weights",
    "node_features",
    "parcel_features",
]

#: ``(self, north, south, west, east)``.
N_SLOTS: int = 5

#: ``REVERSE_SLOT[s]`` is the slot of the neighbour reached through ``s`` that
#: points back at the sender, i.e. ``w_ji`` lives at
#: ``w_out[j, REVERSE_SLOT[s]]`` when ``j`` is ``i``'s neighbour in slot ``s``.
REVERSE_SLOT: tuple[int, ...] = (0, 2, 1, 4, 3)

#: Mean cover of the ten aggregate classes, mean attractiveness, and the share
#: of the district that is still convertible.
NODE_FEATURE_DIM: int = N_AGGREGATE_CLASSES + 2

#: The parcel's own cover and attractiveness plus its 3 x 3 neighbourhood means.
PARCEL_FEATURE_DIM: int = 2 * (N_AGGREGATE_CLASSES + 1)

#: Size of the local window summarised in a parcel feature vector.  Design note:
#: 3 x 3, the smallest window that carries any spatial context; the district
#: embedding supplies the wider view.
_PARCEL_WINDOW: int = 3

#: Row / column offsets of the four rook neighbours, in slot order.
_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class DistrictLattice:
    """A tiling of the parcel grid into LToS agents, plus their adjacency."""

    height: int
    width: int
    district_size: int
    n_rows: int
    n_cols: int
    neighbour_index: np.ndarray
    neighbour_mask: np.ndarray

    @classmethod
    def for_grid(
        cls, height: int, width: int, districts_per_side: int = 10
    ) -> DistrictLattice:
        """Tile a ``height x width`` grid into roughly ``districts_per_side`` columns.

        The district *size* is derived from the grid so that the published
        50 x 50 grid gives 5 x 5 districts.

        Design note: ``size = ceil(max(height, width) / districts_per_side)`` and
        a grid that does not divide evenly keeps a *smaller* district in the
        last row and column rather than dropping parcels.  The note only fixes
        the 50 / 10 case; every parcel has to belong to exactly one agent, so
        truncating the lattice was not an option.
        """
        if height < 1 or width < 1:
            raise ValueError("grid dimensions must be positive")
        if districts_per_side < 1:
            raise ValueError("districts_per_side must be positive")
        size = max(1, math.ceil(max(height, width) / districts_per_side))
        n_rows = math.ceil(height / size)
        n_cols = math.ceil(width / size)
        index, mask = _rook_adjacency(n_rows, n_cols)
        return cls(
            height=int(height),
            width=int(width),
            district_size=int(size),
            n_rows=int(n_rows),
            n_cols=int(n_cols),
            neighbour_index=index,
            neighbour_mask=mask,
        )

    @property
    def n_districts(self) -> int:
        return int(self.n_rows * self.n_cols)

    def index_of(self, district_row: int, district_col: int) -> int:
        """Flat index of the district at ``(district_row, district_col)``."""
        if not (0 <= district_row < self.n_rows and 0 <= district_col < self.n_cols):
            raise IndexError(
                f"district ({district_row}, {district_col}) is outside the "
                f"{self.n_rows} x {self.n_cols} lattice"
            )
        return int(district_row * self.n_cols + district_col)

    def district_of(self, parcel: ParcelIndex) -> int:
        """The district containing ``parcel``."""
        if not (0 <= parcel.row < self.height and 0 <= parcel.col < self.width):
            raise IndexError(f"parcel {parcel} is outside the grid")
        return self.index_of(
            parcel.row // self.district_size, parcel.col // self.district_size
        )

    def districts_of(self, parcels: Sequence[ParcelIndex]) -> np.ndarray:
        """:meth:`district_of` for a whole decision buffer, in order."""
        return np.asarray(
            [self.district_of(parcel) for parcel in parcels], dtype=np.int64
        )

    def bounds(self, district: int) -> tuple[slice, slice]:
        """Row and column slices of the parcels inside ``district``."""
        if not 0 <= district < self.n_districts:
            raise IndexError(f"district {district} is outside the lattice")
        row, col = divmod(district, self.n_cols)
        size = self.district_size
        return (
            slice(row * size, min((row + 1) * size, self.height)),
            slice(col * size, min((col + 1) * size, self.width)),
        )


def _rook_adjacency(n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-width rook adjacency with a self-loop, in :data:`N_SLOTS` slots."""
    n_districts = n_rows * n_cols
    index = np.zeros((n_districts, N_SLOTS), dtype=np.int64)
    mask = np.zeros((n_districts, N_SLOTS), dtype=np.float64)
    for row in range(n_rows):
        for col in range(n_cols):
            district = row * n_cols + col
            for slot, (row_offset, col_offset) in enumerate(_OFFSETS):
                neighbour_row = row + row_offset
                neighbour_col = col + col_offset
                if 0 <= neighbour_row < n_rows and 0 <= neighbour_col < n_cols:
                    index[district, slot] = neighbour_row * n_cols + neighbour_col
                    mask[district, slot] = 1.0
                else:
                    # Absent neighbour: fall back on the district itself so the
                    # gather stays in range, and mask the slot out entirely.
                    index[district, slot] = district
    return index, mask


def incoming_weights(sharing_out: np.ndarray, lattice: DistrictLattice) -> np.ndarray:
    """``w_i^in = {w_ji | j in N_i}`` from the lattice's outgoing vectors.

    This is the "exchange ``w^out`` -> ``w^in``" line of Algorithm 1: the share
    neighbour ``j`` sends to ``i`` sits in the slot of ``j`` that points back at
    ``i``, which is :data:`REVERSE_SLOT`.
    """
    weights = np.asarray(sharing_out, dtype=np.float64)
    if weights.shape != (lattice.n_districts, N_SLOTS):
        raise ValueError(
            f"sharing weights must have shape ({lattice.n_districts}, {N_SLOTS}), "
            f"got {weights.shape}"
        )
    incoming = np.zeros_like(weights)
    for slot in range(N_SLOTS):
        senders = lattice.neighbour_index[:, slot]
        incoming[:, slot] = (
            weights[senders, REVERSE_SLOT[slot]] * lattice.neighbour_mask[:, slot]
        )
    return incoming


def node_features(
    lattice: DistrictLattice, grid: ParcelGrid, attractiveness: AttractivenessField
) -> np.ndarray:
    """``o_i`` for every district: cover, attractiveness and headroom.

    Design note: the paper's environment gives no district-level observation, so
    ``o_i`` is the district's mean land-cover composition, its mean
    attractiveness and the share of its parcels still open to development.
    That is the smallest summary a sharing decision could plausibly need, and
    it mirrors what the QOLU observation carries per parcel.
    """
    features = np.zeros((lattice.n_districts, NODE_FEATURE_DIM), dtype=np.float32)
    convertible = grid.convertible_mask()
    for district in range(lattice.n_districts):
        rows, cols = lattice.bounds(district)
        block = grid.composition[:, rows, cols]
        features[district, :N_AGGREGATE_CLASSES] = block.mean(axis=(1, 2))
        features[district, N_AGGREGATE_CLASSES] = attractiveness.scores[
            rows, cols
        ].mean()
        features[district, N_AGGREGATE_CLASSES + 1] = convertible[rows, cols].mean()
    return features


def parcel_features(
    grid: ParcelGrid,
    attractiveness: AttractivenessField,
    parcels: Sequence[ParcelIndex],
) -> np.ndarray:
    """The low-level policy's per-candidate observation.

    Design note: DGN's Q head needs a compact vector per discrete decision, so a
    candidate is described by its own cover and attractiveness plus the means
    of both over a 3 x 3 window.  The full 9 x 9 QOLU patch would need a
    convolutional encoder, which DGN does not use.
    """
    features = np.zeros((len(parcels), PARCEL_FEATURE_DIM), dtype=np.float32)
    for position, parcel in enumerate(parcels):
        patch = grid.patch(parcel, _PARCEL_WINDOW)
        features[position, :N_AGGREGATE_CLASSES] = grid.composition[
            :, parcel.row, parcel.col
        ]
        features[position, N_AGGREGATE_CLASSES] = attractiveness.at(parcel)
        features[position, N_AGGREGATE_CLASSES + 1 : 2 * N_AGGREGATE_CLASSES + 1] = (
            patch.mean(axis=(1, 2))
        )
        features[position, -1] = attractiveness.patch(parcel, _PARCEL_WINDOW).mean()
    return features
