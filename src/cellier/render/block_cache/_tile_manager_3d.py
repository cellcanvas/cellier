"""Brick keys, slots, and the 3D atlas's slot geometry.

Which brick occupies which slot, what is evicted and what is drawn is the
chunk scheduler's business (``cellier.render.scheduling``, design
``plans/progressive_loading_design_v3.md``); the image residency adapter
(``_image_residency``) writes bricks and paints the LUT.  What is left here
is the part both need: the key and slot records, the flat-slot -> cache grid
position rule, and ``tilemap``, a read-only view of the bricks the LUT draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cellier.render.block_cache._cache_parameters_3d import (
        BlockCacheParameters3D,
    )


@dataclass(frozen=True)
class BlockKey3D:
    """Identifier for a brick at a specific LOAD level and slice position.

    Attributes
    ----------
    level : int
        1-indexed LOAD level (1 = finest).
    g0, g1, g2 : int
        Grid position at this level's resolution.  Indexed in the same
        order as the visual's ``displayed_axes`` -- i.e. ``g0`` is the
        brick coordinate along ``displayed_axes[0]``, etc.  For the
        current 3D case with ``displayed_axes=(0, 1, 2)`` these are
        the grid positions along data axes z, y, x respectively.
    slice_coord : tuple of (data axis, selection) pairs
        Sorted tuple encoding the level-0 selection fetched on each
        collapsed axis when this brick was requested: an integer plane, or
        a ``(start, stop)`` window for a slab.  Bricks from different slice
        positions will have distinct keys, allowing the cache to hold
        bricks from multiple slices simultaneously during a transition,
        while slider positions that fetch the same plane share one.
        Empty for purely 3-D data where all axes are displayed.
    """

    level: int
    g0: int
    g1: int
    g2: int
    slice_coord: tuple[tuple[int, int | tuple[int, int]], ...] = ()


@dataclass
class TileSlot:
    """A slot in the GPU cache.

    Attributes
    ----------
    index : int
        Flat slot index (0 = reserved, never used for data).
    grid_pos : tuple[int, int, int]
        ``(sz, sy, sx)`` in the cache grid.
    timestamp : int
        Unused; kept for callers that construct slots by keyword.
    brick_max : float
        Maximum value in the brick, for the MIP early-out.
    """

    index: int
    grid_pos: tuple[int, int, int]
    timestamp: int = 0
    brick_max: float = 0.0


class TileManager3D:
    """Slot geometry for one 3D brick atlas, and a view of what it draws.

    Slot 0 is reserved: it samples as black (out of bounds), so the LUT's
    zero entry means "nothing here".  Data slots are ``1 .. n_slots - 1``.

    Parameters
    ----------
    cache_parameters :
        Cache sizing metadata (grid dimensions, slot count, etc.).

    Attributes
    ----------
    tilemap : Mapping[BlockKey3D, TileSlot]
        The bricks the LUT currently draws.  Set by the residency adapter
        after every draw rebuild; read-only, for tests and diagnostics.
    """

    def __init__(self, cache_parameters: BlockCacheParameters3D) -> None:
        self.cache_parameters = cache_parameters
        self.tilemap: Mapping[BlockKey3D, TileSlot] = {}

    @property
    def n_data_slots(self) -> int:
        """Slots available for bricks (slot 0 excluded)."""
        return self.cache_parameters.n_slots - 1

    def _slot_grid_pos(self, flat_idx: int) -> tuple[int, int, int]:
        """Convert flat slot index to 3D cache-grid position ``(sz, sy, sx)``."""
        gs = self.cache_parameters.grid_side
        sz, rem = divmod(flat_idx, gs * gs)
        sy, sx = divmod(rem, gs)
        return (sz, sy, sx)

    def clear(self) -> None:
        """Forget the drawn view (the atlas's registry is the scheduler's)."""
        self.tilemap = {}
