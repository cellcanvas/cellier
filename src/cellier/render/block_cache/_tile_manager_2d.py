"""Tile keys, slots, and the 2D atlas's slot geometry.

Which tile occupies which slot, what is evicted and what is drawn is the
chunk scheduler's business (``cellier.render.scheduling``, design
``plans/progressive_loading_design_v3.md``); the image residency adapter
(``_image_residency``) writes tiles and paints the LUT.  What is left here
is the part both need: the key and slot records, the flat-slot -> cache grid
position rule, and ``tilemap``, a read-only view of the tiles the LUT draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cellier.render.block_cache._cache_parameters_2d import (
        BlockCacheParameters2D,
    )


@dataclass(frozen=True)
class BlockKey2D:
    """Identifier for a tile at a specific LOD level and slice position.

    Attributes
    ----------
    level : int
        1-indexed LOD level (1 = finest).
    g0, g1 : int
        Grid position at this level's resolution.  Indexed in the same
        order as the visual's ``displayed_axes`` -- i.e. ``g0`` is the
        brick coordinate along ``displayed_axes[0]``, ``g1`` along
        ``displayed_axes[1]``.
    slice_coord : tuple of (data axis, selection) pairs
        Sorted tuple encoding the level-0 selection fetched on each sliced
        axis when this tile was requested: an integer plane, or a
        ``(start, stop)`` window for a slab.  Tiles from different slice
        positions will have distinct keys, allowing the cache to hold tiles
        from multiple slices simultaneously during a transition.
    """

    level: int
    g0: int
    g1: int
    slice_coord: tuple[tuple[int, int | tuple[int, int]], ...] = ()


@dataclass
class TileSlot:
    """A slot in the GPU cache.

    Attributes
    ----------
    index : int
        Flat slot index (0 = reserved, never used for data).
    grid_pos : tuple[int, int]
        ``(sy, sx)`` in the cache grid.
    timestamp : int
        Unused; kept for callers that construct slots by keyword.
    """

    index: int
    grid_pos: tuple[int, int]
    timestamp: int = 0


class TileManager2D:
    """Slot geometry for one 2D tile atlas, and a view of what it draws.

    Slot 0 is reserved: it samples as black (out of bounds), so the LUT's
    zero entry means "nothing here".  Data slots are ``1 .. n_slots - 1``.

    Parameters
    ----------
    cache_parameters : BlockCacheParameters2D
        Cache sizing metadata (grid dimensions, slot count, etc.).

    Attributes
    ----------
    tilemap : Mapping[BlockKey2D, TileSlot]
        The tiles the LUT currently draws.  Set by the residency adapter
        after every draw rebuild; read-only, for tests and diagnostics.
    """

    def __init__(self, cache_parameters: BlockCacheParameters2D) -> None:
        self.cache_info = cache_parameters
        self.tilemap: Mapping[BlockKey2D, TileSlot] = {}

    @property
    def n_data_slots(self) -> int:
        """Slots available for tiles (slot 0 excluded)."""
        return self.cache_info.n_slots - 1

    def _slot_grid_pos(self, flat_idx: int) -> tuple[int, int]:
        """Convert flat slot index to 2D grid position ``(sy, sx)``."""
        gs = self.cache_info.grid_side
        sy, sx = divmod(flat_idx, gs)
        return (sy, sx)

    def clear(self) -> None:
        """Forget the drawn view (the atlas's registry is the scheduler's)."""
        self.tilemap = {}
