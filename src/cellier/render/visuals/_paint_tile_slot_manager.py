"""Slot allocator for the multiscale-labels GPU paint cache."""

from __future__ import annotations

import logging
import math

_logger = logging.getLogger(__name__)


def paint_cache_grid(n_slots: int) -> tuple[int, int]:
    """Return the ``(rows, cols)`` slot grid of the paint cache texture.

    Slots fill the grid in row-major order: slot ``s`` sits at row
    ``s // cols``, column ``s % cols``.  The grid is as square as it can
    be, so neither texture dimension grows with ``n_slots`` alone.  The
    paint shader derives ``cols`` from the texture width, so the layout
    must stay ``cols * block_size`` pixels wide.

    Parameters
    ----------
    n_slots : int
        Number of paint slots (``paint_max_tiles``).

    Returns
    -------
    rows, cols : tuple[int, int]
        Slots per texture column and per texture row.
    """
    if n_slots <= 0:
        raise ValueError(f"n_slots must be positive, got {n_slots!r}")
    cols = math.isqrt(n_slots - 1) + 1
    rows = -(-n_slots // cols)
    return rows, cols


class PaintTileSlotManager:
    """Allocates fixed-pool paint-cache slots indexed by tile grid coords.

    "Tile grid coords" are 2-D ``(gy, gx)`` integers identifying a
    finest-level tile.  Each tile that gets painted on receives a slot.
    Slots are released only by :meth:`clear` (called when the paint
    session commits or aborts).

    Parameters
    ----------
    max_slots : int
        Pool size (== ``MultiscaleLabelRenderConfig.paint_max_tiles``).
    """

    def __init__(self, max_slots: int) -> None:
        if max_slots <= 0:
            raise ValueError(f"max_slots must be positive, got {max_slots!r}")
        self._max_slots = int(max_slots)
        self._tile_to_slot: dict[tuple[int, int], int] = {}
        self._exhaustion_warned = False

    @property
    def max_slots(self) -> int:
        return self._max_slots

    @property
    def n_allocated(self) -> int:
        return len(self._tile_to_slot)

    @property
    def exhausted(self) -> bool:
        """True if a request was rejected at least once this session."""
        return self._exhaustion_warned

    def get_or_allocate(self, tile_coord: tuple[int, int]) -> int | None:
        """Return the slot index for *tile_coord*, allocating if needed.

        Returns ``None`` when the pool is exhausted; the caller must
        skip the GPU update for that tile (drop-with-warning policy).
        """
        slot = self._tile_to_slot.get(tile_coord)
        if slot is not None:
            return slot
        if len(self._tile_to_slot) >= self._max_slots:
            if not self._exhaustion_warned:
                _logger.warning(
                    "PaintTileSlotManager: paint_max_tiles=%d exhausted; "
                    "subsequent paint will not be visible until commit. "
                    "Increase MultiscaleLabelRenderConfig.paint_max_tiles "
                    "to raise the limit.",
                    self._max_slots,
                )
                self._exhaustion_warned = True
            return None
        slot = len(self._tile_to_slot)
        self._tile_to_slot[tile_coord] = slot
        return slot

    def get(self, tile_coord: tuple[int, int]) -> int | None:
        """Return the slot for *tile_coord* without allocating."""
        return self._tile_to_slot.get(tile_coord)

    def clear(self) -> None:
        """Release every slot.  Resets the exhaustion flag."""
        self._tile_to_slot.clear()
        self._exhaustion_warned = False

    def __repr__(self) -> str:
        return (
            f"<PaintTileSlotManager allocated={self.n_allocated}/"
            f"{self._max_slots} exhausted={self.exhausted}>"
        )
