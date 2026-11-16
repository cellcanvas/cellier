"""The pyramid level a pick landed on, read from the brick or tile table.

A multiscale visual draws each brick or tile at one level of the pyramid, and
the table that says which (``lut_data``) lives on the CPU as well as the GPU.
A pick on a coarse level lands on that level's surface, a voxel or more away
from the level-0 one, so the value under the cursor is the one the drawn level
holds.  `DrawnLevels` answers which level that is.

The pick buffer is the last frame that was drawn, and the table is rewritten by
commit rounds, which the frame that draws it follows.  Between the two the
table no longer says what that frame drew, so the tracker records the table's
version in each frame's tick and answers ``None`` for a canvas whose table has
changed since.  A wrong level reads a wrong value, where level 0 reads the data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pygfx as gfx

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID


def _version(manager: Any) -> int | None:
    return None if manager is None else manager.version


def level_in_3d_table(manager: Any, coordinate: Sequence[float]) -> int | None:
    """The 1-based level of the brick that holds a 3D hit, or ``None``.

    *coordinate* is the hit on the drawn axes in pygfx ``(x, y, z)`` order; the
    table is ``(D, H, W)``.
    """
    block = int(manager._base_layout.block_size)
    grid = manager.lut_data.shape[:3]
    cell = tuple(
        min(max(int(coordinate[2 - k] // block), 0), grid[k] - 1) for k in range(3)
    )
    level = int(manager.lut_data[cell][3])
    return level if level >= 1 else None


def level_in_2d_table(manager: Any, coordinate: Sequence[float]) -> int | None:
    """The 1-based level of the tile that holds a 2D hit, or ``None``.

    *coordinate* is the hit in pygfx ``(x, y)`` order; the table is
    ``(H, W)`` with the level in channel 2.
    """
    block = int(manager._base_layout.block_size)
    grid = manager.lut_data.shape[:2]
    cell = tuple(
        min(max(int(coordinate[1 - k] // block), 0), grid[k] - 1) for k in range(2)
    )
    level = int(manager.lut_data[cell][2])
    return level if level >= 1 else None


class DrawnLevels:
    """Which table version each canvas last drew with, and the level lookups."""

    def __init__(self) -> None:
        # canvas id -> (3D table version, 2D table version) of its last frame.
        self._drawn: dict[UUID, tuple[int | None, int | None]] = {}

    def note(self, canvas_id: UUID, manager_3d: Any, manager_2d: Any) -> None:
        """Record the tables a frame of *canvas_id* is about to draw with.

        Call in the frame, just before it renders, after the commit round that
        writes them.
        """
        self._drawn[canvas_id] = (_version(manager_3d), _version(manager_2d))

    def level(
        self,
        hit_object: gfx.WorldObject,
        coordinate: Sequence[float],
        canvas_id: UUID | None,
        manager_3d: Any,
        manager_2d: Any,
    ) -> int | None:
        """The 1-based level drawn at a hit, or ``None`` if it cannot be said.

        ``None`` when the hit is on neither a volume nor an image node, when
        the node's table does not exist, when no brick or tile covers the hit,
        or, given a *canvas_id*, when that table changed since the canvas drew.
        With no *canvas_id* the table is read as it is.
        """
        if isinstance(hit_object, gfx.Volume):
            manager, which, lookup = manager_3d, 0, level_in_3d_table
        elif isinstance(hit_object, gfx.Image):
            manager, which, lookup = manager_2d, 1, level_in_2d_table
        else:
            return None
        if manager is None:
            return None
        if canvas_id is not None:
            drawn = self._drawn.get(canvas_id)
            if drawn is None or drawn[which] != manager.version:
                return None
        return lookup(manager, coordinate)
