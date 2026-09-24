"""The chunk scheduler's adapters for brick atlases (design v3, 4.4 and 5.8).

:class:`ImageResidency3D` and :class:`ImageResidency2D` are the
:class:`~cellier.render.scheduling.Residency` of one atlas and its LUT --
a :class:`~cellier.render.block_cache.BlockCache3D` or a
:class:`~cellier.render.block_cache.BlockCache2D` -- for image and labels,
one per drawn channel.  The scheduler decides which brick lives in which
slot; these write the brick, paint the LUT, and turn keys back into store
requests.

Keys (design 5.1)
-----------------
A key is a packed ``int64``::

    level:4 | slice id:16 | g0:12 | g1:12 | g2:12

``g0, g1, g2`` is the brick's grid position at its level, in fetch order
(ascending data axes); a 2D tile leaves ``g2`` zero.  The *slice id* interns
the per-level fetched selection on the collapsed axes -- the plane or slab
the fetch reads there,
and, for a channel of a composite image, its channel index.  At a coarse
level two adjacent level-0 planes that read the same coarse plane share one
key.  The interning table is per atlas.  Limits: 15 levels, 65 536 slice
selections per atlas, 4 096 bricks per axis.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from cellier.data.image._image_requests import ChunkRequest
from cellier.render._gpu_lifetime import weak_callback
from cellier.render._level_mapping import level_to_data
from cellier.render.block_cache._tile_manager_2d import BlockKey2D
from cellier.render.block_cache._tile_manager_2d import TileSlot as TileSlot2D
from cellier.render.block_cache._tile_manager_3d import BlockKey3D, TileSlot

if TYPE_CHECKING:
    from collections.abc import Callable

    from cellier.render.block_cache._block_cache import BlockCache3D
    from cellier.render.block_cache._block_cache_2d import BlockCache2D
    from cellier.render.lut_indirection import LutIndirectionManager3D
    from cellier.render.lut_indirection._lut_indirection_manager_2d import (
        LutIndirectionManager2D,
    )
    from cellier.render.scheduling import RegistryView

_LEVEL_SHIFT = 52
_SID_SHIFT = 36
_G0_SHIFT = 24
_G1_SHIFT = 12
_LEVEL_MASK = 0xF
_SID_MASK = 0xFFFF
_G_MASK = 0xFFF

#: Largest level, slice id and grid index a key can hold.
MAX_LEVEL = _LEVEL_MASK
MAX_SLICE_IDS = _SID_MASK + 1
MAX_GRID = _G_MASK + 1

#: A slice selection: one entry per data axis, ``None`` on the displayed
#: (retained) axes and the fetched plane or ``(start, stop)`` slab on the
#: collapsed ones.  ``None`` as a whole means "no region; every axis is
#: displayed" (a visual planned without a selection).
SliceSelection = tuple[int | tuple[int, int] | None, ...] | None

_cache_ids = itertools.count(1)


def new_cache_id() -> int:
    """A process-unique cache id."""
    return next(_cache_ids)


def pack_keys(
    levels: np.ndarray, slice_ids: np.ndarray, grids: np.ndarray
) -> np.ndarray:
    """Pack ``(level, slice id, g0, g1, g2)`` into ``int64`` keys.

    *grids* is ``(N, 3)``, or ``(N, 2)`` for 2D tiles (``g2`` is zero).

    Raises
    ------
    ValueError
        If a field does not fit its bits.
    """
    levels = np.asarray(levels, dtype=np.int64)
    slice_ids = np.asarray(slice_ids, dtype=np.int64)
    grids = np.asarray(grids, dtype=np.int64)
    if grids.ndim == 2 and grids.shape[1] == 2:
        grids = np.column_stack([grids, np.zeros(len(grids), np.int64)])
    grids = grids.reshape(-1, 3)
    if len(levels) and (
        levels.min() < 1
        or levels.max() > MAX_LEVEL
        or slice_ids.max() >= MAX_SLICE_IDS
        or grids.min() < 0
        or grids.max() >= MAX_GRID
    ):
        raise ValueError(
            f"brick keys out of range: levels 1..{MAX_LEVEL}, slice ids below "
            f"{MAX_SLICE_IDS}, grid indices 0..{MAX_GRID - 1}."
        )
    return (
        (levels << _LEVEL_SHIFT)
        | (slice_ids << _SID_SHIFT)
        | (grids[:, 0] << _G0_SHIFT)
        | (grids[:, 1] << _G1_SHIFT)
        | grids[:, 2]
    )


def unpack_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(levels, slice ids, grids)`` from packed keys; grids are ``(N, 3)``."""
    keys = np.asarray(keys, dtype=np.int64)
    levels = (keys >> _LEVEL_SHIFT) & _LEVEL_MASK
    slice_ids = (keys >> _SID_SHIFT) & _SID_MASK
    grids = np.column_stack(
        [(keys >> _G0_SHIFT) & _G_MASK, (keys >> _G1_SHIFT) & _G_MASK, keys & _G_MASK]
    )
    return levels, slice_ids, grids


def _max_value(data: np.ndarray) -> float:
    return float(np.max(data)) if data.size else 0.0


class _DrawnView(Mapping):
    """``BlockKey -> TileSlot`` over the bricks one draw rebuild painted.

    Built lazily from arrays, so a rebuild pays nothing for it unless
    someone reads it (tests and diagnostics).  Holds plain data, not the
    residency: the atlas holds this view, so a back reference would be a
    cycle keeping the atlas's textures alive past ``close()``.
    """

    def __init__(
        self,
        keys: np.ndarray,
        slots: np.ndarray,
        slice_coords: list[tuple[tuple[int, Any], ...]],
        slot_grid: np.ndarray,
        brick_max: np.ndarray | None,
    ) -> None:
        self._keys = keys
        self._slots = slots
        self._slice_coords = slice_coords
        self._slot_grid = slot_grid
        self._brick_max = brick_max
        self._built: dict | None = None

    def _dict(self) -> dict:
        if self._built is None:
            levels, sids, grids = unpack_keys(self._keys)
            built: dict = {}
            for i in range(len(self._keys)):
                slot = int(self._slots[i])
                grid_pos = tuple(int(v) for v in self._slot_grid[i])
                coord = self._slice_coords[int(sids[i])]
                g = [int(v) for v in grids[i]]
                if self._brick_max is None:
                    key: Any = BlockKey2D(
                        level=int(levels[i]), g0=g[0], g1=g[1], slice_coord=coord
                    )
                    built[key] = TileSlot2D(index=slot + 1, grid_pos=grid_pos)
                else:
                    key = BlockKey3D(
                        level=int(levels[i]),
                        g0=g[0],
                        g1=g[1],
                        g2=g[2],
                        slice_coord=coord,
                    )
                    built[key] = TileSlot(
                        index=slot + 1,
                        grid_pos=grid_pos,
                        brick_max=float(self._brick_max[i]),
                    )
            self._built = built
        return self._built

    def __getitem__(self, key: Any) -> Any:
        return self._dict()[key]

    def __iter__(self) -> Iterator:
        return iter(self._dict())

    def __len__(self) -> int:
        return len(self._keys)


class _AtlasResidency:
    """What the 2D and 3D adapters share: slice interning, requests, regions.

    Subclasses set ``_NDIM`` (displayed axes, 2 or 3), fill ``slot_grid``,
    and implement ``write`` and ``rebuild_draw``.
    """

    _NDIM: int = 3

    def __init__(
        self,
        block_cache: Any,
        lut_manager: Any,
        block_size: int,
        level_scales: np.ndarray,
        level_translations: np.ndarray,
        on_write: Callable[[], None] | None = None,
    ) -> None:
        self.cache_id = new_cache_id()
        self.block_cache = block_cache
        self.lut_manager = lut_manager
        self.n_slots = int(block_cache.info.n_slots) - 1
        self._block_size = int(block_size)
        self._overlap = int(block_cache.info.overlap)
        self._level_scales = np.asarray(level_scales, dtype=np.float64)
        self._level_translations = np.asarray(level_translations, dtype=np.float64)
        # The visual owns this residency and passes its own methods: held
        # weakly, so a visual dropped without close() dies by refcount.
        self._on_write = weak_callback(on_write)
        self._selections: list[SliceSelection] = []
        self._selection_ids: dict[SliceSelection, int] = {}
        self._request_id = uuid4()
        #: Cache grid position of each scheduler slot (slow axis first).
        self.slot_grid = np.empty((0, self._NDIM), dtype=np.int64)

    # -- slice selections ------------------------------------------------------

    def intern(self, selection: SliceSelection) -> int:
        """The slice id of *selection*, interning it on first sight.

        Raises
        ------
        RuntimeError
            If the atlas has seen :data:`MAX_SLICE_IDS` distinct selections.
        """
        sid = self._selection_ids.get(selection)
        if sid is None:
            sid = len(self._selections)
            if sid >= MAX_SLICE_IDS:
                raise RuntimeError(
                    f"this atlas has interned {MAX_SLICE_IDS} slice selections; "
                    f"reallocate it to start a new table"
                )
            self._selections.append(selection)
            self._selection_ids[selection] = sid
        return sid

    def selection(self, sid: int) -> SliceSelection:
        """The selection slice id *sid* names."""
        return self._selections[sid]

    def slice_coord(self, sid: int) -> tuple[tuple[int, Any], ...]:
        """``(data axis, selection)`` for the collapsed axes of *sid*."""
        selection = self._selections[sid]
        if selection is None:
            return ()
        return tuple(
            (axis, value) for axis, value in enumerate(selection) if value is not None
        )

    def _slice_coords(self) -> list[tuple[tuple[int, Any], ...]]:
        return [self.slice_coord(sid) for sid in range(len(self._selections))]

    def _drawn(self, view: RegistryView):
        """The painted records: keys, slots, levels, grids and phases."""
        groups = view.paint_groups()
        drawn = np.concatenate(groups) if groups else np.empty(0, np.int64)
        keys = np.asarray(view.key[drawn], dtype=np.int64)
        slots = np.asarray(view.slot[drawn], dtype=np.int64)
        levels, _, grids = unpack_keys(keys)
        # Phases index into the drawn arrays, in painting order.
        bounds = np.cumsum([0, *(len(g) for g in groups)])
        phases = [np.arange(bounds[i], bounds[i + 1]) for i in range(len(groups))]
        return keys, slots, levels, grids[:, : self._NDIM], phases

    # -- requests ------------------------------------------------------------

    def build_request(self, keys: np.ndarray) -> list[ChunkRequest]:
        """One store request per key: the padded brick on its slice."""
        levels, sids, grids = unpack_keys(keys)
        starts = grids * self._block_size - self._overlap
        stops = starts + self._block_size + 2 * self._overlap
        requests: list[ChunkRequest] = []
        for i in range(len(keys)):
            windows = [(int(starts[i, a]), int(stops[i, a])) for a in range(self._NDIM)]
            selection = self._selections[int(sids[i])]
            if selection is None:
                axis_selections: tuple = tuple(windows)
            else:
                retained = iter(windows)
                axis_selections = tuple(
                    next(retained) if value is None else value for value in selection
                )
            requests.append(
                ChunkRequest(
                    chunk_request_id=uuid4(),
                    slice_request_id=self._request_id,
                    scale_index=int(levels[i]) - 1,
                    axis_selections=axis_selections,
                )
            )
        return requests

    def keys_in_region(self, keys: np.ndarray, region: Any) -> np.ndarray:
        """Which *keys*' padded bricks overlap *region*.

        Parameters
        ----------
        keys : np.ndarray
            Packed keys.
        region : Sequence[DataRegion]
            Regions in level-0 data coordinates, one ``(start, stop)`` per
            data axis each (``stop`` exclusive), as a store announces them.

        Returns
        -------
        np.ndarray
            Boolean mask over *keys*.
        """
        keys = np.asarray(keys, dtype=np.int64)
        hit = np.zeros(len(keys), dtype=bool)
        if not len(keys):
            return hit
        regions = [np.asarray(r, dtype=np.float64) for r in region]
        if not regions:
            return hit
        levels, sids, grids = unpack_keys(keys)
        ndim = self._level_scales.shape[1]
        starts = grids * self._block_size - self._overlap
        stops = starts + self._block_size + 2 * self._overlap
        for level, sid in {(int(a), int(b)) for a, b in zip(levels, sids)}:
            rows = np.flatnonzero((levels == level) & (sids == sid))
            low = np.zeros((len(rows), ndim))
            high = np.zeros((len(rows), ndim))
            selection = self._selections[sid]
            retained = (
                list(range(ndim))[-self._NDIM :]
                if selection is None
                else [a for a, v in enumerate(selection) if v is None]
            )
            for position, axis in enumerate(retained):
                low[:, axis] = starts[rows, position]
                high[:, axis] = stops[rows, position]
            if selection is not None:
                for axis, value in enumerate(selection):
                    if value is None:
                        continue
                    if isinstance(value, tuple):
                        low[:, axis], high[:, axis] = value
                    else:
                        low[:, axis], high[:, axis] = value, value + 1
            # Centre convention (plan v2, D1): level voxels [low, high) cover
            # u in [low - 0.5, high - 0.5], and level-0 voxels [start, stop)
            # cover data [start - 0.5, stop - 0.5].
            a, b = (
                level_to_data(
                    low - 0.5,
                    self._level_scales[level - 1],
                    self._level_translations[level - 1],
                ),
                level_to_data(
                    high - 0.5,
                    self._level_scales[level - 1],
                    self._level_translations[level - 1],
                ),
            )
            low0, high0 = np.minimum(a, b), np.maximum(a, b)
            for r in regions:
                inside = (low0 < r[:, 1] - 0.5) & (high0 > r[:, 0] - 0.5)
                hit[rows] |= inside.all(axis=1)
        return hit


class ImageResidency3D(_AtlasResidency):
    """A 3D brick atlas as seen by the chunk scheduler.

    Parameters
    ----------
    block_cache : BlockCache3D
        The atlas texture.  Its slot 0 is reserved (black), so scheduler slot
        ``s`` is atlas slot ``s + 1``.
    lut_manager : LutIndirectionManager3D
        The atlas's LUT.
    block_size : int
        Brick side in voxels.
    level_scales, level_translations : np.ndarray
        ``(n_levels, ndim)``: each level's voxel -> level-0 voxel map per data
        axis (for region invalidation).
    brick_max : Callable[[np.ndarray], float]
        The brick-max value a brick's data writes (image: its maximum; labels:
        whether it holds a label).
    on_write : Callable[[], None] or None
        Called after every brick write (the visual reveals its bounding box
        on the first).

    Attributes
    ----------
    cache_id : int
        The id this atlas is registered with.
    n_slots : int
        Slots the scheduler may fill.
    """

    _NDIM = 3

    def __init__(
        self,
        block_cache: BlockCache3D,
        lut_manager: LutIndirectionManager3D,
        block_size: int,
        level_scales: np.ndarray,
        level_translations: np.ndarray,
        brick_max: Callable[[np.ndarray], float] = _max_value,
        on_write: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            block_cache,
            lut_manager,
            block_size,
            level_scales,
            level_translations,
            on_write=on_write,
        )
        self._brick_max_fn = weak_callback(brick_max, dead_result=0.0)
        side = int(block_cache.info.grid_side)
        flat = np.arange(1, self.n_slots + 1)
        sz, rem = np.divmod(flat, side * side)
        sy, sx = np.divmod(rem, side)
        #: Cache grid position ``(sz, sy, sx)`` of each scheduler slot.
        self.slot_grid = np.column_stack([sz, sy, sx]).astype(np.int64)
        #: The brick max each scheduler slot's data wrote.
        self.brick_max = np.zeros(self.n_slots, dtype=np.float32)

    def write(self, slot: int, key: int, data: Any) -> None:
        """Upload a padded brick into scheduler slot *slot*."""
        data = np.asarray(data)
        self.brick_max[slot] = self._brick_max_fn(data)
        grid_pos = tuple(int(v) for v in self.slot_grid[slot])
        self.block_cache.write_brick(TileSlot(index=slot + 1, grid_pos=grid_pos), data)
        if self._on_write is not None:
            self._on_write()

    def rebuild_draw(self, view: RegistryView) -> None:
        """Paint the LUT: background oldest first, then the foreground (5.8)."""
        keys, slots, levels, grids, phases = self._drawn(view)
        slot_grid = self.slot_grid[slots] if len(slots) else np.empty((0, 3), np.int64)
        brick_max = self.brick_max[slots] if len(slots) else np.empty(0, np.float32)
        self.lut_manager.paint(levels, grids, slot_grid, brick_max, phases)
        self.block_cache.tile_manager.tilemap = _DrawnView(
            keys, slots, self._slice_coords(), slot_grid, brick_max
        )


class ImageResidency2D(_AtlasResidency):
    """A 2D tile atlas as seen by the chunk scheduler.

    Parameters
    ----------
    block_cache : BlockCache2D
        The atlas texture.  Its slot 0 is reserved (black), so scheduler slot
        ``s`` is atlas slot ``s + 1``.
    lut_manager : LutIndirectionManager2D
        The atlas's LUT.
    block_size : int
        Tile side in pixels.
    level_scales, level_translations : np.ndarray
        ``(n_levels, ndim)``: each level's voxel -> level-0 voxel map per data
        axis (for region invalidation).
    on_write : Callable[[], None] or None
        Called after every tile write (the visual reveals its bounding box on
        the first).

    Attributes
    ----------
    cache_id : int
        The id this atlas is registered with.
    n_slots : int
        Slots the scheduler may fill.
    viewport_cells : tuple[int, int, int, int] or None
        Base-grid cell bounds ``(gy0, gx0, gy1, gx1)`` (half open) of the
        latest plan's viewport.  The background (earlier views) is clipped
        to it, so stale tiles outside the view are not referenced; the
        foreground never is.  ``None`` disables the clip.  The planner sets
        it; it is draw state, not the atlas.
    """

    _NDIM = 2

    def __init__(
        self,
        block_cache: BlockCache2D,
        lut_manager: LutIndirectionManager2D,
        block_size: int,
        level_scales: np.ndarray,
        level_translations: np.ndarray,
        on_write: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            block_cache,
            lut_manager,
            block_size,
            level_scales,
            level_translations,
            on_write=on_write,
        )
        side = int(block_cache.info.grid_side)
        sy, sx = np.divmod(np.arange(1, self.n_slots + 1), side)
        #: Cache grid position ``(sy, sx)`` of each scheduler slot.
        self.slot_grid = np.column_stack([sy, sx]).astype(np.int64)
        self.viewport_cells: tuple[int, int, int, int] | None = None

    def write(self, slot: int, key: int, data: Any) -> None:
        """Upload a padded tile into scheduler slot *slot*."""
        grid_pos = tuple(int(v) for v in self.slot_grid[slot])
        self.block_cache.write_tile(
            TileSlot2D(index=slot + 1, grid_pos=grid_pos), np.asarray(data)
        )
        if self._on_write is not None:
            self._on_write()

    def rebuild_draw(self, view: RegistryView) -> None:
        """Paint the LUT: background oldest first and clipped, then foreground."""
        keys, slots, levels, grids, phases = self._drawn(view)
        slot_grid = self.slot_grid[slots] if len(slots) else np.empty((0, 2), np.int64)
        clips = [self.viewport_cells] * (len(phases) - 1) + [None]
        self.lut_manager.paint(levels, grids, slot_grid, phases, clips)
        self.block_cache.tile_manager.tilemap = _DrawnView(
            keys, slots, self._slice_coords(), slot_grid, None
        )
