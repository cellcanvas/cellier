"""Paint controller for multiscale (zarr-backed) label data stores.

Architecture (Phase 3 — pyramid rebuild + autosave loop)
---------------------------------------------------------
Each brush application:

1. Reads pre-paint values from the open ``WriteBuffer`` (read-your-writes
   through a private tensorstore transaction).
2. Stages the new values into the same buffer.
3. Marks the dirty level-0 bricks in the :class:`WriteLayer`.
4. Writes the new values into the GPU paint cache + LUT via
   :meth:`CellierController._patch_painted_tiles_2d`.

The shader composites the paint cache over the base sample, so painted
voxels are visible on the next frame.  No reslice or eviction occurs
during a brush step.

On commit: coarser LOD bricks are rebuilt bottom-up within the open
transaction, then the transaction is flushed atomically; the GPU paint
textures are cleared; the visible cache is evicted for dirty bricks;
``reslice_visual`` repopulates the base cache from disk.

On autosave: same as commit but undo history is preserved and the
session continues with a fresh transaction.

On abort: ``WriteBuffer`` discards the transaction; the GPU paint
textures are cleared.  No reslice is needed because the base cache is
still pre-paint.

The controller is N-dim agnostic at the storage layer (``WriteLayer`` /
``WriteBuffer``).  Visible feedback is currently only wired for 2-D
displayed-axis configurations; 3-D paint feedback raises
``NotImplementedError`` at construction.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from cellier.paint._abstract import AbstractPaintController
from cellier.paint._write_buffer import TensorStoreWriteBuffer
from cellier.paint._write_layer import BrickKey, WriteLayer

if TYPE_CHECKING:
    from uuid import UUID

    import tensorstore as ts

    from cellier.controller import CellierController
    from cellier.paint._history import PaintStrokeCommand


class MultiscalePaintController(AbstractPaintController):
    """Paint controller for OME-Zarr / multiscale-zarr data stores.

    Stages all writes in a tensorstore transaction (RAM-only) until the
    user commits or aborts the session.  Visible feedback is provided by
    writing directly into the GPU paint cache; the shader composites paint
    over the base sample so painted voxels are visible immediately without
    a slicer round-trip.

    Parameters
    ----------
    cellier_controller :
    visual_id :
    scene_id :
    canvas_id :
    data_store :
        The store behind a ``MultiscaleLabelVisual``: an
        ``OMEZarrLabelDataStore`` or a ``MultiscaleZarrDataStore``.
        Must expose ``_ts_stores: list[ts.TensorStore]`` and
        ``level_shapes`` / ``level_scales``.
    visual_block_size :
        Tile / brick side length used by the visual's render config.
        Must match the ``MultiscaleLabelRenderConfig.block_size`` of the
        rendered visual.
    displayed_axes :
        The two data-array axes currently displayed in 2D for the bound
        canvas, in ``(row_axis, col_axis)`` order.  3D paint feedback is
        a follow-up; passing a 3-tuple raises ``NotImplementedError``.
    brush_value : int
    brush_radius_voxels :
    history_depth :
    autosave_interval_s :
        If set, a ``QTimer`` fires every this many seconds to flush staged
        paint to disk, rebuild the pyramid, and reset the GPU paint
        textures.  ``None`` disables autosave.
    """

    def __init__(
        self,
        cellier_controller: CellierController,
        visual_id: UUID,
        scene_id: UUID,
        canvas_id: UUID,
        data_store: Any,
        visual_block_size: int,
        displayed_axes: tuple[int, ...],
        brush_value: int = 1,
        brush_radius_voxels: float = 2.0,
        history_depth: int = 100,
        autosave_interval_s: float | None = None,
    ) -> None:
        if len(displayed_axes) != 2:
            raise NotImplementedError(
                "MultiscalePaintController currently only supports 2-D "
                f"displayed-axis configurations; got {displayed_axes!r}.  "
                "3-D paint feedback is a Phase 3 follow-up."
            )
        if not data_store._ts_stores:
            raise ValueError(
                "data_store has no open tensorstore handles; ensure "
                "model_post_init has run."
            )

        self._data_store = data_store
        self._data_shape: tuple[int, ...] = tuple(
            int(d) for d in data_store.level_shapes[0]
        )
        self._displayed_axes: tuple[int, int] = tuple(displayed_axes)  # type: ignore[assignment]
        self._autosave_count: int = 0
        self._autosave_interval_s: float | None = autosave_interval_s
        self._last_autosave_time: datetime | None = None
        self._autosave_timer = None  # typed below after QTimer import

        self._store_dtype: np.dtype = data_store.dtype
        self._write_buffer: TensorStoreWriteBuffer = TensorStoreWriteBuffer(
            data_store._ts_stores[0]
        )
        # Let the store refuse a cache-pool change while this buffer holds a
        # handle: reopening would bind the store's levels to a new pool while
        # the buffer kept writing through the discarded one.
        data_store.register_paint_writer(self._write_buffer)
        self._write_layer = WriteLayer(
            data_store_id=data_store.id, block_size=int(visual_block_size)
        )

        super().__init__(
            cellier_controller=cellier_controller,
            visual_id=visual_id,
            scene_id=scene_id,
            canvas_id=canvas_id,
            data_store_id=data_store.id,
            brush_value=brush_value,
            brush_radius_voxels=brush_radius_voxels,
            history_depth=history_depth,
        )

        if autosave_interval_s is not None and autosave_interval_s > 0:
            from PySide6.QtCore import QTimer

            self._autosave_timer = QTimer()
            self._autosave_timer.setInterval(int(autosave_interval_s * 1000))
            self._autosave_timer.timeout.connect(self._do_autosave)
            self._autosave_timer.start()

    # ------------------------------------------------------------------
    # AbstractPaintController hooks
    # ------------------------------------------------------------------

    def _apply_brush(self, world_coord: np.ndarray) -> None:
        """Re-order world_coord from pygfx (x, y) to data (row, col) order.

        The labels visuals render ``data[r, c]`` at pygfx world ``(x=c, y=r)``.
        ``CellierController._on_raw_pointer_event`` embeds the mouse position
        verbatim: ``world_coord[ax_row] = pygfx_x`` (column index),
        ``world_coord[ax_col] = pygfx_y`` (row index).  Swapping them here
        produces the correct voxel address before the identity
        ``imap_coordinates`` call in the base.
        """
        ax_row, ax_col = self._displayed_axes
        swapped = world_coord.copy()
        swapped[ax_row], swapped[ax_col] = (
            float(world_coord[ax_col]),
            float(world_coord[ax_row]),
        )
        super()._apply_brush(swapped)

    def _read_old_values(self, voxel_indices: np.ndarray) -> np.ndarray:
        """Read pre-paint values through the open transaction."""
        return self._write_buffer.read_staged(voxel_indices)

    def _write_values(self, voxel_indices: np.ndarray, values: np.ndarray) -> None:
        """Stage writes, mark dirty bricks, patch the GPU paint texture."""
        if voxel_indices.shape[0] == 0:
            return

        cast_values = values.astype(self._store_dtype, copy=False)
        self._write_buffer.stage(voxel_indices, cast_values)

        new_dirty = self._write_layer.voxels_to_brick_keys(voxel_indices)
        for key in new_dirty:
            self._write_layer.mark_dirty(key)

        # GPU paint cache always stores float32 (label IDs as float).
        self._controller._patch_painted_tiles_2d(
            self._visual_id,
            voxel_indices,
            cast_values.astype(np.float32),
            self._displayed_axes,
        )

    def _on_stroke_completed(self, command: PaintStrokeCommand) -> None:
        """No background work — visible feedback already happened in _write_values."""

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def commit(self) -> None:
        """Flush staged paint + rebuild pyramid, clear GPU, repopulate cache."""
        if self._autosave_timer is not None:
            self._autosave_timer.stop()

        # 1. Rebuild coarser LOD levels within the open transaction.
        self._rebuild_pyramid(self._write_buffer.transaction)

        # 2. Flush level-0 + all rebuilt levels atomically.
        self._write_buffer.commit()

        # 3. Drop GPU paint textures — paint is now on disk.
        self._controller._clear_painted_tiles_2d(self._visual_id)

        # 4. Invalidate + reslice so the atlases reflect the committed state.
        self._invalidate_dirty_bricks()
        self._write_layer.clear()
        self._history._undo_stack.clear()
        self._history._redo_stack.clear()
        self._controller.reslice_visual(self._visual_id)

        self._teardown()

    def abort(self) -> None:
        """Discard staged paint and clear the GPU paint textures."""
        # 1. Discard staged writes (disk untouched).
        self._write_buffer.abort()

        # 2. Drop the GPU paint textures.
        self._controller._clear_painted_tiles_2d(self._visual_id)

        # 3. NO reslice — the base cache already shows pre-paint data.
        self._write_layer.clear()
        self._history._undo_stack.clear()
        self._history._redo_stack.clear()

        self._teardown()

    def _teardown(self) -> None:
        """Stop autosave timer, unsubscribe bus events, re-enable camera."""
        if self._autosave_timer is not None:
            self._autosave_timer.stop()
            self._autosave_timer = None
        # Release the store's cache-pool interlock (see __init__).
        self._data_store.unregister_paint_writer()
        super()._teardown()

    # ------------------------------------------------------------------
    # Autosave
    # ------------------------------------------------------------------

    def _do_autosave(self) -> None:
        """Flush staged paint to disk, rebuild pyramid, reset GPU textures.

        Called automatically by the autosave timer.  Does NOT clear undo
        history or tear down the session — the session continues with a
        fresh transaction.
        """
        dirty_count = len(self._write_layer.dirty_keys())
        if dirty_count == 0:
            return

        # 1. Rebuild coarser LOD levels within the open transaction.
        self._rebuild_pyramid(self._write_buffer.transaction)

        # 2. Flush level-0 + rebuilt levels atomically.
        self._write_buffer.commit()

        # 3. Create a fresh transaction for continued staging.
        self._write_buffer = TensorStoreWriteBuffer(self._data_store._ts_stores[0])
        self._data_store.register_paint_writer(self._write_buffer)

        # 4. Drop GPU paint textures — frees the entire slot pool.
        self._controller._clear_painted_tiles_2d(self._visual_id)

        # 5. Invalidate the atlases over dirty bricks; the reslice refetches.
        self._invalidate_dirty_bricks()
        self._write_layer.clear()
        self._controller.reslice_visual(self._visual_id)

        self._autosave_count += 1
        self._last_autosave_time = datetime.now()

    # ------------------------------------------------------------------
    # Autosave state accessors
    # ------------------------------------------------------------------

    @property
    def autosave_count(self) -> int:
        """Number of autosaves completed this session."""
        return self._autosave_count

    @property
    def last_autosave_time(self) -> datetime | None:
        """Datetime of the most recent autosave, or None if none yet."""
        return self._last_autosave_time

    # ------------------------------------------------------------------
    # Pyramid rebuild
    # ------------------------------------------------------------------

    def _rebuild_pyramid(self, txn: ts.Transaction | None) -> dict[int, int]:
        """Rebuild coarser LOD bricks for all dirty level-0 bricks.

        Operates within *txn* so the rebuild is atomic with the level-0
        writes.  Reads level k-1 through the transaction (read-your-writes)
        to build level k.

        Parameters
        ----------
        txn :
            The active tensorstore transaction from ``self._write_buffer``.
            Must still be open (before ``commit_sync``).

        Returns
        -------
        dict[int, int]
            Mapping ``level → n_bricks_rebuilt`` for each level k >= 1.
            Empty if the store has only one level or txn is None.
        """
        if txn is None:
            return {}

        n_levels = self._data_store.n_levels
        if n_levels < 2:
            return {}

        level_shapes = self._data_store.level_shapes
        level_scales = self._data_store.level_scales
        stores = self._data_store._ts_stores
        ax_y, ax_x = self._displayed_axes

        stats: dict[int, int] = {}

        dirty_by_level: dict[int, set[tuple[int, ...]]] = {
            0: {key.grid_coords for key in self._write_layer.dirty_keys()}
        }

        for k in range(1, n_levels):
            stride_y = max(
                1,
                round(level_scales[k][ax_y] / level_scales[k - 1][ax_y]),
            )
            stride_x = max(
                1,
                round(level_scales[k][ax_x] / level_scales[k - 1][ax_x]),
            )

            parent_dirty = dirty_by_level.get(k - 1, set())
            dirty_k: set[tuple[int, ...]] = set()
            for gc in parent_dirty:
                dirty_k.add(
                    tuple(
                        c // stride_y
                        if i == ax_y
                        else c // stride_x
                        if i == ax_x
                        else c
                        for i, c in enumerate(gc)
                    )
                )
            dirty_by_level[k] = dirty_k

            if not dirty_k:
                stats[k] = 0
                continue

            bs = self._write_layer.block_size
            src_store = stores[k - 1].with_transaction(txn)
            dst_store = stores[k].with_transaction(txn)

            shape_km1 = level_shapes[k - 1]
            shape_k = level_shapes[k]

            n_rebuilt = 0
            for gc in dirty_k:
                gy_k, gx_k = gc[ax_y], gc[ax_x]

                src_y0 = gy_k * bs * stride_y
                src_y1 = min((gy_k + 1) * bs * stride_y, shape_km1[ax_y])
                src_x0 = gx_k * bs * stride_x
                src_x1 = min((gx_k + 1) * bs * stride_x, shape_km1[ax_x])

                if src_y0 >= shape_km1[ax_y] or src_x0 >= shape_km1[ax_x]:
                    continue

                dst_y0 = gy_k * bs
                dst_y1 = min((gy_k + 1) * bs, shape_k[ax_y])
                dst_x0 = gx_k * bs
                dst_x1 = min((gx_k + 1) * bs, shape_k[ax_x])

                if dst_y0 >= shape_k[ax_y] or dst_x0 >= shape_k[ax_x]:
                    continue

                src_slices = [slice(None)] * len(shape_km1)
                src_slices[ax_y] = slice(src_y0, src_y1)
                src_slices[ax_x] = slice(src_x0, src_x1)
                block = np.array(
                    src_store[tuple(src_slices)].read().result(),
                    dtype=self._store_dtype,
                )

                # Stride-2 decimation: pick every stride-th voxel.
                # Pure index selection preserves the store dtype exactly.
                dec_slices = [slice(None)] * block.ndim
                dec_slices[ax_y] = slice(None, None, stride_y)
                dec_slices[ax_x] = slice(None, None, stride_x)
                downsampled = block[tuple(dec_slices)]

                actual_h = dst_y1 - dst_y0
                actual_w = dst_x1 - dst_x0
                clip_slices = [slice(None)] * downsampled.ndim
                clip_slices[ax_y] = slice(0, actual_h)
                clip_slices[ax_x] = slice(0, actual_w)
                downsampled = downsampled[tuple(clip_slices)]

                dst_slices = [slice(None)] * len(shape_k)
                dst_slices[ax_y] = slice(dst_y0, dst_y1)
                dst_slices[ax_x] = slice(dst_x0, dst_x1)
                dst_store[tuple(dst_slices)].write(downsampled).result()
                n_rebuilt += 1

            stats[k] = n_rebuilt

        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _invalidate_dirty_bricks(self) -> list[int]:
        """Drop GPU data over every dirty brick, at every level (design 5.14).

        Each dirty level-0 brick becomes a level-0 data region; the chunk
        scheduler drops the resident bricks and tiles of every atlas reading
        this store that overlap one, and fetches the wanted ones again
        (through the open transaction, since rechecks are on while a paint
        writer is registered).  Coarser levels are covered too: the pyramid
        rebuild rewrote them.

        Returns
        -------
        list[int]
            The atlases touched.
        """
        dirty_keys = self._write_layer.dirty_keys()
        if not dirty_keys:
            return []
        bs = self._write_layer.block_size
        regions = tuple(
            tuple((g * bs, (g + 1) * bs) for g in key.grid_coords)
            for key in sorted(dirty_keys)
        )
        return self._controller._invalidate_painted_regions(
            self._data_store.id, regions
        )

    def __repr__(self) -> str:
        return (
            f"<MultiscalePaintController visual={self._visual_id} "
            f"canvas={self._canvas_id} "
            f"dirty_bricks={len(self._write_layer.dirty_keys())} "
            f"history_depth={len(self._history._undo_stack)}>"
        )


# Re-export BrickKey so callers can reach it via the paint package without
# touching private modules.
__all__ = ["BrickKey", "MultiscalePaintController"]
