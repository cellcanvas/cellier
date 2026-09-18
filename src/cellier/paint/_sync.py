"""Synchronous in-memory paint controller for LabelMemoryStore."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellier.paint._abstract import AbstractPaintController

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np

    from cellier.controller import CellierController
    from cellier.data.label._label_memory_store import LabelMemoryStore
    from cellier.paint._history import PaintStrokeCommand


class SyncPaintController(AbstractPaintController):
    """Paint controller for :class:`LabelMemoryStore`.

    Directly and synchronously mutates the backing numpy array on every
    brush application, then announces the write with
    ``store.notify_changed("contents", regions=...)``; the controller
    reslices every visual reading the store, so the display updates
    immediately.  Suitable for testing and in-memory annotation
    workflows.

    Parameters
    ----------
    cellier_controller : CellierController
    visual_id : UUID
    scene_id : UUID
    canvas_id : UUID
    data_store : LabelMemoryStore
        The store whose ``.data`` array is painted directly.
    displayed_axes : tuple[int, int]
        The two data-array axes currently displayed in 2D for the bound
        canvas, in ``(row_axis, col_axis)`` order.
    brush_value : int
    brush_radius_voxels : float
    history_depth : int
    """

    def __init__(
        self,
        cellier_controller: CellierController,
        visual_id: UUID,
        scene_id: UUID,
        canvas_id: UUID,
        data_store: LabelMemoryStore,
        displayed_axes: tuple[int, ...],
        brush_value: int = 1,
        brush_radius_voxels: float = 2.0,
        history_depth: int = 100,
    ) -> None:
        self._data_store: LabelMemoryStore = data_store
        self._data_shape = data_store.shape
        self._displayed_axes: tuple[int, int] = tuple(displayed_axes)  # type: ignore[assignment]
        self._store_dtype: np.dtype = data_store.data.dtype
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

    def _apply_brush(self, world_coord: np.ndarray) -> None:
        """Swap displayed axes before delegating to the base brush logic.

        ``CellierController._on_raw_pointer_event`` embeds the mouse position
        as ``world_coord[0] = pygfx_x``, ``world_coord[1] = pygfx_y``.
        ``GFXImageMemoryVisual`` renders ``data[r, c]`` at pygfx ``(x=c, y=r)``,
        so the row index arrives in ``world_coord[ax_col]`` and the column index
        in ``world_coord[ax_row]``.  Swapping them here produces the correct
        voxel address before the identity ``imap_coordinates`` call in the base.
        """
        ax_row, ax_col = self._displayed_axes
        swapped = world_coord.copy()
        swapped[ax_row], swapped[ax_col] = (
            float(world_coord[ax_col]),
            float(world_coord[ax_row]),
        )
        super()._apply_brush(swapped)

    def _read_old_values(self, voxel_indices: np.ndarray) -> np.ndarray:
        """Read directly from the backing numpy array."""
        idx = tuple(voxel_indices[:, i] for i in range(voxel_indices.shape[1]))
        return self._data_store.data[idx].copy()

    def _write_values(self, voxel_indices: np.ndarray, values: np.ndarray) -> None:
        """Write directly to the backing numpy array and announce the write.

        An in-place write is invisible to the store, so it is announced
        explicitly: a ``contents`` change over the painted voxels' bounding
        box.  The controller reslices every visual reading the store -- not
        the whole scene, as this used to.
        """
        idx = tuple(voxel_indices[:, i] for i in range(voxel_indices.shape[1]))
        self._data_store.data[idx] = values.astype(self._data_store.data.dtype)
        if len(voxel_indices) == 0:
            return
        region = tuple(
            (int(low), int(high) + 1)
            for low, high in zip(
                voxel_indices.min(axis=0), voxel_indices.max(axis=0), strict=True
            )
        )
        self._data_store.notify_changed("contents", regions=[region])

    def _on_stroke_completed(self, command: PaintStrokeCommand) -> None:
        """No background work needed — the array is already up to date."""

    def commit(self) -> None:
        """End the session.  Data is already in the array; just teardown."""
        self._history._undo_stack.clear()
        self._history._redo_stack.clear()
        self._teardown()

    def abort(self) -> None:
        """Revert all strokes by replaying the undo stack in reverse."""
        while self._history.can_undo:
            command = self._history.undo()
            if command is not None:
                self._write_values(command.voxel_indices, command.old_values)
        self._history._redo_stack.clear()
        self._teardown()
