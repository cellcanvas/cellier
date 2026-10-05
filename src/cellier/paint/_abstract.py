"""Shared base class for paint controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np

from cellier.data._plane_clip import kept_points, plane_tuples
from cellier.events._events import (
    CanvasMouseMove2DEvent,
    CanvasMousePress2DEvent,
    CanvasMouseRelease2DEvent,
)
from cellier.paint._history import (
    ActiveStroke,
    CommandHistory,
    PaintStrokeCommand,
)

if TYPE_CHECKING:
    from cellier.controller import CellierController


class AbstractPaintController(ABC):
    """Minimal shared base for paint controllers.

    Owns: EventBus subscriptions, :class:`CommandHistory`,
    :class:`ActiveStroke` accumulation, and camera-controller locking on
    the bound canvas.

    Concrete subclasses override the ``_read_old_values``,
    ``_write_values``, ``_on_stroke_completed``, ``commit``, and
    ``abort`` hooks to plug in storage-specific behaviour.

    Parameters
    ----------
    cellier_controller : CellierController
    visual_id : UUID
    scene_id : UUID
    canvas_id : UUID
        Mouse subscriptions are scoped to this canvas; its camera
        controller is disabled for the session duration.
    data_store_id : UUID
        Recorded on every :class:`PaintStrokeCommand` for history
        attribution.
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
        data_store_id: UUID,
        brush_value: int = 1,
        brush_radius_voxels: float = 2.0,
        history_depth: int = 100,
    ) -> None:
        self._id: UUID = uuid4()
        self._controller = cellier_controller
        self._visual_id = visual_id
        self._scene_id = scene_id
        self._canvas_id = canvas_id
        self._data_store_id = data_store_id
        self._brush_value = int(brush_value)
        self._brush_radius = float(brush_radius_voxels)
        self._history = CommandHistory(max_depth=history_depth)
        self._active_stroke: ActiveStroke | None = None

        # Concrete subclasses must set ``_data_shape`` and ``_displayed_axes``
        # before calling super().__init__; validate here so failures surface
        # in the subclass constructor rather than silently late.
        if not hasattr(self, "_data_shape"):
            raise AttributeError(
                "Concrete paint controllers must set `_data_shape` "
                "before calling AbstractPaintController.__init__."
            )
        if not hasattr(self, "_displayed_axes"):
            raise AttributeError(
                "Concrete paint controllers must set `_displayed_axes` "
                "before calling AbstractPaintController.__init__."
            )
        if not hasattr(self, "_store_dtype"):
            raise AttributeError(
                "Concrete paint controllers must set `_store_dtype` "
                "before calling AbstractPaintController.__init__."
            )

        bus = cellier_controller._outgoing_events
        bus.subscribe(
            CanvasMousePress2DEvent,
            self._on_mouse_press,
            entity_id=canvas_id,
            owner_id=self._id,
        )
        bus.subscribe(
            CanvasMouseMove2DEvent,
            self._on_mouse_move,
            entity_id=canvas_id,
            owner_id=self._id,
        )
        bus.subscribe(
            CanvasMouseRelease2DEvent,
            self._on_mouse_release,
            entity_id=canvas_id,
            owner_id=self._id,
        )

        # Disable camera controller last so a construction failure leaves
        # the camera enabled.
        cellier_controller.set_camera_controller_enabled(canvas_id, False)

    # ------------------------------------------------------------------
    # Brush parameter accessors
    # ------------------------------------------------------------------

    @property
    def brush_value(self) -> int:
        """Scalar value written to every painted voxel."""
        return self._brush_value

    @brush_value.setter
    def brush_value(self, value: int) -> None:
        self._brush_value = int(value)

    @property
    def brush_radius_voxels(self) -> float:
        """Brush radius in voxel units."""
        return self._brush_radius

    @brush_radius_voxels.setter
    def brush_radius_voxels(self, value: float) -> None:
        self._brush_radius = float(value)

    # ------------------------------------------------------------------
    # Mouse event handlers (shared)
    # ------------------------------------------------------------------

    def _on_mouse_press(self, event: CanvasMousePress2DEvent) -> None:
        # A previous stroke may never have received its release (pointer
        # capture broken by focus loss).  Finalise it so its voxels are
        # preserved in history rather than silently overwritten.
        if self._active_stroke is not None:
            command = self._active_stroke.finalise()
            self._active_stroke = None
            self._history.push(command)
            self._on_stroke_completed(command)
        # Label visuals discard background fragments so they never write to the
        # pick buffer when all-background.  Use a data-bounds check instead of
        # pick detection so a blank label array is still paintable.
        if not self._coord_within_data_bounds(event.world_coordinate):
            return
        self._active_stroke = ActiveStroke(
            visual_id=self._visual_id,
            data_store_id=self._data_store_id,
            gesture_id=event.gesture_id,
        )
        self._apply_brush(event.world_coordinate)

    def _coord_within_data_bounds(self, world_coord: np.ndarray) -> bool:
        """True if *world_coord* maps to a valid voxel in the data array.

        Performs the same axis swap as :meth:`_apply_brush` so the bounds
        check uses data-array (row, col) order rather than pygfx world
        (x, y) order.
        """
        ax_row, ax_col = self._displayed_axes
        swapped = world_coord.copy()
        swapped[ax_row], swapped[ax_col] = (
            float(world_coord[ax_col]),
            float(world_coord[ax_row]),
        )
        scene_model = self._controller._model.scenes[self._scene_id]
        visual_model = next(v for v in scene_model.visuals if v.id == self._visual_id)
        voxel_coord = visual_model.transform.imap_coordinates(np.atleast_2d(swapped))[0]
        voxel = np.round(voxel_coord).astype(np.int64)
        return all(0 <= int(voxel[i]) < self._data_shape[i] for i in range(len(voxel)))

    def _on_mouse_move(self, event: CanvasMouseMove2DEvent) -> None:
        if self._active_stroke is None:
            return
        # Ignore moves that belong to a different gesture than the active
        # stroke (guards against interleaved or spurious moves).
        if (
            self._active_stroke.gesture_id is not None
            and event.gesture_id != self._active_stroke.gesture_id
        ):
            return
        self._apply_brush(event.world_coordinate)

    def _on_mouse_release(self, event: CanvasMouseRelease2DEvent) -> None:
        if self._active_stroke is None:
            return
        self._apply_brush(event.world_coordinate)
        command = self._active_stroke.finalise()
        self._active_stroke = None
        self._history.push(command)
        self._on_stroke_completed(command)

    # ------------------------------------------------------------------
    # Brush application
    # ------------------------------------------------------------------

    def _apply_brush(self, world_coord: np.ndarray) -> None:
        """Convert a world coordinate to voxel indices and write the brush."""
        scene_model = self._controller._model.scenes[self._scene_id]
        visual_model = next(v for v in scene_model.visuals if v.id == self._visual_id)
        voxel_coord = visual_model.transform.imap_coordinates(
            np.atleast_2d(world_coord)
        )[0]
        voxel_center = np.round(voxel_coord).astype(np.int64)
        voxel_indices = self._voxels_in_radius(voxel_center, self._brush_radius)
        # The brush is a ball of voxels round the pick and would reach
        # across a clipping plane: write only the voxels that are drawn
        # (clipping planes design D26).
        planes = plane_tuples(visual_model.clipping_planes)
        if planes and voxel_indices.shape[0]:
            voxel_indices = voxel_indices[
                kept_points(voxel_indices.astype(np.float64), planes)
            ]
        if voxel_indices.shape[0] == 0:
            return
        old_values = self._read_old_values(voxel_indices)
        new_values = np.full(
            voxel_indices.shape[0], self._brush_value, dtype=self._store_dtype
        )
        self._write_values(voxel_indices, new_values)
        if self._active_stroke is not None:
            self._active_stroke.record(voxel_indices, old_values, new_values)

    def _voxels_in_radius(self, center: np.ndarray, radius: float) -> np.ndarray:
        """Return integer voxel indices within *radius* of *center*.

        Returns shape ``(N, ndim)`` int64 array in data-array axis order.
        Only returns indices that fall within ``[0, shape)`` along each
        axis.
        """
        ndim = len(center)
        r = int(np.ceil(radius))
        axis_indices: list[np.ndarray] = []
        for i in range(ndim):
            c = int(center[i])
            lo = max(0, c - r)
            hi = min(self._data_shape[i], c + r + 1)
            axis_indices.append(np.arange(lo, hi, dtype=np.int64))
        if any(len(a) == 0 for a in axis_indices):
            return np.zeros((0, ndim), dtype=np.int64)
        grids = np.meshgrid(*axis_indices, indexing="ij")
        candidates = np.stack([g.ravel() for g in grids], axis=1)
        distances = np.linalg.norm(
            candidates.astype(np.float64) - center.astype(np.float64), axis=1
        )
        return candidates[distances <= radius]

    # ------------------------------------------------------------------
    # Undo / redo (shared)
    # ------------------------------------------------------------------

    def undo(self) -> None:
        """Undo the most recent stroke."""
        command = self._history.undo()
        if command is None:
            return
        self._write_values(command.voxel_indices, command.old_values)

    def redo(self) -> None:
        """Redo the most recently undone stroke."""
        command = self._history.redo()
        if command is None:
            return
        self._write_values(command.voxel_indices, command.new_values)

    # ------------------------------------------------------------------
    # Teardown (shared)
    # ------------------------------------------------------------------

    def _teardown(self) -> None:
        """Unsubscribe bus events and re-enable the camera controller."""
        self._controller._outgoing_events.unsubscribe_all(self._id)
        self._controller.set_camera_controller_enabled(self._canvas_id, True)

    # ------------------------------------------------------------------
    # Abstract storage hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _read_old_values(self, voxel_indices: np.ndarray) -> np.ndarray:
        """Return pre-paint values for each voxel in the store's native dtype."""
        ...

    @abstractmethod
    def _write_values(self, voxel_indices: np.ndarray, values: np.ndarray) -> None:
        """Write *values* to the backing store; synchronous and IO-free."""
        ...

    @abstractmethod
    def _on_stroke_completed(self, command: PaintStrokeCommand) -> None:
        """Called after each stroke is finalised and pushed to history."""
        ...

    @abstractmethod
    def commit(self) -> None:
        """End the session, persisting all painted data."""
        ...

    @abstractmethod
    def abort(self) -> None:
        """End the session, reverting all painted data."""
        ...
