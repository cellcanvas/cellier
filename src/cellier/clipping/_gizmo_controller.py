"""Drag a gizmo in a 3D canvas to move one clipping plane of one visual.

The canvas draws the gizmo and reports its pose in rendered space
(``PlaneGizmoMovedEvent``).  ``ClippingPlaneGizmoController`` turns a pose
into the visual's data coordinates and assigns the plane, and moves the
gizmo when the plane is changed from anywhere else (clipping plane gizmo
design v2, 5.2 to 5.5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import numpy as np

from cellier.events import (
    ClippingPlanesChangedEvent,
    DimsChangedEvent,
    PlaneGizmoMovedEvent,
    TransformChangedEvent,
    VisualRemovedEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cellier.controller import CellierController


def handle_changes_plane(kind: str, axis: int | Sequence[int]) -> bool:
    """Whether a drag on a gizmo handle can change the plane.

    In the gizmo's frame axis 0 is the plane's normal.  Sliding the gizmo
    within the plane, or spinning it about the normal, moves the gizmo and
    leaves the plane where it is.

    The answer is taken from the handle and not from comparing planes: such
    a drag leaves the converted plane unequal by float noise.

    Parameters
    ----------
    kind : str
        ``"translate"`` or ``"rotate"``.
    axis : int or Sequence[int]
        The handle's axis, or the pair of axes of a two-axis translate
        handle.

    Returns
    -------
    bool
    """
    if kind == "translate":
        return axis == 0 if isinstance(axis, int) else 0 in tuple(axis)
    if kind == "rotate":
        return axis != 0
    return False


def _closest_on_plane(point: Any, normal: np.ndarray, offset: float) -> np.ndarray:
    """The point of the plane ``normal . r == offset`` nearest *point*."""
    point = np.asarray(point, dtype=np.float64)
    return point - (float(normal @ point) - offset) / float(normal @ normal) * normal


def initial_anchor(
    normal: Sequence[float],
    offset: float,
    orbit_point: Sequence[float] | None,
    bounds: Any | None,
) -> np.ndarray:
    """Where a gizmo is first put on a plane, in rendered space.

    The point of the plane nearest the point the camera orbits about, so a
    gizmo is on screen for a camera zoomed in and panned.  If that is
    outside the visual's bounding box, the point of the plane nearest the
    box's centre.

    Parameters
    ----------
    normal : Sequence[float]
        The plane's normal, any length but zero.
    offset : float
        The plane is ``normal . r == offset``.
    orbit_point : Sequence[float] or None
        The point the camera orbits about; ``None`` if there is none.
    bounds : array-like or None
        ``[[low x, low y, low z], [high x, high y, high z]]`` of the
        visual; ``None`` if it has no bounds yet.

    Returns
    -------
    np.ndarray
        A point on the plane.
    """
    normal = np.asarray(normal, dtype=np.float64)
    box = None if bounds is None else np.asarray(bounds, dtype=np.float64)
    if orbit_point is not None:
        anchor = _closest_on_plane(orbit_point, normal, offset)
        if box is None or bool(np.all((anchor >= box[0]) & (anchor <= box[1]))):
            return anchor
    if box is not None:
        return _closest_on_plane(box.mean(axis=0), normal, offset)
    return _closest_on_plane(np.zeros(3), normal, offset)


class ClippingPlaneGizmoController:
    """One gizmo on one clipping plane of one visual, in one 3D canvas.

    Build one with ``CellierController.add_clipping_plane_gizmo``; end it
    with :meth:`close`.  A canvas holds one at a time.

    - Dragging the arrow along the normal, or a rotation ring that tilts
      the normal, assigns the plane, once per frame.  Sliding the gizmo
      within the plane or spinning it about the normal moves the gizmo
      only.
    - A plane changed from anywhere else (a slider, an assignment) moves
      the gizmo.
    - A drag is one clipping interaction on the visual
      (``PlaneInteractionEvent``).
    - It closes itself when its plane or its visual is removed, when the
      canvas leaves 3D, and when the plane gains a component on an axis the
      view does not show.

    The camera controller is left alone: the gizmo takes only the drags
    that start on a handle.

    Parameters
    ----------
    cellier_controller : CellierController
        The controller the visual and the canvas belong to.
    visual_id : UUID
        The visual whose plane is edited.
    canvas_id : UUID
        A 3D canvas of the visual's scene.
    plane_id : UUID
        The ``id`` of the ``ClippingPlane`` to edit.
    screen_size : float
        The gizmo's size on screen, in logical pixels.
    on_closed : Callable[[ClippingPlaneGizmoController], None] or None
        Called once, when the session has closed.

    Raises
    ------
    KeyError
        If the canvas or the visual is not registered, or the visual has no
        plane with that id.
    ValueError
        If the canvas is in 2D, the visual is not in the canvas's scene, or
        the plane has a component on an axis the view does not show.
    """

    def __init__(
        self,
        cellier_controller: CellierController,
        visual_id: UUID,
        canvas_id: UUID,
        plane_id: UUID,
        *,
        screen_size: float = 100.0,
        on_closed: Callable[[ClippingPlaneGizmoController], None] | None = None,
    ) -> None:
        self._id: UUID = uuid4()
        self._controller = cellier_controller
        self._visual_id = visual_id
        self._canvas_id = canvas_id
        self._plane_id = UUID(str(plane_id))
        self._on_closed = on_closed
        self._closed = False
        self._dragging = False
        self._changes_plane = False

        scene_id = cellier_controller._canvas_to_scene[canvas_id]
        if cellier_controller._visual_to_scene[visual_id] != scene_id:
            raise ValueError(
                f"Visual {visual_id} is not in the scene canvas {canvas_id} shows."
            )
        self._scene_id = scene_id
        render_manager = cellier_controller._render_manager
        if render_manager.canvas_dim(canvas_id) != "3d":
            raise ValueError(
                "A clipping plane gizmo needs a 3D canvas; canvas "
                f"{canvas_id} is in 2D."
            )
        item = cellier_controller.get_clipping_plane(visual_id, self._plane_id)
        self._refuse_hidden_component(item)

        normal, offset = self._rendered_plane(item)
        # The gizmo's state the model does not hold: where on the plane it
        # sits.  (Its spin about the normal is the render gizmo's.)
        self._anchor = initial_anchor(
            normal,
            offset,
            render_manager.canvas_orbit_point(canvas_id),
            render_manager.visual_rendered_bounds(visual_id),
        )
        self._plane = item.plane
        render_manager.add_plane_gizmo(
            canvas_id, self._id, self._anchor, normal, screen_size=screen_size
        )

        bus = cellier_controller._outgoing_events
        for event_type, handler, entity_id in (
            (PlaneGizmoMovedEvent, self._on_gizmo, self._id),
            (ClippingPlanesChangedEvent, self._on_planes_changed, visual_id),
            (TransformChangedEvent, self._on_placement_changed, visual_id),
            (DimsChangedEvent, self._on_placement_changed, scene_id),
            (VisualRemovedEvent, self._on_visual_removed, scene_id),
        ):
            bus.subscribe(event_type, handler, entity_id=entity_id, owner_id=self._id)

    # -- public --------------------------------------------------------------

    @property
    def id(self) -> UUID:
        """The session's id: the ``source_id`` of the changes it makes."""
        return self._id

    @property
    def visual_id(self) -> UUID:
        """The visual whose plane is edited."""
        return self._visual_id

    @property
    def canvas_id(self) -> UUID:
        """The canvas the gizmo is drawn in."""
        return self._canvas_id

    @property
    def plane_id(self) -> UUID:
        """The ``id`` of the plane edited."""
        return self._plane_id

    @property
    def closed(self) -> bool:
        """Whether the session has ended."""
        return self._closed

    @property
    def dragging(self) -> bool:
        """Whether a handle is held."""
        return self._dragging

    @property
    def anchor(self) -> tuple[float, float, float]:
        """Where the gizmo sits on the plane, in rendered ``(x, y, z)``."""
        return tuple(float(v) for v in self._anchor)

    def close(self) -> None:
        """End the session: remove the gizmo and end a drag in progress.

        Safe to call more than once.
        """
        if self._closed:
            return
        self._closed = True
        controller = self._controller
        controller._outgoing_events.unsubscribe_all(self._id)
        self._end_scope()
        controller._render_manager.remove_plane_gizmo(self._canvas_id, self._id)
        if self._on_closed is not None:
            self._on_closed(self)

    # -- the gizmo -> the model ----------------------------------------------

    def _on_gizmo(self, event: PlaneGizmoMovedEvent) -> None:
        if event.phase == "start":
            # A release that never came leaves a scope open: end it first.
            self._end_scope()
            self._dragging = True
            self._changes_plane = handle_changes_plane(
                event.handle_kind, event.handle_axis
            )
            self._controller.begin_plane_interaction(
                self._visual_id, source_id=self._id
            )
        elif event.phase == "move":
            self._anchor = np.asarray(event.point, dtype=np.float64)
            if self._changes_plane:
                self._assign(event.point, event.normal)
        elif event.phase == "end":
            self._end_scope()

    def _assign(self, point: Any, normal: Any) -> None:
        from cellier.transform import Plane

        controller = self._controller
        data_normal, offset = controller._render_manager.expand_rendered_plane(
            self._visual_id, point, normal
        )
        plane = Plane(
            coordinate_system=self._plane.coordinate_system,
            normal=data_normal,
            offset=offset,
        )
        self._plane = plane
        controller.set_clipping_plane(
            self._visual_id, self._plane_id, plane, source_id=self._id
        )

    def _end_scope(self) -> None:
        self._dragging = False
        self._controller.end_plane_interaction(self._visual_id, source_id=self._id)

    # -- the model -> the gizmo ----------------------------------------------

    def _on_planes_changed(self, event: ClippingPlanesChangedEvent) -> None:
        if event.source_id == self._id:
            return  # this session's own move: the gizmo is already there
        item = next(
            (item for item in event.clipping_planes if item.id == self._plane_id),
            None,
        )
        if item is None:
            self.close()
        elif item.plane != self._plane:
            self._follow(item)

    def _on_placement_changed(self, _event: Any) -> None:
        """The view or the visual's transform changed: the plane moved on screen."""
        try:
            item = self._controller.get_clipping_plane(self._visual_id, self._plane_id)
        except KeyError:
            self.close()
            return
        self._follow(item)

    def _on_visual_removed(self, event: VisualRemovedEvent) -> None:
        if event.visual_id == self._visual_id:
            self.close()

    def _follow(self, item: Any) -> None:
        """Put the gizmo on *item*, at the point nearest where it was."""
        render_manager = self._controller._render_manager
        try:
            if render_manager.canvas_dim(self._canvas_id) != "3d":
                raise ValueError("the canvas left 3D")
            self._refuse_hidden_component(item)
            normal, offset = self._rendered_plane(item)
        except (KeyError, ValueError):
            self.close()
            return
        self._plane = item.plane
        self._anchor = _closest_on_plane(self._anchor, normal, offset)
        render_manager.set_plane_gizmo_pose(
            self._canvas_id, self._id, self._anchor, normal
        )

    # -- helpers -------------------------------------------------------------

    def _rendered_plane(self, item: Any) -> tuple[np.ndarray, float]:
        normal, offset = self._controller._render_manager.reduce_clipping_plane(
            self._visual_id, item
        )
        return np.asarray(normal, dtype=np.float64), float(offset)

    def _refuse_hidden_component(self, item: Any) -> None:
        hidden = self._controller._render_manager.clipping_plane_hidden_axes(
            self._visual_id, item
        )
        if hidden:
            raise ValueError(
                f"Clipping plane {self._plane_id} has a component on data "
                f"axes {hidden}, which the 3D view does not show; a gizmo "
                "cannot move it."
            )
