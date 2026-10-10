"""Drag a gizmo in a 3D canvas to move one plane of one visual.

The canvas draws the gizmo and reports its pose in rendered space
(``PlaneGizmoMovedEvent``).  A session turns a pose into the plane's own
coordinates and assigns it, and moves the gizmo when the plane is changed
from anywhere else.  ``PlaneGizmoController`` is the session; its two
kinds differ only in the plane they edit:

- ``ClippingPlaneGizmoController``: a ``ClippingPlane``, in the visual's
  data coordinates (clipping plane gizmo design v2, 5.2 to 5.5).
- ``RenderPlaneGizmoController``: a ``RenderPlane``, in world coordinates,
  with its whole frame and its extents (plane rendering design v3, 8.3).

A canvas has one gizmo, of either kind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal
from uuid import UUID, uuid4

import numpy as np

from cellier.events import (
    ClippingPlanesChangedEvent,
    DimsChangedEvent,
    PlaneGizmoMovedEvent,
    RenderPlanesChangedEvent,
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


class PlaneGizmoController:
    """One gizmo on one plane of one visual, in one 3D canvas.

    The part of a session that does not depend on the kind of plane: the
    drag's interaction scope, the subscriptions and closing.  Build one
    with ``CellierController.add_clipping_plane_gizmo`` or
    ``add_render_plane_gizmo``; end it with :meth:`close`.  A canvas holds
    one at a time.

    A drag is one plane interaction on the visual
    (``PlaneInteractionEvent``).  The camera controller is left alone: the
    gizmo takes only the drags that start on a handle.

    Parameters
    ----------
    cellier_controller : CellierController
        The controller the visual and the canvas belong to.
    visual_id : UUID
        The visual whose plane is edited.
    canvas_id : UUID
        A 3D canvas of the visual's scene.
    plane_id : UUID
        The ``id`` of the plane to edit.
    screen_size : float
        The gizmo's size on screen, in logical pixels.
    on_closed : Callable[[PlaneGizmoController], None] or None
        Called once, when the session has closed.

    Raises
    ------
    KeyError
        If the canvas or the visual is not registered, or the visual has no
        plane with that id.
    ValueError
        If the canvas is in 2D, the visual is not in the canvas's scene, or
        the plane cannot have a gizmo in this view.
    """

    #: Which of the visual's tuples the plane is in.
    kind: ClassVar[Literal["clipping", "render"]]

    def __init__(
        self,
        cellier_controller: CellierController,
        visual_id: UUID,
        canvas_id: UUID,
        plane_id: UUID,
        *,
        screen_size: float = 100.0,
        on_closed: Callable[[Any], None] | None = None,
    ) -> None:
        self._id: UUID = uuid4()
        self._controller = cellier_controller
        self._visual_id = visual_id
        self._canvas_id = canvas_id
        self._plane_id = UUID(str(plane_id))
        self._on_closed = on_closed
        self._closed = False
        self._dragging = False

        scene_id = cellier_controller._canvas_to_scene[canvas_id]
        if cellier_controller._visual_to_scene[visual_id] != scene_id:
            raise ValueError(
                f"Visual {visual_id} is not in the scene canvas {canvas_id} shows."
            )
        self._scene_id = scene_id
        if cellier_controller._render_manager.canvas_dim(canvas_id) != "3d":
            raise ValueError(
                f"A {self.kind} plane gizmo needs a 3D canvas; canvas "
                f"{canvas_id} is in 2D."
            )
        self._open(screen_size)

        bus = cellier_controller._outgoing_events
        for event_type, handler, entity_id in (
            (PlaneGizmoMovedEvent, self._on_gizmo, self._id),
            (DimsChangedEvent, self._on_placement_changed, scene_id),
            (VisualRemovedEvent, self._on_visual_removed, scene_id),
            *self._subscriptions(),
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

    # -- for the kinds -------------------------------------------------------

    def _open(self, screen_size: float) -> None:
        """Check the plane and draw the gizmo on it."""
        raise NotImplementedError

    def _subscriptions(self) -> list[tuple[Any, Any, UUID]]:
        """``(event type, handler, entity id)`` this kind also follows."""
        return []

    def _grabbed(self, event: PlaneGizmoMovedEvent) -> None:
        """A handle was grabbed."""

    def _moved(self, event: PlaneGizmoMovedEvent) -> None:
        """The gizmo moved: assign the plane it describes."""
        raise NotImplementedError

    def _released(self, event: PlaneGizmoMovedEvent) -> None:
        """The handle was released (after the last move)."""

    def _on_placement_changed(self, _event: Any) -> None:
        """The view changed: the plane may have moved on screen."""
        raise NotImplementedError

    # -- the gizmo -> the model ----------------------------------------------

    def _on_gizmo(self, event: PlaneGizmoMovedEvent) -> None:
        if event.phase == "start":
            # A release that never came leaves a scope open: end it first.
            self._end_scope()
            self._dragging = True
            self._grabbed(event)
            self._controller.begin_plane_interaction(
                self._visual_id, source_id=self._id
            )
        elif event.phase == "move":
            self._moved(event)
        elif event.phase == "end":
            self._released(event)
            self._end_scope()

    def _end_scope(self) -> None:
        self._dragging = False
        self._controller.end_plane_interaction(self._visual_id, source_id=self._id)

    def _on_visual_removed(self, event: VisualRemovedEvent) -> None:
        if event.visual_id == self._visual_id:
            self.close()


class ClippingPlaneGizmoController(PlaneGizmoController):
    """One gizmo on one clipping plane of one visual, in one 3D canvas.

    Build one with ``CellierController.add_clipping_plane_gizmo``; end it
    with :meth:`close`.  A canvas holds one gizmo at a time.

    - Dragging the arrow along the normal, or a rotation ring that tilts
      the normal, assigns the plane, once per frame.  Sliding the gizmo
      within the plane or spinning it about the normal moves the gizmo
      only.
    - A plane changed from anywhere else (a slider, an assignment) moves
      the gizmo.
    - A drag is one plane interaction on the visual
      (``PlaneInteractionEvent``).
    - It closes itself when its plane or its visual is removed, when the
      canvas leaves 3D, and when the plane gains a component on an axis the
      view does not show.

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

    kind = "clipping"

    def _open(self, screen_size: float) -> None:
        controller = self._controller
        render_manager = controller._render_manager
        self._changes_plane = False
        item = controller.get_clipping_plane(self._visual_id, self._plane_id)
        self._refuse_hidden_component(item)

        normal, offset = self._rendered_plane(item)
        # The gizmo's state the model does not hold: where on the plane it
        # sits.  (Its spin about the normal is the render gizmo's.)
        self._anchor = initial_anchor(
            normal,
            offset,
            render_manager.canvas_orbit_point(self._canvas_id),
            render_manager.visual_rendered_bounds(self._visual_id),
        )
        self._plane = item.plane
        render_manager.add_plane_gizmo(
            self._canvas_id, self._id, self._anchor, normal, screen_size=screen_size
        )

    def _subscriptions(self) -> list[tuple[Any, Any, UUID]]:
        return [
            (ClippingPlanesChangedEvent, self._on_planes_changed, self._visual_id),
            (TransformChangedEvent, self._on_placement_changed, self._visual_id),
        ]

    @property
    def anchor(self) -> tuple[float, float, float]:
        """Where the gizmo sits on the plane, in rendered ``(x, y, z)``."""
        return tuple(float(v) for v in self._anchor)

    # -- the gizmo -> the model ----------------------------------------------

    def _grabbed(self, event: PlaneGizmoMovedEvent) -> None:
        self._changes_plane = handle_changes_plane(event.handle_kind, event.handle_axis)

    def _moved(self, event: PlaneGizmoMovedEvent) -> None:
        self._anchor = np.asarray(event.point, dtype=np.float64)
        if self._changes_plane:
            self._assign(event.point, event.normal)

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


#: A scale drag that would shrink an extent to under this factor is not
#: assigned: a bounded pair needs ``min < max``.
_MIN_SCALE_FACTOR = 1e-6


def scale_handle_axes(plane: Any) -> tuple[int, ...]:
    """The gizmo axes of a render plane that get a scale handle.

    Gizmo axis 1 is ``in_plane_axis_0`` and axis 2 is ``in_plane_axis_1``.
    An axis has a handle when both sides of its extent are finite; with an
    unbounded side there is no size to scale (plane rendering design v3,
    8.3).

    Parameters
    ----------
    plane : RenderPlane
        The plane the gizmo is on.

    Returns
    -------
    tuple[int, ...]
        A subset of ``(1, 2)``.
    """
    return tuple(
        axis
        for axis, extent in ((1, plane.extent_0), (2, plane.extent_1))
        if extent[0] is not None and extent[1] is not None
    )


def scaled_extent(
    extent: tuple[float | None, float | None], factor: float
) -> tuple[float | None, float | None]:
    """An extent multiplied by a scale handle's factor, about the origin.

    Both sides are multiplied, so the rectangle grows and shrinks about the
    plane's origin.  An extent with an unbounded side, and a factor too
    small to leave ``min < max``, are returned unchanged.
    """
    low, high = extent
    if low is None or high is None or not factor >= _MIN_SCALE_FACTOR:
        return extent
    return (float(low) * factor, float(high) * factor)


class RenderPlaneGizmoController(PlaneGizmoController):
    """One gizmo on one render plane of one visual, in one 3D canvas.

    Build one with ``CellierController.add_render_plane_gizmo``; end it
    with :meth:`close`.  A canvas holds one gizmo at a time.

    The gizmo carries the plane's whole pose: it sits at the plane's
    ``origin``, its axis 0 is the normal and its axes 1 and 2 are
    ``in_plane_axis_0`` and ``in_plane_axis_1``.  Every handle changes the
    model, once per frame:

    - a translate handle moves ``origin``, along the normal or within the
      plane;
    - a rotation ring turns the in-plane axes about ``origin``;
    - a scale handle on an in-plane axis multiplies that axis's extent,
      symmetrically about ``origin``.  On release the factor stays in the
      extent and the gizmo's scale goes back to 1.  An axis with an
      unbounded side has no scale handle.

    A plane changed from anywhere else moves the gizmo.  A drag is one
    plane interaction on the visual (``PlaneInteractionEvent``).  The
    session closes itself when its plane or its visual is removed, when
    the canvas leaves 3D, when the visual leaves the ``"plane"`` render
    mode, and when the plane's axes stop being the displayed ones.

    Parameters
    ----------
    cellier_controller : CellierController
        The controller the visual and the canvas belong to.
    visual_id : UUID
        The visual whose plane is edited.
    canvas_id : UUID
        A 3D canvas of the visual's scene.
    plane_id : UUID
        The ``id`` of the ``RenderPlane`` to edit.  It may be disabled.
    screen_size : float
        The gizmo's size on screen, in logical pixels.
    on_closed : Callable[[RenderPlaneGizmoController], None] or None
        Called once, when the session has closed.

    Raises
    ------
    KeyError
        If the canvas or the visual is not registered, or the visual has no
        render plane with that id.
    ValueError
        If the canvas is in 2D, the visual is not in the canvas's scene or
        not in the ``"plane"`` render mode, or the plane's axes are not the
        ones the view displays.
    """

    kind = "render"

    def _open(self, screen_size: float) -> None:
        controller = self._controller
        plane = controller.get_render_plane(self._visual_id, self._plane_id)
        if not controller._draws_planes(controller.get_visual_model(self._visual_id)):
            raise ValueError(
                f"Visual {self._visual_id} is not in the 'plane' render mode; "
                "a render plane gizmo edits a plane that is drawn."
            )
        origin, axis_0, axis_1 = controller._render_manager.reduce_render_plane(
            self._visual_id, plane
        )
        self._plane = plane
        # The extents a scale drag multiplies: the plane's at the grab.
        self._grab_extents = (plane.extent_0, plane.extent_1)
        self._handle_kind = "translate"
        self._handle_axis: Any = 0
        controller._render_manager.add_plane_gizmo(
            self._canvas_id,
            self._id,
            origin,
            np.cross(axis_0, axis_1),
            screen_size=screen_size,
            in_plane_axes=(axis_0, axis_1),
            scale_axes=scale_handle_axes(plane),
        )

    def _subscriptions(self) -> list[tuple[Any, Any, UUID]]:
        return [(RenderPlanesChangedEvent, self._on_planes_changed, self._visual_id)]

    # -- the gizmo -> the model ----------------------------------------------

    def _grabbed(self, event: PlaneGizmoMovedEvent) -> None:
        self._handle_kind = event.handle_kind
        self._handle_axis = event.handle_axis
        self._grab_extents = (self._plane.extent_0, self._plane.extent_1)

    def _moved(self, event: PlaneGizmoMovedEvent) -> None:
        """Assign the fields the held handle changes, and no other.

        The other fields are kept from the model as they are, so a
        translate leaves the in-plane axes bit for bit and a scale leaves
        the pose.
        """
        plane = self._plane
        update: dict[str, Any] = {}
        if self._handle_kind == "scale":
            # Only the grabbed axis: the gizmo's scale on the other is 1
            # to float noise, not exactly.
            if self._handle_axis == 1:
                update["extent_0"] = scaled_extent(
                    self._grab_extents[0], event.scale[0]
                )
            elif self._handle_axis == 2:
                update["extent_1"] = scaled_extent(
                    self._grab_extents[1], event.scale[1]
                )
        else:
            origin, axis_0, axis_1 = (
                self._controller._render_manager.expand_rendered_frame(
                    self._visual_id,
                    plane.axes,
                    event.point,
                    event.in_plane_axis_0,
                    event.in_plane_axis_1,
                )
            )
            if self._handle_kind == "rotate":
                axis_0 = axis_0 / np.linalg.norm(axis_0)
                axis_1 = axis_1 - float(axis_1 @ axis_0) * axis_0
                axis_1 = axis_1 / np.linalg.norm(axis_1)
                update["in_plane_axis_0"] = tuple(float(v) for v in axis_0)
                update["in_plane_axis_1"] = tuple(float(v) for v in axis_1)
            else:
                update["origin"] = tuple(float(v) for v in origin)
        self._assign(update)

    def _released(self, event: PlaneGizmoMovedEvent) -> None:
        """Fold a scale drag in: the factor is in the extent already."""
        self._grab_extents = (self._plane.extent_0, self._plane.extent_1)
        self._controller._render_manager.reset_plane_gizmo_scale(
            self._canvas_id, self._id
        )

    def _assign(self, update: dict[str, Any]) -> None:
        if "in_plane_axis_0" in update:
            # Validated: the model checks and normalises the axes.
            plane = type(self._plane)(**{**self._plane.model_dump(), **update})
        else:
            # A copy: the axes stay bit for bit under a translate or scale.
            plane = self._plane.model_copy(update=update)
        if plane == self._plane:
            return
        self._plane = self._controller.set_render_plane(
            self._visual_id, self._plane_id, plane, source_id=self._id
        )

    # -- the model -> the gizmo ----------------------------------------------

    def _on_planes_changed(self, event: RenderPlanesChangedEvent) -> None:
        if event.source_id == self._id:
            return  # this session's own move: the gizmo is already there
        plane = next(
            (plane for plane in event.render_planes if plane.id == self._plane_id),
            None,
        )
        if plane is None:
            self.close()
        elif plane != self._plane:
            self._follow(plane)

    def _on_placement_changed(self, _event: Any) -> None:
        """The displayed axes changed: the plane may not be drawn any more."""
        try:
            plane = self._controller.get_render_plane(self._visual_id, self._plane_id)
        except KeyError:
            self.close()
            return
        self._follow(plane)

    def _follow(self, plane: Any) -> None:
        """Put the gizmo on *plane*; close if the view no longer draws it."""
        controller = self._controller
        render_manager = controller._render_manager
        try:
            if render_manager.canvas_dim(self._canvas_id) != "3d":
                raise ValueError("the canvas left 3D")
            if not controller._draws_planes(
                controller.get_visual_model(self._visual_id)
            ):
                raise ValueError("the visual left plane mode")
            origin, axis_0, axis_1 = render_manager.reduce_render_plane(
                self._visual_id, plane
            )
        except (KeyError, ValueError):
            self.close()
            return
        self._plane = plane
        if not self._dragging:
            self._grab_extents = (plane.extent_0, plane.extent_1)
        render_manager.set_plane_gizmo_frame(
            self._canvas_id, self._id, origin, axis_0, axis_1
        )
        render_manager.set_plane_gizmo_scale_axes(
            self._canvas_id, self._id, scale_handle_axes(plane)
        )
