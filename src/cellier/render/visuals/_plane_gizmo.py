"""The render side of a plane gizmo (clipping plane gizmo design v2, 5.1).

A pygfx ``TransformGizmo`` on a proxy object whose position is a point on a
plane and whose local ``+x`` is the plane's normal.  ``CanvasView`` owns it,
draws it as its own pass and turns what it reports into
``PlaneGizmoMovedEvent``.  Nothing here knows about data coordinates or
clipping: the pose is in the scene's rendered space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx as gfx
import pylinalg as la

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from uuid import UUID

#: The mouse button a gizmo drag is made with (pygfx numbering).
_LEFT = 1
#: A drag under this many pixels does not move the object (pygfx's own).
_DEAD_ZONE_PX = 3


class PlaneTransformGizmo(gfx.TransformGizmo):
    """A ``TransformGizmo`` cut down to what a plane has.

    - Every element is drawn with ``depth_test=False``: inside a volume the
      volume would otherwise win the pick and the handles could not be
      grabbed.
    - The centre sphere is hidden: in pygfx a drag on it is a uniform scale
      and a click on it changes the frame.
    - The scale handles are hidden except those of ``scale_axes``.  A
      clipping plane has no size; a render plane has one along each
      in-plane axis that is bounded on both sides (plane rendering design
      v3, 8.3).  Axis 0, the normal, never has one.
    - The frame stays ``"object"``: axis 0 is the normal.
    - The handles point along the object's own axes, whatever the camera
      angle: the arrow of axis 0 is the plane's normal (see
      :meth:`_update_gizmo_transform`).  So a scale drag along a handle's
      arrow always grows the object.
    - It tells its owner when a handle is grabbed, moved and released.

    Relies on private pygfx names: ``_ref``, ``_camera``, ``_viewport``,
    ``_update_visibility``, ``_update_gizmo_transform``, ``_highlight``,
    ``_center_sphere``, ``_scale_children``, ``_object_to_control`` and
    ``gizmo_scale``.

    Parameters
    ----------
    target : gfx.WorldObject
        The object the gizmo moves.
    owner : GFXPlaneGizmo
        Told about grabs, moves and releases.
    screen_size : float
        The gizmo's size on screen, in logical pixels.
    """

    def __init__(
        self, target: gfx.WorldObject, owner: GFXPlaneGizmo, screen_size: float
    ) -> None:
        #: The axes (1 and 2, the in-plane ones) whose scale handle is shown.
        self.scale_axes: frozenset[int] = frozenset()
        super().__init__(target, screen_size=screen_size)
        self._owner = owner
        self.toggle_mode("object")
        for element in self.children:
            element.material.depth_test = False
        # With no depth test the element drawn last is the one seen and the
        # one picked.  A handle that points away from the camera can be
        # behind an in-plane translate square on screen (pygfx turns its
        # handles to the camera, so it never meets this): the one-axis
        # handles are drawn after everything else, so they can be grabbed.
        for element in (*self._translate1_children, *self._scale_children):
            element.render_order = 1

    def _update_visibility(self) -> None:
        super()._update_visibility()
        self._center_sphere.visible = False
        for dim, element in enumerate(self._scale_children):
            if dim not in self.scale_axes:
                element.visible = False

    def _update_gizmo_transform(self) -> None:
        """Place the gizmo as pygfx does, without turning it to the camera.

        pygfx flips every handle that would point away from the camera, so
        an arrow shows where the viewer is and not which way its axis
        goes: it does not turn over when the plane is flipped, and it does
        when the camera passes the plane.  Here the flip is dropped and
        the size on screen kept.

        The rotation is written again, before the scale: a flip that was
        written to the node's matrix does not come back out of it as a
        scale, so a positive scale written over it would leave the gizmo
        turned.
        """
        super()._update_gizmo_transform()
        np.abs(self.gizmo_scale, out=self.gizmo_scale)
        self.world.scale = 1
        self.world.rotation = self._object_to_control.world.rotation
        self.world.scale = self.gizmo_scale

    def process_event(self, event: Any) -> None:
        """Handle the event as pygfx does, then report what it did."""
        was_dragging = self._ref is not None
        super().process_event(event)
        if event.type == "pointer_down":
            if self._ref is not None:
                self._owner._grabbed(self._ref["kind"], self._ref["dim"])
        elif event.type == "pointer_move":
            if self._ref is not None and self._ref["maxdist"] >= _DEAD_ZONE_PX:
                self._owner._moved()
        elif event.type == "pointer_up" and was_dragging:
            self._owner._released()

    @property
    def dragging(self) -> bool:
        """Whether a handle is held."""
        return self._ref is not None

    def cancel_drag(self) -> bool:
        """End a drag whose release did not come.

        Clears the drag state, removes the highlight and gives up the
        pointer capture.  Clearing the state matters as much as the
        capture: pygfx drags on every move while the state is set, so with
        only the capture dropped a hover over a handle would move the
        object with no button held.

        Returns
        -------
        bool
            Whether there was a drag to end.
        """
        was_dragging = self._ref is not None
        self._ref = None
        self._highlight()
        captures = gfx.objects.EventTarget.pointer_captures
        for pointer_id, (target, _root) in list(captures.items()):
            if target() is self:
                captures.pop(pointer_id, None)
        return was_dragging


def _axis(dim: Any) -> int | tuple[int, int]:
    """A pygfx handle ``dim`` as plain Python: an axis or a pair of axes."""
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    first, second = dim
    return int(first), int(second)


class GFXPlaneGizmo:
    """One plane gizmo of one canvas.

    Parameters
    ----------
    gizmo_id : UUID
        Names the gizmo on the events its canvas emits.
    renderer : gfx.WgpuRenderer
        The canvas's renderer; the gizmo resizes itself before each of its
        frames.
    camera : gfx.Camera
        The canvas's 3D camera.
    on_grab, on_move, on_release : Callable[[GFXPlaneGizmo], None]
        Called when a handle is grabbed, on each pointer move that moves
        the proxy, and when the drag ends (a release, or a cancelled drag).
    screen_size : float
        The gizmo's size on screen, in logical pixels.
    """

    def __init__(
        self,
        gizmo_id: UUID,
        renderer: gfx.WgpuRenderer,
        camera: gfx.Camera,
        on_grab: Callable[[GFXPlaneGizmo], None],
        on_move: Callable[[GFXPlaneGizmo], None],
        on_release: Callable[[GFXPlaneGizmo], None],
        screen_size: float = 100.0,
    ) -> None:
        self.gizmo_id = gizmo_id
        self._renderer = renderer
        self._on_grab, self._on_move, self._on_release = on_grab, on_move, on_release
        self._visible = True
        #: The handle of the drag in progress, or of the last one.
        self.handle_kind: str = "translate"
        self.handle_axis: int | tuple[int, int] = 0

        self.proxy = gfx.Group()
        self.gizmo = PlaneTransformGizmo(self.proxy, self, screen_size)
        self.gizmo.add_default_event_handlers(renderer, camera)
        #: What the canvas renders as the gizmo's pass.
        self.scene = gfx.Scene()
        self.scene.add(self.proxy)
        self.scene.add(self.gizmo)

    # -- pose ----------------------------------------------------------------

    def pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The proxy's point and unit normal, in rendered ``(x, y, z)``."""
        point = self.proxy.world.position
        normal = la.vec_transform_quat((1.0, 0.0, 0.0), self.proxy.world.rotation)
        normal = normal / np.linalg.norm(normal)
        return (
            (float(point[0]), float(point[1]), float(point[2])),
            (float(normal[0]), float(normal[1]), float(normal[2])),
        )

    def frame(
        self,
    ) -> tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float]
    ]:
        """The proxy's in-plane axes and its scale along them.

        Returns
        -------
        tuple
            ``(in_plane_axis_0, in_plane_axis_1, scale)``: the proxy's local
            ``+y`` and ``+z`` as unit vectors in rendered ``(x, y, z)``, and
            the size of its scale along each (1.0 unless a scale handle is
            held).
        """
        rotation = self.proxy.world.rotation
        axes = []
        for local in ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            axis = la.vec_transform_quat(local, rotation)
            axis = axis / np.linalg.norm(axis)
            axes.append((float(axis[0]), float(axis[1]), float(axis[2])))
        scale = self.proxy.local.scale
        return axes[0], axes[1], (abs(float(scale[1])), abs(float(scale[2])))

    def set_frame(self, point: Any, in_plane_axis_0: Any, in_plane_axis_1: Any) -> None:
        """Place the proxy at *point* with its whole frame given.

        The proxy's local ``+y`` and ``+z`` become the two in-plane axes
        and its ``+x`` their cross product, the normal.

        Parameters
        ----------
        point : array-like
            A point on the plane, in rendered ``(x, y, z)``.
        in_plane_axis_0, in_plane_axis_1 : array-like
            Two orthogonal directions in the plane.  Any length but zero.

        Raises
        ------
        ValueError
            If the axes are zero or parallel.
        """
        axis_0 = np.asarray(in_plane_axis_0, dtype=np.float64)
        axis_1 = np.asarray(in_plane_axis_1, dtype=np.float64)
        normal = np.cross(axis_0, axis_1)
        if float(np.linalg.norm(normal)) == 0.0:
            raise ValueError("A plane's in-plane axes must not be zero or parallel.")
        normal = normal / np.linalg.norm(normal)
        axis_0 = axis_0 / np.linalg.norm(axis_0)
        # Exactly orthogonal, so the matrix is a rotation.
        axis_1 = np.cross(normal, axis_0)
        matrix = np.eye(4)
        matrix[:3, 0], matrix[:3, 1], matrix[:3, 2] = normal, axis_0, axis_1
        self.proxy.local.rotation = la.quat_from_mat(matrix)
        self.proxy.local.position = tuple(float(v) for v in point)

    def set_scale_axes(self, axes: Any) -> None:
        """Show the scale handles of the in-plane *axes* (1, 2) and no other."""
        self.gizmo.scale_axes = frozenset(int(a) for a in axes if int(a) in (1, 2))

    def reset_scale(self) -> None:
        """Put the proxy's scale back to 1 (a scale drag was folded in)."""
        self.proxy.local.scale = (1.0, 1.0, 1.0)

    def set_pose(self, point: Any, normal: Any) -> None:
        """Place the proxy at *point*, its normal along *normal*.

        The proxy is turned by the smallest rotation that takes its normal
        to the new one, so its spin about the normal is kept.  A flipped
        normal is a half turn about an in-plane axis.

        Parameters
        ----------
        point : array-like
            A point on the plane, in rendered ``(x, y, z)``.
        normal : array-like
            The plane's normal there.  Any length but zero.

        Raises
        ------
        ValueError
            If *normal* is zero.
        """
        target = np.asarray(normal, dtype=np.float64)
        length = float(np.linalg.norm(target))
        if length == 0.0:
            raise ValueError("A plane's normal must not be all zero.")
        target = target / length
        rotation = self.proxy.local.rotation
        current = la.vec_transform_quat((1.0, 0.0, 0.0), rotation)
        cosine = float(np.clip(current @ target, -1.0, 1.0))
        if cosine < -1.0 + 1e-9:
            in_plane = la.vec_transform_quat((0.0, 1.0, 0.0), rotation)
            turn = la.quat_from_axis_angle(in_plane, np.pi)
        else:
            turn = la.quat_from_vecs(current, target)
        self.proxy.local.rotation = la.quat_mul(turn, rotation)
        self.proxy.local.position = tuple(float(v) for v in point)

    # -- state ---------------------------------------------------------------

    @property
    def visible(self) -> bool:
        """Whether the canvas draws the gizmo."""
        return self._visible

    def set_visible(self, visible: bool) -> None:
        """Show or hide the gizmo.  Hiding it ends a drag in progress."""
        visible = bool(visible)
        if visible == self._visible:
            return
        self._visible = visible
        if not visible:
            self.cancel_drag()
        self.gizmo.set_object(self.proxy if visible else None)

    @property
    def dragging(self) -> bool:
        """Whether a handle is held."""
        return self.gizmo.dragging

    def cancel_drag(self) -> bool:
        """End a drag whose release did not come; report it as released.

        Returns
        -------
        bool
            Whether there was a drag to end.
        """
        was_dragging = self.gizmo.cancel_drag()
        if was_dragging:
            self._on_release(self)
        return was_dragging

    def object_ids(self) -> set[int]:
        """The pygfx ids the gizmo's elements can leave in the pick buffer."""
        return {int(element.id) for element in self.gizmo.children}

    def close(self) -> None:
        """End a drag and stop following the renderer's frames."""
        self.gizmo.cancel_drag()
        self._renderer.remove_event_handler(self.gizmo.update_gizmo, "before_render")
        self.gizmo.set_object(None)

    # -- from the gizmo ------------------------------------------------------

    def _grabbed(self, kind: str, dim: Any) -> None:
        self.handle_kind, self.handle_axis = str(kind), _axis(dim)
        self._on_grab(self)

    def _moved(self) -> None:
        self._on_move(self)

    def _released(self) -> None:
        self._on_release(self)


def drop_stale_gizmo_capture(event: Mapping[str, Any]) -> bool:
    """Drop a pointer capture a plane gizmo kept past its release.

    For a ``pointer_down`` as the canvas delivers it, ahead of the renderer.
    pygfx keeps captures in one table for every canvas, so a capture whose
    ``pointer_up`` never came makes every canvas deaf to that pointer.

    The capture is stale when the left button is not already held: a left
    press always qualifies, and a press of another button does when the
    left one is not among ``buttons``.  A press of another button during a
    live left drag is left alone.

    Parameters
    ----------
    event : Mapping[str, Any]
        The canvas's ``pointer_down`` event.

    Returns
    -------
    bool
        Whether a capture was dropped.
    """
    held = gfx.objects.EventTarget.pointer_captures.get(event.get("pointer_id"))
    if held is None:
        return False
    target = held[0]()
    if not isinstance(target, PlaneTransformGizmo):
        return False
    if event.get("button") != _LEFT and _LEFT in event.get("buttons", ()):
        return False
    target._owner.cancel_drag()
    # With no drag in progress there was still a capture to give up.
    target.cancel_drag()
    return True
