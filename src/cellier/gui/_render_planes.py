"""The "Render planes" control of a visual, decided once for both toolkits.

A visual in the ``"plane"`` render mode draws its data on a tuple of
:class:`~cellier.visuals.RenderPlane`, in the scene's world coordinates.
The Qt and anywidget controls (``QtRenderPlanesControls``,
``AnywidgetRenderPlanesControls``) draw one row per plane, as close to the
"Clipping planes" control as they can be (plane rendering design v3, 9.2):
an enabled checkbox, the normal (one entry per axis of the plane, under that
axis's name, with a button to face the plane along either direction of an
axis), a flip button, a position slider along the normal, a remove button,
and what a clipping plane has not: the extents, four fields with an
"unbounded" box each.  An add button appends a plane.

A row is plain data, so the anywidget control syncs the whole list in one
trait and constructs nothing when a plane is added::

    {
        "id": "0f6c...",
        "enabled": True,
        "axes": ["z", "y", "x"],
        "normal": [1.0, 0.0, 0.0],
        "position": 12.0,
        "extent_0": [None, None],
        "extent_1": [-20.0, 20.0],
    }

``normal`` and ``position`` are on the plane's own three world axes, named
by ``axes``; ``position`` is where the plane sits along its unit normal.
``extent_0`` and ``extent_1`` are ``[min, max]`` distances from the plane's
origin along its two in-plane axes, in world units; ``None`` is unbounded.

**The in-plane frame has no entry.**  A row does not hold the plane: the
editor keeps the planes the model last reported and applies each edit to
them.  A normal edit turns the frame about the plane's origin by the
smallest rotation and so keeps its spin; a position edit moves the origin
along the normal and keeps its in-plane coordinates.  The spin is set with
the gizmo's ring or from code.

Nothing here imports a toolkit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple
from uuid import UUID

import numpy as np

from cellier.events import (
    AppearanceChangedEvent,
    ChannelAppearanceChangedEvent,
    DataStoreMetadataChangedEvent,
    DimsChangedEvent,
    ImageCompositeChangedEvent,
    PlaneGizmoChangedEvent,
    PlaneGizmoUpdateEvent,
    RenderPlanesChangedEvent,
    RenderPlanesUpdateEvent,
    SingleAppearanceChangedEvent,
    SubscriptionSpec,
    TransformChangedEvent,
)
from cellier.gui._clipping_planes import facing_of, position_range
from cellier.gui._loading import error_message
from cellier.visuals._render_plane import MAX_RENDER_PLANES, RenderPlane

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

RENDER_PLANES_TITLE = "Render planes"
"""The control's name: its ``DEFAULT_TITLE`` on both toolkits."""

RENDER_PLANE_GIZMO_TOOLTIP = (
    "Drag a gizmo in the 3D view to move, turn and resize this plane."
)
"""The gizmo toggle's tooltip while it can be used."""

Row = dict[str, Any]
Bounds = list[list[float]]


# --------------------------------------------------------------------------- #
# Planes, edited                                                               #
# --------------------------------------------------------------------------- #


def _unit_normal(plane: RenderPlane) -> np.ndarray:
    normal = np.cross(plane.in_plane_axis_0, plane.in_plane_axis_1)
    return normal / np.linalg.norm(normal)


def _floats(vector: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(v) for v in vector)


def with_position(plane: RenderPlane, position: float) -> RenderPlane:
    """*plane* moved along its normal to sit at *position*.

    The origin keeps its in-plane coordinates.
    """
    normal = _unit_normal(plane)
    origin = np.asarray(plane.origin, dtype=np.float64)
    moved = origin + (float(position) - float(origin @ normal)) * normal
    return plane.model_copy(update={"origin": _floats(moved)})


def with_normal(plane: RenderPlane, normal: Sequence[float]) -> RenderPlane:
    """*plane* turned about its origin so its normal is *normal*.

    The frame is turned by the smallest rotation that takes the old normal
    to the new one, so its spin about the normal is kept.  An opposite
    normal is a half turn about ``in_plane_axis_0``.

    Raises
    ------
    ValueError
        If *normal* is zero or not finite.
    """
    target = np.asarray([float(v) for v in normal], dtype=np.float64)
    length = float(np.linalg.norm(target))
    if not math.isfinite(length) or length == 0.0:
        raise ValueError("A normal must not be all zero.")
    target = target / length
    current = _unit_normal(plane)
    axis_0 = np.asarray(plane.in_plane_axis_0, dtype=np.float64)
    cosine = float(np.clip(current @ target, -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return plane
    if cosine < -1.0 + 1e-12:
        new_0 = axis_0  # the half turn's own axis
    else:
        # Rodrigues, about the axis that is perpendicular to both normals.
        axis = np.cross(current, target)
        sine = float(np.linalg.norm(axis))
        axis = axis / sine
        new_0 = (
            axis_0 * cosine
            + np.cross(axis, axis_0) * sine
            + axis * float(axis @ axis_0) * (1.0 - cosine)
        )
    new_0 = new_0 - float(new_0 @ target) * target
    new_0 = new_0 / np.linalg.norm(new_0)
    new_1 = np.cross(target, new_0)
    return RenderPlane(
        **{
            **plane.model_dump(),
            "in_plane_axis_0": _floats(new_0),
            "in_plane_axis_1": _floats(new_1),
        }
    )


def flipped(plane: RenderPlane) -> RenderPlane:
    """*plane* with its normal reversed and nothing drawn changed.

    ``in_plane_axis_1`` is reversed and its extent mirrored with it, so the
    rectangle stays where it is (design 9.2, P15).
    """
    low, high = plane.extent_1
    return plane.model_copy(
        update={
            "in_plane_axis_1": _floats(-np.asarray(plane.in_plane_axis_1)),
            "extent_1": (
                None if high is None else -float(high),
                None if low is None else -float(low),
            ),
        }
    )


def box_cut(
    plane: RenderPlane, bounds: Sequence[Sequence[float]]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Where the data box cuts a plane, along the plane's two in-plane axes.

    Parameters
    ----------
    plane : RenderPlane
        The plane.
    bounds : Sequence[Sequence[float]]
        ``(low, high)`` of the box on each of the plane's three axes.

    Returns
    -------
    tuple or None
        ``((min_0, max_0), (min_1, max_1))``: the span of the polygon the
        box cuts out of the plane, as distances from the plane's origin
        along ``in_plane_axis_0`` and ``in_plane_axis_1``.  ``None`` when
        the plane misses the box.
    """
    box = np.asarray(bounds, dtype=np.float64)
    if box.shape != (3, 2) or not np.all(np.isfinite(box)):
        return None
    origin = np.asarray(plane.origin, dtype=np.float64)
    normal = _unit_normal(plane)
    corners = np.array(
        [
            [box[0, i], box[1, j], box[2, k]]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ]
    )
    side = (corners - origin) @ normal
    points = []
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") != 1:
                continue  # not an edge of the box
            if side[a] == 0.0:
                points.append(corners[a])
            if side[b] == 0.0:
                points.append(corners[b])
            if side[a] * side[b] < 0.0:
                t = side[a] / (side[a] - side[b])
                points.append(corners[a] + t * (corners[b] - corners[a]))
    if not points:
        return None
    relative = np.asarray(points) - origin
    along_0 = relative @ np.asarray(plane.in_plane_axis_0)
    along_1 = relative @ np.asarray(plane.in_plane_axis_1)
    return (
        (float(along_0.min()), float(along_0.max())),
        (float(along_1.min()), float(along_1.max())),
    )


def with_side_bounded(
    plane: RenderPlane,
    axis: int,
    side: int,
    bounded: bool,
    bounds: Sequence[Sequence[float]],
) -> RenderPlane:
    """*plane* with one extent side made finite or unbounded.

    Making a side finite fills in where the data box cuts the plane on that
    side, so the picture does not jump (design 9.2).  When the plane misses
    the box, or the cut would not leave ``min < max`` against the other
    side, the side is put a little past the other one.

    Parameters
    ----------
    plane : RenderPlane
        The plane.
    axis : {0, 1}
        The in-plane axis.
    side : {0, 1}
        ``0`` the minimum, ``1`` the maximum.
    bounded : bool
        ``False`` clears the side to ``None``.
    bounds : Sequence[Sequence[float]]
        ``(low, high)`` of the data box on each of the plane's three axes.
    """
    field = "extent_1" if axis else "extent_0"
    extent = list(getattr(plane, field))
    if not bounded:
        extent[side] = None
        return plane.model_copy(update={field: tuple(extent)})
    box = np.asarray(bounds, dtype=np.float64)
    diagonal = float(np.linalg.norm(box[:, 1] - box[:, 0])) if box.size else 1.0
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        diagonal = 1.0
    cut = box_cut(plane, bounds)
    value = (-0.5 if side == 0 else 0.5) * diagonal if cut is None else cut[axis][side]
    other = extent[1 - side]
    if other is not None:
        gap = 0.01 * diagonal
        if side == 0 and not value < other:
            value = float(other) - gap
        elif side == 1 and not value > other:
            value = float(other) + gap
    extent[side] = float(value)
    return plane.model_copy(update={field: tuple(extent)})


def with_side(plane: RenderPlane, axis: int, side: int, value: float) -> RenderPlane:
    """*plane* with one bounded extent side set to *value*.

    Raises
    ------
    ValueError
        If the pair would not have ``min < max``, or *value* is not finite.
    """
    field = "extent_1" if axis else "extent_0"
    extent = list(getattr(plane, field))
    extent[side] = float(value)
    return RenderPlane(**{**plane.model_dump(), field: tuple(extent)})


# --------------------------------------------------------------------------- #
# Rows                                                                         #
# --------------------------------------------------------------------------- #


def _extent_row(extent: Sequence[float | None]) -> list[float | None]:
    return [None if side is None else float(side) for side in extent]


def rows_from_planes(
    planes: Iterable[RenderPlane], axis_names: Mapping[str, str] | None = None
) -> list[Row]:
    """The rows that show *planes* (a visual's ``render_planes``).

    Parameters
    ----------
    planes : Iterable[RenderPlane]
        The planes.
    axis_names : Mapping[str, str] or None
        World axis id (as a string) -> name.  An axis not in it is shown as
        it is written on the plane.
    """
    names = axis_names or {}
    rows = []
    for plane in planes:
        normal = _unit_normal(plane)
        rows.append(
            {
                "id": str(plane.id),
                "enabled": bool(plane.enabled),
                "axes": [names.get(str(axis), str(axis)) for axis in plane.axes],
                # ``+ 0.0`` turns a negative zero into zero.
                "normal": [float(v) + 0.0 for v in normal],
                "position": float(np.asarray(plane.origin) @ normal),
                "extent_0": _extent_row(plane.extent_0),
                "extent_1": _extent_row(plane.extent_1),
            }
        )
    return rows


class RenderPlaneGizmoTarget(NamedTuple):
    """Where the gizmo of a render planes control is drawn.

    Parameters
    ----------
    visual_id : UUID
        The visual whose plane the gizmo edits.
    canvas_id : UUID
        The 3D canvas the gizmo is drawn in.
    scene_id : UUID
        That canvas's scene.
    """

    visual_id: UUID
    canvas_id: UUID
    scene_id: UUID


def _plain_bounds(extent: Iterable[Sequence[float]]) -> Bounds:
    """``[low, high]`` per axis; ``[0, 1]`` where the visual has no extent."""
    out = []
    for low, high in extent:
        low, high = float(low), float(high)
        out.append(
            [low, high] if math.isfinite(low) and math.isfinite(high) else [0.0, 1.0]
        )
    return out


def get_render_planes_data_from_visual(
    controller: Any,
    visual_id: UUID,
    *,
    blocked: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """What a render planes control is built with, read through the controller.

    Unlike a clipping planes control, a render planes control reads the
    scene as well as the visual: its planes are in world coordinates, a new
    one is put on the axes the 3D view displays, and the control is
    disabled while the planes are not drawn.

    Parameters
    ----------
    controller : CellierController
        The controller the control is wired to.
    visual_id : UUID
        The visual.  For a group of visuals edited together, the first.
    blocked : Callable[[], str] or None
        Why the control is disabled now, or ``""``.  ``None`` uses
        ``controller.render_planes_blocked(visual_id)``: the visual is not
        in the ``"plane"`` render mode, or the view is 2D.

    Returns
    -------
    dict[str, Any]
        Keyword arguments of the control: ``coordinate_system`` (the world
        system's id, as a string), ``axis_ids`` and ``axis_names`` (the
        world axes, in order), ``planes`` (the visual's ``render_planes``),
        ``scene_id``, ``data_store_id`` and ``state_source``, a reader of
        ``{"blocked", "displayed_axes", "bounds"}`` as they are now.

    Raises
    ------
    KeyError
        If the visual is not registered.
    """
    visual_id = UUID(str(visual_id))
    visual = controller.get_visual_model(visual_id)
    scene_id = controller.get_visual_scene_id(visual_id)
    world = controller.get_scene(scene_id).dims.world_coordinate_system

    def state() -> dict[str, Any]:
        dims = controller.get_scene(scene_id).dims
        return {
            "blocked": (
                controller.render_planes_blocked(visual_id)
                if blocked is None
                else blocked()
            ),
            "displayed_axes": [
                str(world.axes[axis].id) for axis in dims.selection.displayed_axes
            ],
            "bounds": _plain_bounds(controller.visual_world_extent(visual_id)),
        }

    return {
        "coordinate_system": str(world.id),
        "axis_ids": [str(axis.id) for axis in world.axes],
        "axis_names": [str(name) for name in world.axis_names()],
        "planes": tuple(visual.render_planes),
        "scene_id": str(scene_id),
        "data_store_id": str(visual.data_store_id),
        "state_source": state,
    }


def get_render_plane_gizmo_data(
    controller: Any, visual_id: UUID, canvas_id: UUID
) -> dict[str, Any]:
    """The gizmo arguments of a render planes control, for one visual and canvas.

    Selects nothing: the caller names the visual whose plane the gizmo
    edits and the canvas it is drawn in, as for
    ``CellierController.add_render_plane_gizmo``.  The canvas must exist:
    build the canvas before the control.

    Returns
    -------
    dict[str, Any]
        ``gizmo_target`` (a :class:`RenderPlaneGizmoTarget`), ``gizmo_plane``
        (the id of the render plane that has the canvas's gizmo now, as a
        string, if it is one of this visual's; otherwise ``None``) and
        ``gizmo_blocked`` (a reader of why a plane cannot have one, over
        ``controller.render_plane_gizmo_blocked``).

    Raises
    ------
    KeyError
        If the visual or the canvas is not registered.
    ValueError
        If the canvas does not show the visual's scene.
    """
    visual_id, canvas_id = UUID(str(visual_id)), UUID(str(canvas_id))
    scene_id = controller.get_visual_scene_id(visual_id)
    if canvas_id not in controller.get_canvas_ids(scene_id):
        controller.get_canvas_view(canvas_id)  # KeyError for an unknown canvas
        raise ValueError(
            f"Canvas {canvas_id} does not show scene "
            f"{controller.get_scene(scene_id).name!r}, the scene of visual "
            f"{visual_id}."
        )
    session = controller.get_plane_gizmo(canvas_id)
    on_this = (
        session is not None
        and session.kind == "render"
        and session.visual_id == visual_id
    )

    def blocked(plane_id: str) -> str:
        return controller.render_plane_gizmo_blocked(
            visual_id, canvas_id, UUID(str(plane_id))
        )

    return {
        "gizmo_target": RenderPlaneGizmoTarget(visual_id, canvas_id, scene_id),
        "gizmo_plane": str(session.plane_id) if on_this else None,
        "gizmo_blocked": blocked,
    }


# --------------------------------------------------------------------------- #
# The editor                                                                   #
# --------------------------------------------------------------------------- #


class RenderPlanesEditor:
    """The toolkit-neutral half of a render planes control.

    Holds the planes the model last reported, sends edits, and tells the
    widget what to show.  An edit is one ``RenderPlanesUpdateEvent`` per
    visual, carrying the whole new tuple.  If the controller refuses it,
    the widget is told to show the last rows again, with the error.

    Parameters
    ----------
    visual_ids : Iterable[UUID]
        The visuals edited together.  One update is sent per visual.
    coordinate_system : UUID or str
        The scene's world coordinate system.
    axis_ids, axis_names : Sequence[str]
        The world axes, in order: their ids and their names.
    planes : Sequence[RenderPlane]
        The visual's current ``render_planes``.
    source_id : UUID
        The widget's id, stamped on each edit.
    emit : Callable[[Any], None]
        Sends an edit (the widget's ``changed.emit``).
    show : Callable[[list[Row], str], None]
        Draws the rows and an error message ("" for none).
    displayed_axes : Sequence[str] or None
        The ids of the three world axes a new plane is put on.  ``None``
        means the last three world axes.  Read from *state_source* when
        that is given.
    bounds : Sequence[Sequence[float]] or None
        ``(low, high)`` of the visual's data on each world axis.  ``None``
        means ``(0, 1)`` on each.  Read from *state_source* when given.
    state_source : Callable[[], Mapping[str, Any]] or None
        Reads ``{"blocked", "displayed_axes", "bounds"}`` as they are now
        (:func:`get_render_planes_data_from_visual`).  The control is
        disabled, with the reason shown, while ``blocked`` is not ``""``.
    scene_id, data_store_id : UUID, str or None
        The visuals' scene and store, whose changes move what
        *state_source* reads.
    gizmo_target : RenderPlaneGizmoTarget or None
        Where a plane's gizmo is drawn.  ``None`` gives the rows no gizmo
        toggle.  See :func:`get_render_plane_gizmo_data`.
    gizmo_plane : str or None
        The id of the plane that has the canvas's gizmo now.
    gizmo_blocked : Callable[[str], str] or None
        Given a plane's id, why it cannot have a gizmo now, or ``""``.

    Raises
    ------
    ValueError
        If the visual of *gizmo_target* is not one of *visual_ids*.
    """

    def __init__(
        self,
        visual_ids: Iterable[UUID],
        coordinate_system: UUID | str,
        axis_ids: Sequence[str],
        axis_names: Sequence[str],
        planes: Sequence[RenderPlane],
        source_id: UUID,
        emit: Callable[[Any], None],
        show: Callable[[list[Row], str], None],
        displayed_axes: Sequence[str] | None = None,
        bounds: Sequence[Sequence[float]] | None = None,
        state_source: Callable[[], Mapping[str, Any]] | None = None,
        scene_id: UUID | str | None = None,
        data_store_id: UUID | str | None = None,
        gizmo_target: RenderPlaneGizmoTarget | None = None,
        gizmo_plane: str | None = None,
        gizmo_blocked: Callable[[str], str] | None = None,
    ) -> None:
        self._visual_ids = tuple(visual_ids)
        self._coordinate_system = UUID(str(coordinate_system))
        self._axis_ids = [str(axis) for axis in axis_ids]
        self.axis_names = [str(name) for name in axis_names]
        self._names = dict(zip(self._axis_ids, self.axis_names, strict=True))
        self._index = {axis: i for i, axis in enumerate(self._axis_ids)}
        self._index.update({name: i for i, name in enumerate(self.axis_names)})
        self.planes: tuple[RenderPlane, ...] = tuple(planes)
        self._source_id = source_id
        self._emit = emit
        self._show = show
        self._state_source = state_source
        self._scene_id = None if scene_id is None else UUID(str(scene_id))
        self._data_store_id = (
            None if data_store_id is None else UUID(str(data_store_id))
        )
        self._displayed_axes = (
            self._axis_ids[-3:]
            if displayed_axes is None
            else [str(axis) for axis in displayed_axes]
        )
        self._bounds: Bounds = (
            [[0.0, 1.0] for _ in self._axis_ids]
            if bounds is None
            else _plain_bounds(bounds)
        )
        #: Why the control is disabled now, or "".
        self.blocked: str = ""
        self._read_state()
        self._gizmo: RenderPlaneGizmoTarget | None = None
        if gizmo_target is not None:
            self._gizmo = RenderPlaneGizmoTarget(
                *(UUID(str(value)) for value in gizmo_target)
            )
            if self._gizmo.visual_id not in self._visual_ids:
                raise ValueError(
                    f"The gizmo target's visual {self._gizmo.visual_id} is not "
                    "one of the visuals this control edits."
                )
        self.gizmo_plane: str | None = None if gizmo_plane is None else str(gizmo_plane)
        self._gizmo_blocked = gizmo_blocked

    # -- state ---------------------------------------------------------------

    @property
    def rows(self) -> list[Row]:
        """The rows of the planes the model last reported."""
        return rows_from_planes(self.planes, self._names)

    @property
    def has_gizmo(self) -> bool:
        """Whether the rows have a gizmo toggle."""
        return self._gizmo is not None

    @property
    def can_add(self) -> bool:
        """Whether a plane can be added: enabled, and under four planes."""
        return not self.blocked and len(self.planes) < MAX_RENDER_PLANES

    def _read_state(self) -> bool:
        """Read the scene's state again; whether anything shown changed."""
        if self._state_source is None:
            return False
        state = self._state_source()
        before = (self.blocked, self._displayed_axes, self._bounds)
        self.blocked = str(state.get("blocked", ""))
        self._displayed_axes = [str(axis) for axis in state["displayed_axes"]]
        self._bounds = _plain_bounds(state["bounds"])
        return before != (self.blocked, self._displayed_axes, self._bounds)

    def _plane_bounds(self, plane: RenderPlane) -> Bounds:
        """The data box on *plane*'s three axes."""
        return [
            self._bounds[self._index[str(axis)]]
            if str(axis) in self._index
            else [0.0, 1.0]
            for axis in plane.axes
        ]

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """What the control follows on the bus.

        Per visual: its ``RenderPlanesChangedEvent``, and the events that
        can change whether its planes are drawn or where its data is (a
        render mode, ``composite``, its transform).  The scene's
        ``DimsChangedEvent`` and the store's extent.  With a gizmo toggle,
        the canvas's ``PlaneGizmoChangedEvent``.
        """
        specs = []
        for visual_id in self._visual_ids:
            specs.append(
                SubscriptionSpec(
                    RenderPlanesChangedEvent, self.on_changed, entity_id=visual_id
                )
            )
            for event_type in (
                AppearanceChangedEvent,
                SingleAppearanceChangedEvent,
                ChannelAppearanceChangedEvent,
                ImageCompositeChangedEvent,
                TransformChangedEvent,
            ):
                specs.append(
                    SubscriptionSpec(
                        event_type, self.on_state_changed, entity_id=visual_id
                    )
                )
        if self._scene_id is not None:
            specs.append(
                SubscriptionSpec(
                    DimsChangedEvent, self.on_dims_changed, entity_id=self._scene_id
                )
            )
        if self._data_store_id is not None:
            specs.append(
                SubscriptionSpec(
                    DataStoreMetadataChangedEvent,
                    self.on_state_changed,
                    entity_id=self._data_store_id,
                )
            )
        if self._gizmo is not None:
            specs.append(
                SubscriptionSpec(
                    PlaneGizmoChangedEvent,
                    self.on_gizmo_changed,
                    entity_id=self._gizmo.canvas_id,
                )
            )
        return specs

    # -- edits ---------------------------------------------------------------

    def set_planes(self, planes: Iterable[RenderPlane]) -> None:
        """Send *planes* as the new tuple; nothing when they are the current.

        The comparison is by value: a host that delivers one front-end edit
        twice (marimo does) sends one update.
        """
        try:
            planes = tuple(planes)
            if planes == self.planes:
                return
            if self.blocked:
                raise ValueError(self.blocked)
            for visual_id in self._visual_ids:
                self._emit(
                    RenderPlanesUpdateEvent(
                        source_id=self._source_id,
                        visual_id=visual_id,
                        render_planes=planes,
                    )
                )
        except Exception as error:
            self._show(self.rows, error_message(error))
            return
        self.planes = planes
        self._show(self.rows, "")

    def _edit(self, index: int, edit: Callable[[RenderPlane], RenderPlane]) -> None:
        """Replace plane *index* by ``edit(plane)``; a refusal is shown."""
        try:
            planes = list(self.planes)
            planes[index] = edit(planes[index])
        except Exception as error:
            self._show(self.rows, error_message(error))
            return
        self.set_planes(planes)

    def add(self) -> None:
        """Append a plane through the data's middle, facing the first axis.

        The plane is on the three axes the 3D view displays and unbounded.
        """
        try:
            if len(self.planes) >= MAX_RENDER_PLANES:
                raise ValueError(
                    f"A visual draws at most {MAX_RENDER_PLANES} render planes."
                )
            axes = list(self._displayed_axes)
            if len(axes) != 3:
                raise ValueError(
                    "A render plane is on the three axes a 3D view displays."
                )
            centre = [0.5 * sum(self._bounds[self._index[axis]]) for axis in axes]
            plane = RenderPlane(
                coordinate_system=self._coordinate_system,
                axes=tuple(UUID(axis) for axis in axes),
                origin=tuple(centre),
                in_plane_axis_0=(0.0, 1.0, 0.0),
                in_plane_axis_1=(0.0, 0.0, 1.0),
            )
        except Exception as error:
            self._show(self.rows, error_message(error))
            return
        self.set_planes([*self.planes, plane])

    def remove(self, index: int) -> None:
        """Remove plane *index*."""
        self.set_planes([p for i, p in enumerate(self.planes) if i != index])

    def set_enabled(self, index: int, enabled: bool) -> None:
        """Switch plane *index* on or off; it stays in the list."""
        self._edit(index, lambda p: p.model_copy(update={"enabled": bool(enabled)}))

    def set_position(self, index: int, position: float) -> None:
        """Move plane *index* along its normal."""
        self._edit(index, lambda p: with_position(p, position))

    def flip(self, index: int) -> None:
        """Reverse the normal of plane *index*; nothing drawn changes."""
        self._edit(index, flipped)

    def set_normal(self, index: int, normal: Sequence[float]) -> None:
        """Turn plane *index* about its origin to a new normal; keeps its spin."""
        self._edit(index, lambda p: with_normal(p, normal))

    def set_facing(self, index: int, axis: int, sign: int) -> None:
        """Face plane *index* along its axis *axis* (0 to 2), toward *sign*."""
        normal = [0.0, 0.0, 0.0]
        normal[axis] = -1.0 if sign < 0 else 1.0
        self.set_normal(index, normal)

    def set_component(self, index: int, axis: int, value: float) -> None:
        """Set the normal's entry on the plane's axis *axis*; the others stay."""
        normal = list(self.rows[index]["normal"])
        normal[axis] = float(value)
        self.set_normal(index, normal)

    def set_side_bounded(self, index: int, axis: int, side: int, bounded: bool) -> None:
        """Make one extent side of plane *index* finite or unbounded.

        Making it finite fills in where the data box cuts the plane there.
        """
        self._edit(
            index,
            lambda p: with_side_bounded(p, axis, side, bounded, self._plane_bounds(p)),
        )

    def set_side(self, index: int, axis: int, side: int, value: float) -> None:
        """Set one bounded extent side of plane *index*, in world units."""
        self._edit(index, lambda p: with_side(p, axis, side, value))

    def set_gizmo(self, index: int, enabled: bool) -> None:
        """Put the canvas's gizmo on plane *index*, or take it off.

        Sends a ``PlaneGizmoUpdateEvent`` of kind ``"render"``.  The toggle
        is drawn from what the controller answers (:meth:`on_gizmo_changed`),
        not from the click: a refused request leaves it off, with the
        reason shown.
        """
        if self._gizmo is None:
            return
        visual_id, canvas_id, _scene_id = self._gizmo
        error = ""
        try:
            if self.blocked:
                raise ValueError(self.blocked)
            self._emit(
                PlaneGizmoUpdateEvent(
                    source_id=self._source_id,
                    visual_id=visual_id,
                    plane_id=self.planes[index].id,
                    canvas_id=canvas_id,
                    kind="render",
                    enabled=bool(enabled),
                )
            )
        except Exception as refused:
            error = error_message(refused)
        self._show(self.rows, error)

    # -- model -> widget -----------------------------------------------------

    def on_changed(self, event: RenderPlanesChangedEvent) -> None:
        """The model changed (this widget's edit or anyone's): show it."""
        planes = tuple(event.render_planes)
        if planes == self.planes:
            return
        self.planes = planes
        self._show(self.rows, "")

    def on_gizmo_changed(self, event: PlaneGizmoChangedEvent) -> None:
        """The canvas's gizmo moved to another plane, or closed: show it."""
        if self._gizmo is None:
            return
        on_this = (
            event.kind == "render"
            and event.visual_id == self._gizmo.visual_id
            and event.plane_id is not None
        )
        plane = str(event.plane_id) if on_this else None
        if plane == self.gizmo_plane:
            return
        self.gizmo_plane = plane
        self._show(self.rows, "")

    def on_dims_changed(self, event: DimsChangedEvent) -> None:
        """The displayed axes changed: the planes may not be drawn now."""
        if event.displayed_axes_changed:
            self.refresh()

    def on_state_changed(self, _event: Any = None) -> None:
        """A render mode, the transform or the store's extent changed."""
        if self._read_state():
            self._show(self.rows, "")

    def refresh(self) -> None:
        """Read the scene's state again and show the rows."""
        self._read_state()
        self._show(self.rows, "")

    def describe(self) -> list[Row]:
        """The rows with what a front end needs to draw each one.

        Adds ``facing`` (the signed axis the normal lies along, or
        ``None``) and the position slider's ``low`` / ``high``.  With a
        gizmo toggle, also ``gizmo`` (whether this plane has the canvas's
        gizmo) and ``gizmo_blocked`` (why it cannot have one, or ``""``);
        without, neither key.
        """
        described = []
        for plane, row in zip(self.planes, self.rows, strict=True):
            low, high = position_range(row["normal"], self._plane_bounds(plane))
            entry = {
                **row,
                "facing": facing_of(row["normal"]),
                "low": min(low, row["position"]),
                "high": max(high, row["position"]),
            }
            if self._gizmo is not None:
                entry["gizmo"] = row["id"] == self.gizmo_plane
                entry["gizmo_blocked"] = self.blocked or (
                    ""
                    if self._gizmo_blocked is None
                    else self._gizmo_blocked(row["id"])
                )
            described.append(entry)
        return described
