"""The "Clipping planes" control of a visual, decided once for both toolkits.

A visual carries a tuple of :class:`~cellier.visuals.ClippingPlane` in its
level-0 data coordinates.  The Qt and anywidget controls
(``QtClippingPlanesControls``, ``AnywidgetClippingPlanesControls``) draw one
row per plane: an enabled checkbox, a normal (one entry per data axis, under
that axis's name, with a button to face the plane along either direction of
an axis), a flip button, a position slider along the normal and a remove
button; an add button appends a plane.  This module holds the rows, turns
them into planes and back, and carries edits to the bus and model changes
back.

A row can also carry a "Gizmo" toggle, which puts a gizmo on that plane in
the viewer's 3D canvas.  A canvas has one gizmo, so the toggles of every
control that names the canvas behave as one set of radio buttons: the
controller says which plane has it (``ClippingPlaneGizmoChangedEvent``) and
each control draws that.

A row is plain data, so the anywidget control syncs the whole list in one
trait and constructs nothing when a plane is added::

    {"id": "0f6c...", "enabled": True, "normal": [0.0, 0.0, 1.0], "position": 12.0}

``id`` is the plane's id, which an edit keeps: whatever follows a plane (a
gizmo) keeps following it.  ``normal`` has one entry per data axis.  ``position`` is where the plane
sits along its unit normal: the plane is ``normal . p == position *
|normal|``.  Values are in data units (voxels for an image).

Nothing here imports a toolkit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from cellier.events import (
    ClippingPlaneGizmoChangedEvent,
    ClippingPlaneGizmoUpdateEvent,
    ClippingPlanesChangedEvent,
    ClippingPlanesUpdateEvent,
    DataStoreMetadataChangedEvent,
    DimsChangedEvent,
    SubscriptionSpec,
)
from cellier.gui._appearance_fields import normalize_visual_ids
from cellier.gui._loading import error_message

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

CLIPPING_PLANES_TITLE = "Clipping planes"
"""The control's name: its ``DEFAULT_TITLE`` on both toolkits."""

Row = dict[str, Any]


def _length(normal: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in normal))


def rows_from_planes(planes: Iterable[Any]) -> list[Row]:
    """The rows that show *planes* (a visual's ``clipping_planes``)."""
    rows = []
    for item in planes:
        normal = [float(v) for v in item.plane.normal]
        rows.append(
            {
                "id": str(item.id),
                "enabled": bool(item.enabled),
                "normal": normal,
                "position": float(item.plane.offset) / _length(normal),
            }
        )
    return rows


def planes_from_rows(rows: Iterable[Mapping[str, Any]], coordinate_system: UUID):
    """Build the ``ClippingPlane`` tuple that *rows* describe.

    Raises
    ------
    ValueError
        If a row's normal is zero or not finite.
    """
    from cellier.transform import Plane
    from cellier.visuals import ClippingPlane

    planes = []
    for row in rows:
        normal = [float(v) for v in row["normal"]]
        identity = {"id": UUID(str(row["id"]))} if row.get("id") else {}
        planes.append(
            ClippingPlane(
                **identity,
                plane=Plane(
                    coordinate_system=coordinate_system,
                    normal=normal,
                    offset=float(row["position"]) * _length(normal),
                ),
                enabled=bool(row.get("enabled", True)),
            )
        )
    return tuple(planes)


def normalized_rows(rows: Iterable[Mapping[str, Any]]) -> list[Row]:
    """Rows as plain Python values, so two lists compare by value.

    A row with no ``id`` is given a new one.
    """
    return [
        {
            "id": str(row.get("id") or uuid4()),
            "enabled": bool(row.get("enabled", True)),
            "normal": [float(v) for v in row["normal"]],
            "position": float(row["position"]),
        }
        for row in rows
    ]


def position_range(
    normal: Sequence[float], bounds: Sequence[Sequence[float]]
) -> tuple[float, float]:
    """The slider's range: the data bounding box projected onto the normal.

    Parameters
    ----------
    normal : Sequence[float]
        One entry per data axis.
    bounds : Sequence[Sequence[float]]
        ``(low, high)`` per data axis.  An axis the normal has no component
        on does not matter.

    Returns
    -------
    tuple[float, float]
        Positions along the unit normal at which the plane touches the
        box; ``(0.0, 1.0)`` for a zero normal.
    """
    length = _length(normal)
    if length == 0.0:
        return 0.0, 1.0
    low = high = 0.0
    for component, (axis_low, axis_high) in zip(normal, bounds):
        unit = float(component) / length
        if unit == 0.0:
            continue
        a, b = unit * float(axis_low), unit * float(axis_high)
        low += min(a, b)
        high += max(a, b)
    if high <= low:
        high = low + 1.0
    return low, high


def facing_of(normal: Sequence[float]) -> list[int] | None:
    """The signed data axis a normal lies along.

    Returns
    -------
    list[int] or None
        ``[axis index, sign]`` with sign ``1`` or ``-1``; ``None`` when the
        normal has a component on more than one axis.
    """
    along = [index for index, value in enumerate(normal) if float(value) != 0.0]
    if len(along) != 1:
        return None
    return [along[0], -1 if float(normal[along[0]]) < 0 else 1]


def new_row(axis_names: Sequence[str], bounds: Sequence[Sequence[float]]) -> Row:
    """A plane across the last data axis, through the middle of the data."""
    normal = [0.0] * len(axis_names)
    normal[-1] = 1.0
    low, high = position_range(normal, bounds)
    return {
        "id": str(uuid4()),
        "enabled": True,
        "normal": normal,
        "position": 0.5 * (low + high),
    }


def store_bounds(store: Any, ndim: int) -> list[list[float]]:
    """``[low, high]`` of *store*'s data on each of its *ndim* axes.

    ``[0, 1]`` per axis for a store that cannot say (an empty one).
    """
    extents = store.axis_extents
    if extents is None:
        return [[0.0, 1.0] for _ in range(ndim)]
    return [[float(low), float(high)] for low, high in extents]


def get_clipping_planes_data_from_visual(visual: Any, store: Any) -> dict[str, Any]:
    """What a clipping planes control is built with, read off the models.

    Parameters
    ----------
    visual : BaseVisual
        The visual (for an ``OrthoViewer`` group, any one of them: they
        read one store and carry the same planes).
    store : BaseDataStore
        The store the visual reads.

    Returns
    -------
    dict[str, Any]
        ``coordinate_system`` (the store's level-0 system id, as a string),
        ``data_store_id`` (as a string), ``axis_names``, ``bounds``
        (``[low, high]`` per data axis) and ``planes`` (the rows).
    """
    system = store.data_coordinate_system
    names = [str(name) for name in system.axis_names()]
    return {
        "coordinate_system": str(system.id),
        "data_store_id": str(store.id),
        "axis_names": names,
        "bounds": store_bounds(store, len(names)),
        "planes": rows_from_planes(visual.clipping_planes),
    }


def seed_bounds_source(
    controller: Any, seed: Mapping[str, Any]
) -> Callable[[], list[list[float]]] | None:
    """A reader of the current bounds of the store a seed was read off.

    Parameters
    ----------
    controller : CellierController or None
        Looks the store up by id each time, so the reader holds no store.
    seed : Mapping[str, Any]
        From :func:`get_clipping_planes_data_from_visual`.

    Returns
    -------
    Callable or None
        The ``bounds_source`` of a clipping planes control; ``None``
        without a controller.
    """
    if controller is None:
        return None
    store_id = UUID(str(seed["data_store_id"]))
    ndim = len(seed["axis_names"])
    return lambda: store_bounds(controller.get_data_store(store_id), ndim)


GIZMO_TOOLTIP = "Drag a gizmo in the 3D view to move and tilt this plane."
"""The gizmo toggle's tooltip while it can be used."""


def gizmo_seed(controller: Any, visual_ids: Iterable[UUID]) -> dict[str, Any]:
    """The gizmo arguments of a clipping planes control, read off a controller.

    The gizmo of a control's planes is drawn in the first canvas of the
    first of *visual_ids* whose scene can be shown in 3D (for an
    ``OrthoViewer`` group, its 3D panel).  That canvas must exist: build
    the canvas before the control.

    Parameters
    ----------
    controller : CellierController or None
        The controller the control is wired to.
    visual_ids : Iterable[UUID]
        The visuals the control edits together.

    Returns
    -------
    dict[str, Any]
        ``gizmo`` (``visual_id``, ``canvas_id`` and ``scene_id``, as
        strings), ``gizmo_plane`` (the id of the plane that has the
        canvas's gizmo now, or ``None``) and ``gizmo_blocked`` (a reader of
        why a plane cannot have one).  Empty without a controller, or when
        none of the visuals is in a scene that can show 3D: the control
        then draws no toggle.

    Raises
    ------
    ValueError
        If the scene that can show 3D has no canvas yet.
    """
    if controller is None:
        return {}
    for visual_id in normalize_visual_ids(visual_ids):
        try:
            scene_id = controller.get_visual_scene_id(visual_id)
        except KeyError:
            continue
        scene = controller.get_scene(scene_id)
        if "3d" not in scene.render_modes:
            continue
        canvases = controller.get_canvas_ids(scene_id)
        if not canvases:
            raise ValueError(
                f"Scene {scene.name!r} has no canvas to draw a clipping plane "
                "gizmo in.  Build the canvas before the clipping planes "
                "control."
            )
        canvas_id = canvases[0]
        session = controller.get_clipping_plane_gizmo(canvas_id)
        on_this = session is not None and session.visual_id == visual_id

        def blocked(plane_id: str, _visual=visual_id, _canvas=canvas_id) -> str:
            return controller.clipping_plane_gizmo_blocked(
                _visual, _canvas, UUID(str(plane_id))
            )

        return {
            "gizmo": {
                "visual_id": str(visual_id),
                "canvas_id": str(canvas_id),
                "scene_id": str(scene_id),
            },
            "gizmo_plane": str(session.plane_id) if on_this else None,
            "gizmo_blocked": blocked,
        }
    return {}


class ClippingPlanesEditor:
    """The toolkit-neutral half of a clipping planes control.

    Holds the rows the model last reported, sends edits, and tells the
    widget what to show.  An edit is one ``ClippingPlanesUpdateEvent`` per
    visual, carrying the whole new tuple.  If the controller refuses it,
    the widget is told to show the last rows again, with the error.

    Parameters
    ----------
    visual_ids : Iterable[UUID]
        The visuals edited together (an ``OrthoViewer`` panel group).
    coordinate_system : UUID or str
        The level-0 data coordinate system of the store they read.
    axis_names : Sequence[str]
        Its axis names, in order.
    bounds : Sequence[Sequence[float]]
        ``(low, high)`` of the data on each axis, for the position range.
    planes : Sequence[Mapping[str, Any]]
        The current rows.
    source_id : UUID
        The widget's id, stamped on each edit.
    emit : Callable[[ClippingPlanesUpdateEvent], None]
        Sends an edit (the widget's ``changed.emit``).
    show : Callable[[list[Row], str], None]
        Draws the rows and an error message ("" for none).
    data_store_id : UUID, str or None
        The store the visuals read.  With *bounds_source*, the position
        range follows the store's extent.
    bounds_source : Callable[[], Sequence[Sequence[float]]] or None
        Reads the store's current ``(low, high)`` per axis.  ``None`` keeps
        *bounds* for the life of the control.
    gizmo : Mapping[str, Any] or None
        Where a plane's gizmo is drawn: ``visual_id`` (the one of the
        visuals the gizmo edits), ``canvas_id`` and ``scene_id``.  ``None``
        gives the rows no gizmo toggle.  See :func:`gizmo_seed`.
    gizmo_plane : str or None
        The id of the plane that has the canvas's gizmo now.
    gizmo_blocked : Callable[[str], str] or None
        Given a plane's id, why it cannot have a gizmo now, or ``""``.
        ``None`` means never blocked.
    """

    def __init__(
        self,
        visual_ids: Iterable[UUID],
        coordinate_system: UUID | str,
        axis_names: Sequence[str],
        bounds: Sequence[Sequence[float]],
        planes: Sequence[Mapping[str, Any]],
        source_id: UUID,
        emit: Callable[[ClippingPlanesUpdateEvent], None],
        show: Callable[[list[Row], str], None],
        data_store_id: UUID | str | None = None,
        bounds_source: Callable[[], Sequence[Sequence[float]]] | None = None,
        gizmo: Mapping[str, Any] | None = None,
        gizmo_plane: str | None = None,
        gizmo_blocked: Callable[[str], str] | None = None,
    ) -> None:
        self._visual_ids = tuple(visual_ids)
        self._coordinate_system = UUID(str(coordinate_system))
        self.axis_names = [str(name) for name in axis_names]
        self.bounds = [[float(low), float(high)] for low, high in bounds]
        self.rows: list[Row] = normalized_rows(planes)
        self._source_id = source_id
        self._emit = emit
        self._show = show
        self._data_store_id = (
            None if data_store_id is None else UUID(str(data_store_id))
        )
        self._bounds_source = bounds_source
        #: (visual, canvas, scene) of the gizmo, or ``None`` for no toggle.
        self._gizmo: tuple[UUID, UUID, UUID] | None = (
            None
            if gizmo is None
            else (
                UUID(str(gizmo["visual_id"])),
                UUID(str(gizmo["canvas_id"])),
                UUID(str(gizmo["scene_id"])),
            )
        )
        #: The id of the plane that has the canvas's gizmo, if it is one of
        #: this control's.
        self.gizmo_plane: str | None = None if gizmo_plane is None else str(gizmo_plane)
        self._gizmo_blocked = gizmo_blocked

    @property
    def has_gizmo(self) -> bool:
        """Whether the rows have a gizmo toggle."""
        return self._gizmo is not None

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``ClippingPlanesChangedEvent`` subscription per visual.

        And the store's ``DataStoreMetadataChangedEvent``, when the control
        was given a way to read the store's bounds.  With a gizmo toggle:
        the canvas's ``ClippingPlaneGizmoChangedEvent`` and the scene's
        ``DimsChangedEvent``.
        """
        specs = [
            SubscriptionSpec(
                ClippingPlanesChangedEvent, self.on_changed, entity_id=visual_id
            )
            for visual_id in self._visual_ids
        ]
        if self._data_store_id is not None and self._bounds_source is not None:
            specs.append(
                SubscriptionSpec(
                    DataStoreMetadataChangedEvent,
                    self.on_store_changed,
                    entity_id=self._data_store_id,
                )
            )
        if self._gizmo is not None:
            _visual_id, canvas_id, scene_id = self._gizmo
            specs.append(
                SubscriptionSpec(
                    ClippingPlaneGizmoChangedEvent,
                    self.on_gizmo_changed,
                    entity_id=canvas_id,
                )
            )
            specs.append(
                SubscriptionSpec(
                    DimsChangedEvent, self.on_dims_changed, entity_id=scene_id
                )
            )
        return specs

    # -- edits ---------------------------------------------------------------

    def set_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Send *rows* as the new planes; nothing when they are the current.

        The comparison is by value: a host that delivers one front-end edit
        twice (marimo does) sends one update.
        """
        try:
            rows = normalized_rows(rows)
            if rows == self.rows:
                return
            planes = planes_from_rows(rows, self._coordinate_system)
            for visual_id in self._visual_ids:
                self._emit(
                    ClippingPlanesUpdateEvent(
                        source_id=self._source_id,
                        visual_id=visual_id,
                        clipping_planes=planes,
                    )
                )
        except Exception as error:
            self._show(list(self.rows), error_message(error))
            return
        self.rows = rows
        self._show(list(self.rows), "")

    def _edited(self, index: int, **changes: Any) -> list[Row]:
        rows = [dict(row) for row in self.rows]
        rows[index].update(changes)
        return rows

    def add(self) -> None:
        """Append a plane across the last axis, through the data's middle."""
        self.set_rows([*self.rows, new_row(self.axis_names, self.bounds)])

    def remove(self, index: int) -> None:
        """Remove plane *index*."""
        self.set_rows([row for i, row in enumerate(self.rows) if i != index])

    def set_enabled(self, index: int, enabled: bool) -> None:
        """Switch plane *index* on or off; it stays in the list."""
        self.set_rows(self._edited(index, enabled=bool(enabled)))

    def set_position(self, index: int, position: float) -> None:
        """Move plane *index* along its normal."""
        self.set_rows(self._edited(index, position=float(position)))

    def flip(self, index: int) -> None:
        """Keep the other side of plane *index*; the plane does not move."""
        row = self.rows[index]
        self.set_rows(
            self._edited(
                index,
                normal=[-v if v != 0.0 else 0.0 for v in row["normal"]],
                position=-row["position"],
            )
        )

    def set_normal(self, index: int, normal: Sequence[float]) -> None:
        """Give plane *index* a new normal, through the point it passed.

        The plane keeps the point of the old plane nearest the middle of
        the data, so changing its direction turns it in place.
        """
        row = self.rows[index]
        old = row["normal"]
        old_length = _length(old)
        centre = [0.5 * (low + high) for low, high in self.bounds]
        # The old plane's point nearest the centre.
        distance = (
            sum(n * c for n, c in zip(old, centre)) / old_length - row["position"]
        )
        anchor = [c - distance * n / old_length for n, c in zip(old, centre)]
        normal = [float(v) for v in normal]
        length = _length(normal)
        if length == 0.0:
            self._show(list(self.rows), "A normal must not be all zero.")
            return
        position = sum(n * a for n, a in zip(normal, anchor)) / length
        self.set_rows(self._edited(index, normal=normal, position=position))

    def set_facing(self, index: int, axis: int, sign: int) -> None:
        """Face plane *index* along data axis *axis*, keeping side *sign*.

        ``sign`` ``1`` keeps the side toward higher values on the axis,
        ``-1`` the side toward lower ones.  The plane turns in place.
        """
        normal = [0.0] * len(self.axis_names)
        normal[axis] = -1.0 if sign < 0 else 1.0
        self.set_normal(index, normal)

    def set_component(self, index: int, axis: int, value: float) -> None:
        """Set the normal's entry on data axis *axis*; the others stay.

        The normal is not rescaled, so the other entries read as they were
        typed.  The plane turns in place.
        """
        normal = list(self.rows[index]["normal"])
        normal[axis] = float(value)
        self.set_normal(index, normal)

    def set_gizmo(self, index: int, enabled: bool) -> None:
        """Put the canvas's gizmo on plane *index*, or take it off.

        Sends a ``ClippingPlaneGizmoUpdateEvent``.  The toggle is drawn from
        what the controller answers (:meth:`on_gizmo_changed`), not from the
        click: a refused request leaves it off, with the reason shown.
        """
        if self._gizmo is None:
            return
        visual_id, canvas_id, _scene_id = self._gizmo
        error = ""
        try:
            self._emit(
                ClippingPlaneGizmoUpdateEvent(
                    source_id=self._source_id,
                    visual_id=visual_id,
                    plane_id=UUID(self.rows[index]["id"]),
                    canvas_id=canvas_id,
                    enabled=bool(enabled),
                )
            )
        except Exception as refused:
            error = error_message(refused)
        self._show(list(self.rows), error)

    # -- model -> widget -----------------------------------------------------

    def on_changed(self, event: ClippingPlanesChangedEvent) -> None:
        """The model changed (this widget's edit or anyone's): show it."""
        rows = rows_from_planes(event.clipping_planes)
        if rows == self.rows:
            return
        self.rows = rows
        self._show(list(self.rows), "")

    def on_gizmo_changed(self, event: ClippingPlaneGizmoChangedEvent) -> None:
        """The canvas's gizmo moved to another plane, or closed: show it."""
        if self._gizmo is None:
            return
        on_this = event.visual_id == self._gizmo[0] and event.plane_id is not None
        plane = str(event.plane_id) if on_this else None
        if plane == self.gizmo_plane:
            return
        self.gizmo_plane = plane
        self._show(list(self.rows), "")

    def on_dims_changed(self, event: DimsChangedEvent) -> None:
        """The view changed between 2D and 3D: a gizmo may be blocked now."""
        if event.displayed_axes_changed:
            self._show(list(self.rows), "")

    def on_store_changed(self, event: DataStoreMetadataChangedEvent) -> None:
        """The store's extent changed: the position ranges follow it."""
        if self._bounds_source is None:
            return
        bounds = [[float(low), float(high)] for low, high in self._bounds_source()]
        if bounds == self.bounds:
            return
        self.bounds = bounds
        self._show(list(self.rows), "")

    def describe(self) -> list[Row]:
        """The rows with what a front end needs to draw each one.

        Adds ``facing`` (:func:`facing_of`) and the slider's ``low`` /
        ``high``.  With a gizmo toggle, also ``gizmo`` (whether this plane
        has the canvas's gizmo) and ``gizmo_blocked`` (why it cannot have
        one, or ``""``); without, neither key.
        """
        described = []
        for row in self.rows:
            low, high = position_range(row["normal"], self.bounds)
            entry = {
                **row,
                "facing": facing_of(row["normal"]),
                "low": min(low, row["position"]),
                "high": max(high, row["position"]),
            }
            if self._gizmo is not None:
                entry["gizmo"] = row["id"] == self.gizmo_plane
                entry["gizmo_blocked"] = (
                    ""
                    if self._gizmo_blocked is None
                    else self._gizmo_blocked(row["id"])
                )
            described.append(entry)
        return described
