"""The control models of the ``controls=`` list.

``controls=`` on every ``Viewer`` / ``OrthoViewer`` ``add_*`` method is a
list of the models in this module, or ``None`` for no controls.  An item in
the list means its control is shown, its position is its place in the dock,
and its options are its own fields::

    viewer.add_image_multiscale(
        store,
        controls=[
            ImageControl(fields=["visible", "color_map", "clim"], decimals=0),
            LevelOfDetailControl(),
            BoundingBoxControl(),
            ClippingPlanesControl(),
            DatasetInfoControl(),
        ],
    )

The order of the list is the order of the dock: to reorder the controls,
move the lines; to drop one, delete its line.  Nothing is added that the
list does not name, and there is no default list.

An image's fields are drawn by one control, ``ImageControl``; every other
visual's by ``FieldControls``, one control per name.  ``FIELD_KINDS`` holds
the field names of each kind of visual and ``ALLOWED_CONTROLS`` the models
its list may hold; ``validate_controls`` checks a list against both when
the visual is added.

These are convenience-layer objects only: they are not part of the core
cellier model and are not serialized.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from cellier.visuals._removed import REMOVED_CONTROL_KEYS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cellier.data._dataset_info import DatasetInfo

VisualKind = Literal[
    "image",
    "multiscale_image",
    "labels",
    "multiscale_labels",
    "mesh",
    "multiscale_mesh",
    "points",
    "lines",
    "graph",
]
"""The kinds of visual, one per ``add_*`` method of the viewers."""


def _check_fields(fields: object, owner: str) -> list[str]:
    """Check the shape of a ``fields`` list and return it as a list.

    Only the shape: which names are valid depends on the kind of visual,
    which :func:`validate_controls` checks.
    """
    if isinstance(fields, str) or not isinstance(fields, (list, tuple)):
        raise TypeError(
            f"{owner}.fields must be a list of field names, got {fields!r}."
        )
    if not fields:
        raise ValueError(
            f"{owner}.fields must name at least one field. There is no "
            f"default list; to show no field control, leave {owner} out of "
            "the controls list."
        )
    seen: set[str] = set()
    for name in fields:
        if not isinstance(name, str):
            raise TypeError(
                f"{owner}.fields must be a list of field names, got the entry {name!r}."
            )
        if name in seen:
            raise ValueError(f"{owner}.fields names {name!r} more than once.")
        seen.add(name)
    return list(fields)


@dataclass
class Control:
    """Base class of the control models.  It has no fields."""


@dataclass
class ImageControl(Control):
    """The image control: every image field in one widget.

    A shared section, a composite switch when the visual has a channel
    axis, and a page per mode.  The blending and interpolation rows and the
    switch are always there.

    Parameters
    ----------
    fields : list[str]
        The rows to show, e.g. ``["visible", "color_map", "clim"]``.
        Required and not empty.  The list chooses rows; it does not order
        them: the control has its own layout.  Names: ``visible``,
        ``opacity``, ``color_map``, ``clim``, ``render_mode``,
        ``iso_threshold``, and on a multiscale image ``attenuation``.
    colormap_names : list[str] or None
        Names available in the colormap dropdowns.  A curated list when
        ``None``.
    clim_range : tuple[float, float] or None
        ``(min, max)`` bounds of the contrast-limits sliders.  Inferred from
        the current limits when ``None``.
    channel_labels : dict[int, str] or None
        Per-channel names on the composite page.  ``"Channel {i}"`` when
        ``None``.
    decimals : int
        Decimal places shown for values in data units: the contrast limits
        and the iso threshold.  Default 2.  Set it to suit the data, e.g. 0
        for integer images; fractions such as opacity always show 2.
    """

    fields: list[str]
    colormap_names: list[str] | None = None
    clim_range: tuple[float, float] | None = None
    channel_labels: dict[int, str] | None = None
    decimals: int = 2

    def __post_init__(self) -> None:
        """Check the shape of ``fields`` and ``decimals``."""
        self.fields = _check_fields(self.fields, type(self).__name__)
        if isinstance(self.decimals, bool) or not isinstance(self.decimals, int):
            raise TypeError(
                f"{type(self).__name__}.decimals must be an int, got {self.decimals!r}."
            )
        if self.decimals < 0:
            raise ValueError(
                f"{type(self).__name__}.decimals must be >= 0, got {self.decimals}."
            )


@dataclass
class FieldControls(Control):
    """One control per appearance field, for every visual but an image.

    Parameters
    ----------
    fields : list[str]
        The fields to show, one control each, drawn in the order of this
        list.  Required and not empty.  The names a visual accepts depend
        on its kind (see ``FIELD_KINDS``); a mesh carries the flat fields
        (``wireframe``, ``wireframe_thickness``) or the phong fields
        (``shininess``, ``flat_shading``), not both, and a field the
        visual's appearance model lacks warns and is not built.

    Notes
    -----
    On labels, ``render_mode`` is the labels render mode
    (``iso_categorical`` / ``flat_categorical``), not the image one.
    """

    fields: list[str]

    def __post_init__(self) -> None:
        """Check the shape of ``fields``."""
        self.fields = _check_fields(self.fields, type(self).__name__)


@dataclass
class LevelOfDetailControl(Control):
    """The level-of-detail control of a multiscale image or labels visual.

    The settled bias and "coarsest while moving".
    """


@dataclass
class BoundingBoxControl(Control):
    """The bounding box control: on or off, line width and colour."""


@dataclass
class ClippingPlanesControl(Control):
    """The visual's clipping planes control.

    A row per plane (on or off, the normal, a position slider along it) and
    a button to add one.  Values are in the visual's data coordinates.

    On a ``Viewer`` it is drawn in the appearance dock, at its place in the
    list.  On an ``OrthoViewer`` it is drawn in the
    ``OrthoClippingControls()`` dock, one control per group of linked
    panels, and its place in the list is ignored.
    """


@dataclass
class RenderPlanesControl(Control):
    """The control of the planes of the ``"plane"`` render mode.

    A row per plane (on or off, the normal, a position slider, the extents)
    and a button to add one.  Values are in world units.  It is disabled,
    with the reason shown, while the visual is not in plane mode or the
    view is 2D.

    On an ``OrthoViewer`` it is drawn for the 3D view's group only: the 2D
    views draw no render planes.
    """


@dataclass
class OutlineControl(Control):
    """The visual's outline control.

    The palette slot it is outlined in and the placement; for a labels
    visual, the per-label selection.  The outline reads the pick buffer:
    choosing a slot turns ``pick_write`` on with a warning.  List
    :class:`PickingControl` to show that flag.
    """


@dataclass
class AmbientOcclusionControl(Control):
    """The visual's ambient-occlusion checkbox.

    Ambient occlusion reads the pick buffer.  List :class:`PickingControl`
    to show that flag.
    """


@dataclass
class PickingControl(Control):
    """The ``pick_write`` checkbox: whether the visual is in the pick buffer."""


@dataclass
class LoadingIndicator(Control):
    """How far the visual's data has loaded: a bar and a status line.

    For a multiscale image or labels visual, and for a mesh, which is not
    drawn while a new position loads.
    """


@dataclass
class LoadingConfigControl(Control):
    """A control for every ``ProgressiveLoadingConfig`` field.

    Advanced settings of a multiscale image or labels visual.
    """


@dataclass
class MeshSectionControl(Control):
    """The "2D section" group of a mesh.

    How the mesh is drawn in a 2D view: outline, fill, outline width, and
    whether the cut is the slice plane or the scene's slab (see
    :class:`~cellier.visuals.MeshSectionConfig`).
    """


@dataclass
class MeshLodControl(Control):
    """The "LOD" group of a multiscale mesh.

    What a dims scrub loads and draws, and what a moving camera draws (see
    :class:`~cellier.visuals.GeometryLodConfig`).  A multiscale mesh with
    one level has no such settings and gets no group.
    """


@dataclass
class TrailControl(Control):
    """The trail-window controls of a graph.

    On or off, the extent before and after the slice position, and fade,
    for one or more of the graph's data axes.

    Parameters
    ----------
    axes : list[int | str] or None
        ``None`` (default) offers the axes that already have a window in
        ``GraphVisual.trail``, or the store's time axes when none do.  A
        list names the axes by data-axis index or name, which is how to
        offer an axis that has no window yet.  A name or index is resolved
        against the store when the dock is built.
    """

    axes: list[int | str] | None = None

    def __post_init__(self) -> None:
        """Check the shape of ``axes``.

        Only the shape: whether an axis exists depends on the graph's
        store, which is checked when the dock is built.
        """
        axes = self.axes
        if axes is None:
            return
        if (
            not isinstance(axes, (list, tuple))
            or not axes
            or not all(
                isinstance(axis, (int, str)) and not isinstance(axis, bool)
                for axis in axes
            )
        ):
            raise TypeError(
                "TrailControl.axes must be None or a list of data-axis indices "
                f"or names that is not empty; got {axes!r}."
            )


@dataclass
class DatasetInfoControl(Control):
    """The read-only dataset-info block.

    Parameters
    ----------
    info : None or DatasetInfo or Sequence[tuple[str, str]]
        ``None`` (default) asks the visual's data store to describe itself
        via ``BaseDataStore.dataset_info()``: it stays correct as the store
        changes and costs no array reads.  A
        :class:`~cellier.data.DatasetInfo` is displayed as given.  A
        sequence of ``(label, value)`` pairs is for rows that no store
        knows about; both halves are coerced to ``str``.

        Values are displayed, never interpreted as markup.
    """

    info: DatasetInfo | Sequence[tuple[str, str]] | None = None


_IMAGE_FIELDS: dict[str, str] = {
    "visible": "image",
    "opacity": "image",
    "color_map": "image",
    "clim": "image",
    "render_mode": "image",
    "iso_threshold": "image",
}
_LABELS_FIELDS: dict[str, str] = {
    "visible": "visible",
    "opacity": "opacity",
    "render_mode": "labels_render_mode",
    "smooth_geometry_level": "smooth_geometry_level",
    "salt": "salt",
    "background_label": "background_label",
}
_MESH_FIELDS: dict[str, str] = {
    "visible": "visible",
    "opacity": "opacity",
    "color": "color",
    "side": "side",
    "wireframe": "wireframe",
    "wireframe_thickness": "wireframe_thickness",
    "shininess": "shininess",
    "flat_shading": "flat_shading",
}

FIELD_KINDS: dict[str, dict[str, str]] = {
    "image": _IMAGE_FIELDS,
    "multiscale_image": {**_IMAGE_FIELDS, "attenuation": "image"},
    "labels": _LABELS_FIELDS,
    "multiscale_labels": _LABELS_FIELDS,
    "mesh": _MESH_FIELDS,
    "multiscale_mesh": _MESH_FIELDS,
    "points": {
        "visible": "visible",
        "opacity": "opacity",
        "color": "color",
        "size": "size",
        "size_space": "size_space",
    },
    "lines": {
        "visible": "visible",
        "opacity": "opacity",
        "color": "color",
        "thickness": "thickness",
        "thickness_space": "thickness_space",
    },
    "graph": {
        "visible": "visible",
        "opacity": "opacity",
        "node_visible": "node_visible",
        "node_color": "node_color",
        "node_size": "node_size",
        "node_size_space": "node_size_space",
        "edge_visible": "edge_visible",
        "edge_color": "edge_color",
        "edge_thickness": "edge_thickness",
        "edge_thickness_space": "edge_thickness_space",
    },
}
"""Per visual kind: field name to the kind of control that drives it.

Keyed by visual kind because names collide: ``render_mode`` is
``mip``/``iso``/``minip`` on an image and
``iso_categorical``/``flat_categorical`` on labels, and one global table
would hand a labels visual the image widget.

A name is here if and only if a widget exists for it.
"""

_COMMON_CONTROLS: tuple[type[Control], ...] = (
    BoundingBoxControl,
    ClippingPlanesControl,
    OutlineControl,
    AmbientOcclusionControl,
    PickingControl,
    DatasetInfoControl,
)

ALLOWED_CONTROLS: dict[str, tuple[type[Control], ...]] = {
    "image": (ImageControl, RenderPlanesControl, *_COMMON_CONTROLS),
    "multiscale_image": (
        ImageControl,
        LevelOfDetailControl,
        RenderPlanesControl,
        LoadingIndicator,
        LoadingConfigControl,
        *_COMMON_CONTROLS,
    ),
    "labels": (FieldControls, RenderPlanesControl, *_COMMON_CONTROLS),
    "multiscale_labels": (
        FieldControls,
        LevelOfDetailControl,
        RenderPlanesControl,
        LoadingIndicator,
        LoadingConfigControl,
        *_COMMON_CONTROLS,
    ),
    "mesh": (FieldControls, MeshSectionControl, LoadingIndicator, *_COMMON_CONTROLS),
    "multiscale_mesh": (
        FieldControls,
        MeshSectionControl,
        MeshLodControl,
        LoadingIndicator,
        *_COMMON_CONTROLS,
    ),
    "points": (FieldControls, *_COMMON_CONTROLS),
    "lines": (FieldControls, *_COMMON_CONTROLS),
    "graph": (FieldControls, TrailControl, *_COMMON_CONTROLS),
}
"""Per visual kind: the control models its ``controls=`` list may hold."""

_ADD_METHODS: dict[str, str] = {
    "image": "add_image",
    "multiscale_image": "add_image_multiscale",
    "labels": "add_labels",
    "multiscale_labels": "add_labels_multiscale",
    "mesh": "add_mesh",
    "multiscale_mesh": "add_multiscale_mesh",
    "points": "add_points",
    "lines": "add_lines",
    "graph": "add_graph",
}

_ACCEPTED_FORMS = (
    "a list of control models, e.g. controls=[BoundingBoxControl(), "
    "ClippingPlanesControl()], or None for no controls"
)


def _check_field_names(item: Control, kind: str, name: str) -> None:
    """Raise on a field name of *item* that *kind* has no control for."""
    owner = type(item).__name__
    valid = list(FIELD_KINDS[kind])
    lod_allowed = LevelOfDetailControl in ALLOWED_CONTROLS[kind]
    for field in item.fields:
        if field in valid:
            continue
        # The level-of-detail control was a field name once, and so were
        # the keys it replaced.  It is its own item of the list now.
        if (
            field == "level_of_detail"
            or REMOVED_CONTROL_KEYS.get(field) == "level_of_detail"
        ):
            where = (
                "add LevelOfDetailControl() to the controls list"
                if lod_allowed
                else "LevelOfDetailControl() is for a multiscale image or labels visual"
            )
            raise ValueError(
                f"{field!r} is not a field of {owner} (controls of {name!r}): "
                f"the level-of-detail control is its own item; {where}."
            )
        if field in REMOVED_CONTROL_KEYS:
            raise ValueError(
                f"{field!r} was removed as a field of {owner}: use "
                f"{REMOVED_CONTROL_KEYS[field]!r}."
            )
        close = difflib.get_close_matches(field, valid, n=1)
        suggestion = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(
            f"{field!r} is not a valid field of {owner} for "
            f"{_ADD_METHODS[kind]} (controls of {name!r}).{suggestion} "
            f"Valid fields: {valid}."
        )


def validate_controls(controls: object, kind: str, name: str) -> list[Control] | None:
    """Check the ``controls=`` of one ``add_*`` call.

    Called by every ``add_*`` method before anything is added.

    Parameters
    ----------
    controls : list[Control] or None
        What was passed as ``controls=``.
    kind : VisualKind
        The kind of visual the method adds.
    name : str
        What the visual is called, for the messages.

    Returns
    -------
    list[Control] or None
        A new list holding the items, or ``None`` when *controls* is
        ``None`` or empty: no controls.

    Raises
    ------
    TypeError
        If *controls* is not a list or ``None`` (``True`` and ``False``
        included), or an item is not a control model.
    ValueError
        If an item is not available for this kind of visual, a type is
        listed twice, or a field name has no control for this kind.
    """
    if controls is None:
        return None
    method = _ADD_METHODS[kind]
    if isinstance(controls, bool):
        raise TypeError(
            f"controls={controls!r} is not accepted ({method} of {name!r}): "
            "list the controls to show, e.g. controls=[BoundingBoxControl()], "
            "or pass None for no controls. There is no default list."
        )
    if isinstance(controls, Control):
        raise TypeError(
            f"controls= of {name!r} must be a list: write "
            f"controls=[{type(controls).__name__}(...)]."
        )
    if not isinstance(controls, (list, tuple)):
        raise TypeError(
            f"controls= of {name!r} must be {_ACCEPTED_FORMS}; got "
            f"{type(controls).__name__}."
        )

    allowed = ALLOWED_CONTROLS[kind]
    seen: set[type] = set()
    for item in controls:
        if isinstance(item, type) and issubclass(item, Control):
            raise TypeError(
                f"controls= of {name!r} holds the class {item.__name__}; "
                f"write {item.__name__}() to make the control."
            )
        if not isinstance(item, Control):
            raise TypeError(
                f"controls= of {name!r} must be {_ACCEPTED_FORMS}; got the "
                f"item {item!r}."
            )
        if type(item) not in allowed:
            raise ValueError(
                f"{type(item).__name__} is not available for {method} "
                f"(controls of {name!r}).{_unavailable_hint(item, kind)} "
                f"Available: {[cls.__name__ for cls in allowed]}."
            )
        if type(item) in seen:
            raise ValueError(
                f"controls= of {name!r} lists {type(item).__name__} more than "
                "once; a control is shown once."
            )
        seen.add(type(item))
        if isinstance(item, (ImageControl, FieldControls)):
            _check_field_names(item, kind, name)
    return list(controls) or None


def _unavailable_hint(item: Control, kind: str) -> str:
    """One sentence saying what to use in place of *item*, or nothing."""
    if isinstance(item, FieldControls) and ImageControl in ALLOWED_CONTROLS[kind]:
        return " An image's fields are drawn by ImageControl(fields=[...])."
    if isinstance(item, ImageControl):
        return " Use FieldControls([...]) for this visual's fields."
    if isinstance(item, MeshLodControl):
        return " It is for add_multiscale_mesh: an in-memory mesh has one level."
    return ""
