"""Shared, host-agnostic resolvers used by both layout renderers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple
from uuid import UUID

from cellier.convenience.gui._controls_config import (
    FIELD_KINDS,
    AmbientOcclusionControl,
    BoundingBoxControl,
    ClippingPlanesControl,
    DatasetInfoControl,
    FieldControls,
    ImageControl,
    LevelOfDetailControl,
    LoadingConfigControl,
    LoadingIndicator,
    MeshLodControl,
    MeshSectionControl,
    OutlineControl,
    PickingControl,
    RenderPlanesControl,
    TrailControl,
)
from cellier.gui._clipping_planes import (
    CLIPPING_PLANES_TITLE,
    get_clipping_planes_data_from_visual,
)
from cellier.gui._level_of_detail import (
    LEVEL_OF_DETAIL_TITLE,
    level_of_detail_values,
)
from cellier.gui._loading import LOADING_CONFIG_TITLE, LOADING_TITLE
from cellier.gui._lod import LOD_CONFIG_TITLE
from cellier.gui._mesh_section import MESH_SECTION_TITLE
from cellier.gui._render_controls import VISUAL_RENDER_TITLES
from cellier.gui._render_planes import RENDER_PLANES_TITLE

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cellier.convenience.gui._controls_config import Control

# ── Appearance controls: the toolkit-neutral decision layer ──────────────────
#
# Which controls a panel contains, in what order, seeded with what values, is
# one question with one answer for every toolkit.  ``appearance_specs`` is
# that answer, as pure data: no controller, no widgets, no toolkit import, so
# the whole decision layer is unit-testable with no fixtures.  The order is
# the order of the visual's ``controls=`` list.  Each renderer keeps only a
# dispatch table mapping ``ControlSpec.kind`` to its own widget class.


@dataclass(frozen=True)
class ControlSpec:
    """One control to build, described without reference to any toolkit.

    Parameters
    ----------
    kind : str
        Which control to build: ``image``, ``level_of_detail``, ``aabb``,
        ``loading``, ``loading_config``, ``mesh_section``,
        ``clipping_planes``, ``render_planes``, ``lod_config``,
        ``dataset_info``, or the kind of a single-field control such as
        ``opacity``.  A renderer with no builder for a kind skips it.
    title : str
        What the control is called, e.g. ``"Contrast limits"``.  Both front
        ends pass it to the widget, which draws it itself -- as a label beside
        a single control, or as a heading over a block of rows
        (``plans/label_ownership_unification.md``).
    values : dict[str, object]
        Construction keywords for the widget, already read off the model.
        Toolkit-neutral: a keyword only one toolkit accepts (Qt's
        ``dtype_max``) is derived inside that toolkit's builder from what is
        here.
    """

    kind: str
    title: str
    values: dict[str, object]


class AppearanceSpecs(NamedTuple):
    """The controls to build for one visual, plus what was asked for and dropped.

    Parameters
    ----------
    specs : list[ControlSpec]
        The controls, in display order.
    skipped : list[str]
        Listed field names that produced no control because the visual's
        appearance model does not carry them (``shininess`` on a mesh with
        the flat model).  Reported rather than silently dropped; a name the
        kind of visual has no control for at all is refused earlier, by
        ``validate_controls``.
    """

    specs: list[ControlSpec]
    skipped: list[str]


_CONTROL_TITLES = {
    "image": "Image",
    "level_of_detail": LEVEL_OF_DETAIL_TITLE,
    "aabb": "Bounding box",
    "trail": "Trail",
    "loading": LOADING_TITLE,
    "loading_config": LOADING_CONFIG_TITLE,
    "mesh_section": MESH_SECTION_TITLE,
    "clipping_planes": CLIPPING_PLANES_TITLE,
    "render_planes": RENDER_PLANES_TITLE,
    "lod_config": LOD_CONFIG_TITLE,
    # Read rather than restated: the per-visual groups name themselves in
    # the shared control spec, beside the controls they hold.
    **VISUAL_RENDER_TITLES,
    "dataset_info": "Dataset info",
}
"""Titles for the control kinds that need one written out.

A kind not listed here is a plain single-field control, and its title is
derived from the field name.

Each of these is also a widget class's ``DEFAULT_TITLE`` on both toolkits,
pinned by ``test_composite_default_titles_match_the_shared_vocabulary``: this
is the name the renderers pass in, that is the name a directly-constructed
widget uses.
"""


class RenderPlanesTarget(NamedTuple):
    """What a dock's render planes control edits, as its viewer names it.

    Parameters
    ----------
    visual_ids : list[UUID]
        The visuals whose ``render_planes`` the control edits.
    canvas_id : UUID or None
        The 3D canvas its gizmo toggle draws in.  ``None`` gives the rows
        no toggle.
    blocked : Callable[[], str] or None
        Why the control is disabled now, or ``""``.  ``None`` uses the
        controller's answer: the visual is not in plane mode, or the view
        is 2D.
    watch : Callable[[Callable[[], None]], None] or None
        Given the control's ``refresh``, arranges for it to be called when
        what *blocked* reads may have changed.
    """

    visual_ids: list
    canvas_id: Any = None
    blocked: Any = None
    watch: Any = None


_LEVEL_OF_DETAIL_FIELD = "settled_lod_bias"
"""The appearance field whose presence says a visual has levels of detail."""

_IMAGE_FIELD_ORDER = tuple(FIELD_KINDS["multiscale_image"])
"""Every image field name, in the order the image control is given them."""

_FIELD_CONTROL_KINDS = {
    field: kind
    for visual_kind, fields in FIELD_KINDS.items()
    if visual_kind not in ("image", "multiscale_image")
    for field, kind in fields.items()
}
"""Field name to control kind, for the single-field controls.

An image's fields are not here: they are all drawn by the image control,
and ``render_mode`` would collide with the labels field of that name.
"""


APPEARANCE_DOCK_GAP_PX = 6
"""Spacing between the stacked controls in an appearance dock, in pixels.

How far apart two controls sit is a property of the panel, not of a toolkit,
so both renderers read it here.  They used to disagree -- Qt spaced its
column by 6 and the anywidget composer by 4, under a comment claiming it was
"mirroring the 6px ``setSpacing`` the Qt dock column uses".  Six is what both
now use, since that is the value the comment says was intended.
"""


STATIC_CONTROL_KINDS = frozenset({"dataset_info"})
"""Kinds that are pure display, with nothing on the event bus.

Whether a control can be driven by the model is a fact about the control, not
about a toolkit, so it is declared once here and read by both renderers.  A
kind listed here is built and stacked like any other and then *not* passed to
``CellierController.connect_widget``: it has no ``changed``/``closed`` to
connect and no subscriptions to register.
"""


def _default_title(field: str) -> str:
    """Title for a single-field control, e.g. ``wireframe_thickness``."""
    return field.replace("_", " ").capitalize()


def _resolve_data_store(controller: object, visual: object) -> object | None:
    """Return the visual's backing data store, or ``None`` if there is none.

    A visual with no controller or no
    ``data_store_id`` yields ``None`` quietly -- there is nothing to look up.
    A lookup that *fails* is different: the visual names a store the
    controller does not have, which is a wiring bug, and it warns rather
    than silently producing a panel with the block missing.

    ``BaseVisual.data_store_id`` is a ``str`` while
    ``CellierController._model.data.stores`` is keyed by ``UUID``, so the id
    is coerced here; passing the string through looks up nothing and finds
    nothing.
    """
    store_id = getattr(visual, "data_store_id", None)
    if controller is None or store_id is None:
        return None

    key = UUID(store_id) if isinstance(store_id, str) else store_id
    try:
        return controller.get_data_store(key)
    except (KeyError, AttributeError):
        warnings.warn(
            f"Visual {getattr(visual, 'name', visual)!r} names data store "
            f"{store_id} but the controller has no such store; the "
            f"dataset-info block was not built.",
            UserWarning,
            stacklevel=3,
        )
        return None


def appearance_specs(
    visual: object,
    controls: Sequence[Control] | None,
    store: object | None = None,
    palette: object = (),
) -> AppearanceSpecs:
    """Decide which appearance controls *visual* gets from its *controls* list.

    Pure: reads the visual and the list, and returns data.  Both renderers
    walk the result and build widgets from it.

    The specs follow the list: each item is handed to its builder in
    ``_SPEC_BUILDERS``, which returns the specs it stands for, and they are
    appended in turn.  So the order of the list is the order of the dock.
    A builder returns nothing when the visual or the store cannot support
    its control (no store for the clipping planes, a mesh with one level
    for the mesh LOD group); the rest of the list keeps its order.

    Parameters
    ----------
    visual : BaseVisual
        The visual whose appearance the panel will drive.
    controls : list[Control] or None
        The recorded ``controls=`` list.  ``None`` or empty gives no specs.
    store : BaseDataStore or None
        The visual's backing data store, needed by the clipping planes,
        trail, loading config and dataset info controls.  Passed in rather
        than looked up so this stays pure: the shared decision layer takes
        no controller and imports no toolkit.  A renderer resolves it from
        ``visual.data_store_id``.
    palette : sequence of RGBA
        The outline palette, needed only by the outline control, whose slot
        swatches offer exactly the slots it holds.  Passed in for the same
        reason *store* is.

    Returns
    -------
    AppearanceSpecs
    """
    specs: list[ControlSpec] = []
    skipped: list[str] = []
    for item in controls or ():
        specs.extend(_SPEC_BUILDERS[type(item)](item, visual, store, palette, skipped))
    return AppearanceSpecs(specs, skipped)


# -- One builder per control model -------------------------------------------
#
# Each takes ``(item, visual, store, palette, skipped)`` and returns the specs
# the item stands for: none, one or several.  *skipped* collects the field
# names the visual's appearance model does not carry.


def _carried_fields(visual: object, fields: list[str]) -> list[str]:
    """The names of *fields* the visual's appearance model carries.

    An image's mode-dependent fields live on ``single`` (unified image
    design 3.1), not on the shared appearance.
    """
    app = getattr(visual, "appearance", None)
    single = getattr(visual, "single", None)
    return [
        field
        for field in fields
        if hasattr(app, field) or (single is not None and hasattr(single, field))
    ]


def _image_specs(
    item: ImageControl, visual: object, store: object, palette: object, skipped: list
) -> list[ControlSpec]:
    """The one image control, showing the rows *item* names."""
    from cellier.gui._image_controls import image_control_values

    carried = set(_carried_fields(visual, item.fields))
    skipped.extend(sorted(set(item.fields) - carried))
    if not carried:
        return []
    # The list chooses rows; the control has its own layout (design D1).
    fields = [field for field in _IMAGE_FIELD_ORDER if field in carried]
    return [
        ControlSpec(
            "image",
            _CONTROL_TITLES["image"],
            image_control_values(
                visual,
                fields=fields,
                colormap_names=item.colormap_names,
                clim_range=item.clim_range,
                channel_labels=item.channel_labels,
                decimals=item.decimals,
                axis_names=_axis_names(store),
            ),
        )
    ]


def _axis_names(store: object) -> tuple[str, ...]:
    """The level-0 axis names of *store*, or none when it has no systems."""
    systems = getattr(store, "data_coordinate_systems", None) or ()
    if not systems:
        return ()
    return tuple(systems[0].axis_names())


def _field_specs(
    item: FieldControls, visual: object, store: object, palette: object, skipped: list
) -> list[ControlSpec]:
    """One single-field control per name, in the order of *item*'s list."""
    from cellier.gui._appearance_fields import (
        APPEARANCE_FIELD_WIDGETS,
        literal_choices,
    )

    carried = set(_carried_fields(visual, item.fields))
    skipped.extend(sorted(set(item.fields) - carried))
    app = getattr(visual, "appearance", None)
    specs = []
    for field in item.fields:
        if field not in carried:
            continue
        kind = _FIELD_CONTROL_KINDS.get(field, field)
        values = {"initial_value": getattr(app, field)}
        # A Literal field's options come from the model's own annotation,
        # so the in-memory and multiscale variants of a field each offer
        # their own set without a list restated here.
        choices = literal_choices(app, field)
        if choices:
            values["choices"] = choices
        stem_title = APPEARANCE_FIELD_WIDGETS.get(kind)
        title = stem_title[1] if stem_title else _default_title(field)
        specs.append(ControlSpec(kind, title, values))
    return specs


def _level_of_detail_specs(
    item: object, visual: object, store: object, palette: object, skipped: list
) -> list[ControlSpec]:
    """The level-of-detail control, for a model that has the settled bias."""
    app = getattr(visual, "appearance", None)
    if not hasattr(app, _LEVEL_OF_DETAIL_FIELD):
        return []
    return [
        ControlSpec(
            "level_of_detail",
            _CONTROL_TITLES["level_of_detail"],
            level_of_detail_values(app),
        )
    ]


def _aabb_specs(
    item: object, visual: object, store: object, palette: object, skipped: list
) -> list[ControlSpec]:
    """The bounding box control.

    ``aabb`` is on ``BaseVisual`` with a default factory, so every visual
    has one.
    """
    aabb = getattr(visual, "aabb", None)
    if aabb is None:
        return []
    return [
        ControlSpec(
            "aabb",
            _CONTROL_TITLES["aabb"],
            {
                "initial_enabled": aabb.enabled,
                "initial_line_width": aabb.line_width,
                "initial_color": aabb.color,
            },
        )
    ]


def _mesh_section_specs(
    item: object, visual: object, store: object, palette: object, skipped: list
) -> list[ControlSpec]:
    """A mesh's 2D section: outline, fill, outline width, mode."""
    section = getattr(visual, "section", None)
    if section is None:
        return []
    return [
        ControlSpec(
            "mesh_section",
            _CONTROL_TITLES["mesh_section"],
            {"section": section.model_dump()},
        )
    ]


def _clipping_planes_specs(
    item: object, visual: object, store: object, palette: object, skipped: list
) -> list[ControlSpec]:
    """The visual's clipping planes: a row per plane.

    Every visual type has them; they are edited in the store's data
    coordinates, so there is no control without a store.
    """
    if store is None:
        return []
    return [
        ControlSpec(
            "clipping_planes",
            _CONTROL_TITLES["clipping_planes"],
            get_clipping_planes_data_from_visual(visual, store),
        )
    ]


def _render_planes_specs(
    item: object, visual: object, store: object, palette: object, skipped: list
) -> list[ControlSpec]:
    """The planes of the "plane" render mode: a row per plane.

    Only for a visual that has the mode (an image or labels).  The control
    reads the scene through the controller, so its builder fills it in.
    """
    if not hasattr(visual, "render_planes"):
        return []
    return [ControlSpec("render_planes", _CONTROL_TITLES["render_planes"], {})]


def _mesh_lod_specs(
    item: object, visual: object, store: object, palette: object, skipped: list
) -> list[ControlSpec]:
    """A multiscale mesh's level-of-detail settings.

    What a dims scrub loads and draws, and what a moving camera draws.  A
    mesh with one level has none.
    """
    lod = getattr(visual, "lod", None)
    if lod is None:
        return []
    return [
        ControlSpec(
            "lod_config", _CONTROL_TITLES["lod_config"], {"lod": lod.model_dump()}
        )
    ]


def _outline_specs(
    item: object, visual: object, store: object, palette: object, skipped: list
) -> list[ControlSpec]:
    """The visual's outline control.

    Which control is emitted depends on the visual type, because
    ``outline.slot`` means two different things.  On most visuals it chooses
    the colour, since the selection layer draws the region in
    ``palette[slot - 1]``.  On a labels visual it only decides whether the
    volume participates -- the colour comes from ``outline_selected_labels``,
    per label value -- so that one gets a checkbox and a row editor instead
    of a swatch picker.
    """
    from cellier.controller import _default_placement
    from cellier.visuals._label_memory import BaseLabelsVisual

    outline = getattr(visual, "outline", None)
    if outline is None:
        return []
    is_labels = isinstance(visual, BaseLabelsVisual)
    kind = "labels_outline" if is_labels else "visual_outline"
    values = {
        "outline.slot": outline.slot,
        "outline.placement": outline.placement,
        "default_placement": _default_placement(visual),
        # The palette travels in the spec so both front ends read the
        # slots on offer from one place; each widget then follows it
        # live through RenderConfigChangedEvent.
        "palette": [tuple(entry) for entry in palette],
    }
    if is_labels:
        values["outline_selected_labels"] = dict(visual.outline_selected_labels)
        # The mode decides which controls the panel draws (slot swatches
        # or the per-label rows), and both front ends fall back to
        # ``per_label`` without it -- which misdrew every visual in
        # another mode.
        values["outline_mode"] = visual.outline_mode
        # What the depth range row shows in ``depth_threshold`` mode.  It
        # travels in every mode, because the mode can be entered from the
        # panel.
        values["outline_depth_range"] = tuple(visual.outline_depth_range)
    return [ControlSpec(kind, _CONTROL_TITLES[kind], values)]


def _occlusion_specs(
    item: object, visual: object, store: object, palette: object, skipped: list
) -> list[ControlSpec]:
    """The visual's ambient-occlusion checkbox."""
    if getattr(visual, "outline", None) is None:
        return []
    return [
        ControlSpec(
            "visual_occlusion",
            _CONTROL_TITLES["visual_occlusion"],
            {"ambient_occlusion": visual.ambient_occlusion},
        )
    ]


def _picking_specs(
    item: object, visual: object, store: object, palette: object, skipped: list
) -> list[ControlSpec]:
    """The ``pick_write`` checkbox.

    The outline and ambient occlusion both read the pick buffer, so this
    is the flag that gates them.
    """
    if getattr(visual, "outline", None) is None:
        return []
    return [
        ControlSpec(
            "visual_picking",
            _CONTROL_TITLES["visual_picking"],
            {"pick_write": visual.pick_write},
        )
    ]


def _loading_specs(
    item: object, visual: object, store: object, palette: object, skipped: list
) -> list[ControlSpec]:
    """How far the visual's data has loaded.

    A multiscale visual loads progressively; its render config says so by
    carrying ``loading``.  The widget reads the current progress off the
    controller when it is built, so the spec carries nothing.  A mesh loads
    through the same scheduler and is not drawn while it loads, so it shows
    the same indicator, worded in levels.
    """
    specs = []
    if _loading_config(visual) is not None:
        specs.append(ControlSpec("loading", _CONTROL_TITLES["loading"], {}))
    if getattr(visual, "section", None) is not None:
        specs.append(
            ControlSpec("loading", _CONTROL_TITLES["loading"], {"levels": True})
        )
    return specs


def _loading_config_specs(
    item: object, visual: object, store: object, palette: object, skipped: list
) -> list[ControlSpec]:
    """A control for every field of the visual's progressive loading config."""
    loading = _loading_config(visual)
    if loading is None:
        return []
    level_shapes = getattr(store, "level_shapes", None)
    return [
        ControlSpec(
            "loading_config",
            _CONTROL_TITLES["loading_config"],
            {
                "loading": loading.model_dump(),
                "n_levels": len(level_shapes) if level_shapes else None,
            },
        )
    ]


def _loading_config(visual: object) -> object | None:
    """The visual's progressive loading config, or ``None`` if it has none."""
    return getattr(getattr(visual, "render_config", None), "loading", None)


def _trail_specs(
    item: TrailControl, visual: object, store: object, palette: object, skipped: list
) -> list[ControlSpec]:
    """The trail-window control of a graph.

    Only a graph visual has a trail.  The axes are resolved here, against
    the store's own data axes, because the list is written before it is
    paired with a store.
    """
    trail = getattr(visual, "trail", None)
    if trail is None:
        return []

    systems = getattr(store, "data_coordinate_systems", None) or []
    system = systems[0] if systems else None
    names = system.axis_names() if system is not None else ()

    if item.axes is None:
        axes = sorted(trail)
        if not axes and system is not None:
            axes = [
                index
                for index, axis in enumerate(system.axes)
                if axis.axis_type == "time"
            ]
    else:
        axes = [_resolve_trail_axis(ref, names) for ref in item.axes]
    if not axes:
        warnings.warn(
            f"TrailControl() on {getattr(visual, 'name', 'a graph')!r} found "
            "no axis to offer: the graph has no trail window and no time axis. "
            "Name the axes instead, e.g. TrailControl(axes=[0]).",
            UserWarning,
            stacklevel=3,
        )
        return []

    return [
        ControlSpec(
            "trail",
            _CONTROL_TITLES["trail"],
            {
                "axes": [
                    (axis, names[axis] if axis < len(names) else f"axis {axis}")
                    for axis in dict.fromkeys(axes)
                ],
                # Copies: the widget keeps what it was seeded with, and must
                # never hold the objects the controller is wired to.
                "trail": {axis: window.model_copy() for axis, window in trail.items()},
            },
        )
    ]


def _resolve_trail_axis(ref: int | str, names: tuple[str, ...]) -> int:
    """Resolve one ``TrailControl.axes`` entry to a data-axis index."""
    if isinstance(ref, str):
        matches = [index for index, name in enumerate(names) if name == ref]
        if len(matches) != 1:
            problem = "is not one of" if not matches else "is ambiguous among"
            raise ValueError(
                f"TrailControl axis {ref!r} {problem} the graph's data axes "
                f"{list(names)}."
            )
        return matches[0]
    if not 0 <= ref < len(names):
        raise ValueError(
            f"TrailControl axis {ref} is out of range for a {len(names)}-axis graph."
        )
    return int(ref)


def _dataset_info_specs(
    item: DatasetInfoControl,
    visual: object,
    store: object,
    palette: object,
    skipped: list,
) -> list[ControlSpec]:
    """The read-only dataset-info block, from ``item.info``.

    ``None``
        Ask *store* to describe itself.  Yields nothing when no store was
        resolved -- the caller could not find one, and a block asserting
        that a store has no metadata would be worse than no block.
    a ``DatasetInfo``
        Displayed as given.
    a sequence of pairs
        The hand-authored escape hatch, carried through as flat rows.
    """
    from cellier.data._dataset_info import DatasetInfo

    requested = item.info
    if requested is None:
        if store is None or not hasattr(store, "dataset_info"):
            return []
        info: DatasetInfo | None = store.dataset_info()
    elif isinstance(requested, DatasetInfo):
        info = requested
    else:
        rows = [(str(label), str(value)) for label, value in requested]
        if not rows:
            return []
        return [
            ControlSpec("dataset_info", _CONTROL_TITLES["dataset_info"], {"rows": rows})
        ]

    if info is None or not info.sections:
        return []
    return [
        ControlSpec("dataset_info", _CONTROL_TITLES["dataset_info"], {"info": info})
    ]


_SPEC_BUILDERS: dict[type[Control], Callable[..., list[ControlSpec]]] = {
    ImageControl: _image_specs,
    FieldControls: _field_specs,
    LevelOfDetailControl: _level_of_detail_specs,
    BoundingBoxControl: _aabb_specs,
    ClippingPlanesControl: _clipping_planes_specs,
    RenderPlanesControl: _render_planes_specs,
    OutlineControl: _outline_specs,
    AmbientOcclusionControl: _occlusion_specs,
    PickingControl: _picking_specs,
    LoadingIndicator: _loading_specs,
    LoadingConfigControl: _loading_config_specs,
    MeshSectionControl: _mesh_section_specs,
    MeshLodControl: _mesh_lod_specs,
    TrailControl: _trail_specs,
    DatasetInfoControl: _dataset_info_specs,
}
"""Control model to the function that turns one into its specs."""


def warn_skipped_appearance_fields(skipped: list[str], visual: object) -> None:
    """Warn about listed fields that produced no control.

    ``validate_controls`` refuses a name the kind of visual has no control
    for.  That leaves one case it cannot catch: a name valid for the kind
    whose value the particular *visual*'s appearance model does not carry
    -- ``shininess`` on a mesh with the flat model, say.  The renderer
    reports it because only the renderer has both halves.
    """
    if not skipped:
        return
    warnings.warn(
        f"The controls list of {getattr(visual, 'name', visual)!r} names "
        f"field(s) {skipped} that {type(visual).__name__}'s appearance model "
        "does not have; no control was built for them.",
        UserWarning,
        stacklevel=3,
    )


class ControlTarget(NamedTuple):
    """One configured visual a controls dock can drive.

    Parameters
    ----------
    key : UUID
        The representative visual id the controls list is recorded under.  Stable
        for as long as the visual exists, so it is what a dock's selection
        is kept by.
    label : str
        What the dock's visual selector calls this target.  Unique among the
        targets of one dock.
    visual : BaseVisual
        The representative visual.  Its appearance model is what the controls
        are seeded from; on an ``OrthoViewer`` it is the group's first panel
        visual.
    config : list[Control]
        The recorded ``controls=`` list.
    visual_ids : list[UUID]
        Every visual the controls write to -- one on a ``Viewer``; on an
        ``OrthoViewer``, the three 2D panel siblings or the 3D panel's
        visual.  The widgets accept this directly (see
        ``cellier.gui._appearance_fields.VisualIdGroup``), so the spec walk is
        identical either way and only the id list differs.
    """

    key: UUID
    label: str
    visual: object
    config: object
    visual_ids: list


def appearance_targets(viewer: object) -> list[ControlTarget]:
    """Every visual an ``AppearanceControls()`` dock can drive, in add order.

    A group is offered only when at least one item of its controls list
    draws in the appearance dock for that group
    (:func:`draws_in_appearance_dock`): offering one with nothing to draw
    would only lead to an empty panel.  Multi-scene aware: an ``OrthoViewer``
    records two groups per fanned-out add, the 2D panels and the 3D panel,
    each keyed by its first visual and labelled by the viewer; the
    ``_visual_groups`` record expands each key to its visuals.
    """
    return _control_targets(viewer)


def draws_in_appearance_dock(
    viewer: object, controls: Sequence[Control], visual_ids: list
) -> bool:
    """Whether any item of *controls* draws in the appearance dock.

    For the visual group *visual_ids* of *viewer*.  Every control model
    draws there but two:

    * ``ClippingPlanesControl`` does on a ``Viewer``; an ``OrthoViewer``
      draws it in its ``OrthoClippingControls()`` dock
      (``_CLIPPING_DOCK_NODE``).
    * ``RenderPlanesControl`` does unless the viewer names no target for
      the group (``_render_planes_target`` returns ``None``): the 2D views
      of an ``OrthoViewer``.

    The viewer is asked; no widget is built to find out.
    """
    for item in controls:
        if isinstance(item, ClippingPlanesControl):
            clipping_node = getattr(viewer, "_CLIPPING_DOCK_NODE", "AppearanceControls")
            if clipping_node == "AppearanceControls":
                return True
        elif isinstance(item, RenderPlanesControl):
            target = getattr(viewer, "_render_planes_target", None)
            if target is None or target(list(visual_ids)) is not None:
                return True
        else:
            return True
    return False


def _control_targets(viewer: object) -> list[ControlTarget]:
    """Resolve the recorded controls lists into labelled targets.

    Registration order, not scene order: it is the order the user added
    things in, and it is the same on a ``Viewer`` and an ``OrthoViewer``.  An
    id the controller no longer knows is skipped rather than raised on; the
    viewer prunes removed visuals, so reaching one means a stale read, and a
    dock should show what exists.
    """
    controller = getattr(viewer, "controller", None)
    controls_configs: dict = getattr(viewer, "_controls_configs", {}) or {}
    if controller is None or not controls_configs:
        return []
    groups: dict = getattr(viewer, "_visual_groups", {}) or {}
    given_labels: dict = getattr(viewer, "_controls_labels", {}) or {}

    resolved = []
    for rep_id, config in controls_configs.items():
        if not config:
            continue
        try:
            visual = controller.get_visual_model(rep_id)
        except KeyError:
            continue
        visual_ids = list(groups.get(rep_id, [rep_id]))
        if not draws_in_appearance_dock(viewer, config, visual_ids):
            continue
        resolved.append(
            (
                rep_id,
                given_labels.get(rep_id) or _group_name(controller, visual, visual_ids),
                visual,
                config,
                visual_ids,
            )
        )

    labels = unique_labels([name for _, name, _, _, _ in resolved])
    return [
        ControlTarget(rep_id, label, visual, config, visual_ids)
        for (rep_id, _, visual, config, visual_ids), label in zip(resolved, labels)
    ]


OVERLAY_PLACEHOLDER = "No overlays"
"""What an overlay dock says while the viewer has no overlays."""


def overlay_targets(viewer: object) -> list[ControlTarget]:
    """Every overlay an ``OverlayControls()`` dock can drive.

    In :attr:`Viewer.overlays` order.  Overlays of a type with no controls
    table entry are skipped rather than offered with an empty panel.  The
    target's ``visual`` slot holds the overlay model and ``config`` is
    ``None``: an overlay has no ``controls=`` list, and its controls are
    fixed by its type.
    """
    from cellier.gui._overlay_fields import OVERLAY_CONTROLS

    overlays = [
        overlay
        for overlay in getattr(viewer, "overlays", ()) or ()
        if getattr(overlay, "overlay_type", None) in OVERLAY_CONTROLS
    ]
    labels = unique_labels([str(overlay.name) for overlay in overlays])
    return [
        ControlTarget(overlay.id, label, overlay, None, [overlay.id])
        for overlay, label in zip(overlays, labels)
    ]


def overlay_control_specs(overlay: object) -> list[ControlSpec]:
    """The controls for one overlay, in display order.

    Each spec's ``kind`` is the dotted field path the control writes, which
    is also the key of ``OVERLAY_FIELD_WIDGETS``.  ``values`` carries the
    initial value read off the model and, for a ``Literal`` field, the
    choices it admits.
    """
    from cellier.gui._appearance_fields import literal_choices
    from cellier.gui._overlay_fields import (
        OVERLAY_CONTROLS,
        OVERLAY_FIELD_WIDGETS,
        overlay_field_value,
    )

    specs = []
    for field in OVERLAY_CONTROLS.get(getattr(overlay, "overlay_type", ""), ()):
        values: dict[str, object] = {
            "initial_value": overlay_field_value(overlay, field)
        }
        parent_path, _, attribute = field.rpartition(".")
        if parent_path:
            choices = literal_choices(
                overlay_field_value(overlay, parent_path), attribute
            )
            if choices:
                values["choices"] = choices
        specs.append(ControlSpec(field, OVERLAY_FIELD_WIDGETS[field][1], values))
    return specs


def _group_name(controller: object, visual: object, visual_ids: list) -> str:
    """The name a visual group goes by.

    An ``OrthoViewer`` names each panel's visual ``f"{name}_{panel}"``, so a
    group whose members all share the part before the last underscore is
    called by that part.  Anything else is called by its representative.
    """
    name = str(getattr(visual, "name", "") or "")
    if len(visual_ids) < 2:
        return name
    names = []
    for visual_id in visual_ids:
        try:
            names.append(str(controller.get_visual_model(visual_id).name))
        except KeyError:
            continue
    bases = {sibling.rsplit("_", 1)[0] for sibling in names if "_" in sibling}
    if len(bases) == 1 and len(names) == sum("_" in n for n in names):
        return bases.pop()
    return name


def unique_labels(names: list[str]) -> list[str]:
    """Make *names* unique, keeping the first of each as it is.

    ``["image", "image", "mesh"]`` becomes ``["image", "image (2)", "mesh"]``.
    Visual names are not unique -- every ``add_image`` defaults to
    ``"image"`` -- and a selector offering two identical entries cannot be
    used.  An empty name is called ``"visual"``.
    """
    taken: set[str] = set()
    seen: dict[str, int] = {}
    labels = []
    for raw in names:
        base = raw or "visual"
        count = seen.get(base, 0) + 1
        seen[base] = count
        label = base if count == 1 else f"{base} ({count})"
        while label in taken:
            count += 1
            seen[base] = count
            label = f"{base} ({count})"
        taken.add(label)
        labels.append(label)
    return labels


def next_selection(
    previous: ControlTarget | None, targets: list[ControlTarget]
) -> ControlTarget | None:
    """Which target a dock shows after its targets change.

    The previous selection is kept while it exists -- matched by key, or by
    controls list when an ``OrthoViewer`` re-keyed its group after the
    representative panel's visual was removed.  A newly added visual does not
    take the selection.  When the selection is gone, or there was none, the
    first target is shown; with no targets, nothing is.
    """
    if not targets:
        return None
    if previous is not None:
        for target in targets:
            if target.key == previous.key:
                return target
        for target in targets:
            if target.config is previous.config:
                return target
    return targets[0]


def render_panel_kwargs(section: str, controller: object) -> dict:
    """Extra constructor arguments one render-settings panel needs.

    The two front ends build different classes but hand them the same
    things, so the wiring lives here rather than twice in the renderers.

    Each is a *callable* rather than a value: the effective occlusion radius
    and the accumulated frame count are derived state that changes without
    any config field changing, so a panel has to be able to re-read them.

    Parameters
    ----------
    section :
        ``"outline"``, ``"ambient_occlusion"`` or ``"temporal"``.
    controller :
        The viewer's ``CellierController``.

    Returns
    -------
    dict
        Keyword arguments for that section's panel; empty for sections
        with no derived state to show.
    """
    if section == "outline":
        # ``{slot: how many visuals use it}``, so a palette editor can show
        # that slot 2 is three visuals rather than leaving the reader to hold
        # it in their head.  Both panel bases have accepted this since they
        # were written and both front ends draw it; nothing ever passed it,
        # so the readout was dead on arrival on both toolkits.
        return {"slot_usage": controller.slot_usage}
    if section == "ambient_occlusion":
        return {
            "effective_radius": lambda: controller.ambient_occlusion_effective_radius
        }
    if section == "temporal":
        return {
            "frame_count": lambda: controller.render_manager.temporal_frame_count,
            "on_reset": controller.reset_temporal_accumulation,
        }
    return {}


def dock_node_names(layout: object) -> frozenset[str]:
    """The class names of every dock node of *layout*, stacks walked through.

    What a viewer checks its controls lists against
    (``plans/clipping_planes_linking_design.md`` section 6.5).
    """
    names: set[str] = set()

    def walk(node: object) -> None:
        if node is None:
            return
        items = getattr(node, "items", None)
        if isinstance(items, list):
            for item in items:
                walk(item)
        else:
            names.add(type(node).__name__)

    for side in ("left", "right", "top", "bottom"):
        walk(getattr(layout, f"{side}_dock", None))
    return frozenset(names)


def missing_dock_node(
    controls: Sequence[Control] | None, nodes: frozenset[str], *, clipping_node: str
) -> str | None:
    """What is wrong with *controls* for a layout with dock *nodes*, if anything.

    An item of the list says a control is shown; a dock node says where it
    goes.  An item with no node for it would build nothing and say nothing,
    so it is an error.  (A node with no item is not: the dock shows its
    placeholder.)

    Where each model draws:

    * ``ClippingPlanesControl``: in *clipping_node*.
    * every other model: in ``AppearanceControls()``.  (On an
      ``OrthoViewer`` the render planes control is drawn for the 3D view's
      group only, which is still that dock.)

    Parameters
    ----------
    controls : list[Control] or None
        The ``controls=`` of one visual.
    nodes : frozenset[str]
        From :func:`dock_node_names`.
    clipping_node : str
        The node that shows clipping planes controls on this viewer:
        ``"AppearanceControls"`` on a ``Viewer``, where the control is part
        of the appearance dock, and ``"OrthoClippingControls"`` on an
        ``OrthoViewer``.

    Returns
    -------
    str or None
        The rest of a sentence that starts with the visual's name and
        "was added with", or ``None`` when every item has its node.
    """
    if not controls:
        return None
    in_appearance = [
        type(item).__name__
        for item in controls
        if clipping_node == "AppearanceControls"
        or not isinstance(item, ClippingPlanesControl)
    ]
    if in_appearance and "AppearanceControls" not in nodes:
        return (
            f"controls for the appearance dock ({', '.join(in_appearance)}), "
            "but the layout has no AppearanceControls() dock to show them in. "
            "Add AppearanceControls() to a dock of the Layout, or leave them "
            "out of controls=."
        )
    has_clipping = any(isinstance(item, ClippingPlanesControl) for item in controls)
    if has_clipping and clipping_node not in nodes:
        return (
            f"ClippingPlanesControl(), but the layout has no {clipping_node}() "
            f"dock to show its clipping planes controls in. Add "
            f"{clipping_node}() to a dock of the Layout, or leave "
            "ClippingPlanesControl() out of controls=."
        )
    return None


def unsupported_dock_node(spec: object) -> TypeError:
    """The error for a spec node no dock can render.

    Both renderers walked their known dock specs and then ``return``-ed
    ``None``, so a ``Grid`` in a dock -- or anything else unrecognised --
    produced no widget, no dock and no complaint, identically and silently on
    both toolkits.  Silence is the one behaviour that cannot be right here:
    the caller asked for something and got nothing back.

    ``Grid`` is called out by name because it is the plausible mistake: it is
    a valid *center* node, so reaching for it in a dock is a reasonable thing
    to try.
    """
    return TypeError(
        f"Cannot render {type(spec).__name__!r} in a dock. Docks accept "
        "AppearanceControls, OverlayControls, RenderControls, "
        "OrthoClippingControls, or an HStack / VStack of those. Grid is a "
        "center-only node."
    )


def render_panel_sections(spec: object) -> tuple[str, ...]:
    """Validate and return the sections a ``RenderControls`` spec asks for."""
    from cellier.gui._render_controls import RENDER_CONTROLS

    sections = tuple(getattr(spec, "sections", ()) or ())
    unknown = [name for name in sections if name not in RENDER_CONTROLS]
    if unknown:
        raise ValueError(
            f"unknown render control section(s): {unknown}; valid sections "
            f"are {sorted(RENDER_CONTROLS)}."
        )
    return sections
