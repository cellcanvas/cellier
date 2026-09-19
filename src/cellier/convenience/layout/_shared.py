"""Shared, host-agnostic resolvers used by both layout renderers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import NamedTuple
from uuid import UUID

from cellier.gui._render_controls import VISUAL_RENDER_TITLES

# ── Appearance controls: the toolkit-neutral decision layer ──────────────────
#
# Design section 7.3.  Which controls a panel contains, in what order, seeded
# with what values, is one question with one answer -- it was previously
# answered twice, once per toolkit, in ~110 near-identical lines each
# (section 4.2).  These two functions are that answer, as pure data: no
# controller, no widgets, no toolkit import, so the whole decision layer is
# unit-testable with no fixtures.  Each renderer keeps only a dispatch table
# mapping ``ControlSpec.kind`` to its own widget class.


@dataclass(frozen=True)
class ControlSpec:
    """One control to build, described without reference to any toolkit.

    Parameters
    ----------
    kind : str
        Which control to build: ``color_map``, ``clim``, ``render``,
        ``lod_bias``, ``aabb`` or ``dataset_info``.  A renderer with no
        builder for a kind skips it.
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
        Requested field names that produced no control, either because the
        config class has no control for them or because the visual's
        appearance model does not carry them.  Reported rather than silently
        dropped (design section 9.5 step 4); stage 3 turns the first case
        into an error at config-construction time, which leaves this for the
        second.
    """

    specs: list[ControlSpec]
    skipped: list[str]


_CONTROL_TITLES = {
    "image": "Image",
    "lod_bias": "LOD bias",
    "aabb": "Bounding box",
    "trail": "Trail",
    # Read rather than restated: the per-visual groups name themselves in
    # the shared control spec, beside the controls they hold.
    **VISUAL_RENDER_TITLES,
    "dataset_info": "Dataset info",
}
"""Titles for the control kinds that need one written out.

A kind not listed here is a plain single-field control, and its title is
derived from the field name -- so a config class can name a new kind without
registering anything.

Each of these is also a widget class's ``DEFAULT_TITLE`` on both toolkits,
pinned by ``test_composite_default_titles_match_the_shared_vocabulary``: this
is the name the renderers pass in, that is the name a directly-constructed
widget uses.
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

    Only ``dataset_info=True`` needs it.  A visual with no controller or no
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
    config: object,
    store: object | None = None,
    palette: object = (),
) -> AppearanceSpecs:
    """Decide which appearance controls *visual* should get under *config*.

    Pure: reads the visual's appearance model and the config, and returns
    data.  Both renderers walk the result and build widgets from it.

    The requested field names are matched against the config class's
    ``APPEARANCE_CONTROLS`` map (design section 6.5.2 decision 6), so which
    widget a name means is decided by the kind of visual being configured
    rather than by the name alone.  Group order is the map's order, not the
    caller's -- the ``appearance`` docstring's "in display order" has never
    been true and stage 1 does not change that.

    Parameters
    ----------
    visual : BaseVisual
        The visual whose appearance the panel will drive.
    config : BaseControlsConfig
        The recorded controls config.
    store : BaseDataStore or None
        The visual's backing data store, needed only to honour
        ``dataset_info=True``, which asks the store to describe itself.
        Passed in rather than looked up so this stays pure: the shared
        decision layer takes no controller and imports no toolkit.  A
        renderer resolves it from ``visual.data_store_id``.
    palette : sequence of RGBA
        The outline palette, needed only by the per-visual outline
        controls, whose slot swatches offer exactly the slots it holds.
        Passed in for the same reason *store* is: this stays pure, and
        both front ends then read the slots on offer from one place.

    Returns
    -------
    AppearanceSpecs
    """
    from cellier.gui._appearance_fields import (
        APPEARANCE_FIELD_WIDGETS,
        literal_choices,
    )

    appearance = getattr(config, "appearance", False)
    if appearance is True:
        # "Use this config class's default panel" -- every field it can drive
        # (design section 11.2).  ``True`` was a dead value before stage 5:
        # the annotation admitted it and it took the same branch as ``False``.
        field_list = list(type(config).default_appearance_fields())
        explicit = False
    else:
        field_list = appearance if isinstance(appearance, list) and appearance else None
        explicit = True
    if not field_list or not hasattr(visual, "appearance"):
        return AppearanceSpecs([], list(field_list or []) if explicit else [])

    app = visual.appearance
    controls_map: dict[str, str] = type(config).APPEARANCE_CONTROLS
    requested = set(field_list)

    # A field is honoured when the config class has a control for it *and* the
    # visual's model actually carries it.  Both halves matter: a config can be
    # paired with a visual whose model is narrower (a multiscale config on an
    # in-memory image has no ``lod_bias`` to drive).
    # An image's mode-dependent fields live on ``single`` (unified image
    # design 3.1), not on the shared appearance.
    single = getattr(visual, "single", None)
    honoured = {
        field
        for field in requested
        if field in controls_map
        and (hasattr(app, field) or (single is not None and hasattr(single, field)))
    }
    # A default list is "everything this config *can* drive", and the visual's
    # model is often narrower on purpose -- a flat mesh has no ``shininess``.
    # Reporting those would warn on every ``appearance=True`` panel, so only
    # fields the caller named are reported as dropped.
    skipped = sorted(requested - honoured) if explicit else []

    image_fields = [
        field
        for field in controls_map
        if field in honoured and controls_map[field] == "image"
    ]

    def _image_values():
        from cellier.gui._image_controls import image_control_values

        return image_control_values(
            visual,
            fields=image_fields,
            colormap_names=getattr(config, "colormap_names", None),
            clim_range=getattr(config, "clim_range", None),
            channel_labels=getattr(config, "channel_labels", None),
        )

    values_for = {
        "image": _image_values,
        "lod_bias": lambda: {"initial_lod_bias": float(getattr(app, "lod_bias", 1.0))},
    }

    specs: list[ControlSpec] = []
    seen: set[str] = set()
    for field, kind in controls_map.items():
        if field not in honoured or kind in seen:
            continue
        seen.add(kind)
        # A kind with no bespoke entry is a plain single-field control: its
        # value is the field's current value, and its title comes from the
        # shared widget table.  The bespoke entries exist for the controls
        # that read several fields at once, or that need something the model
        # does not hold directly.
        if kind in values_for:
            values = values_for[kind]()
            title = _CONTROL_TITLES[kind]
        else:
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

    trail_spec = _trail_spec(visual, config, store)
    if trail_spec is not None:
        specs.append(trail_spec)

    # The bounding box is not a field of the appearance model and is not
    # requested by name: ``aabb`` is on ``BaseVisual`` with a default factory,
    # so every visual has one and every configured panel gets the control.
    # This is what the anywidget path already did; stage 1 gives Qt the same
    # (design section 7.1).
    aabb = getattr(visual, "aabb", None)
    if aabb is not None:
        specs.append(
            ControlSpec(
                "aabb",
                _CONTROL_TITLES["aabb"],
                {
                    "initial_enabled": aabb.enabled,
                    "initial_line_width": aabb.line_width,
                    "initial_color": aabb.color,
                },
            )
        )

    specs.extend(_visual_render_specs(visual, config, palette))

    dataset_info_spec = _dataset_info_spec(
        getattr(config, "dataset_info", False), store
    )
    if dataset_info_spec is not None:
        specs.append(dataset_info_spec)

    return AppearanceSpecs(specs, skipped)


def _trail_spec(visual: object, config: object, store: object) -> ControlSpec | None:
    """The trail-window control *config* asks for, or ``None``.

    Only a graph visual has a trail, and only ``GraphControlsConfig`` can ask
    for its control.  The axes are resolved here, against the store's own
    data axes, because the config is built before it is paired with a store.
    """
    requested = getattr(config, "trail_controls", False)
    trail = getattr(visual, "trail", None)
    if not requested or trail is None:
        return None

    systems = getattr(store, "data_coordinate_systems", None) or []
    system = systems[0] if systems else None
    names = system.axis_names() if system is not None else ()

    if requested is True:
        axes = sorted(trail)
        if not axes and system is not None:
            axes = [
                index
                for index, axis in enumerate(system.axes)
                if axis.axis_type == "time"
            ]
    else:
        axes = [_resolve_trail_axis(ref, names) for ref in requested]
    if not axes:
        warnings.warn(
            f"trail_controls=True on {getattr(visual, 'name', 'a graph')!r} found "
            "no axis to offer: the graph has no trail window and no time axis. "
            "Name the axes instead, e.g. trail_controls=[0].",
            UserWarning,
            stacklevel=3,
        )
        return None

    return ControlSpec(
        "trail",
        _CONTROL_TITLES["trail"],
        {
            "axes": [
                (axis, names[axis] if axis < len(names) else f"axis {axis}")
                for axis in dict.fromkeys(axes)
            ],
            # Copies: the widget keeps what it was seeded with, and must never
            # hold the objects the controller is wired to.
            "trail": {axis: window.model_copy() for axis, window in trail.items()},
        },
    )


def _resolve_trail_axis(ref: int | str, names: tuple[str, ...]) -> int:
    """Resolve one ``trail_controls`` entry to a data-axis index."""
    if isinstance(ref, str):
        matches = [index for index, name in enumerate(names) if name == ref]
        if len(matches) != 1:
            problem = "is not one of" if not matches else "is ambiguous among"
            raise ValueError(
                f"trail_controls axis {ref!r} {problem} the graph's data axes "
                f"{list(names)}."
            )
        return matches[0]
    if not 0 <= ref < len(names):
        raise ValueError(
            f"trail_controls axis {ref} is out of range for a {len(names)}-axis graph."
        )
    return int(ref)


def _visual_render_specs(
    visual: object, config: object, palette: object = ()
) -> list[ControlSpec]:
    """The per-visual outline / occlusion / picking controls *config* asks for.

    Opt-in, unlike the bounding box: both passes are off by default, so an
    always-present group would lengthen every appearance panel for people
    who never enable either feature.

    Which outline control is emitted depends on the visual type, because
    ``outline.slot`` means two different things.  On most visuals it chooses
    the colour, since the selection layer draws the region in
    ``palette[slot - 1]``.  On a labels visual it only decides whether the
    volume participates -- the colour comes from ``outline_selected_labels``,
    per label value -- so that one gets a checkbox and a row editor instead
    of a swatch picker.
    """
    from cellier.controller import _default_placement
    from cellier.visuals._label_memory import BaseLabelsVisual

    wants_outline = bool(getattr(config, "outline_controls", False))
    wants_occlusion = bool(getattr(config, "ambient_occlusion_controls", False))
    if not (wants_outline or wants_occlusion):
        return []

    outline = getattr(visual, "outline", None)
    if outline is None:
        return []

    specs: list[ControlSpec] = []
    if wants_outline:
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
        specs.append(ControlSpec(kind, _CONTROL_TITLES[kind], values))

    if wants_occlusion:
        specs.append(
            ControlSpec(
                "visual_occlusion",
                _CONTROL_TITLES["visual_occlusion"],
                {"ambient_occlusion": visual.ambient_occlusion},
            )
        )

    # Both features read the pick buffer, so the flag that gates them both
    # comes along with either.
    specs.append(
        ControlSpec(
            "visual_picking",
            _CONTROL_TITLES["visual_picking"],
            {"pick_write": visual.pick_write},
        )
    )
    return specs


def _dataset_info_spec(requested: object, store: object) -> ControlSpec | None:
    """Resolve a config's ``dataset_info`` setting into a spec, or ``None``.

    The setting has three useful forms and one that hides the block:

    ``True``
        Ask *store* to describe itself.  Yields nothing when no store was
        resolved -- the caller could not find one, and a block asserting
        that a store has no metadata would be worse than no block.
    a ``DatasetInfo``
        Displayed as given.
    a sequence of pairs
        The hand-authored escape hatch, carried through as flat rows.
    ``False`` / empty
        No block.
    """
    from cellier.data._dataset_info import DatasetInfo

    if requested is False or requested is None:
        return None

    if requested is True:
        if store is None or not hasattr(store, "dataset_info"):
            return None
        info: DatasetInfo | None = store.dataset_info()
    elif isinstance(requested, DatasetInfo):
        info = requested
    else:
        rows = [(str(label), str(value)) for label, value in requested]
        if not rows:
            return None
        return ControlSpec(
            "dataset_info", _CONTROL_TITLES["dataset_info"], {"rows": rows}
        )

    if info is None or not info.sections:
        return None
    return ControlSpec("dataset_info", _CONTROL_TITLES["dataset_info"], {"info": info})


def warn_skipped_appearance_fields(
    skipped: list[str], visual: object, config: object
) -> None:
    """Warn about requested fields that produced no control.

    Stage 3 makes a name the *config class* cannot drive raise at
    construction, which leaves one residual case this cannot catch: a name
    valid for the config class whose value the particular *visual*'s
    appearance model does not carry -- a ``MultiscaleImageControlsConfig``
    paired with an in-memory image, say.  That is still a silent drop unless
    something says so, and it is the renderer's to surface because only the
    renderer has both halves (design section 9.5 step 4).
    """
    if not skipped:
        return
    warnings.warn(
        f"{type(config).__name__} requested appearance field(s) "
        f"{skipped} that {type(visual).__name__}'s appearance model does not "
        f"have; no control was built for them.",
        UserWarning,
        stacklevel=3,
    )


class ControlTarget(NamedTuple):
    """One configured visual a controls dock can drive.

    Parameters
    ----------
    key : UUID
        The representative visual id the config is recorded under.  Stable
        for as long as the visual exists, so it is what a dock's selection
        is kept by.
    label : str
        What the dock's visual selector calls this target.  Unique among the
        targets of one dock.
    visual : BaseVisual
        The representative visual.  Its appearance model is what the controls
        are seeded from; on an ``OrthoViewer`` it is the first panel's visual
        and the other three are guaranteed equal to it.
    config : BaseControlsConfig
        The recorded controls config.
    visual_ids : list[UUID]
        Every visual the controls write to -- one on a ``Viewer``, the four
        panel siblings on an ``OrthoViewer``.  The widgets accept this
        directly (see ``cellier.gui._appearance_fields.VisualIdGroup``), so
        the spec walk is identical either way and only the id list differs.
    """

    key: UUID
    label: str
    visual: object
    config: object
    visual_ids: list


def appearance_targets(viewer: object) -> list[ControlTarget]:
    """Every visual an ``AppearanceControls()`` dock can drive, in add order.

    A config whose ``appearance`` is falsy (``False``, ``None``, ``[]``) is skipped: it
    asks for no panel, and :func:`appearance_specs` would build none, so offering it in
    the selector would only lead to an empty dock.  Multi-scene aware: an
    ``OrthoViewer`` records one config per fanned-out add, keyed by the first panel's
    visual, and ``_visual_groups`` expands it to all four.
    """
    return _control_targets(viewer)


def _control_targets(viewer: object) -> list[ControlTarget]:
    """Resolve the recorded configs of one kind into labelled targets.

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

    resolved = []
    for rep_id, config in controls_configs.items():
        if not getattr(config, "appearance", False):
            continue
        try:
            visual = controller.get_visual_model(rep_id)
        except KeyError:
            continue
        visual_ids = list(groups.get(rep_id, [rep_id]))
        resolved.append(
            (
                rep_id,
                _group_name(controller, visual, visual_ids),
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
    ``None``: an overlay has no ``controls=`` config, and its controls are
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
    config when an ``OrthoViewer`` re-keyed its group after the
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
        "AppearanceControls, OverlayControls, RenderControls, or an HStack / "
        "VStack of those. Grid is a center-only node."
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
