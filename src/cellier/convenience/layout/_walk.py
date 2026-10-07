"""The one layout walk, shared by every GUI backend.

Deliberately not in ``_shared.py``: that module promises "no controller, no
widgets, no toolkit import, so the whole decision layer is unit-testable with
no fixtures", and this walk needs the controller (to wire widgets to the bus)
and handles widgets.  Keeping them apart keeps that promise true.

Everything toolkit-specific is reached through the injected
:class:`~cellier.convenience._hosts.LayoutHost` and the
:class:`~cellier.convenience._backend.GuiBackend` it carries, so this file
imports neither Qt nor anywidget (``plans/gui_backend_seam.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    from cellier.convenience._hosts import LayoutHost


class RenderedView(NamedTuple):
    """What a layout walk produced: the root, and what has to be closed."""

    root: object
    closeables: list


def render_layout(layout: object, viewer: object, host: LayoutHost) -> RenderedView:
    """Walk a ``Layout`` spec and compose it through *host*.

    The whole traversal, for every toolkit.  Everything backend-specific is
    reached through *host* and the ``GuiBackend`` it carries.

    Before anything is built, the viewer checks its recorded ``controls=``
    flags against the layout's dock nodes: a flag with no dock to show its
    controls in raises ``ValueError``.  The viewer keeps the dock nodes, for
    the ``add_*`` calls that follow, only once the whole layout is built: a
    render that raises leaves it as it was.
    """
    from cellier.convenience.layout._shared import dock_node_names

    nodes = dock_node_names(layout)
    check = getattr(viewer, "_check_rendered_layout", None)
    if check is not None:
        check(nodes)
    closeables: list = []
    center = render_center(layout.center, host, closeables)
    docks = {
        name: render_dock(getattr(layout, f"{name}_dock"), viewer, host, closeables)
        for name in ("left", "right", "top", "bottom")
    }
    root = host.assemble(
        center, docks, closeables, dock_min_widths=layout.dock_min_widths()
    )
    record = getattr(viewer, "_record_rendered_layout", None)
    if record is not None:
        record(nodes)
    return RenderedView(root, closeables)


def render_center(node: object, host: LayoutHost, closeables: list) -> object:
    """Recursively render a center spec node to a composed widget.

    A leaf is anything with ``compose(host)`` -- ``AnywidgetCanvasView``,
    ``OrthoAnywidgetCanvases``, ``QtCanvasWidget``, ``OrthoCanvasWidgets``.
    Leaves that own bus subscriptions are collected into *closeables* so the
    caller can release them.
    """
    from cellier.convenience.layout._spec import Grid, HStack, VStack

    if isinstance(node, HStack):
        items = [render_center(item, host, closeables) for item in node.items]
        return host.stack(items, direction="h")
    if isinstance(node, VStack):
        items = [render_center(item, host, closeables) for item in node.items]
        return host.stack(items, direction="v")
    if isinstance(node, Grid):
        # An empty cell stays empty rather than closing the gap: ``Grid.cells``
        # promises ``None`` leaves a cell empty, and dropping it here would
        # shift every later cell in that row one column left.  The host places
        # the hole.
        rows = [
            [
                None if cell is None else render_center(cell, host, closeables)
                for cell in row
            ]
            for row in node.cells
        ]
        return host.grid(rows)

    compose = getattr(node, "compose", None)
    if compose is None:
        raise TypeError(
            f"Cannot render {type(node).__name__!r} as a center node. A leaf "
            "must provide compose(host); containers are HStack, VStack or Grid."
        )
    if hasattr(node, "close"):
        closeables.append(node)
    return compose(host)


def render_dock(
    spec: object, viewer: object, host: LayoutHost, closeables: list
) -> object | None:
    """Render one dock spec, or ``None`` when it builds nothing.

    ``AppearanceControls`` and ``OrthoClippingControls`` always build something:
    they follow the viewer, so they render a placeholder until a configured
    visual exists.  Only a ``RenderControls``
    with no sections, or a stack of nothing but those, builds nothing.

    Which controls a dock contains is decided in ``_shared.py`` and is the same
    on every toolkit; *which widget class* serves each one comes from
    ``host.backend``; how the column is shaped comes from ``host.dock_panel``.
    That is the whole per-toolkit surface of a dock.
    """
    from cellier.convenience.layout._shared import unsupported_dock_node
    from cellier.convenience.layout._spec import (
        AppearanceControls,
        HStack,
        OrthoClippingControls,
        OverlayControls,
        RenderControls,
        VStack,
    )

    if spec is None:
        return None
    if isinstance(spec, AppearanceControls):
        return _render_appearance_dock(spec, viewer, host, closeables)
    if isinstance(spec, OrthoClippingControls):
        return _render_ortho_clipping_dock(viewer, host, closeables)
    if isinstance(spec, OverlayControls):
        return _render_overlay_dock(spec, viewer, host, closeables)
    if isinstance(spec, RenderControls):
        return _render_render_dock(spec, viewer, host, closeables)
    if isinstance(spec, (HStack, VStack)):
        items = [render_dock(item, viewer, host, closeables) for item in spec.items]
        items = [item for item in items if item is not None]
        if not items:
            # A stack whose contents all resolved to nothing is an empty dock,
            # and an empty dock is a titled rectangle beside the canvas.
            return None
        direction = "h" if isinstance(spec, HStack) else "v"
        return host.stack(items, direction=direction)

    raise unsupported_dock_node(spec)


def build_appearance_widgets(
    visual: object,
    config: object,
    controller: object,
    visual_ids: list | None = None,
    *,
    backend: object,
    clipping_gizmo_target: Callable[[list], tuple | None] | None = None,
    clipping_controls: bool = True,
    render_planes_target: Callable[[list], object | None] | None = None,
) -> list:
    """Build and wire the appearance controls for *visual*, on any backend.

    Returns the widgets in display order, each already ``connect_widget``-wired
    where it has a bus contract, and each carrying the name the shared spec
    gave it.

    *clipping_gizmo_target* is how the viewer names where the clipping planes
    control draws its gizmo: given the control's visual ids it returns
    ``(visual_id, canvas_id)``, or ``None`` for no gizmo toggle.  It is asked
    only when a clipping planes control is built.  Without it the control has
    no toggle: nothing here picks a visual or a canvas.

    *clipping_controls* ``False`` builds no clipping planes control whatever
    the config says: on an ``OrthoViewer`` they are in the
    ``OrthoClippingControls()`` dock, one per link group, and not here.

    *render_planes_target* is how the viewer names what a render planes
    control edits: given the control's visual ids it returns a
    ``RenderPlanesTarget`` (the visuals whose planes it edits, the canvas of
    its gizmo toggle, why it is disabled), or ``None`` for no control (an
    ``OrthoViewer``'s 2D views).  Without it the control edits every visual
    of the group and has no gizmo toggle.
    """
    from cellier.convenience.layout._shared import (
        STATIC_CONTROL_KINDS,
        RenderPlanesTarget,
        _resolve_data_store,
        appearance_specs,
        warn_skipped_appearance_fields,
    )
    from cellier.gui._appearance_fields import APPEARANCE_FIELD_WIDGETS

    specs, skipped = appearance_specs(
        visual,
        config,
        _resolve_data_store(controller, visual),
        palette=controller.render_config.outline.palette,
    )
    warn_skipped_appearance_fields(skipped, visual, config)
    # Every visual the controls write to: one on a ``Viewer``, a group of
    # panel siblings on an ``OrthoViewer``.  Defaults to *visual* alone.
    visual_ids = [visual.id] if visual_ids is None else list(visual_ids)

    built: list = []
    for spec in specs:
        if spec.kind == "clipping_planes" and not clipping_controls:
            continue
        builder = backend.builders.get(spec.kind)
        if builder is not None and spec.kind == "render_planes":
            target = (
                RenderPlanesTarget(list(visual_ids))
                if render_planes_target is None
                else render_planes_target(list(visual_ids))
            )
            if target is None:
                continue
            widget = builder(spec, list(target.visual_ids), controller, target)
            if target.watch is not None:
                target.watch(widget.refresh)
        elif builder is not None and spec.kind == "clipping_planes":
            gizmo = (
                None
                if clipping_gizmo_target is None
                else clipping_gizmo_target(list(visual_ids))
            )
            widget = builder(spec, list(visual_ids), controller, gizmo)
        elif builder is not None:
            widget = builder(spec, list(visual_ids), controller)
        elif spec.kind in APPEARANCE_FIELD_WIDGETS:
            widget = backend.field_widget(spec, list(visual_ids))
        else:
            continue
        # A static control has no ``changed``/``closed`` and no subscriptions,
        # so it is built and stacked like any other and then not wired.
        if spec.kind not in STATIC_CONTROL_KINDS:
            controller.connect_widget(
                widget, subscription_specs=widget.subscription_specs()
            )
        built.append(widget)
    return built


def _render_appearance_dock(
    spec: object, viewer: object, host: LayoutHost, closeables: list
) -> object:
    """Appearance controls for the configured visuals.

    ``spec.presentation`` decides whether a selector chooses one visual at a
    time or every visual gets a collapsible section.  Either way the dock
    follows the viewer as visuals are added and removed (see
    ``_controls_dock.py``).
    """
    from cellier.convenience.layout._controls_dock import (
        APPEARANCE_PLACEHOLDER,
        ControlsDock,
    )
    from cellier.convenience.layout._shared import appearance_targets

    controller = viewer.controller
    # Where this viewer shows clipping planes controls: here, or (an
    # ``OrthoViewer``) in the ortho clipping widget.
    clipping_node = getattr(viewer, "_CLIPPING_DOCK_NODE", "AppearanceControls")

    def build(target) -> list:
        return build_appearance_widgets(
            target.visual,
            target.config,
            controller,
            target.visual_ids,
            backend=host.backend,
            clipping_gizmo_target=getattr(viewer, "_clipping_gizmo_target", None),
            clipping_controls=clipping_node == "AppearanceControls",
            render_planes_target=getattr(viewer, "_render_planes_target", None),
        )

    dock = ControlsDock(
        viewer,
        host,
        resolve=appearance_targets,
        build=build,
        placeholder=APPEARANCE_PLACEHOLDER,
        presentation=spec.presentation,
    )
    closeables.append(dock)
    return dock.root


def _render_ortho_clipping_dock(
    viewer: object, host: LayoutHost, closeables: list
) -> object:
    """The link mode selector over an ``OrthoViewer``'s clipping controls."""
    from cellier.convenience.gui._ortho_clipping import OrthoClippingWidget

    if not hasattr(viewer, "clipping_controller"):
        raise TypeError(
            "OrthoClippingControls() needs an OrthoViewer; "
            f"got {type(viewer).__name__}. On a Viewer the clipping planes "
            "control is part of AppearanceControls()."
        )
    widget = OrthoClippingWidget(viewer, host)
    closeables.append(widget)
    return widget.root


def build_overlay_widgets(
    overlay: object,
    controller: object,
    overlay_ids: list | None = None,
    *,
    backend: object,
) -> list:
    """Build and wire the controls for one overlay, on any backend.

    Returns the widgets in display order, each ``connect_widget``-wired.
    """
    from cellier.convenience.layout._shared import overlay_control_specs

    overlay_ids = [overlay.id] if overlay_ids is None else list(overlay_ids)
    built: list = []
    for spec in overlay_control_specs(overlay):
        widget = backend.overlay_field_widget(spec, list(overlay_ids))
        controller.connect_widget(
            widget, subscription_specs=widget.subscription_specs()
        )
        built.append(widget)
    return built


def _render_overlay_dock(
    spec: object, viewer: object, host: LayoutHost, closeables: list
) -> object:
    """Controls for every overlay on the viewer, following adds and removes."""
    from cellier.convenience.layout._controls_dock import (
        OVERLAY_SELECTOR_TITLE,
        ControlsDock,
    )
    from cellier.convenience.layout._shared import (
        OVERLAY_PLACEHOLDER,
        overlay_targets,
    )

    controller = viewer.controller

    def build(target) -> list:
        return build_overlay_widgets(
            target.visual, controller, target.visual_ids, backend=host.backend
        )

    dock = ControlsDock(
        viewer,
        host,
        resolve=overlay_targets,
        build=build,
        placeholder=OVERLAY_PLACEHOLDER,
        presentation=spec.presentation,
        selector_title=OVERLAY_SELECTOR_TITLE,
    )
    closeables.append(dock)
    return dock.root


def _render_render_dock(
    spec: object, viewer: object, host: LayoutHost, closeables: list
) -> object | None:
    """One panel per render-config section the spec names.

    Needs no configured visual: render settings belong to the renderer, so
    unlike the appearance dock this never returns ``None`` for want of a
    target.
    """
    from cellier.convenience.layout._shared import (
        render_panel_kwargs,
        render_panel_sections,
    )
    from cellier.gui._render_controls import RENDER_DOCK_TITLE

    sections = render_panel_sections(spec)
    if not sections:
        return None

    controller = viewer.controller
    panels = []
    for section in sections:
        panel = host.backend.render_panel(
            section,
            getattr(controller.render_config, section),
            **render_panel_kwargs(section, controller),
        )
        controller.connect_widget(panel, subscription_specs=panel.subscription_specs())
        closeables.append(panel)
        panels.append(host.leaf(panel))

    # One heading over the whole dock, naming its scope.  Without it these read
    # as the same kind of thing as the per-visual groups on the other side of
    # the canvas -- "Outline" beside "Outlines" is not a distinction anyone
    # should have to notice.
    return host.dock_panel(panels, title=RENDER_DOCK_TITLE)
