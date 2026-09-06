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
    from cellier.convenience._hosts import LayoutHost


class RenderedView(NamedTuple):
    """What a layout walk produced: the root, and what has to be closed."""

    root: object
    closeables: list


def render_layout(layout: object, viewer: object, host: LayoutHost) -> RenderedView:
    """Walk a ``Layout`` spec and compose it through *host*.

    The whole traversal, for every toolkit.  Everything backend-specific is
    reached through *host* and the ``GuiBackend`` it carries.
    """
    closeables: list = []
    center = render_center(layout.center, host, closeables)
    docks = {
        name: render_dock(getattr(layout, f"{name}_dock"), viewer, host, closeables)
        for name in ("left", "right", "top", "bottom")
    }
    return RenderedView(host.assemble(center, docks, closeables), closeables)


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

    Which controls a dock contains is decided in ``_shared.py`` and is the same
    on every toolkit; *which widget class* serves each one comes from
    ``host.backend``; how the column is shaped comes from ``host.dock_panel``.
    That is the whole per-toolkit surface of a dock.
    """
    from cellier.convenience.layout._shared import unsupported_dock_node
    from cellier.convenience.layout._spec import (
        AppearanceControls,
        ChannelControls,
        HStack,
        RenderControls,
        VStack,
    )

    if spec is None:
        return None
    if isinstance(spec, AppearanceControls):
        return _render_appearance_dock(viewer, host, closeables)
    if isinstance(spec, ChannelControls):
        return _render_channel_dock(viewer, host, closeables)
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
) -> list:
    """Build and wire the appearance controls for *visual*, on any backend.

    Returns the widgets in display order, each already ``connect_widget``-wired
    where it has a bus contract, and each carrying the name the shared spec
    gave it.
    """
    from cellier.convenience.layout._shared import (
        STATIC_CONTROL_KINDS,
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
    # Every visual the controls write to: one on a ``Viewer``, the four panel
    # siblings on an ``OrthoViewer``.  Defaults to *visual* alone.
    visual_ids = [visual.id] if visual_ids is None else list(visual_ids)

    built: list = []
    for spec in specs:
        builder = backend.builders.get(spec.kind)
        if builder is not None:
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
    viewer: object, host: LayoutHost, closeables: list
) -> object | None:
    """The appearance controls for the first configured visual."""
    from cellier.convenience.layout._shared import select_appearance_target

    target = select_appearance_target(viewer)
    if target is None:
        return None

    widgets = build_appearance_widgets(
        target.visual,
        target.config,
        viewer.controller,
        target.visual_ids,
        backend=host.backend,
    )
    if not widgets:
        return None
    closeables.extend(widget for widget in widgets if hasattr(widget, "close"))
    return host.dock_panel([host.leaf(widget) for widget in widgets])


def _render_channel_dock(
    viewer: object, host: LayoutHost, closeables: list
) -> object | None:
    """Per-channel controls for the configured multichannel visual(s).

    Multi-scene aware: on an ``OrthoViewer`` the one widget drives every
    panel's sibling visual through the fan-out ``visual_ids``.
    """
    from cellier.convenience.layout._shared import (
        _resolve_channel_visual_ids,
        channel_widget_kwargs,
    )

    resolved = _resolve_channel_visual_ids(viewer)
    if resolved is None:
        return None
    config, visual_ids, channels = resolved

    widget = host.backend.channel_list(
        visual_ids, channels, **channel_widget_kwargs(config, channels)
    )
    viewer.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )
    closeables.append(widget)
    return host.leaf(widget)


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
