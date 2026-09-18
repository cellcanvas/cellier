"""Which widget classes a toolkit provides (the ``GuiBackend`` seam).

A *backend* answers "which widget class serves this control"; a *host*
(:mod:`cellier.convenience._hosts`) answers "how are widgets composed and
presented".  The split is what lets one anywidget backend serve two hosts,
Jupyter and marimo, and it is what lets the layout walk in
:mod:`cellier.convenience.layout._walk` be written once
(``plans/gui_backend_seam.md``).

Deliberately **not** a registry.  There are exactly two backends, named here
and looked up through :func:`backend_for`; a third would be three more lines,
and designing a plugin interface from two similar cases would bake in
assumptions neither has tested.

Not every ``gui`` value has a backend: ``gui="offscreen"`` is headless capture
with no widgets at all, and :func:`backend_for` refuses it by name rather than
failing on a missing key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from cellier.convenience.layout._shared import ControlSpec
    from cellier.gui._constants import GuiName
    from cellier.gui._protocol import WidgetView

#: A builder takes one ``ControlSpec``, the visual ids it drives, and the
#: controller, and returns a wired-able widget.  Qt builders additionally
#: accept a ``parent``; the backend supplies ``None`` and Qt reparents the
#: widget when it is added to a layout.
ControlBuilder = Callable[["ControlSpec", list, Any], "WidgetView"]


@runtime_checkable
class GuiBackend(Protocol):
    """The widget classes one toolkit provides."""

    #: The ``gui`` value this backend serves.
    name: str

    #: ``ControlSpec.kind`` -> constructor, for the controls that need a
    #: bespoke builder.  Single-field controls fall through to
    #: :meth:`field_widget`.
    builders: dict[str, ControlBuilder]

    def field_widget(self, spec: ControlSpec, visual_ids: list) -> WidgetView:
        """Build one of the single-field appearance controls."""
        ...

    def overlay_field_widget(self, spec: ControlSpec, overlay_ids: list) -> WidgetView:
        """Build one overlay control; ``spec.kind`` is the field path."""
        ...

    def render_panel(self, section: str, config: Any, **kwargs) -> WidgetView:
        """Build the panel for one render-config section."""
        ...

    def target_selector(
        self, labels: list[str], index: int = 0, *, title: str = "Visual"
    ) -> object:
        """Build the selector a controls dock shows over several visuals.

        Not a ``WidgetView``: it is local UI state with nothing on the bus.  It
        exposes ``widget``, a psygnal ``selected(int)``, ``set_choices(labels,
        index)``, ``select(index)`` and ``close()``.
        """
        ...

    def collapsible_section(
        self, title: str, widgets: list, *, expanded: bool = False
    ) -> object:
        """Build a titled section that shows or hides *widgets* when clicked.

        What a controls dock wraps each visual's controls in when presented as
        collapsible sections.  Not a ``WidgetView``: like the selector it is
        local UI state with nothing on the bus.  It exposes ``widget``,
        ``title``, ``expanded``, ``set_title(str)``, ``set_expanded(bool)`` and
        ``close()``.  *widgets* are backend widgets, stacked top to bottom.
        """
        ...

    def canvas_view(self, scene, canvas_view, axis_values: dict, **kwargs) -> object:
        """Wrap an **already-created** ``CanvasView`` in a toolkit widget.

        Takes a canvas rather than making one: creating the render surface is
        the render layer's job and forks on ``gui`` there
        (``render/canvas_view.py``), which is deliberately outside this seam.
        A backend wraps a canvas; it does not own one.
        """
        ...


class _QtBackend:
    """Qt widget classes."""

    name = "qt"

    @property
    def builders(self) -> dict[str, ControlBuilder]:
        from cellier.convenience.gui._appearance_widgets_qt import QT_BUILDERS

        return QT_BUILDERS

    def field_widget(self, spec: ControlSpec, visual_ids: list) -> WidgetView:
        """Build a single-field control from the shared table.

        ``parent=None`` is safe: every widget built here is added to a layout
        immediately, and Qt reparents it then.
        """
        from cellier.gui._appearance_fields import field_widget_class

        widget_class = field_widget_class(spec.kind, "qt")
        kwargs: dict[str, Any] = {"initial_value": spec.values["initial_value"]}
        if "choices" in spec.values:
            kwargs["choices"] = spec.values["choices"]
        return widget_class(visual_ids, parent=None, **kwargs)

    def target_selector(
        self, labels: list[str], index: int = 0, *, title: str = "Visual"
    ) -> object:
        from cellier.gui.qt._target_selector import QtTargetSelector

        return QtTargetSelector(labels, index, title=title)

    def collapsible_section(
        self, title: str, widgets: list, *, expanded: bool = False
    ) -> object:
        from cellier.convenience.layout._shared import APPEARANCE_DOCK_GAP_PX
        from cellier.gui.qt._collapsible_section import QtCollapsibleSection

        return QtCollapsibleSection(
            title, widgets, expanded=expanded, gap=APPEARANCE_DOCK_GAP_PX
        )

    def overlay_field_widget(self, spec: ControlSpec, overlay_ids: list) -> WidgetView:
        """Build one overlay control from ``OVERLAY_FIELD_WIDGETS``."""
        from cellier.gui._overlay_fields import overlay_field_widget_class

        widget_class = overlay_field_widget_class(spec.kind, "qt")
        kwargs: dict[str, Any] = {"initial_value": spec.values["initial_value"]}
        if "choices" in spec.values:
            kwargs["choices"] = spec.values["choices"]
        return widget_class(overlay_ids, parent=None, **kwargs)

    def render_panel(self, section: str, config: Any, **kwargs) -> WidgetView:
        from cellier.gui.qt.render import (
            QtAmbientOcclusionControls,
            QtOutlineControls,
            QtTemporalControls,
        )

        panel_types = {
            "outline": QtOutlineControls,
            "ambient_occlusion": QtAmbientOcclusionControls,
            "temporal": QtTemporalControls,
        }
        return panel_types[section](config, **kwargs)

    def canvas_view(self, scene, canvas_view, axis_values: dict, **kwargs) -> object:
        """Wrap the canvas in a ``QtCanvasWidget``.

        ``canvas_size`` is accepted and ignored: Qt sizes the canvas through
        its layout.
        """
        from cellier.gui.qt import QtCanvasWidget

        return QtCanvasWidget.from_scene_and_canvas(scene, canvas_view, axis_values)


class _AnywidgetBackend:
    """anywidget widget classes, shared by the Jupyter and marimo hosts."""

    name = "anywidget"

    @property
    def builders(self) -> dict[str, ControlBuilder]:
        from cellier.convenience.gui._appearance_widgets import ANYWIDGET_BUILDERS

        return ANYWIDGET_BUILDERS

    def field_widget(self, spec: ControlSpec, visual_ids: list) -> WidgetView:
        from cellier.gui._appearance_fields import field_widget_class

        widget_class = field_widget_class(spec.kind, "anywidget")
        kwargs: dict[str, Any] = {"initial_value": spec.values["initial_value"]}
        if "choices" in spec.values:
            kwargs["choices"] = spec.values["choices"]
        return widget_class(visual_ids, **kwargs)

    def target_selector(
        self, labels: list[str], index: int = 0, *, title: str = "Visual"
    ) -> object:
        from cellier.gui.anywidget._target_selector import AnywidgetTargetSelector

        return AnywidgetTargetSelector(labels, index, title=title)

    def collapsible_section(
        self, title: str, widgets: list, *, expanded: bool = False
    ) -> object:
        from cellier.convenience.layout._shared import APPEARANCE_DOCK_GAP_PX
        from cellier.gui.anywidget._collapsible_section import (
            AnywidgetCollapsibleSection,
        )

        return AnywidgetCollapsibleSection(
            title, widgets, expanded=expanded, gap=APPEARANCE_DOCK_GAP_PX
        )

    def overlay_field_widget(self, spec: ControlSpec, overlay_ids: list) -> WidgetView:
        """Build one overlay control from ``OVERLAY_FIELD_WIDGETS``."""
        from cellier.gui._overlay_fields import overlay_field_widget_class

        widget_class = overlay_field_widget_class(spec.kind, "anywidget")
        kwargs: dict[str, Any] = {"initial_value": spec.values["initial_value"]}
        if "choices" in spec.values:
            kwargs["choices"] = spec.values["choices"]
        return widget_class(overlay_ids, **kwargs)

    def render_panel(self, section: str, config: Any, **kwargs) -> WidgetView:
        from cellier.gui.anywidget.render import (
            AnywidgetAmbientOcclusionControls,
            AnywidgetOutlineControls,
            AnywidgetTemporalControls,
        )

        panel_types = {
            "outline": AnywidgetOutlineControls,
            "ambient_occlusion": AnywidgetAmbientOcclusionControls,
            "temporal": AnywidgetTemporalControls,
        }
        return panel_types[section](config, **kwargs)

    def canvas_view(self, scene, canvas_view, axis_values: dict, **kwargs) -> object:
        """Build the canvas + dims leaf pair."""
        from cellier.convenience.gui._canvas import AnywidgetCanvasView
        from cellier.gui.anywidget._dims_panel import AnywidgetDimsPanel

        dims = AnywidgetDimsPanel.from_scene(scene, axis_values)
        return AnywidgetCanvasView(
            canvas=canvas_view.widget,
            dims=dims,
            canvas_size=kwargs.get("canvas_size") or (600, 600),
        )


QT_BACKEND = _QtBackend()
ANYWIDGET_BACKEND = _AnywidgetBackend()

_BACKENDS: dict[str, GuiBackend] = {
    QT_BACKEND.name: QT_BACKEND,
    ANYWIDGET_BACKEND.name: ANYWIDGET_BACKEND,
}


def backend_for(
    gui: GuiName, *, lacks: str = "widgets", what: str = "gui"
) -> GuiBackend:
    """Return the backend serving *gui*.

    Parameters
    ----------
    gui :
        The viewer's ``gui`` value.
    lacks :
        What ``"offscreen"`` has none of, phrased for the caller -- an
        embeddable widget when a builder asks, a window when ``run`` does.
        The advice that follows is the same either way, so only the lead
        varies: a reader who called ``build_canvas_widget`` should not be told
        about windows, and a reader who called ``run`` should not be told
        about embeddable widgets.
    what :
        How to name the offending argument in the unknown-value message,
        e.g. ``"viewer.gui"``.

    Raises
    ------
    ValueError
        For ``"offscreen"``, which has no widgets by design, and for any
        unrecognised name.  A backend-less ``gui`` is a supported state, so it
        is refused by name rather than as a missing key.
    """
    try:
        return _BACKENDS[gui]
    except KeyError:
        if gui == "offscreen":
            raise ValueError(
                f"gui='offscreen' has {lacks}. Offscreen viewers are for "
                "headless capture: call viewer.screenshot() directly."
            ) from None
        raise ValueError(
            f"Unknown {what} {gui!r}. Expected one of {sorted(_BACKENDS)}."
        ) from None
