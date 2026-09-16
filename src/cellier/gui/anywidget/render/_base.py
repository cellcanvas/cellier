"""Shared plumbing for the anywidget render-settings panels.

The notebook twin of ``cellier.gui.qt.render._base``.  Both front ends draw
their controls from the same spec
(:data:`cellier.gui._render_controls.RENDER_CONTROLS`), so the labels,
ranges and explanations exist once and neither toolkit can quietly grow a
control the other lacks.

One panel is one ``AnyWidget`` with one flattened, synced trait per field,
following the former per-channel list rather than a single nested dict: a
scalar trait per control is what lets the JS observe exactly the one that
changed.  A dotted config path (``selection.inward_thickness``) is not a
legal trait name, so it is spelled with a double underscore
(``selection__inward_thickness``) on the way in and translated back on the
way out.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.events import (
    RenderConfigChangedEvent,
    RenderConfigUpdateEvent,
    SubscriptionSpec,
)
from cellier.gui._render_controls import (
    RENDER_CONTROLS,
    render_config_path,
    with_api_path,
)
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from cellier.gui._render_controls import RenderControl

_STATIC = Path(__file__).parent / "static"


def trait_name(field: str) -> str:
    """Spell a dotted config path as a legal trait name."""
    return field.replace(".", "__")


def field_name(trait: str) -> str:
    """Recover the dotted config path from a trait name."""
    return trait.replace("__", ".")


def _trait_for(control: RenderControl, value: Any):
    """Build the synced trait one control needs, seeded with *value*."""
    if control.kind == "bool":
        return traitlets.Bool(bool(value))
    if control.kind == "int":
        return traitlets.Int(int(value))
    if control.kind == "float":
        # Allow None: the occlusion radius uses it for "derive from the
        # scene", and there is no separate flag.
        return traitlets.Float(value, allow_none=True)
    if control.kind == "color":
        return traitlets.List([float(component) for component in value])
    if control.kind == "palette":
        return traitlets.List(
            [[float(component) for component in entry] for entry in value]
        )
    raise ValueError(f"unknown render control kind: {control.kind!r}")


def _read_dotted(model: Any, field: str) -> Any:
    target = model
    for name in field.split("."):
        target = getattr(target, name)
    return target


class AnywidgetRenderConfigPanel(anywidget.AnyWidget):
    """Base for a notebook panel driving one section of the render config.

    Subclasses set :attr:`section` and pass the live config section to
    ``super().__init__``.

    Wire to the controller after construction::

        panel = AnywidgetAmbientOcclusionControls(
            controller.render_config.ambient_occlusion
        )
        controller.connect_widget(panel, subscription_specs=panel.subscription_specs())
    """

    #: Which render-config section this panel drives.
    section: ClassVar[str] = ""

    _esm = _STATIC / "render_panel.js"
    _css = _STATIC / "render_panel.css"

    # psygnal outward signals (the WidgetView contract); not traitlets.
    changed: Signal = Signal(object)
    closed: Signal = Signal()

    #: The control spec, handed to the JS so it can draw the panel.
    controls = traitlets.List([]).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    #: Read-only lines the panel shows beneath its controls, as
    #: ``[[label, text], ...]``.  Derived state -- the effective occlusion
    #: radius, the accumulated frame count -- changes without any field
    #: changing, so nothing on the bus announces it.
    readouts = traitlets.List([]).tag(sync=True)
    #: ``{slot: visual count}`` for the palette control, as strings because
    #: JSON object keys are strings.  Derived state -- nothing on the bus
    #: announces a visual joining a slot -- so it is refreshed on demand.
    slot_usage = traitlets.Dict({}).tag(sync=True)
    #: Incremented by JS when an action button is clicked.
    _action_clicks = traitlets.Int(0).tag(sync=True)
    #: Label of the action button, or "" for none.
    action_label = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self, config: Any, *, title: str = "", slot_usage=None, **kwargs
    ) -> None:
        # ``_describe`` runs inside the super().__init__ below, before the
        # traits exist, so the config it reads is stashed first.
        self._describe_config = config
        self._slot_usage = slot_usage
        specs = [c for c in RENDER_CONTROLS[self.section] if self._include(c)]
        super().__init__(
            controls=[self._describe(control) for control in specs],
            title=title,
            **kwargs,
        )
        self._id = uuid4()
        self._config = config
        self._specs = {control.field: control for control in specs}
        self._applying = False
        self._readout_sources: list[tuple[str, Any]] = []

        traits = {
            trait_name(control.field): _trait_for(
                control, _read_dotted(config, control.field)
            )
            for control in specs
        }
        self.add_traits(**{k: t.tag(sync=True) for k, t in traits.items()})
        self._trait_names = list(traits)
        self.observe(self._on_trait_change, names=self._trait_names)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _include(self, control: RenderControl) -> bool:
        """Whether to draw *control*.  Subclasses may drop one."""
        return True

    def _describe(self, control: RenderControl) -> dict:
        """Serialise one control for the JS side."""
        if control.kind == "palette":
            from cellier.gui._render_controls import next_palette_color
            from cellier.render._config import MAX_OUTLINE_SLOT

            entries = _read_dotted(self._describe_config, control.field)
            return {
                **self._describe_common(control),
                "max_slots": MAX_OUTLINE_SLOT,
                # Precomputed rather than derived in JS: the golden-angle walk
                # that keeps a new slot distinct from the last one lives in
                # one place, and both front ends read it from there.
                "next_color": list(next_palette_color(entries)),
            }
        return self._describe_common(control)

    def _describe_common(self, control: RenderControl) -> dict:
        return {
            "field": control.field,
            "trait": trait_name(control.field),
            "label": control.label,
            "kind": control.kind,
            "min": control.minimum,
            "max": control.maximum,
            "step": self._step_for(control),
            "tooltip": with_api_path(
                control.tooltip, render_config_path(self.section, control.field)
            ),
            "group": control.group or "",
        }

    @staticmethod
    def _step_for(control: RenderControl) -> float:
        if control.kind == "int":
            return 1
        if control.kind == "float":
            return 10.0**-control.decimals
        return 0

    def _add_readout(self, label: str, read) -> None:
        """Register a read-only line, refreshed by :meth:`refresh_readouts`."""
        self._readout_sources.append((label, read))
        self.refresh_readouts()

    # ------------------------------------------------------------------
    # WidgetView contract
    # ------------------------------------------------------------------

    @property
    def widget(self) -> AnywidgetRenderConfigPanel:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    def close(self) -> None:
        """Unsubscribe from the bus and release the widget.

        ``closed`` tells the controller to drop this widget's
        subscriptions; the rest actually releases the widget.  See
        ``cellier.gui.anywidget._teardown`` for why both steps are needed.
        """
        self.closed.emit()
        close_aux_widgets(self)
        super().close()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Subscribe to every render-config change; filter by section here.

        ``entity_id`` is deliberately ``None``: there is no UUID a render
        config change could be keyed by.
        """
        return [
            SubscriptionSpec(
                event_type=RenderConfigChangedEvent,
                handler=self._on_render_config_changed,
            )
        ]

    def refresh_readouts(self) -> None:
        """Re-read any derived values this panel displays."""
        self.readouts = [[label, str(read())] for label, read in self._readout_sources]
        if self._slot_usage is not None:
            self.slot_usage = {
                str(slot): count for slot, count in self._slot_usage().items()
            }

    # ------------------------------------------------------------------
    # widget -> model (outbound)
    # ------------------------------------------------------------------

    def _on_trait_change(self, change) -> None:
        if self._applying:
            return  # bus -> widget write; do not echo back
        field = field_name(change.name)
        value = change.new
        kind = self._specs[field].kind
        if kind == "color":
            value = tuple(float(component) for component in value)
        elif kind == "palette":
            value = [tuple(float(c) for c in entry) for entry in value]
        self.changed.emit(
            RenderConfigUpdateEvent(
                source_id=self._id,
                section=self.section,
                field=field,
                value=value,
            )
        )

    # ------------------------------------------------------------------
    # model -> widget (inbound)
    # ------------------------------------------------------------------

    def _on_render_config_changed(self, event: RenderConfigChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.section != self.section:
            return  # another panel's section
        if event.field_name is None:
            # Whole-section replacement: refresh everything this panel shows.
            for field in self._specs:
                self._apply(field, _read_dotted(event.config, field))
            self.refresh_readouts()
            return
        if event.field_name not in self._specs:
            return  # a field this panel does not display
        self._apply(event.field_name, event.new_value)
        self.refresh_readouts()

    def _apply(self, field: str, value: Any) -> None:
        kind = self._specs[field].kind
        if kind == "color":
            value = [float(component) for component in value]
        elif kind == "palette":
            value = [[float(c) for c in entry] for entry in value]
        self._applying = True
        try:
            setattr(self, trait_name(field), value)
        finally:
            self._applying = False
