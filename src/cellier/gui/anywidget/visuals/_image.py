"""Volume-render controls wired to the cellier v2 event bus (anywidget).

Render mode, ISO threshold, and attenuation, combined into one widget since
the latter two are mode-dependent (mirrors ``QtVolumeRenderControls``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.events import (
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    SubscriptionSpec,
)
from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui.anywidget._teardown import close_aux_widgets
from cellier.gui.anywidget.visuals._base import (
    AnywidgetBoundedSlider,
    AnywidgetChoice,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

_STATIC = Path(__file__).parent / "static"

_FIELDS = ("render_mode", "iso_threshold", "attenuation")


class AnywidgetVolumeRenderControls(VisualIdGroup, anywidget.AnyWidget):
    """Combined render-mode, ISO-threshold, and attenuation widget.

    Mirrors ``QtVolumeRenderControls``: a mode select plus two mode-dependent
    sliders (ISO threshold visible for ``"iso"``/``"smooth_iso"``, attenuation
    visible for ``"attenuated_mip"``), sharing one UUID so a single
    subscription covers all three fields.

    Wire to the controller after construction::

        controls = AnywidgetVolumeRenderControls(visual_id, initial_render_mode="mip", ...)
        controller.connect_widget(controls, subscription_specs=controls.subscription_specs())

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``render_mode``, ``iso_threshold``, and
        ``attenuation`` fields this widget controls.
        A sequence drives every listed visual in lock-step -- the
        ``OrthoViewer``'s four panel siblings (design section 8.1).
    initial_render_mode :
        Starting render mode. Default ``"mip"``.
    initial_threshold :
        Starting ISO threshold. Default ``0.2``.
    initial_attenuation :
        Starting attenuation coefficient. Default ``1.0``.
    """

    _esm = _STATIC / "volume_render_controls.js"
    _css = _STATIC / "volume_render_controls.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    DEFAULT_TITLE = "Render mode"
    """Name shown when no ``title=`` is given.

    The renderer passes the title from the shared control vocabulary; this is
    what a directly-constructed widget calls itself, and
    ``test_composite_default_titles_match_the_shared_vocabulary`` pins the two
    together.
    """

    title = traitlets.Unicode(DEFAULT_TITLE).tag(sync=True)
    """What this control calls itself, drawn by its own front end.

    A control names itself rather than being named by whatever lays it out
    (``plans/label_ownership_unification.md``), which is what lets the dock
    stack controls and stop.
    """

    render_mode = traitlets.Unicode("mip").tag(sync=True)
    iso_threshold = traitlets.Float(0.2).tag(sync=True)
    attenuation = traitlets.Float(1.0).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_render_mode: str = "mip",
        initial_threshold: float = 0.2,
        initial_attenuation: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            render_mode=str(initial_render_mode),
            iso_threshold=float(initial_threshold),
            attenuation=float(initial_attenuation),
            **kwargs,
        )
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._applying = False
        self.observe(self._on_trait_change, names=list(_FIELDS))

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> AnywidgetVolumeRenderControls:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    def close(self) -> None:
        """Unsubscribe from the bus and release the widget.

        ``closed`` tells the controller to drop this widget's subscriptions;
        the rest actually releases the widget.  See
        ``cellier.gui.anywidget._teardown`` for why both steps are needed --
        ``ipywidgets`` holds every widget, and every widget's ``layout``, in a
        process-global table that only ``close()`` clears.
        """
        self.closed.emit()
        close_aux_widgets(self)
        super().close()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound subscription this widget requires."""
        return self._group_specs(AppearanceChangedEvent, self._on_appearance_changed)

    # ── model -> widget ──────────────────────────────────────────────────────

    def _on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.field_name not in _FIELDS:
            return
        self._set_field(event.field_name, event.new_value)

    def _set_field(self, name: str, value) -> None:
        self._applying = True
        try:
            setattr(self, name, value)
        finally:
            self._applying = False

    # ── widget -> model ──────────────────────────────────────────────────────

    def _on_trait_change(self, change) -> None:
        if self._applying:
            return
        self._emit_group(AppearanceUpdateEvent, change["name"], change["new"])


class AnywidgetRenderModeCombo(AnywidgetChoice):
    """Render mode for an image visual (anywidget).

    The twin of ``QtRenderModeCombo``; see it for why the options come from
    the model rather than from a list here.
    """

    _field: ClassVar[str] = "render_mode"
    _label: ClassVar[str] = "Render mode"
    _default_value: ClassVar[str] = "mip"
    _default_choices: ClassVar[tuple[str, ...]] = ("mip", "iso", "minip")


class AnywidgetIsoThresholdSlider(AnywidgetBoundedSlider):
    """ISO surface threshold for an image visual (anywidget).

    The twin of ``QtIsoThresholdSlider``; see it for why this bounded slider's
    range is a convention rather than a model constraint.
    """

    _field: ClassVar[str] = "iso_threshold"
    _label: ClassVar[str] = "Iso threshold"
    _default_value: ClassVar[float] = 0.2
    _default_range: ClassVar[tuple[float, float]] = (0.0, 1.0)
