"""Image-specific widgets wired to the cellier v2 event bus."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from uuid import uuid4

from psygnal import Signal

from cellier.events import (
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    SubscriptionSpec,
)
from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui.qt.visuals._base import QtBoundedSlider, QtChoice

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID


def _fix_label_width(label, lo: float, hi: float, decimals: int) -> None:
    """Force a superqt SliderLabel to be wide enough for the formatted value.

    SliderLabel._get_size() sizes from str(float) repr ("0.0") but
    setDecimals(2) displays "0.00" -- one char wider, clipping the text.
    Patching _update_size to a no-op prevents showEvent from re-shrinking.
    """
    from qtpy.QtGui import QFontMetrics

    fm = QFontMetrics(label.font())
    w = (
        max(
            fm.horizontalAdvance(f"{lo:.{decimals}f}"),
            fm.horizontalAdvance(f"{hi:.{decimals}f}"),
        )
        + 12
    )
    label._update_size = lambda *_: None
    label.setFixedWidth(w)


class QtRenderModeCombo(QtChoice):
    """Render mode for an image visual (Qt).

    A layer-3 field control like every other: the options come from the
    model's own ``Literal`` annotation, so the in-memory variant
    (``mip``/``iso``/``minip``) and the multiscale one
    (``iso``/``mip``/``smooth_iso``/``attenuated_mip``) each offer their own
    set with no list restated here.

    The composite :class:`QtVolumeRenderControls` draws its own mode selector
    and does not use this; this is the standalone control, for a panel that
    wants render mode on its own.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically ``visual.appearance.render_mode``.
    """

    _field: ClassVar[str] = "render_mode"
    _label: ClassVar[str] = "Render mode"
    _default_value: ClassVar[str] = "mip"
    _default_choices: ClassVar[tuple[str, ...]] = ("mip", "iso", "minip")


class QtIsoThresholdSlider(QtBoundedSlider):
    """ISO surface threshold for an image visual (Qt).

    **The one bounded slider whose range is a convention rather than a model
    constraint.**  Every other one takes its range from the field's ``ge``/``le``
    metadata; ``iso_threshold`` carries none, because a threshold is only
    bounded relative to the data's intensity range.  ``(0.0, 1.0)`` is right
    for the normalised images cellier renders, and is stated here rather than
    derived -- see :class:`~cellier.gui.qt.visuals._base.QtBoundedSlider`.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically ``visual.appearance.iso_threshold``.
    """

    _field: ClassVar[str] = "iso_threshold"
    _label: ClassVar[str] = "Iso threshold"
    _default_value: ClassVar[float] = 0.2
    _default_range: ClassVar[tuple[float, float]] = (0.0, 1.0)


class QtVolumeRenderControls(VisualIdGroup):
    """Combined render-mode, ISO-threshold, and attenuation widget wired to the cellier v2 bus.

    Contains a render-mode ``QComboBox`` (``"iso"`` / ``"smooth_iso"`` /
    ``"mip"`` / ``"attenuated_mip"``) and mode-dependent parameter sliders:

    - ISO threshold slider — visible when mode is ``"iso"`` or ``"smooth_iso"``.
    - Attenuation slider — visible when mode is ``"attenuated_mip"``.

    All controls share one UUID so a single ``on_visual_changed`` subscription
    handles all fields.  Follows the v2 widget pattern: source-ID echo
    filtering and signal blocking on model-driven updates.

    Wire to the controller after construction::

        controls = QtVolumeRenderControls(visual_id, dtype_max=255, ...)
        controller.connect_widget(
            controls, subscription_specs=controls.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``render_mode``, ``iso_threshold``, and
        ``attenuation`` fields this widget controls.
        A sequence drives every listed visual in lock-step -- the
        ``OrthoViewer``'s four panel siblings (design section 8.1).
    dtype_max :
        Upper bound of the threshold slider — typically the dtype maximum
        (e.g. 65535 for uint16, 1.0 for float32).
    initial_render_mode :
        Starting render mode — typically ``visual_model.appearance.render_mode``.
    initial_threshold :
        Starting threshold — typically ``visual_model.appearance.iso_threshold``.
    initial_attenuation :
        Starting attenuation coefficient — typically
        ``visual_model.appearance.attenuation``.  Default is ``1.0``.
    decimals :
        Number of decimal places shown in the threshold slider label.  Use
        ``0`` for integer dtypes and ``2`` (or similar) for float data.
        Default is ``2``.
    title :
        The name shown on the group frame.  Defaults to
        :data:`DEFAULT_TITLE`.
    parent :
        Optional Qt parent widget.
    """

    DEFAULT_TITLE = "Render mode"
    """Name shown when no ``title=`` is given.

    The renderer passes the title from the shared control vocabulary; this
    is what a directly-constructed widget calls itself, and
    ``test_composite_default_titles_match_the_shared_vocabulary`` pins the
    two together.
    """

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        dtype_max: float,
        initial_render_mode: str,
        initial_threshold: float,
        initial_attenuation: float = 1.0,
        decimals: int = 2,
        title: str | None = None,
        parent=None,
    ) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QComboBox, QFormLayout, QWidget
        from superqt import QLabeledDoubleSlider

        from cellier.gui.qt.visuals._chrome import titled_group

        # ── Cellier layer ────────────────────────────────────────────────────
        self._id = uuid4()
        self._init_visual_ids(visual_id)

        # ── Qt seam 1: widget creation and signal wiring ─────────────────────
        self._container = QWidget(parent)
        layout = QFormLayout(self._container)
        layout.setContentsMargins(12, 0, 12, 0)

        self._combo = QComboBox()
        self._combo.addItems(["iso", "smooth_iso", "mip", "attenuated_mip"])
        self._combo.setCurrentText(initial_render_mode)
        self._combo.currentTextChanged.connect(self._on_combo_changed)
        layout.addRow("Mode", self._combo)

        self._slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0.0, dtype_max)
        self._slider.setValue(initial_threshold)
        self._slider.setDecimals(decimals)
        _fix_label_width(self._slider._label, 0.0, dtype_max, decimals)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addRow("Threshold", self._slider)

        self._attenuation_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self._attenuation_slider.setRange(0.0, 10.0)
        self._attenuation_slider.setValue(initial_attenuation)
        self._attenuation_slider.setDecimals(2)
        _fix_label_width(self._attenuation_slider._label, 0.0, 10.0, 2)
        self._attenuation_slider.valueChanged.connect(
            self._on_attenuation_slider_changed
        )
        layout.addRow("Attenuation", self._attenuation_slider)

        # Show mode-specific controls; hide the others.
        self._update_mode_controls_visibility(initial_render_mode)

        self._group = titled_group(
            self.DEFAULT_TITLE if title is None else title, self._container, parent
        )

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self):
        """The titled group to insert into a layout.

        The rows inside only mean something together, so the group names them
        (``plans/label_ownership_unification.md``); reach for :attr:`control`
        for the bare rows.
        """
        return self._group

    @property
    def control(self):
        """The bare row container inside the group."""
        return self._container

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound subscription this widget requires.

        Pass the result to ``CellierController.connect_widget``.
        """
        return self._group_specs(AppearanceChangedEvent, self._on_visual_changed)

    # ── Cellier layer: model → widget ────────────────────────────────────────

    def _on_visual_changed(self, event) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.field_name == "render_mode":
            self._set_render_mode(event.new_value)
        elif event.field_name == "iso_threshold":
            self._set_threshold(event.new_value)
        elif event.field_name == "attenuation":
            self._set_attenuation(event.new_value)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_combo_changed(self, text: str) -> None:
        self._update_mode_controls_visibility(text)
        self._emit_group(AppearanceUpdateEvent, "render_mode", text)

    def _on_slider_changed(self, value: float) -> None:
        self._emit_group(AppearanceUpdateEvent, "iso_threshold", value)

    def _on_attenuation_slider_changed(self, value: float) -> None:
        self._emit_group(AppearanceUpdateEvent, "attenuation", value)

    # ── Qt seam 2: push values without re-firing signals ─────────────────────

    def _set_render_mode(self, value: str) -> None:
        self._combo.blockSignals(True)
        self._combo.setCurrentText(value)
        self._combo.blockSignals(False)
        self._update_mode_controls_visibility(value)

    def _set_threshold(self, value: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(value)
        self._slider.blockSignals(False)

    def _set_attenuation(self, value: float) -> None:
        self._attenuation_slider.blockSignals(True)
        self._attenuation_slider.setValue(value)
        self._attenuation_slider.blockSignals(False)

    def _update_mode_controls_visibility(self, render_mode: str) -> None:
        threshold_visible = render_mode in ("iso", "smooth_iso")
        attenuation_visible = render_mode == "attenuated_mip"
        layout = self._container.layout()
        self._slider.setVisible(threshold_visible)
        layout.labelForField(self._slider).setVisible(threshold_visible)
        self._attenuation_slider.setVisible(attenuation_visible)
        layout.labelForField(self._attenuation_slider).setVisible(attenuation_visible)
