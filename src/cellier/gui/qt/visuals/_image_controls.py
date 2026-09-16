"""The unified image control for Qt (unified image design 3.10).

One control per image visual: a shared section (visibility, blending,
interpolation, and attenuation on a multiscale image), a composite switch
shown only when the visual has a channel axis, and a page per mode -- the
single appearance, or one group per channel.  An empty channel list is a
valid composite page.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from psygnal import Signal

from cellier.events import ImageCompositeUpdateEvent
from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._image_controls import (
    FIELD_LABELS,
    INBOUND_EVENT_TYPES,
    MODE_FIELDS,
    image_update_event,
    inbound_target,
    mode_values,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID


class QtImageControls(VisualIdGroup):
    """Bidirectional image appearance control on the cellier bus.

    Parameters
    ----------
    visual_id :
        The visual, or an ``OrthoViewer``'s panel siblings, driven together.
    values :
        The seed from :func:`cellier.gui._image_controls.image_control_values`.
    title :
        The group frame's title.  Defaults to :data:`DEFAULT_TITLE`.
    parent :
        Optional Qt parent widget.
    """

    DEFAULT_TITLE = "Image"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        values: dict[str, Any],
        *,
        title: str | None = None,
        parent=None,
    ) -> None:
        from qtpy.QtWidgets import (
            QCheckBox,
            QFormLayout,
            QLabel,
            QStackedWidget,
            QVBoxLayout,
            QWidget,
        )

        from cellier.gui.qt.visuals._chrome import titled_group

        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._values = values
        self._fields = set(values["fields"])
        # (page, channel, field) -> apply a value with the control's signals
        # blocked, for the inbound path.
        self._appliers: dict[tuple[str, int | None, str], Callable[[Any], None]] = {}
        # (page, channel, field) -> the Qt control, for callers that need to
        # drive or inspect one directly.
        self._controls: dict[tuple[str, int | None, str], object] = {}

        self._container = QWidget(parent)
        layout = QVBoxLayout(self._container)
        layout.setContentsMargins(12, 0, 12, 0)

        # Rows are form rows rather than self-named labelled rows: the group
        # frame names the whole control, as the volume-render group did, so
        # both toolkits report one "Image" control.
        shared_rows = QWidget()
        shared_form = QFormLayout(shared_rows)
        shared_form.setContentsMargins(0, 0, 0, 0)
        self._shared_rows(shared_form, values)
        layout.addWidget(shared_rows)

        self._composite_box = QCheckBox("Composite channels")
        self._composite_box.setChecked(bool(values["composite"]))
        self._composite_box.toggled.connect(self._on_composite_toggled)
        self._composite_box.setVisible(bool(values["has_channel_axis"]))
        layout.addWidget(self._composite_box)

        self._pages = QStackedWidget()
        self._single_page = self._mode_rows("single", None, values["single"], values)
        self._pages.addWidget(self._single_page)

        self._composite_page = QWidget()
        composite_layout = QVBoxLayout(self._composite_page)
        composite_layout.setContentsMargins(0, 0, 0, 0)
        self._channel_groups: dict[int, object] = {}
        if not values["channels"]:
            self._empty_label = QLabel("No channels")
            composite_layout.addWidget(self._empty_label)
        for index in sorted(values["channels"]):
            self._add_channel_group(composite_layout, index, values)
        self._pages.addWidget(self._composite_page)
        self._pages.setCurrentIndex(1 if values["composite"] else 0)
        layout.addWidget(self._pages)

        self._group = titled_group(
            self.DEFAULT_TITLE if title is None else title, self._container, parent
        )

    # ── Public interface ────────────────────────────────────────────────

    @property
    def widget(self):
        """The titled group to insert into a layout."""
        return self._group

    @property
    def composite(self) -> bool:
        """Whether the control shows the composite page."""
        return self._pages.currentIndex() == 1

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list:
        """One subscription per inbound event type per driven visual."""
        specs = []
        for event_type in INBOUND_EVENT_TYPES:
            specs.extend(self._group_specs(event_type, self._on_event))
        return specs

    # ── Building ────────────────────────────────────────────────────────

    def _shared_rows(self, layout, values) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QCheckBox, QComboBox
        from superqt import QLabeledDoubleSlider

        shared = values["shared"]
        if "visible" in self._fields:
            box = QCheckBox()
            box.setChecked(shared["visible"])
            box.toggled.connect(lambda v: self._emit("shared", "visible", bool(v)))
            self._appliers[("shared", None, "visible")] = _checkbox_applier(box)
            self._controls[("shared", None, "visible")] = box
            layout.addRow(FIELD_LABELS["visible"], box)

        for field, choices in (
            ("transparency_mode", values["transparency_modes"]),
            ("interpolation", values["interpolations"]),
        ):
            combo = QComboBox()
            combo.addItems(list(choices))
            combo.setCurrentText(shared[field])
            combo.currentTextChanged.connect(
                lambda text, f=field: self._emit("shared", f, text)
            )
            self._appliers[("shared", None, field)] = _combo_applier(combo)
            self._controls[("shared", None, field)] = combo
            layout.addRow(FIELD_LABELS[field], combo)

        if "attenuation" in shared and "attenuation" in self._fields:
            slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
            slider.setRange(0.0, 10.0)
            slider.setValue(shared["attenuation"])
            slider.valueChanged.connect(
                lambda v: self._emit("shared", "attenuation", float(v))
            )
            self._appliers[("shared", None, "attenuation")] = _value_applier(slider)
            self._controls[("shared", None, "attenuation")] = slider
            layout.addRow(FIELD_LABELS["attenuation"], slider)

    def _mode_rows(self, page: str, channel: int | None, mode: dict, values: dict):
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QCheckBox, QComboBox, QFormLayout, QWidget
        from superqt import (
            QColormapComboBox,
            QLabeledDoubleRangeSlider,
            QLabeledDoubleSlider,
        )

        rows = QWidget()
        layout = QFormLayout(rows)
        layout.setContentsMargins(0, 0, 0, 0)

        def emit(field):
            return lambda value: self._emit(page, field, value, channel)

        if page == "channel":
            box = QCheckBox()
            box.setChecked(bool(mode["visible"]))
            box.toggled.connect(lambda v: self._emit(page, "visible", bool(v), channel))
            self._appliers[(page, channel, "visible")] = _checkbox_applier(box)
            self._controls[(page, channel, "visible")] = box
            layout.addRow(FIELD_LABELS["visible"], box)

        for field in MODE_FIELDS:
            if field not in self._fields:
                continue
            if field == "color_map":
                control = QColormapComboBox()
                control.addColormaps(values["colormap_names"])
                # The model's own colormap, not its name: superqt adds and
                # selects a Colormap object it does not hold, but cannot
                # resolve the name of one built inline (design 3.10).
                control.setCurrentColormap(_mode_colormap(values, page, channel, mode))
                control.currentColormapChanged.connect(
                    lambda cm, f=field: self._emit(page, f, _colormap_name(cm), channel)
                )
                applier = _colormap_applier(control)
            elif field == "clim":
                control = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
                control.setDecimals(2)
                control.setRange(*values["clim_range"])
                control.setValue(tuple(mode["clim"]))
                control.valueChanged.connect(
                    lambda v: self._emit(page, "clim", list(v), channel)
                )
                applier = _value_applier(control, tuple)
            elif field == "render_mode":
                control = QComboBox()
                control.addItems(values["render_modes"])
                control.setCurrentText(mode["render_mode"])
                control.currentTextChanged.connect(emit(field))
                applier = _combo_applier(control)
            else:  # opacity, iso_threshold
                control = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
                control.setRange(0.0, 1.0)
                control.setValue(float(mode[field]))
                control.valueChanged.connect(
                    lambda v, f=field: self._emit(page, f, float(v), channel)
                )
                applier = _value_applier(control)
            self._appliers[(page, channel, field)] = applier
            self._controls[(page, channel, field)] = control
            layout.addRow(FIELD_LABELS[field], control)
        return rows

    def _add_channel_group(self, layout, index: int, values: dict) -> None:
        from qtpy.QtGui import QFont
        from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

        group = QWidget()
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(0, 4, 0, 4)
        header = QLabel(values["channel_labels"].get(index, f"Channel {index}"))
        font = QFont(header.font())
        font.setBold(True)
        header.setFont(font)
        group_layout.addWidget(header)
        group_layout.addWidget(
            self._mode_rows("channel", index, values["channels"][index], values)
        )
        layout.addWidget(group)
        self._channel_groups[index] = group

    # ── widget -> model ─────────────────────────────────────────────────

    def _emit(self, page: str, field: str, value: Any, channel: int | None = None):
        for visual_id in self._visual_ids:
            self.changed.emit(
                image_update_event(self._id, visual_id, page, field, value, channel)
            )

    def _on_composite_toggled(self, checked: bool) -> None:
        """Switch pages and ask for the mode; go back if the switch is refused."""
        checked = bool(checked)
        self._pages.setCurrentIndex(1 if checked else 0)
        try:
            for visual_id in self._visual_ids:
                self.changed.emit(
                    ImageCompositeUpdateEvent(self._id, visual_id, checked)
                )
        except Exception:
            self._set_composite(not checked)
            raise

    # ── model -> widget ─────────────────────────────────────────────────

    def _set_composite(self, composite: bool) -> None:
        self._composite_box.blockSignals(True)
        self._composite_box.setChecked(composite)
        self._composite_box.blockSignals(False)
        self._pages.setCurrentIndex(1 if composite else 0)

    def _on_event(self, event) -> None:
        if event.source_id == self._id:
            return
        target = inbound_target(event)
        if target is None:
            return
        page, channel, field, value = target
        if page == "composite":
            self._set_composite(value)
            return
        if page == "single" and field is None:
            for name, new in mode_values(value).items():
                applier = self._appliers.get(("single", None, name))
                if applier is not None:
                    applier(new)
            return
        applier = self._appliers.get((page, channel, field))
        if applier is not None:
            applier(value)


def _mode_colormap(values: dict, page: str, channel: int | None, mode: dict):
    """The model's own colormap for this page, falling back to its name."""
    colormaps = values.get("colormaps") or {}
    if page == "channel":
        value = (colormaps.get("channels") or {}).get(channel)
    else:
        value = colormaps.get("single")
    return mode["color_map"] if value is None else value


def _colormap_name(colormap) -> str:
    from cellier.gui._colormap_util import colormap_to_str

    return colormap_to_str(colormap)


def _blocked(control, setter: Callable[[], None]) -> None:
    control.blockSignals(True)
    try:
        setter()
    finally:
        control.blockSignals(False)


def _checkbox_applier(control):
    return lambda value: _blocked(control, lambda: control.setChecked(bool(value)))


def _combo_applier(control):
    return lambda value: _blocked(control, lambda: control.setCurrentText(str(value)))


def _colormap_applier(control):
    def _apply(value) -> None:
        try:
            _blocked(control, lambda: control.setCurrentColormap(value))
        except ValueError:
            # An inline colormap that reached the bus as its name cannot be
            # resolved; leave the combo as it is rather than raise into the
            # event bus, which would take the emitting edit down with it.
            pass

    return _apply


def _value_applier(control, convert=float):
    return lambda value: _blocked(control, lambda: control.setValue(convert(value)))
