"""The Qt "Clipping planes" control of a visual.

One row per plane, described and edited by
:class:`cellier.gui._clipping_planes.ClippingPlanesEditor`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from psygnal import Signal

from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._clipping_planes import (
    CLIPPING_PLANES_TITLE,
    GIZMO_TOOLTIP,
    ClippingPlanesEditor,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from uuid import UUID

    from cellier.events import SubscriptionSpec

#: Steps of the position slider (Qt sliders are integers).
_SLIDER_STEPS = 1000


class _PlaneRow:
    """The widgets of one plane.  Updated in place; never rebuilt by a move.

    The normal is laid out one column per data axis: two buttons named
    after the axis (``+z`` and ``-z`` for an axis ``z``) that face the plane
    along it, and under them the normal's entry on it.
    """

    def __init__(self, owner: QtClippingPlanesControls, index: int, parent) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QCheckBox,
            QDoubleSpinBox,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSlider,
            QToolButton,
            QWidget,
        )

        self.index = index
        self._range = (0.0, 1.0)
        editor = owner._editor
        self._editor = editor
        self.widget = QWidget(parent)
        grid = QGridLayout(self.widget)
        grid.setContentsMargins(0, 2, 0, 6)
        n_axes = len(editor.axis_names)

        self.enabled = QCheckBox(f"Plane {index + 1}", self.widget)
        self.enabled.setToolTip("Whether this plane clips. It stays in the list.")
        self.enabled.toggled.connect(lambda v: editor.set_enabled(self.index, v))

        # Drawn from what the controller says, not from the click: one
        # canvas has one gizmo, so switching one on switches another off.
        self.gizmo = QPushButton("Gizmo", self.widget)
        self.gizmo.setCheckable(True)
        self.gizmo.setToolTip(GIZMO_TOOLTIP)
        self.gizmo.setVisible(editor.has_gizmo)
        self.gizmo.clicked.connect(lambda on: editor.set_gizmo(self.index, on))

        self.flip = QPushButton("Flip", self.widget)
        self.flip.setToolTip("Keep the other side of the plane.")
        self.flip.clicked.connect(lambda: editor.flip(self.index))

        self.remove = QPushButton("Remove", self.widget)
        self.remove.clicked.connect(lambda: editor.remove(self.index))

        header = QHBoxLayout()
        header.addWidget(self.enabled)
        header.addStretch(1)
        header.addWidget(self.gizmo)
        header.addWidget(self.flip)
        header.addWidget(self.remove)
        grid.addLayout(header, 0, 0, 1, n_axes + 1)

        normal = QLabel("Normal", self.widget)
        normal.setToolTip(
            "The normal, one entry per data axis. It points to the side that "
            "is kept. Only its direction matters. A button faces the plane "
            "along its axis and keeps that side."
        )
        # Beside both of its rows: the buttons and the entries.
        grid.addWidget(normal, 1, 0, 2, 1, Qt.AlignmentFlag.AlignVCenter)

        #: ``(axis index, sign)`` -> the button that faces the plane that way.
        self.facing: dict[tuple[int, int], Any] = {}
        #: The normal's entry on each data axis, in axis order.
        self.components: list[Any] = []
        for axis, name in enumerate(editor.axis_names):
            column = axis + 1
            buttons = QHBoxLayout()
            buttons.setSpacing(2)
            for sign, text, side in ((1, "+", "higher"), (-1, "-", "lower")):
                button = QToolButton(self.widget)
                button.setText(f"{text}{name}")
                button.setCheckable(True)
                button.setToolTip(
                    f"Cut across {name} and keep the side toward {side} {name}."
                )
                button.clicked.connect(
                    lambda _checked, a=axis, s=sign: self._on_facing(a, s)
                )
                buttons.addWidget(button)
                self.facing[(axis, sign)] = button
            grid.addLayout(buttons, 1, column, Qt.AlignmentFlag.AlignCenter)

            component = QDoubleSpinBox(self.widget)
            component.setDecimals(3)
            component.setRange(-100.0, 100.0)
            component.setSingleStep(0.1)
            component.setKeyboardTracking(False)
            component.setToolTip(f"The normal's entry on {name}.")
            component.valueChanged.connect(
                lambda v, a=axis: editor.set_component(self.index, a, v)
            )
            grid.addWidget(component, 2, column)
            grid.setColumnStretch(column, 1)
            self.components.append(component)

        self.slider = QSlider(Qt.Orientation.Horizontal, self.widget)
        self.slider.setRange(0, _SLIDER_STEPS)
        self.slider.setToolTip("Where the plane sits along its normal. Data units.")
        self.slider.valueChanged.connect(self._on_slider)

        self.position = QDoubleSpinBox(self.widget)
        self.position.setDecimals(2)
        self.position.setRange(-1e9, 1e9)
        self.position.setKeyboardTracking(False)
        self.position.valueChanged.connect(lambda v: editor.set_position(self.index, v))

        grid.addWidget(QLabel("Position", self.widget), 3, 0)
        along = QHBoxLayout()
        along.addWidget(self.slider, 1)
        along.addWidget(self.position)
        grid.addLayout(along, 3, 1, 1, n_axes)

        self._inputs = (
            self.enabled,
            *self.facing.values(),
            *self.components,
            self.slider,
            self.position,
        )

    def _on_slider(self, step: int) -> None:
        low, high = self._range
        self._editor.set_position(self.index, low + (high - low) * step / _SLIDER_STEPS)

    def _on_facing(self, axis: int, sign: int) -> None:
        # A click on the button already shown un-checks it and changes
        # nothing, so nothing would draw it again.
        self.facing[(axis, sign)].setChecked(True)
        self._editor.set_facing(self.index, axis, sign)

    def show(self, row: Mapping[str, Any]) -> None:
        """Show *row* (from ``ClippingPlanesEditor.describe``); emits nothing."""
        for widget in self._inputs:
            widget.blockSignals(True)
        try:
            self.enabled.setChecked(bool(row["enabled"]))
            facing = None if row["facing"] is None else tuple(row["facing"])
            for key, button in self.facing.items():
                button.setChecked(key == facing)
            for component, value in zip(self.components, row["normal"]):
                component.setValue(float(value))
            low, high = float(row["low"]), float(row["high"])
            self._range = (low, high)
            span = high - low
            step = (
                0
                if span <= 0
                else round((row["position"] - low) / span * _SLIDER_STEPS)
            )
            self.slider.setValue(int(min(max(step, 0), _SLIDER_STEPS)))
            self.position.setSingleStep(max(span / 200.0, 0.01))
            self.position.setValue(float(row["position"]))
            blocked = str(row.get("gizmo_blocked", ""))
            self.gizmo.setChecked(bool(row.get("gizmo", False)))
            self.gizmo.setEnabled(not blocked)
            self.gizmo.setToolTip(blocked or GIZMO_TOOLTIP)
        finally:
            for widget in self._inputs:
                widget.blockSignals(False)


class QtClippingPlanesControls(VisualIdGroup):
    """A visual's clipping planes: one row per plane, and an add button.

    Each row has an enabled checkbox, a flip button, a remove button, the
    normal and a position slider along the normal.  The normal has one
    column per data axis: two buttons named after the axis as the store's
    coordinate system gives it (``+z`` and ``-z``) that face the plane
    along the axis, and under them the normal's entry on it.  Values are in the visual's
    data coordinates.  Given *gizmo*, a row also has a "Gizmo" toggle that
    puts a gizmo on its plane in the 3D canvas; one plane of a canvas has
    it at a time.  An edit is sent as ``ClippingPlanesUpdateEvent`` with the
    whole new tuple.  A moved plane updates its row in place, so a slider is
    not destroyed while it is dragged.

    Wire to the controller after construction::

        seed = get_clipping_planes_data_from_visual(visual, store)
        controls = QtClippingPlanesControls(visual.id, **seed)
        controller.connect_widget(
            controls, subscription_specs=controls.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        The visual, or a group of them (an ``OrthoViewer``'s panel
        siblings), edited together.
    coordinate_system :
        The level-0 data coordinate system of the store they read (its id).
    axis_names :
        The data axis names, in order.
    bounds :
        ``(low, high)`` of the data on each axis.
    planes :
        The current rows (``rows_from_planes(visual.clipping_planes)``).
    data_store_id :
        The store the visuals read.
    bounds_source :
        Reads the store's current ``(low, high)`` per axis.  Given with
        *data_store_id*, the position ranges follow the store's extent;
        without it they stay at *bounds*.
    gizmo, gizmo_plane, gizmo_blocked :
        Where a plane's gizmo is drawn, the plane that has it now, and why
        a plane cannot have one: ``gizmo_seed(controller, visual_ids)``.
        Without *gizmo* the rows have no gizmo toggle.
    title :
        The group's name.  Defaults to :data:`DEFAULT_TITLE`.
    parent :
        Optional Qt parent widget.
    """

    DEFAULT_TITLE = CLIPPING_PLANES_TITLE
    """Name shown when no ``title=`` is given."""

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        coordinate_system: UUID | str,
        axis_names: Sequence[str],
        bounds: Sequence[Sequence[float]],
        planes: Sequence[Mapping[str, Any]] = (),
        data_store_id: UUID | str | None = None,
        bounds_source: Callable[[], Sequence[Sequence[float]]] | None = None,
        gizmo: Mapping[str, Any] | None = None,
        gizmo_plane: str | None = None,
        gizmo_blocked: Callable[[str], str] | None = None,
        title: str | None = None,
        parent=None,
    ) -> None:
        from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

        from cellier.gui.qt.visuals._chrome import titled_group

        self._id = uuid4()
        self._init_visual_ids(visual_id)

        self._content = QWidget(parent)
        column = QVBoxLayout(self._content)
        column.setContentsMargins(0, 0, 0, 0)
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        column.addLayout(self._rows_layout)
        self._rows: list[_PlaneRow] = []

        self._add = QPushButton("Add plane", self._content)
        self._add.setToolTip("Add a plane across the last axis, through the middle.")
        column.addWidget(self._add)

        self._error = QLabel(self._content)
        self._error.setWordWrap(True)
        self._error.setStyleSheet("color: #d04040")
        self._error.setVisible(False)
        column.addWidget(self._error)

        self._group = titled_group(
            self.DEFAULT_TITLE if title is None else title, self._content, parent
        )
        self._editor = ClippingPlanesEditor(
            self._visual_ids,
            coordinate_system,
            axis_names,
            bounds,
            planes,
            self._id,
            self.changed.emit,
            self._show,
            data_store_id=data_store_id,
            bounds_source=bounds_source,
            gizmo=gizmo,
            gizmo_plane=gizmo_plane,
            gizmo_blocked=gizmo_blocked,
        )
        self._add.clicked.connect(lambda: self._editor.add())
        self._show(self._editor.rows, "")

    # -- Public interface ------------------------------------------------------

    @property
    def widget(self):
        """The widget to insert into a layout: the titled group."""
        return self._group

    @property
    def planes(self) -> list[dict[str, Any]]:
        """The rows as last reported by the model."""
        return [dict(row) for row in self._editor.rows]

    @property
    def editor(self) -> ClippingPlanesEditor:
        """The toolkit-neutral half (for tests and scripting)."""
        return self._editor

    @property
    def error(self) -> str:
        """The reason the last edit was refused, or ""."""
        return self._error.text()

    def row(self, index: int) -> _PlaneRow:
        """The widgets of plane *index* (for tests and scripting)."""
        return self._rows[index]

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``ClippingPlanesChangedEvent`` per visual, and the store's extent."""
        return self._editor.subscription_specs()

    # -- model -> widget -------------------------------------------------------

    def _show(self, rows: list[dict[str, Any]], error: str) -> None:
        described = self._editor.describe()
        while len(self._rows) > len(described):
            row = self._rows.pop()
            self._rows_layout.removeWidget(row.widget)
            row.widget.setParent(None)
            row.widget.deleteLater()
        while len(self._rows) < len(described):
            row = _PlaneRow(self, len(self._rows), self._content)
            self._rows.append(row)
            self._rows_layout.addWidget(row.widget)
        for row, values in zip(self._rows, described):
            row.show(values)
        self._error.setText(error)
        self._error.setVisible(bool(error))
