"""The Qt "Render planes" control of a visual.

One row per plane, described and edited by
:class:`cellier.gui._render_planes.RenderPlanesEditor`.  The rows are those
of the "Clipping planes" control with the extents added.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from psygnal import Signal

from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._clipping_planes import OUTSIDE_TOOLTIP, position_tooltip
from cellier.gui._render_planes import (
    RENDER_PLANE_GIZMO_TOOLTIP,
    RENDER_PLANES_TITLE,
    RenderPlanesEditor,
)
from cellier.gui.qt.visuals._plane_outline import OutlineInputs

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from uuid import UUID

    from cellier.events import SubscriptionSpec
    from cellier.gui._render_planes import RenderPlaneGizmoTarget

#: Steps of the position slider (Qt sliders are integers).
_SLIDER_STEPS = 1000
#: The colour of the position's number when the plane is outside the
#: bounding box (it is also set in italics).
_OUTSIDE_COLOR = "#d08020"
#: The label of each extent side, by ``side`` (0 the minimum, 1 the maximum).
_SIDE_NAMES = ("min", "max")
#: The least width of an extent side's number, in pixels.  Its range would
#: otherwise size it for a ten digit number and widen the whole dock.
_SIDE_WIDTH = 90
#: The gap between a side's number and its toggle, and what the gap between
#: the two sides has on top of it, in pixels.
_SIDE_GAP = 4
_SIDES_GAP = 6
#: The text of an extent side's "unbounded" toggle.
_UNBOUNDED_TEXT = "inf"


class _PlaneRow:
    """The widgets of one plane.  Updated in place; never rebuilt by a move.

    The normal is laid out one column per axis of the plane: two buttons
    named after the axis (``+z`` and ``-z`` for an axis ``z``) that face the
    plane along it, and under them the normal's entry on it.  Under the
    position, the extents: a column per side (``min`` and ``max``) and a
    line per in-plane axis, each side a number and an ``inf`` toggle that
    makes it unbounded.  Last the plane's outline: a box that draws it and
    its colour.
    """

    def __init__(self, owner: QtRenderPlanesControls, index: int, parent) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QCheckBox,
            QDoubleSpinBox,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSizePolicy,
            QSlider,
            QToolButton,
            QWidget,
        )

        self.index = index
        #: The slider's far end: the box's size along the normal.
        self._span = 1.0
        editor = owner._editor
        self._editor = editor
        self.widget = QWidget(parent)
        grid = QGridLayout(self.widget)
        grid.setContentsMargins(0, 2, 0, 6)

        self.enabled = QCheckBox(f"Plane {index + 1}", self.widget)
        self.enabled.setToolTip("Whether this plane is drawn. It stays in the list.")
        self.enabled.toggled.connect(lambda v: editor.set_enabled(self.index, v))

        # Drawn from what the controller says, not from the click: one
        # canvas has one gizmo, so switching one on switches another off.
        self.gizmo = QPushButton("Gizmo", self.widget)
        self.gizmo.setCheckable(True)
        self.gizmo.setToolTip(RENDER_PLANE_GIZMO_TOOLTIP)
        self.gizmo.setVisible(editor.has_gizmo)
        self.gizmo.clicked.connect(lambda on: editor.set_gizmo(self.index, on))

        self.flip = QPushButton("Flip", self.widget)
        self.flip.setToolTip(
            "Reverse the normal. A plane is drawn the same from both sides, "
            "so the picture does not change."
        )
        self.flip.clicked.connect(lambda: editor.flip(self.index))

        self.remove = QPushButton("Remove", self.widget)
        self.remove.clicked.connect(lambda: editor.remove(self.index))

        header = QHBoxLayout()
        header.addWidget(self.enabled)
        header.addStretch(1)
        header.addWidget(self.gizmo)
        header.addWidget(self.flip)
        header.addWidget(self.remove)
        grid.addLayout(header, 0, 0, 1, 4)

        normal = QLabel("Normal", self.widget)
        normal.setToolTip(
            "The plane's normal, one entry per world axis of the plane. Only "
            "its direction matters. A button faces the plane along its axis."
        )
        # Beside both of its rows: the buttons and the entries.
        grid.addWidget(normal, 1, 0, 2, 1, Qt.AlignmentFlag.AlignVCenter)

        #: ``(axis index, sign)`` -> the button that faces the plane that way.
        self.facing: dict[tuple[int, int], Any] = {}
        #: The normal's entry on each of the plane's three axes.
        self.components: list[Any] = []
        for axis in range(3):
            column = axis + 1
            buttons = QHBoxLayout()
            buttons.setSpacing(2)
            for sign in (1, -1):
                button = QToolButton(self.widget)
                button.setCheckable(True)
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
            component.valueChanged.connect(
                lambda v, a=axis: editor.set_component(self.index, a, v)
            )
            grid.addWidget(component, 2, column)
            grid.setColumnStretch(column, 1)
            self.components.append(component)

        self.slider = QSlider(Qt.Orientation.Horizontal, self.widget)
        self.slider.setRange(0, _SLIDER_STEPS)
        self._tooltip = position_tooltip("World")
        self.slider.setToolTip(self._tooltip)
        self.slider.valueChanged.connect(self._on_slider)

        # The number is the depth into the box (see ``depth_of``).  Its range
        # is not the slider's: a gizmo can take a plane out of the box, and
        # a spin box cannot show a value outside its range.
        self.position = QDoubleSpinBox(self.widget)
        self.position.setDecimals(2)
        self.position.setRange(-1e9, 1e9)
        self.position.setKeyboardTracking(False)
        self.position.setToolTip(self._tooltip)
        self.position.valueChanged.connect(lambda v: editor.set_depth(self.index, v))
        #: Whether the number is drawn as a plane's outside the box.
        self.outside = False

        grid.addWidget(QLabel("Position", self.widget), 3, 0)
        along = QHBoxLayout()
        along.addWidget(self.slider, 1)
        along.addWidget(self.position)
        grid.addLayout(along, 3, 1, 1, 3)

        #: ``(in-plane axis, side)`` -> the side's value, and its
        #: "unbounded" toggle.
        self.sides: dict[tuple[int, int], Any] = {}
        self.unbounded: dict[tuple[int, int], Any] = {}

        def narrow(widget) -> None:
            # As wide as its share of the line, and no narrower than
            # ``_SIDE_WIDTH``; its own size hint is not asked.
            widget.setMinimumWidth(_SIDE_WIDTH)
            widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        toggle_width = 0
        for axis in (0, 1):
            label = QLabel(f"Extent {axis}", self.widget)
            label.setToolTip(
                f"How far the plane reaches from its origin along in-plane "
                f"axis {axis}, in world units. An unbounded side ends at the "
                "data."
            )
            grid.addWidget(label, 5 + axis, 0)
            line = QHBoxLayout()
            line.setSpacing(_SIDE_GAP)
            for side in (0, 1):
                if side:
                    line.addSpacing(_SIDES_GAP)
                value = QDoubleSpinBox(self.widget)
                value.setDecimals(2)
                value.setRange(-1e9, 1e9)
                value.setKeyboardTracking(False)
                value.setToolTip(f"The {_SIDE_NAMES[side]} along in-plane axis {axis}.")
                value.valueChanged.connect(
                    lambda v, a=axis, s=side: editor.set_side(self.index, a, s, v)
                )
                narrow(value)
                box = QToolButton(self.widget)
                box.setText(_UNBOUNDED_TEXT)
                box.setCheckable(True)
                box.setToolTip(
                    f"Unbounded: no {_SIDE_NAMES[side]}, the plane ends at the "
                    "data on this side. Switching it off fills in where the "
                    "data ends now."
                )
                box.toggled.connect(
                    lambda on, a=axis, s=side: editor.set_side_bounded(
                        self.index, a, s, not on
                    )
                )
                toggle_width = max(toggle_width, box.sizeHint().width())
                line.addWidget(value, 1)
                line.addWidget(box)
                self.sides[(axis, side)] = value
                self.unbounded[(axis, side)] = box
            grid.addLayout(line, 5 + axis, 1, 1, 3)

        # The column heads: each as wide as the number under it, then the
        # width of the toggle beside that.
        heads = QHBoxLayout()
        heads.setSpacing(0)
        for side in (0, 1):
            if side:
                heads.addSpacing(_SIDES_GAP + _SIDE_GAP)
            head = QLabel(_SIDE_NAMES[side], self.widget)
            narrow(head)
            heads.addWidget(head, 1)
            heads.addSpacing(_SIDE_GAP + toggle_width)
        for box in self.unbounded.values():
            box.setFixedWidth(toggle_width)
        grid.addLayout(heads, 4, 1, 1, 3)

        #: The outline's check box and colour button.
        self.outline = OutlineInputs(
            self.widget,
            lambda on: editor.set_outline_enabled(self.index, on),
            lambda color: editor.set_outline_color(self.index, color),
        )
        grid.addWidget(self.outline.enabled, 7, 0)
        grid.addLayout(self.outline.layout, 7, 1, 1, 3)

        self._inputs = (
            self.enabled,
            *self.facing.values(),
            *self.components,
            self.slider,
            self.position,
            *self.sides.values(),
            *self.unbounded.values(),
        )

    def _on_slider(self, step: int) -> None:
        self._editor.set_depth(self.index, self._span * step / _SLIDER_STEPS)

    def _mark_outside(self, outside: bool) -> None:
        """Draw the number as a plane's that misses the box, or as usual."""
        if outside == self.outside:
            return
        from qtpy.QtGui import QColor, QPalette

        self.outside = outside
        # Through the palette and the font, not a style sheet: a style
        # sheet would take the spin box out of the platform's own style.
        palette = QPalette()
        if outside:
            palette.setColor(QPalette.ColorRole.Text, QColor(_OUTSIDE_COLOR))
        self.position.setPalette(palette)
        font = self.position.font()
        font.setItalic(outside)
        self.position.setFont(font)
        self.position.setToolTip(
            f"{OUTSIDE_TOOLTIP} {self._tooltip}" if outside else self._tooltip
        )

    def _on_facing(self, axis: int, sign: int) -> None:
        # A click on the button already shown un-checks it and changes
        # nothing, so nothing would draw it again.
        self.facing[(axis, sign)].setChecked(True)
        self._editor.set_facing(self.index, axis, sign)

    def show(self, row: Mapping[str, Any]) -> None:
        """Show *row* (from ``RenderPlanesEditor.describe``); emits nothing."""
        for widget in self._inputs:
            widget.blockSignals(True)
        try:
            self.enabled.setChecked(bool(row["enabled"]))
            facing = None if row["facing"] is None else tuple(row["facing"])
            names = [str(name) for name in row["axes"]]
            for (axis, sign), button in self.facing.items():
                name = names[axis]
                button.setText(f"{'+' if sign > 0 else '-'}{name}")
                button.setToolTip(f"Face the plane along {name}.")
                button.setChecked((axis, sign) == facing)
            for axis, (component, value) in enumerate(
                zip(self.components, row["normal"])
            ):
                component.setToolTip(f"The normal's entry on {names[axis]}.")
                component.setValue(float(value))
            depth, span = float(row["depth"]), float(row["span"])
            self._span = span
            # Outside the box the slider is pinned at the nearer end and
            # the number shows how far past it the plane is.
            step = 0 if span <= 0 else round(depth / span * _SLIDER_STEPS)
            self.slider.setValue(int(min(max(step, 0), _SLIDER_STEPS)))
            self.position.setSingleStep(max(span / 200.0, 0.01))
            self.position.setValue(depth)
            self._mark_outside(bool(row["outside"]))
            for (axis, side), value in self.sides.items():
                amount = row[f"extent_{axis}"][side]
                self.unbounded[(axis, side)].setChecked(amount is None)
                value.setEnabled(amount is not None)
                value.setSingleStep(max(span / 200.0, 0.01))
                if amount is not None:
                    value.setValue(float(amount))
            self.outline.show(row)
            blocked = str(row.get("gizmo_blocked", ""))
            self.gizmo.setChecked(bool(row.get("gizmo", False)))
            self.gizmo.setEnabled(not blocked)
            self.gizmo.setToolTip(blocked or RENDER_PLANE_GIZMO_TOOLTIP)
        finally:
            for widget in self._inputs:
                widget.blockSignals(False)


class QtRenderPlanesControls(VisualIdGroup):
    """A visual's render planes: one row per plane, and an add button.

    The planes a visual in the ``"plane"`` render mode draws its data on.
    Each row is a "Clipping planes" row with the extents added: an enabled
    checkbox, a flip button, a remove button, the normal (two buttons and an
    entry per axis of the plane), a position slider along the normal, and
    for each of the plane's two in-plane axes a minimum and a maximum, each
    with an "inf" toggle that makes it unbounded.  Values are in world
    units.  Given
    *gizmo_target*, a row also has a "Gizmo" toggle that puts the canvas's
    gizmo on its plane; one plane of a canvas has it at a time, clipping
    planes included.

    An edit is sent as ``RenderPlanesUpdateEvent`` with the whole new tuple.
    A moved plane updates its row in place, so a slider is not destroyed
    while it is dragged.  A visual draws at most four planes: the add button
    is disabled at four.  While the planes are not drawn (the visual is not
    in plane mode, or the view is 2D) the control is disabled and says why.

    Wire to the controller after construction::

        data = get_render_planes_data_from_visual(controller, visual.id)
        controls = QtRenderPlanesControls(visual.id, **data)
        controller.connect_widget(
            controls, subscription_specs=controls.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        The visual, or a group of them, edited together.
    coordinate_system :
        The scene's world coordinate system (its id).
    axis_ids, axis_names :
        The world axes, in order: their ids and names.
    planes :
        The visual's current ``render_planes``.
    displayed_axes, bounds, state_source, scene_id, data_store_id :
        What the scene and the visual's data are now, and a reader of it:
        ``get_render_planes_data_from_visual(controller, visual_id)``.
    gizmo_target, gizmo_plane, gizmo_blocked :
        Where a plane's gizmo is drawn (a ``RenderPlaneGizmoTarget``), the
        plane that has it now, and why a plane cannot have one:
        ``get_render_plane_gizmo_data(controller, visual_id, canvas_id)``.
        Without *gizmo_target* the rows have no gizmo toggle.
    title :
        The group's name.  Defaults to :data:`DEFAULT_TITLE`.
    parent :
        Optional Qt parent widget.
    """

    DEFAULT_TITLE = RENDER_PLANES_TITLE
    """Name shown when no ``title=`` is given."""

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        coordinate_system: UUID | str,
        axis_ids: Sequence[str],
        axis_names: Sequence[str],
        planes: Sequence[Any] = (),
        displayed_axes: Sequence[str] | None = None,
        bounds: Sequence[Sequence[float]] | None = None,
        state_source: Callable[[], Mapping[str, Any]] | None = None,
        scene_id: UUID | str | None = None,
        data_store_id: UUID | str | None = None,
        gizmo_target: RenderPlaneGizmoTarget | None = None,
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

        # Why the control is disabled; outside the part that is disabled.
        self._reason = QLabel(self._content)
        self._reason.setWordWrap(True)
        self._reason.setVisible(False)
        column.addWidget(self._reason)

        self._body = QWidget(self._content)
        self._rows_layout = QVBoxLayout(self._body)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self._body)
        self._rows: list[_PlaneRow] = []

        self._add = QPushButton("Add plane", self._content)
        self._add.setToolTip(
            "Add a plane through the middle of the data, facing the first "
            "displayed axis. A visual draws at most four."
        )
        column.addWidget(self._add)

        self._error = QLabel(self._content)
        self._error.setWordWrap(True)
        self._error.setStyleSheet("color: #d04040")
        self._error.setVisible(False)
        column.addWidget(self._error)

        self._group = titled_group(
            self.DEFAULT_TITLE if title is None else title, self._content, parent
        )
        self._editor = RenderPlanesEditor(
            self._visual_ids,
            coordinate_system,
            axis_ids,
            axis_names,
            planes,
            self._id,
            self.changed.emit,
            self._show,
            displayed_axes=displayed_axes,
            bounds=bounds,
            state_source=state_source,
            scene_id=scene_id,
            data_store_id=data_store_id,
            gizmo_target=gizmo_target,
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
        return self._editor.rows

    @property
    def editor(self) -> RenderPlanesEditor:
        """The toolkit-neutral half (for tests and scripting)."""
        return self._editor

    @property
    def error(self) -> str:
        """The reason the last edit was refused, or ""."""
        return self._error.text()

    @property
    def blocked(self) -> str:
        """Why the control is disabled now, or ""."""
        return self._editor.blocked

    def row(self, index: int) -> _PlaneRow:
        """The widgets of plane *index* (for tests and scripting)."""
        return self._rows[index]

    def refresh(self) -> None:
        """Read the scene's state again (whether the control is disabled)."""
        self._editor.refresh()

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """What the control follows on the bus (see the editor)."""
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
            row = _PlaneRow(self, len(self._rows), self._body)
            self._rows.append(row)
            self._rows_layout.addWidget(row.widget)
        for row, values in zip(self._rows, described):
            row.show(values)
        blocked = self._editor.blocked
        self._reason.setText(blocked)
        self._reason.setVisible(bool(blocked))
        self._body.setEnabled(not blocked)
        self._add.setEnabled(self._editor.can_add)
        self._error.setText(error)
        self._error.setVisible(bool(error))
