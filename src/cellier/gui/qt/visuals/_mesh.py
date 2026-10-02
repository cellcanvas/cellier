"""Appearance controls specific to mesh visuals (Qt).

Layer 3 of the three-layer design in ``plans/convenience_cleanup.md`` section
10.2: each class is a field name, a label, and the per-field defaults.  All
behaviour -- the bus contract, the echo filter, the fan-out over an
``OrthoViewer``'s panel group -- lives in the layer-1 base, and the control
itself in the layer-2 type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

from psygnal import Signal

from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._mesh_section import (
    MESH_SECTION_TITLE,
    MeshSectionEditor,
    SectionVisibility,
    mesh_section_fields,
)
from cellier.gui.qt.visuals._base import (
    QtChoice,
    QtFloatSpin,
    QtToggle,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from cellier.events import SubscriptionSpec


class QtSideCombo(QtChoice):
    """Which face windings are drawn.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.side``.
    """

    _field: ClassVar[str] = "side"
    _label: ClassVar[str] = "Side"
    _default_value: ClassVar[str] = "both"
    _default_choices: ClassVar[tuple[str, ...]] = ("both", "front", "back")


class QtWireframeToggle(QtToggle):
    """Draw the mesh as edges only.  Flat meshes.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.wireframe``.
    """

    _field: ClassVar[str] = "wireframe"
    _label: ClassVar[str] = "Wireframe"
    _default_value: ClassVar[bool] = False


class QtWireframeThicknessSpin(QtFloatSpin):
    """Edge thickness of the mesh wireframe, in screen pixels.

    Below 0.1 the wireframe is invisible; above roughly 20 it
    swamps the mesh.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.wireframe_thickness``.
    """

    _field: ClassVar[str] = "wireframe_thickness"
    _label: ClassVar[str] = "Wireframe thickness"
    _default_value: ClassVar[float] = 1.0
    _default_range: ClassVar[tuple[float, float]] = (0.1, 20.0)
    _default_step: ClassVar[float] = 0.1


class QtShininessSpin(QtFloatSpin):
    """Phong specular exponent.  Phong meshes.

    128 is the conventional practical ceiling.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.shininess``.
    """

    _field: ClassVar[str] = "shininess"
    _label: ClassVar[str] = "Shininess"
    _default_value: ClassVar[float] = 30.0
    _default_range: ClassVar[tuple[float, float]] = (0.0, 128.0)
    _default_step: ClassVar[float] = 1.0


class QtFlatShadingToggle(QtToggle):
    """Use face normals instead of smooth vertex normals.  Phong meshes.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.flat_shading``.
    """

    _field: ClassVar[str] = "flat_shading"
    _label: ClassVar[str] = "Flat shading"
    _default_value: ClassVar[bool] = False


class QtMeshSectionControls(VisualIdGroup):
    """How a mesh is drawn in a 2D view: every ``MeshSectionConfig`` field.

    One row per field: the outline on or off, the fill on or off, the
    outline's width in screen pixels, and whether the cut is the slice plane
    (``cut``) or the scene's slab (``slab``).  An edit is sent as
    ``MeshSectionUpdateEvent``.  Turning a part on or off, or changing the
    mode, reads the mesh again; the width applies at once.

    The settings act only in a 2D view, so the group is hidden while none of
    the scenes in ``displayed_dimensions`` displays two dimensions.

    Wire to the controller after construction::

        controls = QtMeshSectionControls(visual.id, section=visual.section.model_dump())
        controller.connect_widget(
            controls, subscription_specs=controls.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        The mesh visual, or a group of them (an ``OrthoViewer``'s panel
        siblings), edited together.
    section :
        The current config, as ``MeshSectionConfig.model_dump()``.
    title :
        The group's name.  Defaults to :data:`DEFAULT_TITLE`.
    displayed_dimensions :
        Scene id to how many dimensions it displays now
        (``section_display_seed``).  The control follows these scenes'
        ``DimsChangedEvent``.  Empty (the default) keeps it always shown.
    parent :
        Optional Qt parent widget.
    """

    DEFAULT_TITLE = MESH_SECTION_TITLE
    """Name shown when no ``title=`` is given."""

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        section: Mapping[str, Any],
        title: str | None = None,
        displayed_dimensions: Mapping[UUID, int] | None = None,
        parent=None,
    ) -> None:
        from qtpy.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QLabel,
            QVBoxLayout,
            QWidget,
        )

        from cellier.gui.qt.visuals._chrome import titled_group

        self._id = uuid4()
        self._init_visual_ids(visual_id)

        content = QWidget(parent)
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        column.addLayout(form)

        self._inputs: dict[str, Any] = {}
        for field in mesh_section_fields():
            name = field.name
            if field.kind == "bool":
                widget = QCheckBox(content)
                widget.toggled.connect(lambda v, n=name: self._editor.edit(n, v))
            elif field.kind == "choice":
                widget = QComboBox(content)
                widget.addItems(list(field.choices))
                widget.currentTextChanged.connect(
                    lambda v, n=name: self._editor.edit(n, v)
                )
            else:
                widget = QDoubleSpinBox(content)
                widget.setRange(field.minimum, field.maximum)
                widget.setSingleStep(field.step)
                widget.setDecimals(1)
                widget.setSuffix(" px")
                widget.valueChanged.connect(lambda v, n=name: self._editor.edit(n, v))
            self._inputs[name] = widget
            form.addRow(field.label, widget)
            widget.setToolTip(field.tooltip)
            form.labelForField(widget).setToolTip(field.tooltip)

        self._error = QLabel(content)
        self._error.setWordWrap(True)
        self._error.setStyleSheet("color: #d04040")
        self._error.setVisible(False)
        column.addWidget(self._error)

        # The group sits in a bare holder and is shown or hidden inside it.
        # Showing a widget that has no parent opens it as a window, so the
        # widget handed to layouts is the holder, which is never touched.
        self._holder = QWidget(parent)
        holder_layout = QVBoxLayout(self._holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        self._group = titled_group(
            self.DEFAULT_TITLE if title is None else title, content, self._holder
        )
        holder_layout.addWidget(self._group)
        self._editor = MeshSectionEditor(
            self._visual_ids, section, self._id, self.changed.emit, self._show
        )
        self._show(dict(section), "")
        self._visibility = SectionVisibility(
            displayed_dimensions, self._group.setVisible
        )
        self._group.setVisible(self._visibility.visible)

    # -- Public interface ------------------------------------------------------

    @property
    def widget(self):
        """The widget to insert into a layout: it holds the titled group."""
        return self._holder

    @property
    def shown(self) -> bool:
        """Whether the group is shown: a followed scene displays 2D."""
        return self._visibility.visible

    @property
    def config(self) -> dict[str, Any]:
        """The settings as last reported by the model."""
        return dict(self._editor.config)

    @property
    def error(self) -> str:
        """The reason the last edit was refused, or ""."""
        return self._error.text()

    def input(self, field: str):
        """The control for one field (for tests and scripting)."""
        return self._inputs[field]

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``MeshSectionChangedEvent`` per visual, one dims per scene."""
        return [
            *self._editor.subscription_specs(),
            *self._visibility.subscription_specs(),
        ]

    # -- model -> widget -------------------------------------------------------

    def _show(self, config: dict[str, Any], error: str) -> None:
        for name, widget in self._inputs.items():
            if name not in config:
                continue
            value = config[name]
            widget.blockSignals(True)
            try:
                if hasattr(widget, "setChecked"):
                    widget.setChecked(bool(value))
                elif hasattr(widget, "setCurrentText"):
                    widget.setCurrentText(str(value))
                else:
                    widget.setValue(float(value))
            finally:
                widget.blockSignals(False)
        self._error.setText(error)
        self._error.setVisible(bool(error))
