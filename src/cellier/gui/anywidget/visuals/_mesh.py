"""Appearance controls specific to mesh visuals (anywidget).

Layer 3 of the three-layer design in ``plans/convenience_cleanup.md`` section
10.2: each class is a field name, a label, and the per-field defaults.  All
behaviour -- the bus contract, the echo filter, the fan-out over an
``OrthoViewer``'s panel group -- lives in the layer-1 base, and the control
itself in the layer-2 type.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._mesh_section import (
    MESH_SECTION_TITLE,
    MeshSectionEditor,
    SectionVisibility,
    mesh_section_fields,
)
from cellier.gui.anywidget._teardown import close_aux_widgets
from cellier.gui.anywidget.visuals._base import (
    AnywidgetChoice,
    AnywidgetFloatSpin,
    AnywidgetToggle,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from cellier.events import SubscriptionSpec

_STATIC = Path(__file__).parent / "static"


class AnywidgetSideCombo(AnywidgetChoice):
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


class AnywidgetWireframeToggle(AnywidgetToggle):
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


class AnywidgetWireframeThicknessSpin(AnywidgetFloatSpin):
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


class AnywidgetShininessSpin(AnywidgetFloatSpin):
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


class AnywidgetFlatShadingToggle(AnywidgetToggle):
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


class AnywidgetMeshSectionControls(VisualIdGroup, anywidget.AnyWidget):
    """How a mesh is drawn in a 2D view: every ``MeshSectionConfig`` field.

    Mirrors ``QtMeshSectionControls``.  The front end is the settings-rows
    module the loading settings use (``loading_config.js``): it draws one
    row per entry of ``fields`` from the values in ``config``, and reports
    an edit by setting ``edit``; a refused edit comes back as ``error``,
    with ``config`` unchanged.  ``hidden`` is true while none of the
    scenes in ``displayed_dimensions`` displays two dimensions.

    Parameters
    ----------
    visual_id :
        The mesh visual, or a group of them, edited together.
    section :
        The current config, as ``MeshSectionConfig.model_dump()``.
    displayed_dimensions :
        Scene id to how many dimensions it displays now
        (``section_display_seed``).  The control follows these scenes'
        ``DimsChangedEvent``.  Empty (the default) keeps it always shown.
    """

    _esm = _STATIC / "loading_config.js"
    _css = _STATIC / "loading_config.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    DEFAULT_TITLE = MESH_SECTION_TITLE
    """Name shown when no ``title=`` is given."""

    title = traitlets.Unicode(DEFAULT_TITLE).tag(sync=True)
    #: One description per row: name, label, kind, choices, min, max, step.
    fields = traitlets.List([]).tag(sync=True)
    #: The settings as their controls hold them.
    config = traitlets.Dict({}).tag(sync=True)
    #: The reason the last edit was refused, or "".
    error = traitlets.Unicode("").tag(sync=True)
    #: Set by the front end: ``{"field", "value", "serial"}``.
    edit = traitlets.Dict({}).tag(sync=True)
    #: Read by the shared front end for a level row; a section has none.
    coarsest_text = traitlets.Unicode("").tag(sync=True)
    #: Whether the front end hides the whole control (no scene is 2D).
    hidden = traitlets.Bool(False).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        section: Mapping[str, Any],
        displayed_dimensions: Mapping[UUID, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            fields=[field._asdict() for field in mesh_section_fields()], **kwargs
        )
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._editor = MeshSectionEditor(
            self._visual_ids, section, self._id, self.changed.emit, self._show
        )
        self._show(dict(section), "")
        self._visibility = SectionVisibility(displayed_dimensions, self._set_shown)
        self.hidden = not self._visibility.visible
        self.observe(self._on_edit, names="edit")

    # -- Public interface ------------------------------------------------------

    @property
    def widget(self) -> AnywidgetMeshSectionControls:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    def close(self) -> None:
        """Unsubscribe from the bus and release the widget."""
        self.closed.emit()
        close_aux_widgets(self)
        super().close()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``MeshSectionChangedEvent`` per visual, one dims per scene."""
        return [
            *self._editor.subscription_specs(),
            *self._visibility.subscription_specs(),
        ]

    # -- widget -> model -------------------------------------------------------

    def _on_edit(self, change) -> None:
        edit = change["new"] or {}
        if "field" in edit:
            self._editor.edit(edit["field"], edit.get("value"))

    # -- model -> widget -------------------------------------------------------

    def _set_shown(self, shown: bool) -> None:
        self.hidden = not shown

    def _show(self, config: dict[str, Any], error: str) -> None:
        with self.hold_sync():
            self.config = dict(config)
            self.error = error
