"""Appearance controls specific to graph visuals (anywidget).

Layer 3 of the appearance widgets' three layers (layer 1 talks to the bus,
layer 2 is one class per control type, layer 3 binds a field and a label):
each class is a field name, a label, and the per-field defaults.  All
behaviour -- the bus contract, the echo filter, the fan-out over an
``OrthoViewer``'s panel group -- lives in the layer-1 base, and the control
itself in the layer-2 type.
"""

from __future__ import annotations

from typing import ClassVar

from cellier.gui.anywidget.visuals._base import (
    AnywidgetChoice,
    AnywidgetColorPicker,
    AnywidgetFloatSpin,
    AnywidgetToggle,
)


class AnywidgetNodeVisibleToggle(AnywidgetToggle):
    """Show the node sub-visual.  Nests under ``visible``.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.node_visible``.
    """

    _field: ClassVar[str] = "node_visible"
    _label: ClassVar[str] = "Nodes visible"
    _default_value: ClassVar[bool] = True


class AnywidgetNodeColorPicker(AnywidgetColorPicker):
    """Uniform RGBA node colour.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.node_color``.
    """

    _field: ClassVar[str] = "node_color"
    _label: ClassVar[str] = "Node color"
    _default_value: ClassVar[tuple[float, float, float, float]] = (1.0, 1.0, 1.0, 1.0)


class AnywidgetNodeSizeSpin(AnywidgetFloatSpin):
    """Uniform node size, in ``node_size_space`` units.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.node_size``.
    """

    _field: ClassVar[str] = "node_size"
    _label: ClassVar[str] = "Node size"
    _default_value: ClassVar[float] = 5.0
    _default_range: ClassVar[tuple[float, float]] = (0.1, 100.0)
    _default_step: ClassVar[float] = 0.5


class AnywidgetNodeSizeSpaceCombo(AnywidgetChoice):
    """Coordinate space node size is interpreted in.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.node_size_space``.
    """

    _field: ClassVar[str] = "node_size_space"
    _label: ClassVar[str] = "Node size space"
    _default_value: ClassVar[str] = "screen"
    _default_choices: ClassVar[tuple[str, ...]] = ("screen", "world")


class AnywidgetEdgeVisibleToggle(AnywidgetToggle):
    """Show the edge sub-visual.  Nests under ``visible``.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.edge_visible``.
    """

    _field: ClassVar[str] = "edge_visible"
    _label: ClassVar[str] = "Edges visible"
    _default_value: ClassVar[bool] = True


class AnywidgetEdgeColorPicker(AnywidgetColorPicker):
    """Uniform RGBA edge colour.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.edge_color``.
    """

    _field: ClassVar[str] = "edge_color"
    _label: ClassVar[str] = "Edge color"
    _default_value: ClassVar[tuple[float, float, float, float]] = (0.7, 0.7, 0.7, 1.0)


class AnywidgetEdgeThicknessSpin(AnywidgetFloatSpin):
    """Edge thickness, in ``edge_thickness_space`` units.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.edge_thickness``.
    """

    _field: ClassVar[str] = "edge_thickness"
    _label: ClassVar[str] = "Edge thickness"
    _default_value: ClassVar[float] = 2.0
    _default_range: ClassVar[tuple[float, float]] = (0.1, 50.0)
    _default_step: ClassVar[float] = 0.5


class AnywidgetEdgeThicknessSpaceCombo(AnywidgetChoice):
    """Coordinate space edge thickness is interpreted in.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.edge_thickness_space``.
    """

    _field: ClassVar[str] = "edge_thickness_space"
    _label: ClassVar[str] = "Edge thickness space"
    _default_value: ClassVar[str] = "screen"
    _default_choices: ClassVar[tuple[str, ...]] = ("screen", "world")
