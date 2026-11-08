"""Appearance controls specific to lines visuals (anywidget).

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
    AnywidgetFloatSpin,
)


class AnywidgetThicknessSpin(AnywidgetFloatSpin):
    """Line thickness, in ``thickness_space`` units.

    Screen pixels by default; see ``*SizeSpin`` for the ``world``
    case.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.thickness``.
    """

    _field: ClassVar[str] = "thickness"
    _label: ClassVar[str] = "Thickness"
    _default_value: ClassVar[float] = 2.0
    _default_range: ClassVar[tuple[float, float]] = (0.1, 50.0)
    _default_step: ClassVar[float] = 0.5


class AnywidgetThicknessSpaceCombo(AnywidgetChoice):
    """Coordinate space line thickness is interpreted in.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.thickness_space``.
    """

    _field: ClassVar[str] = "thickness_space"
    _label: ClassVar[str] = "Thickness space"
    _default_value: ClassVar[str] = "screen"
    _default_choices: ClassVar[tuple[str, ...]] = ("screen", "world")
