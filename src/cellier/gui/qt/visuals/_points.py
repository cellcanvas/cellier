"""Appearance controls specific to points visuals (Qt).

Layer 3 of the appearance widgets' three layers (layer 1 talks to the bus,
layer 2 is one class per control type, layer 3 binds a field and a label):
each class is a field name, a label, and the per-field defaults.  All
behaviour -- the bus contract, the echo filter, the fan-out over an
``OrthoViewer``'s panel group -- lives in the layer-1 base, and the control
itself in the layer-2 type.
"""

from __future__ import annotations

from typing import ClassVar

from cellier.gui.qt.visuals._base import (
    QtChoice,
    QtFloatSpin,
)


class QtSizeSpin(QtFloatSpin):
    """Uniform point size, in ``size_space`` units.

    The default range is in screen pixels.  With
    ``size_space="world"`` the useful range depends on the data
    extent, and ``value_range=`` is the escape hatch (design
    section 6.5.1 proposal 3).

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.size``.
    """

    _field: ClassVar[str] = "size"
    _label: ClassVar[str] = "Size"
    _default_value: ClassVar[float] = 5.0
    _default_range: ClassVar[tuple[float, float]] = (0.1, 100.0)
    _default_step: ClassVar[float] = 0.5


class QtSizeSpaceCombo(QtChoice):
    """Coordinate space point size is interpreted in.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.size_space``.
    """

    _field: ClassVar[str] = "size_space"
    _label: ClassVar[str] = "Size space"
    _default_value: ClassVar[str] = "screen"
    _default_choices: ClassVar[tuple[str, ...]] = ("screen", "world")
