"""Uniform-colour control, shared by mesh, points and lines (anywidget).

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
    AnywidgetColorPicker,
)


class AnywidgetUniformColorPicker(AnywidgetColorPicker):
    """Uniform RGBA colour.

    One class for mesh, points and lines: they spell the field
    ``color`` alike, so the control is shared rather than
    triplicated.  Only used when the visual's ``color_mode`` is
    ``"uniform"``; per-vertex colours come from the data store.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.color``.
    """

    _field: ClassVar[str] = "color"
    _label: ClassVar[str] = "Color"
    _default_value: ClassVar[tuple[float, float, float, float]] = (1.0, 1.0, 1.0, 1.0)
