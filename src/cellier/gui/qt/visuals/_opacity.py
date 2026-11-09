"""Opacity control for any visual (Qt).

Layer 3 of the appearance widgets' three layers (layer 1 talks to the bus,
layer 2 is one class per control type, layer 3 binds a field and a label):
each class is a field name, a label, and the per-field defaults.  All
behaviour -- the bus contract, the echo filter, the fan-out over an
``OrthoViewer``'s panel group -- lives in the layer-1 base, and the control
itself in the layer-2 type.
"""

from __future__ import annotations

from typing import ClassVar

from cellier.gui._appearance_fields import field_bounds
from cellier.gui.qt.visuals._base import (
    QtBoundedSlider,
)
from cellier.visuals._base_visual import BaseAppearance

_OPACITY_RANGE = field_bounds(BaseAppearance, "opacity") or (0.0, 1.0)
"""Read off the model, so the slider cannot offer a value pydantic rejects."""


class QtOpacitySlider(QtBoundedSlider):
    """Master opacity multiplier.

    Applies to every visual type -- ``opacity`` is on
    ``BaseAppearance``.  It is also the **only** appearance field
    carrying ``ge``/``le``, which is why it is the only
    ``BoundedSlider``: the range is read off the model rather than
    picked.

    Parameters
    ----------
    visual_id :
        UUID of the visual, or a sequence of UUIDs to drive as one
        group.
    initial_value :
        Starting value -- typically ``visual.appearance.opacity``.
    """

    _field: ClassVar[str] = "opacity"
    _label: ClassVar[str] = "Opacity"
    _default_value: ClassVar[float] = 1.0
    _default_range: ClassVar[tuple[float, float]] = _OPACITY_RANGE
