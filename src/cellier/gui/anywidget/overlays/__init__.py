"""Overlay controls (anywidget).

Layer 3 for overlays: each class is the overlay mixin, a layer-2 control
type from ``cellier.gui.anywidget.visuals._base``, a dotted field path and a label.  The
mixin swaps the bus contract -- ``OverlayChangedEvent`` in,
``OverlayUpdateEvent`` out, keyed by overlay id -- and everything else is the
appearance controls' own behaviour (``cellier.gui._overlay_fields``).

Wire to the controller after construction::

    toggle = AnywidgetOverlayVisibleToggle(overlay.id, initial_value=overlay.visible)
    controller.connect_widget(toggle, subscription_specs=toggle.subscription_specs())
"""

from __future__ import annotations

from typing import ClassVar

from cellier.gui._overlay_fields import OverlayFieldMixin
from cellier.gui.anywidget.visuals._base import (
    AnywidgetChoice,
    AnywidgetColorPicker,
    AnywidgetFloatSpin,
    AnywidgetToggle,
)


class AnywidgetOverlayVisibleToggle(OverlayFieldMixin, AnywidgetToggle):
    """Whether the overlay is drawn.

    Parameters
    ----------
    visual_id :
        UUID of the overlay, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically read off the overlay model.
    """

    _field: ClassVar[str] = "visible"
    _label: ClassVar[str] = "Visible"
    _default_value: ClassVar[bool] = True


class AnywidgetOverlayColorPicker(OverlayFieldMixin, AnywidgetColorPicker):
    """Line colour of a :class:`~cellier.visuals.SceneBoundingBox`.

    Parameters
    ----------
    visual_id :
        UUID of the overlay, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically read off the overlay model.
    """

    _field: ClassVar[str] = "appearance.color"
    _label: ClassVar[str] = "Color"
    _default_value: ClassVar[tuple[float, float, float, float]] = (0.5, 0.5, 0.5, 1.0)


class AnywidgetOverlayThicknessSpin(OverlayFieldMixin, AnywidgetFloatSpin):
    """Line thickness of a :class:`~cellier.visuals.SceneBoundingBox`, in screen pixels.

    Parameters
    ----------
    visual_id :
        UUID of the overlay, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically read off the overlay model.
    """

    _field: ClassVar[str] = "appearance.thickness"
    _label: ClassVar[str] = "Thickness"
    _default_value: ClassVar[float] = 1.5
    _default_range: ClassVar[tuple[float, float]] = (0.1, 20.0)
    _default_step: ClassVar[float] = 0.5


class AnywidgetOverlayCornerCombo(OverlayFieldMixin, AnywidgetChoice):
    """Canvas corner a :class:`~cellier.visuals.CenteredAxes2D` is anchored to.

    Parameters
    ----------
    visual_id :
        UUID of the overlay, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically read off the overlay model.
    """

    _field: ClassVar[str] = "appearance.corner"
    _label: ClassVar[str] = "Corner"
    _default_value: ClassVar[str] = "center"
    _default_choices: ClassVar[tuple[str, ...]] = (
        "bottom_left",
        "bottom_right",
        "top_left",
        "top_right",
        "center",
    )


class AnywidgetOverlayLengthSpin(OverlayFieldMixin, AnywidgetFloatSpin):
    """Axis segment length of a :class:`~cellier.visuals.CenteredAxes2D`, in screen pixels.

    Parameters
    ----------
    visual_id :
        UUID of the overlay, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically read off the overlay model.
    """

    _field: ClassVar[str] = "appearance.length_px"
    _label: ClassVar[str] = "Length"
    _default_value: ClassVar[float] = 60.0
    _default_range: ClassVar[tuple[float, float]] = (5.0, 500.0)
    _default_step: ClassVar[float] = 5.0


class AnywidgetOverlayLineThicknessSpin(OverlayFieldMixin, AnywidgetFloatSpin):
    """Axis line thickness of a :class:`~cellier.visuals.CenteredAxes2D`, in screen pixels.

    Parameters
    ----------
    visual_id :
        UUID of the overlay, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically read off the overlay model.
    """

    _field: ClassVar[str] = "appearance.line_thickness_px"
    _label: ClassVar[str] = "Thickness"
    _default_value: ClassVar[float] = 2.0
    _default_range: ClassVar[tuple[float, float]] = (0.5, 20.0)
    _default_step: ClassVar[float] = 0.5


class AnywidgetOverlayAxisAColorPicker(OverlayFieldMixin, AnywidgetColorPicker):
    """Colour of a :class:`~cellier.visuals.CenteredAxes2D`'s first axis.

    Parameters
    ----------
    visual_id :
        UUID of the overlay, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically read off the overlay model.
    """

    _field: ClassVar[str] = "appearance.axis_a_color"
    _label: ClassVar[str] = "Axis A color"
    _default_value: ClassVar[tuple[float, float, float, float]] = (
        0.23,
        0.67,
        0.23,
        1.0,
    )


class AnywidgetOverlayAxisBColorPicker(OverlayFieldMixin, AnywidgetColorPicker):
    """Colour of a :class:`~cellier.visuals.CenteredAxes2D`'s second axis.

    Parameters
    ----------
    visual_id :
        UUID of the overlay, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically read off the overlay model.
    """

    _field: ClassVar[str] = "appearance.axis_b_color"
    _label: ClassVar[str] = "Axis B color"
    _default_value: ClassVar[tuple[float, float, float, float]] = (
        0.80,
        0.40,
        0.00,
        1.0,
    )


class AnywidgetOverlayShowLabelsToggle(OverlayFieldMixin, AnywidgetToggle):
    """Whether a :class:`~cellier.visuals.CenteredAxes2D` draws its axis labels.

    Parameters
    ----------
    visual_id :
        UUID of the overlay, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically read off the overlay model.
    """

    _field: ClassVar[str] = "appearance.show_labels"
    _label: ClassVar[str] = "Labels"
    _default_value: ClassVar[bool] = True


class AnywidgetOverlayLabelColorPicker(OverlayFieldMixin, AnywidgetColorPicker):
    """Text colour of a :class:`~cellier.visuals.CenteredAxes2D`'s labels.

    Parameters
    ----------
    visual_id :
        UUID of the overlay, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically read off the overlay model.
    """

    _field: ClassVar[str] = "appearance.label_color"
    _label: ClassVar[str] = "Label color"
    _default_value: ClassVar[tuple[float, float, float, float]] = (1.0, 1.0, 1.0, 1.0)


class AnywidgetOverlayFontSizeSpin(OverlayFieldMixin, AnywidgetFloatSpin):
    """Label font size of a :class:`~cellier.visuals.CenteredAxes2D`, in screen pixels.

    Parameters
    ----------
    visual_id :
        UUID of the overlay, or a sequence of UUIDs to drive as one group.
    initial_value :
        Starting value -- typically read off the overlay model.
    """

    _field: ClassVar[str] = "appearance.font_size_px"
    _label: ClassVar[str] = "Font size"
    _default_value: ClassVar[float] = 12.0
    _default_range: ClassVar[tuple[float, float]] = (4.0, 72.0)
    _default_step: ClassVar[float] = 1.0


__all__ = [
    "AnywidgetOverlayAxisAColorPicker",
    "AnywidgetOverlayAxisBColorPicker",
    "AnywidgetOverlayColorPicker",
    "AnywidgetOverlayCornerCombo",
    "AnywidgetOverlayFontSizeSpin",
    "AnywidgetOverlayLabelColorPicker",
    "AnywidgetOverlayLengthSpin",
    "AnywidgetOverlayLineThicknessSpin",
    "AnywidgetOverlayShowLabelsToggle",
    "AnywidgetOverlayThicknessSpin",
    "AnywidgetOverlayVisibleToggle",
]
