"""Cellier v2 GUI widgets for visual (appearance) controls.

Control-type bases (layer 2) live in ``_base``; the field-specific
classes (layer 3) are grouped by the visual family they apply to, with
``_visible``, ``_opacity`` and ``_color`` shared across families.
"""

from cellier.gui.qt.visuals._aabb import QtAABBWidget
from cellier.gui.qt.visuals._base import (
    QtAppearanceField,
    QtBoundedSlider,
    QtChoice,
    QtColorPicker,
    QtFloatSpin,
    QtIntSpin,
    QtToggle,
)
from cellier.gui.qt.visuals._channel import QtChannelList
from cellier.gui.qt.visuals._color import QtUniformColorPicker
from cellier.gui.qt.visuals._colormap import QtColormapCombo
from cellier.gui.qt.visuals._contrast_limits import QtClimRangeSlider
from cellier.gui.qt.visuals._graph import (
    QtEdgeColorPicker,
    QtEdgeThicknessSpaceCombo,
    QtEdgeThicknessSpin,
    QtEdgeVisibleToggle,
    QtNodeColorPicker,
    QtNodeSizeSpaceCombo,
    QtNodeSizeSpin,
    QtNodeVisibleToggle,
)
from cellier.gui.qt.visuals._image import (
    QtIsoThresholdSlider,
    QtRenderModeCombo,
    QtVolumeRenderControls,
)
from cellier.gui.qt.visuals._labels import (
    QtBackgroundLabelSpin,
    QtLabelsRenderModeCombo,
    QtSaltSpin,
)
from cellier.gui.qt.visuals._lines import (
    QtThicknessSpaceCombo,
    QtThicknessSpin,
)
from cellier.gui.qt.visuals._lod_bias import QtLodBiasSlider
from cellier.gui.qt.visuals._mesh import (
    QtFlatShadingToggle,
    QtShininessSpin,
    QtSideCombo,
    QtWireframeThicknessSpin,
    QtWireframeToggle,
)
from cellier.gui.qt.visuals._opacity import QtOpacitySlider
from cellier.gui.qt.visuals._points import (
    QtSizeSpaceCombo,
    QtSizeSpin,
)
from cellier.gui.qt.visuals._visible import QtVisibleToggle

__all__ = [
    "QtAABBWidget",
    "QtAppearanceField",
    "QtBackgroundLabelSpin",
    "QtBoundedSlider",
    "QtChannelList",
    "QtChoice",
    "QtClimRangeSlider",
    "QtColorPicker",
    "QtColormapCombo",
    "QtEdgeColorPicker",
    "QtEdgeThicknessSpaceCombo",
    "QtEdgeThicknessSpin",
    "QtEdgeVisibleToggle",
    "QtFlatShadingToggle",
    "QtFloatSpin",
    "QtIntSpin",
    "QtIsoThresholdSlider",
    "QtLabelsRenderModeCombo",
    "QtLodBiasSlider",
    "QtNodeColorPicker",
    "QtNodeSizeSpaceCombo",
    "QtNodeSizeSpin",
    "QtNodeVisibleToggle",
    "QtOpacitySlider",
    "QtRenderModeCombo",
    "QtSaltSpin",
    "QtShininessSpin",
    "QtSideCombo",
    "QtSizeSpaceCombo",
    "QtSizeSpin",
    "QtThicknessSpaceCombo",
    "QtThicknessSpin",
    "QtToggle",
    "QtUniformColorPicker",
    "QtVisibleToggle",
    "QtVolumeRenderControls",
    "QtWireframeThicknessSpin",
    "QtWireframeToggle",
]
