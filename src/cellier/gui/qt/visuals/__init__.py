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
from cellier.gui.qt.visuals._chrome import labelled_row, titled_group
from cellier.gui.qt.visuals._color import QtUniformColorPicker
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
from cellier.gui.qt.visuals._image_controls import QtImageControls
from cellier.gui.qt.visuals._labels import (
    QtBackgroundLabelSpin,
    QtLabelsRenderModeCombo,
    QtSaltSpin,
)
from cellier.gui.qt.visuals._lines import (
    QtThicknessSpaceCombo,
    QtThicknessSpin,
)
from cellier.gui.qt.visuals._loading import QtLoadingIndicator
from cellier.gui.qt.visuals._loading_config import QtLoadingConfigControls
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
from cellier.gui.qt.visuals._trail import QtTrailControls
from cellier.gui.qt.visuals._visible import QtVisibleToggle

__all__ = [
    "QtAABBWidget",
    "QtAppearanceField",
    "QtBackgroundLabelSpin",
    "QtBoundedSlider",
    "QtChoice",
    "QtColorPicker",
    "QtEdgeColorPicker",
    "QtEdgeThicknessSpaceCombo",
    "QtEdgeThicknessSpin",
    "QtEdgeVisibleToggle",
    "QtFlatShadingToggle",
    "QtFloatSpin",
    "QtImageControls",
    "QtIntSpin",
    "QtLabelsRenderModeCombo",
    "QtLoadingConfigControls",
    "QtLoadingIndicator",
    "QtLodBiasSlider",
    "QtNodeColorPicker",
    "QtNodeSizeSpaceCombo",
    "QtNodeSizeSpin",
    "QtNodeVisibleToggle",
    "QtOpacitySlider",
    "QtSaltSpin",
    "QtShininessSpin",
    "QtSideCombo",
    "QtSizeSpaceCombo",
    "QtSizeSpin",
    "QtThicknessSpaceCombo",
    "QtThicknessSpin",
    "QtToggle",
    "QtTrailControls",
    "QtUniformColorPicker",
    "QtVisibleToggle",
    "QtWireframeThicknessSpin",
    "QtWireframeToggle",
    "labelled_row",
    "titled_group",
]
