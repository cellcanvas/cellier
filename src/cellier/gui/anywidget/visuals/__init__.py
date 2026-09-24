"""Cellier v2 anywidget GUI widgets for visual (appearance) controls.

Control-type bases (layer 2) live in ``_base``; the field-specific
classes (layer 3) are grouped by the visual family they apply to, with
``_visible``, ``_opacity`` and ``_color`` shared across families.
"""

from cellier.gui.anywidget.visuals._aabb import AnywidgetAABBWidget
from cellier.gui.anywidget.visuals._base import (
    AnywidgetAppearanceField,
    AnywidgetBoundedSlider,
    AnywidgetChoice,
    AnywidgetColorPicker,
    AnywidgetFloatSpin,
    AnywidgetIntSpin,
    AnywidgetToggle,
)
from cellier.gui.anywidget.visuals._color import AnywidgetUniformColorPicker
from cellier.gui.anywidget.visuals._graph import (
    AnywidgetEdgeColorPicker,
    AnywidgetEdgeThicknessSpaceCombo,
    AnywidgetEdgeThicknessSpin,
    AnywidgetEdgeVisibleToggle,
    AnywidgetNodeColorPicker,
    AnywidgetNodeSizeSpaceCombo,
    AnywidgetNodeSizeSpin,
    AnywidgetNodeVisibleToggle,
)
from cellier.gui.anywidget.visuals._image_controls import AnywidgetImageControls
from cellier.gui.anywidget.visuals._labels import (
    AnywidgetBackgroundLabelSpin,
    AnywidgetLabelsRenderModeCombo,
    AnywidgetSaltSpin,
)
from cellier.gui.anywidget.visuals._lines import (
    AnywidgetThicknessSpaceCombo,
    AnywidgetThicknessSpin,
)
from cellier.gui.anywidget.visuals._loading import AnywidgetLoadingIndicator
from cellier.gui.anywidget.visuals._loading_config import AnywidgetLoadingConfigControls
from cellier.gui.anywidget.visuals._lod_bias import AnywidgetLodBiasSlider
from cellier.gui.anywidget.visuals._mesh import (
    AnywidgetFlatShadingToggle,
    AnywidgetShininessSpin,
    AnywidgetSideCombo,
    AnywidgetWireframeThicknessSpin,
    AnywidgetWireframeToggle,
)
from cellier.gui.anywidget.visuals._opacity import AnywidgetOpacitySlider
from cellier.gui.anywidget.visuals._points import (
    AnywidgetSizeSpaceCombo,
    AnywidgetSizeSpin,
)
from cellier.gui.anywidget.visuals._trail import AnywidgetTrailControls
from cellier.gui.anywidget.visuals._visible import AnywidgetVisibleToggle

__all__ = [
    "AnywidgetAABBWidget",
    "AnywidgetAppearanceField",
    "AnywidgetBackgroundLabelSpin",
    "AnywidgetBoundedSlider",
    "AnywidgetChoice",
    "AnywidgetColorPicker",
    "AnywidgetEdgeColorPicker",
    "AnywidgetEdgeThicknessSpaceCombo",
    "AnywidgetEdgeThicknessSpin",
    "AnywidgetEdgeVisibleToggle",
    "AnywidgetFlatShadingToggle",
    "AnywidgetFloatSpin",
    "AnywidgetImageControls",
    "AnywidgetIntSpin",
    "AnywidgetLabelsRenderModeCombo",
    "AnywidgetLoadingConfigControls",
    "AnywidgetLoadingIndicator",
    "AnywidgetLodBiasSlider",
    "AnywidgetNodeColorPicker",
    "AnywidgetNodeSizeSpaceCombo",
    "AnywidgetNodeSizeSpin",
    "AnywidgetNodeVisibleToggle",
    "AnywidgetOpacitySlider",
    "AnywidgetSaltSpin",
    "AnywidgetShininessSpin",
    "AnywidgetSideCombo",
    "AnywidgetSizeSpaceCombo",
    "AnywidgetSizeSpin",
    "AnywidgetThicknessSpaceCombo",
    "AnywidgetThicknessSpin",
    "AnywidgetToggle",
    "AnywidgetTrailControls",
    "AnywidgetUniformColorPicker",
    "AnywidgetVisibleToggle",
    "AnywidgetWireframeThicknessSpin",
    "AnywidgetWireframeToggle",
]
