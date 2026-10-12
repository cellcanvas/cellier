"""Convenience canvas/grid builders for the cellier Viewer (Qt + anywidget)."""

from cellier.convenience.gui._canvas import (
    AnywidgetCanvasView,
    build_canvas_view,
    build_canvas_widget,
)
from cellier.convenience.gui._controls_config import (
    AmbientOcclusionControl,
    BoundingBoxControl,
    ClippingPlanesControl,
    Control,
    DatasetInfoControl,
    FieldControls,
    ImageControl,
    LevelOfDetailControl,
    LoadingConfigControl,
    LoadingIndicator,
    MeshLodControl,
    MeshSectionControl,
    OutlineControl,
    PickingControl,
    RenderPlanesControl,
    TrailControl,
)
from cellier.convenience.gui._ortho import (
    PANEL_LAYOUT,
    OrthoAnywidgetCanvases,
    OrthoCanvasGrid,
    OrthoCanvasWidgets,
    build_ortho_grid_widget,
)

__all__ = [
    "PANEL_LAYOUT",
    "AmbientOcclusionControl",
    "AnywidgetCanvasView",
    "BoundingBoxControl",
    "ClippingPlanesControl",
    "Control",
    "DatasetInfoControl",
    "FieldControls",
    "ImageControl",
    "LevelOfDetailControl",
    "LoadingConfigControl",
    "LoadingIndicator",
    "MeshLodControl",
    "MeshSectionControl",
    "OrthoAnywidgetCanvases",
    "OrthoCanvasGrid",
    "OrthoCanvasWidgets",
    "OutlineControl",
    "PickingControl",
    "RenderPlanesControl",
    "TrailControl",
    "build_canvas_view",
    "build_canvas_widget",
    "build_ortho_grid_widget",
]
