"""Utilities to make it easier to construct Cellier viewers."""

from cellier.convenience._geometry import (
    axis_values_from_ortho,
    axis_values_from_viewer,
)
from cellier.convenience._launch import DisplayHandle, display, launch, run, show
from cellier.convenience._ortho_viewer import OrthoViewer
from cellier.convenience._sidecar import SidecarOptions
from cellier.convenience._startup import StartupState
from cellier.convenience._viewer import Viewer
from cellier.convenience._window_capture import screenshot_window
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
from cellier.convenience.layout import (
    AppearanceControls,
    Grid,
    HStack,
    Layout,
    OrthoClippingControls,
    OverlayControls,
    RenderControls,
    VStack,
)
from cellier.gui._axis_values import ContinuousAxisValues, DiscreteAxisValues
from cellier.scene.dims import spatial_axes, world_coordinate_system

__all__ = [
    "AmbientOcclusionControl",
    "AppearanceControls",
    "BoundingBoxControl",
    "ClippingPlanesControl",
    "ContinuousAxisValues",
    "Control",
    "DatasetInfoControl",
    "DiscreteAxisValues",
    "DisplayHandle",
    "FieldControls",
    "Grid",
    "HStack",
    "ImageControl",
    "Layout",
    "LevelOfDetailControl",
    "LoadingConfigControl",
    "LoadingIndicator",
    "MeshLodControl",
    "MeshSectionControl",
    "OrthoClippingControls",
    "OrthoViewer",
    "OutlineControl",
    "OverlayControls",
    "PickingControl",
    "RenderControls",
    "RenderPlanesControl",
    "SidecarOptions",
    "StartupState",
    "TrailControl",
    "VStack",
    "Viewer",
    "axis_values_from_ortho",
    "axis_values_from_viewer",
    "display",
    "launch",
    "run",
    "screenshot_window",
    "show",
    "spatial_axes",
    "world_coordinate_system",
]
