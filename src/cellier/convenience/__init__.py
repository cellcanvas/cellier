"""Utilities to make it easier to construct Cellier viewers."""

from cellier.convenience._geometry import (
    axis_ranges_from_ortho,
    axis_ranges_from_viewer,
)
from cellier.convenience._launch import DisplayHandle, display, launch, run, show
from cellier.convenience._ortho_viewer import OrthoViewer
from cellier.convenience._sidecar import SidecarOptions
from cellier.convenience._startup import StartupState
from cellier.convenience._viewer import Viewer
from cellier.convenience._window_capture import screenshot_window
from cellier.convenience.gui._controls_config import (
    BaseControlsConfig,
    ChannelControlsConfig,
    GraphControlsConfig,
    InMemoryImageControlsConfig,
    LabelsControlsConfig,
    LinesControlsConfig,
    MeshControlsConfig,
    MultiscaleImageControlsConfig,
    MultiscaleLabelsControlsConfig,
    PointsControlsConfig,
)
from cellier.convenience.layout import (
    AppearanceControls,
    ChannelControls,
    Grid,
    HStack,
    Layout,
    RenderControls,
    VStack,
)

__all__ = [
    "AppearanceControls",
    "BaseControlsConfig",
    "ChannelControls",
    "ChannelControlsConfig",
    "DisplayHandle",
    "GraphControlsConfig",
    "Grid",
    "HStack",
    "InMemoryImageControlsConfig",
    "LabelsControlsConfig",
    "Layout",
    "LinesControlsConfig",
    "MeshControlsConfig",
    "MultiscaleImageControlsConfig",
    "MultiscaleLabelsControlsConfig",
    "OrthoViewer",
    "PointsControlsConfig",
    "RenderControls",
    "SidecarOptions",
    "StartupState",
    "VStack",
    "Viewer",
    "axis_ranges_from_ortho",
    "axis_ranges_from_viewer",
    "display",
    "launch",
    "run",
    "screenshot_window",
    "show",
]
