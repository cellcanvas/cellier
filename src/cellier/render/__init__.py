"""Components for the rendering backend."""

from cellier.render._config import (
    CameraConfig,
    OutlineConfig,
    OutlineLayerConfig,
    RenderManagerConfig,
    SlicingConfig,
    TemporalAccumulationConfig,
)
from cellier.render._outline import OutlinePass
from cellier.render._outline_lut import OutlineLut
from cellier.render._requests import DimsState, ReslicingRequest
from cellier.render._scene_config import VisualRenderConfig
from cellier.render._temporal_accumulation import TemporalAccumulationPass
from cellier.render.canvas_view import CanvasView
from cellier.render.render_manager import RenderManager
from cellier.render.scene_manager import SceneManager
from cellier.render.slice_coordinator import SliceCoordinator

__all__ = [
    "CameraConfig",
    "CanvasView",
    "DimsState",
    "OutlineConfig",
    "OutlineLayerConfig",
    "OutlineLut",
    "OutlinePass",
    "RenderManager",
    "RenderManagerConfig",
    "ReslicingRequest",
    "SceneManager",
    "SliceCoordinator",
    "SlicingConfig",
    "TemporalAccumulationConfig",
    "TemporalAccumulationPass",
    "VisualRenderConfig",
]
