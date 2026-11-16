"""Components for the rendering backend."""

from cellier.render._config import (
    AmbientOcclusionConfig,
    CameraConfig,
    OutlineConfig,
    OutlineLayerConfig,
    RenderManagerConfig,
    SchedulerConfig,
    TemporalAccumulationConfig,
)
from cellier.render._requests import DimsState, ReslicingRequest
from cellier.render._scene_config import VisualRenderConfig
from cellier.render._temporal_accumulation import TemporalAccumulationPass
from cellier.render.canvas_view import CanvasView
from cellier.render.render_manager import RenderManager
from cellier.render.scene_manager import SceneManager
from cellier.render.slice_coordinator import SliceCoordinator

__all__ = [
    "AmbientOcclusionConfig",
    "CameraConfig",
    "CanvasView",
    "DimsState",
    "OutlineConfig",
    "OutlineLayerConfig",
    "RenderManager",
    "RenderManagerConfig",
    "ReslicingRequest",
    "SceneManager",
    "SchedulerConfig",
    "SliceCoordinator",
    "TemporalAccumulationConfig",
    "TemporalAccumulationPass",
    "VisualRenderConfig",
]
