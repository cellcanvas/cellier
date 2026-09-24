"""Visuals for representing multiscale label data."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cellier.transform import AffineTransform
from cellier.visuals._label_memory import BaseLabelsAppearance, BaseLabelsVisual
from cellier.visuals._loading import ProgressiveLoadingConfig


class MultiscaleLabelsAppearance(BaseLabelsAppearance):
    """Appearance for multiscale label visuals.

    Extends ``BaseLabelsAppearance`` with LOD-control fields required for
    brick-streamed multiscale rendering.

    Parameters
    ----------
    colormap_mode : "random" | "direct"
        Inherited from ``BaseLabelsAppearance``. Frozen.
    background_label : int
        Inherited from ``BaseLabelsAppearance``.
    salt : int
        Inherited from ``BaseLabelsAppearance``.
    color_dict : dict
        Inherited from ``BaseLabelsAppearance``.
    render_mode : str
        Overrides ``BaseLabelsAppearance`` to widen the Literal to include
        ``"gradient_debug"`` and ``"smooth_iso"``. Not frozen — live-mutable.
    lod_bias : float
        Multiplier on the screen-space LOD threshold. Default 1.0.
    force_level : int | None
        Overrides automatic LOD selection when set. Default None.
    frustum_cull : bool
        Skip bricks outside the camera frustum. Default True.
    ray_steps_per_voxel : float
        3D ray-march samples per voxel of the drawn level, measured along
        the ray.  Lower is faster but can skip labels a voxel or two
        thick.  Must be in [0.5, 8]: below 0.5 a bisection probe can
        overshoot the brick's ghost border.  Default 1.0.
    """

    render_mode: Literal[
        "iso_categorical", "flat_categorical", "gradient_debug", "smooth_iso"
    ] = "iso_categorical"
    lod_bias: float = 1.0
    force_level: int | None = None
    frustum_cull: bool = True
    ray_steps_per_voxel: float = Field(default=1.0, ge=0.5, le=8.0)


class MultiscaleLabelRenderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    """Render-layer configuration for a multiscale label visual.

    Parameters
    ----------
    block_size : int
        Brick / tile side length in voxels. Default 32.
    gpu_budget_bytes : int
        Maximum GPU memory for the 3-D brick cache. Default 1 GiB.
    gpu_budget_bytes_2d : int
        Maximum GPU memory for the 2-D tile cache. Default 64 MiB.
    paint_max_tiles : int
        Tiles the 2-D paint overlay can hold. Default 512.
    loading : ProgressiveLoadingConfig
        The coarse backstop loaded ahead of the target level.  Changing
        only this reslices; the other fields reallocate GPU resources.
    """

    block_size: int = 32
    gpu_budget_bytes: int = 1 * 1024**3
    gpu_budget_bytes_2d: int = 64 * 1024**2
    paint_max_tiles: int = 512
    loading: ProgressiveLoadingConfig = Field(default_factory=ProgressiveLoadingConfig)


class MultiscaleLabelVisual(BaseLabelsVisual):
    """Model for a multiscale label visual.

    Parameters
    ----------
    level_transforms : list[AffineTransform]
        Per-level transforms mapping level-k voxel coords to level-0.
    appearance : MultiscaleLabelsAppearance
        Visual appearance configuration.
    render_config : MultiscaleLabelRenderConfig
        GPU resource configuration.
    requires_camera_reslice : bool
        Always True (camera movement triggers reslice for LOD). Frozen.
    """

    visual_type: Literal["multiscale_label"] = "multiscale_label"
    level_transforms: list[AffineTransform]
    appearance: MultiscaleLabelsAppearance
    render_config: MultiscaleLabelRenderConfig = Field(
        default_factory=MultiscaleLabelRenderConfig
    )
    requires_camera_reslice: bool = Field(default=True, frozen=True)
