"""Multiscale image visual models (unified image design 3.1)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cellier.transform import AffineTransform
from cellier.visuals._image_memory import (
    BaseImageAppearance,
    BaseImageSingleAppearance,
    BaseImageVisual,
)
from cellier.visuals._loading import ProgressiveLoadingConfig


class MultiscaleImageAppearance(BaseImageAppearance):
    """Shared appearance of a multiscale image visual.

    Adds the fields both modes share on a brick-streamed visual.

    Parameters
    ----------
    attenuation : float
        Depth attenuation coefficient for ``"attenuated_mip"``.  Default 1.0.
    lod_bias : float
        Divisor on the screen-space LOD threshold: higher is coarser.
        Default 1.0.
    force_level : int or None
        Overrides automatic LOD selection when set.  Default None.
    frustum_cull : bool
        Skip bricks outside the camera frustum.  Default True.
    ray_steps_per_voxel : float
        3D ray-march samples per voxel of the drawn level, measured along
        the ray.  Lower is faster but can skip features a voxel or two
        thick and dims MIP peaks (a one-voxel spot can lose up to
        ``1 / (2 * ray_steps_per_voxel)`` of its value).  Must be in
        [0.5, 8]: below 0.5 a bisection probe can overshoot the brick's
        ghost border.  Default 1.0.
    """

    attenuation: float = 1.0
    lod_bias: float = 1.0
    force_level: int | None = None
    frustum_cull: bool = True
    ray_steps_per_voxel: float = Field(default=1.0, ge=0.5, le=8.0)


class MultiscaleImageSingleAppearance(BaseImageSingleAppearance):
    """Single-mode appearance of a multiscale image visual.

    Parameters
    ----------
    render_mode : str
        ``"iso"`` (default), ``"mip"``, ``"smooth_iso"`` or
        ``"attenuated_mip"``.
    iso_threshold : float
        Isosurface threshold.  Default 0.2.
    """

    render_mode: Literal["iso", "mip", "smooth_iso", "attenuated_mip"] = "iso"
    iso_threshold: float = 0.2


class MultiscaleImageChannelAppearance(MultiscaleImageSingleAppearance):
    """One channel's appearance on a multiscale image, in composite mode.

    Parameters
    ----------
    visible : bool
        Whether composite mode draws this channel.  Default ``True``.
    """

    visible: bool = True


class MultiscaleImageRenderConfig(BaseModel):
    """Render-layer configuration for a multiscale image visual.

    These parameters control GPU resource allocation.  They are stored in the
    model so they round-trip through serialization.

    Parameters
    ----------
    block_size : int
        Brick / tile side length in voxels. Default 32.
    gpu_budget_bytes : int
        Maximum GPU memory for the 3-D brick caches, split evenly between the
        channels the visual can draw -- its ``channels``, not the slot pool's
        ``max_channels``, so an unused slot takes no budget from a drawing
        one.  A visual with no channel axis, or none configured, gets all of
        it. Default 1 GiB.
    gpu_budget_bytes_2d : int
        Maximum GPU memory for the 2-D tile caches, split likewise.
        Default 64 MiB.
    loading : ProgressiveLoadingConfig
        The coarse backstop loaded ahead of the target level.  Changing
        only this reslices; the other fields reallocate GPU resources.
    """

    model_config = ConfigDict(frozen=True)

    block_size: int = 32
    gpu_budget_bytes: int = 1 * 1024**3
    gpu_budget_bytes_2d: int = 64 * 1024**2
    loading: ProgressiveLoadingConfig = Field(default_factory=ProgressiveLoadingConfig)


class MultiscaleImageVisual(BaseImageVisual):
    """Model for a multiscale image visual.

    Parameters
    ----------
    visual_type : Literal["multiscale_image"]
        Discriminator field. Always ``"multiscale_image"``.
    level_transforms : list[AffineTransform]
        Per-level transforms mapping level-k voxel coords to level-0 voxel
        coords.  Copied from the data store by the controller.
    appearance : MultiscaleImageAppearance
        Shared by both modes.
    single : MultiscaleImageSingleAppearance
        Single mode's appearance.
    channels : dict[int, MultiscaleImageChannelAppearance]
        Composite mode's per-channel appearances.
    render_config : MultiscaleImageRenderConfig
        GPU cache configuration.
    """

    visual_type: Literal["multiscale_image"] = "multiscale_image"
    level_transforms: list[AffineTransform]
    appearance: MultiscaleImageAppearance = Field(
        default_factory=MultiscaleImageAppearance
    )
    single: MultiscaleImageSingleAppearance = Field(
        default_factory=MultiscaleImageSingleAppearance
    )
    channels: dict[int, MultiscaleImageChannelAppearance] = Field(default_factory=dict)
    render_config: MultiscaleImageRenderConfig = Field(
        default_factory=MultiscaleImageRenderConfig
    )
    requires_camera_reslice: bool = Field(default=True, frozen=True)
