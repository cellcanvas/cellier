"""Visuals for representing multiscale label data."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cellier.transform import AffineTransform
from cellier.visuals._label_memory import BaseLabelsAppearance, BaseLabelsVisual
from cellier.visuals._loading import ProgressiveLoadingConfig
from cellier.visuals._removed import (
    REMOVED_APPEARANCE_FIELDS,
    refuse_removed_fields,
)


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
        The modes of ``BaseLabelsAppearance``, plus ``"gradient_debug"``,
        which draws the surface normal as RGB to diagnose shading across
        brick boundaries.  Not frozen -- live-mutable.  In
        ``"smooth_geometry"`` the surface is found exactly rather than by
        ray steps, so ``ray_steps_per_voxel`` has no effect on it.
    settled_lod_bias : float
        Level-of-detail bias of a settled view, 2D and 3D.  Divisor on the
        screen-space LOD threshold: higher is coarser.  Default 1.0.
    coarsest_while_moving_3d : bool
        In a 3D view, plan no target while the visual moves.  A visual moves
        while its scene's dims are scrubbed, while its clipping planes are
        dragged (``CellierController.plane_interaction``), or while its
        transform is dragged (``CellierController.transform_interaction``).
        A dims tick then loads the new slice's coarse backstop only; a plane
        or transform drag plans nothing and keeps the last plan, whose
        bricks move with the transform.  The target is planned once every
        motion has ended.  ``False`` plans in full on every tick and nothing
        more at the end.  Default ``True``.
    coarsest_while_moving_2d : bool
        The same, for a 2D view: each tick of a dims scrub, of a transform
        drag (inside ``transform_interaction``) or of a thickness drag
        (inside ``dims_interaction``, or sent with ``update_thickness(...,
        interactive=True)``) loads only the new slice's coarse backstop.
        This includes a tilted 2D view drawn from 3D bricks (the brick
        route).  ``True`` saves most of a drag's reads, is the setting for a
        slow store, and is what keeps a tilted view at 30 fps at native
        zoom.  Default ``False``.
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

    @model_validator(mode="before")
    @classmethod
    def _refuse_removed_fields(cls, data: Any) -> Any:
        return refuse_removed_fields(data, REMOVED_APPEARANCE_FIELDS, cls.__name__)

    render_mode: Literal[
        "iso_categorical",
        "flat_categorical",
        "gradient_debug",
        "smooth_iso",
        "smooth_geometry",
        "plane",
    ] = "iso_categorical"
    settled_lod_bias: float = 1.0
    coarsest_while_moving_3d: bool = True
    coarsest_while_moving_2d: bool = False
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
        The tiles share one near-square texture, so
        ``ceil(sqrt(paint_max_tiles)) * block_size`` must not exceed the
        device's maximum 2-D texture dimension.
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

    def plans_coarse_while_moving(self, n_displayed_dims: int) -> bool:
        """The moving setting of the view being planned.

        ``appearance.coarsest_while_moving_3d`` for a 3D view,
        ``appearance.coarsest_while_moving_2d`` for a 2D one.
        """
        if n_displayed_dims == 3:
            return self.appearance.coarsest_while_moving_3d
        return self.appearance.coarsest_while_moving_2d
