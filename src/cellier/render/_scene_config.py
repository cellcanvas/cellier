"""Per-visual render configuration for the RenderManager pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellier.render.scheduling._types import PlanMode
from cellier.visuals._loading import ProgressiveLoadingConfig


@dataclass
class VisualRenderConfig:
    """Mutable render settings for one visual.

    Passed through the call stack at reslice time.  When a visual's ID is
    absent from the ``visual_configs`` dict supplied to
    ``SceneManager.build_slice_requests``, a default instance is used.

    Parameters
    ----------
    settled_lod_bias : float
        Level-of-detail bias of a settled plan: a divisor on the LOD
        thresholds, so values greater than 1.0 favour coarser levels and
        values less than 1.0 finer ones.  Default ``1.0`` (no bias).
    force_level : int or None
        When set, all bricks are assigned this 1-based LOD level,
        bypassing distance-based selection entirely.  ``None`` restores
        automatic selection.  Default ``None``.
    frustum_cull : bool
        When ``True``, bricks outside the camera frustum are skipped.
        When ``False``, all bricks in the scene are submitted regardless
        of visibility.  Default ``True``.
    slicing_enabled : bool
        When ``False``, ``SceneManager`` plans no requests for the visual at
        all, so it keeps showing whatever it last loaded.  The controller
        turns this off for hidden image visuals.  Default ``True``.
    loading : ProgressiveLoadingConfig
        A multiscale visual's backstop settings (its
        ``render_config.loading``).  Ignored by every other visual.
    plan_mode : PlanMode
        What a multiscale visual plans this reslice: ``FULL`` (default), or
        ``BACKSTOP_ONLY`` for any reslice while the scene's dims are being
        scrubbed, when the visual opted in (its ``coarsest_while_moving``
        setting for the view, or a mesh's ``lod.dims_drag="coarse"``).
    """

    settled_lod_bias: float = 1.0
    force_level: int | None = None
    frustum_cull: bool = True
    slicing_enabled: bool = True
    loading: ProgressiveLoadingConfig = field(default_factory=ProgressiveLoadingConfig)
    plan_mode: PlanMode = PlanMode.FULL
