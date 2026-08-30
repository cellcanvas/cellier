"""Rendering performance configuration models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

DEFAULT_CAMERA_SETTLE_THRESHOLD_S: float = 0.3


class SlicingConfig(BaseModel):
    """Configuration for the async chunk-slicing pipeline.

    These parameters are construction-time only. Changing them after
    ``RenderManager`` is created has no effect.

    Parameters
    ----------
    batch_size : int
        Number of chunks fetched concurrently in each async batch.
        Higher values increase throughput but raise peak memory pressure.
    render_every : int
        Number of completed batches between progressive redraws.
        1 = redraw after every batch (lowest latency to first pixels);
        higher values reduce GPU upload overhead on fast I/O.
    """

    batch_size: int = Field(default=8, gt=0)
    render_every: int = Field(default=1, gt=0)


class TemporalAccumulationConfig(BaseModel):
    """Configuration for the temporal accumulation post-processing pass.

    Parameters
    ----------
    enabled : bool
        When ``False`` the pass is bypassed entirely and each frame is
        shown raw. Useful for debugging or when jitter is disabled.
    alpha : float
        Minimum EMA blend weight for the current frame. During warm-up
        the weight is ``1 / (frame_count + 1)``; once that falls below
        ``alpha`` the weight clamps to ``alpha``. Lower values give
        smoother steady-state but slower convergence after a camera
        move. Must be in ``(0, 1]``.
    """

    enabled: bool = True
    alpha: float = Field(default=0.1, gt=0.0, le=1.0)


RGBA = tuple[float, float, float, float]

#: Highest selection slot the LUT's 4-bit field can carry.
MAX_OUTLINE_SLOT: int = 15

DEFAULT_OUTLINE_PALETTE: list[RGBA] = [
    (1.0, 0.0, 1.0, 1.0),
    (0.0, 1.0, 1.0, 1.0),
    (1.0, 1.0, 0.0, 1.0),
    (0.0, 1.0, 0.0, 1.0),
]


class OutlineLayerConfig(BaseModel):
    """Configuration for one screen-space outline layer.

    Two layers run in the same fragment invocation and can both be active:
    ``boundaries`` draws every outlined region, ``selection`` draws only
    regions with a nonzero LUT slot.  Precedence is selection > contrast
    band > boundaries > fill.

    Parameters
    ----------
    enabled : bool
        Whether this layer contributes.  A uniform; toggling it does not
        recompile the shader.
    inward_thickness : int
        Band width for inward-placed visuals, in **internal pixels**.
        Effect passes run before the output pass's SSAA downsample, so at
        ``pixel_ratio > 1`` the on-screen band is thinner than this number
        by that factor.  0 disables the inward branch for this layer.
        A shader template var: changing it recompiles.
    outward_thickness : int
        Band width for outward-placed visuals, same units and the same
        recompile behaviour.
    color : tuple[float, float, float, float]
        RGBA used by the **boundaries** layer.  The selection layer takes
        its colour from the palette slot carried in the LUT, so this field
        is unused there.  Alpha below 1 blends over the fill instead of
        replacing it.  Values are in the renderer's linear working space,
        the same convention as pygfx material colours.

    Notes
    -----
    Thickness is a property of the layer and the placement group, not of
    the individual visual.  Outward placement makes per-visual thickness
    impossible: finding an outward-outlined neighbour means sampling at
    *its* thickness, which is not known until after the tap.
    """

    enabled: bool = True
    inward_thickness: int = Field(default=1, ge=0)
    outward_thickness: int = Field(default=1, ge=0)
    color: RGBA = (1.0, 1.0, 1.0, 0.4)


class OutlineConfig(BaseModel):
    """Configuration for the screen-space outline pass.

    Parameters
    ----------
    enabled : bool
        Master switch.  Defaults to ``False``; when off the pass is skipped
        entirely by ``flush()`` and the frame is pixel-identical to one
        rendered without the feature.
    boundaries : OutlineLayerConfig
        The every-region layer.  Thin and translucent by default.
    selection : OutlineLayerConfig
        The selected-region layer.  Thick and opaque by default, coloured
        from ``palette``.
    inner_thickness : int
        Width of the contrast band drawn immediately inside the selection
        outline, in internal pixels.  0 disables it.  A template var:
        changing it recompiles.
    inner_color : tuple[float, float, float, float]
        Contrast band colour.  Exists so a coloured outline stays legible
        against an arbitrary colormapped fill.
    palette : list[tuple[float, float, float, float]]
        Selection palette.  LUT slot ``v`` uses ``palette[v - 1]``; slot 0
        means "not selected".  At most ``MAX_OUTLINE_SLOT`` entries, the
        limit of the LUT's 4-bit slot field.

    Examples
    --------
    >>> config = OutlineConfig(
    ...     enabled=True,
    ...     boundaries=OutlineLayerConfig(enabled=True, inward_thickness=1),
    ...     selection=OutlineLayerConfig(inward_thickness=2, outward_thickness=2),
    ... )
    >>> restored = OutlineConfig.model_validate_json(config.model_dump_json())
    >>> restored == config
    True
    """

    enabled: bool = False
    boundaries: OutlineLayerConfig = Field(
        default_factory=lambda: OutlineLayerConfig(
            enabled=True,
            inward_thickness=1,
            outward_thickness=1,
            color=(1.0, 1.0, 1.0, 0.4),
        )
    )
    selection: OutlineLayerConfig = Field(
        default_factory=lambda: OutlineLayerConfig(
            enabled=True, inward_thickness=2, outward_thickness=2
        )
    )
    inner_thickness: int = Field(default=2, ge=0)
    inner_color: RGBA = (0.0, 0.0, 0.0, 1.0)
    palette: list[RGBA] = Field(default_factory=lambda: list(DEFAULT_OUTLINE_PALETTE))

    @field_validator("palette")
    @classmethod
    def _check_palette_length(cls, value: list[RGBA]) -> list[RGBA]:
        if len(value) > MAX_OUTLINE_SLOT:
            raise ValueError(
                f"palette holds at most {MAX_OUTLINE_SLOT} entries, got {len(value)}"
            )
        return value


class SSAOConfig(BaseModel):
    """Configuration for the screen-space ambient occlusion pass.

    The pass reads the depth buffer (and, where a shader wrote one, the
    ``normal`` render target), samples a rotated hemisphere around every
    visible fragment, and multiplies the resulting occlusion into the
    composited colour.  It is **3D only**: a 2D cellier scene is a plane at
    near-constant depth, where the occlusion comes out uniform.

    Parameters
    ----------
    enabled : bool
        Master switch.  Defaults to ``False``; when off the pass is skipped
        entirely by ``flush()`` and the frame is pixel-identical to one
        rendered without the feature.
    n_samples : int
        Hemisphere samples per pixel.  A shader template var: changing it
        recompiles.  16 rather than learnopengl's 64 because the pass runs
        *before* ``TemporalAccumulationPass``, whose EMA averages the
        per-frame kernel rotation away once the camera settles.
    blur_radius : int
        Half-width of the box blur applied to the occlusion field, in
        internal pixels, so the filter is ``(2 * blur_radius + 1)`` square.
        0 disables the blur.  Also a template var.
    radius : float or None
        Hemisphere radius in **scene units**.  ``None`` (the default) derives
        it from the scene bounding box; see ``auto_radius_fraction``.  A fixed
        default is meaningless across cellier's coordinate systems, where a
        bounding box may be 96 units or 0.0003.
    auto_radius_fraction : float
        Fraction of the scene bounding box diagonal used when ``radius`` is
        ``None``.
    bias : float
        Depth-comparison bias that stops a flat surface occluding itself from
        depth quantisation -- shadow acne under another name.  Expressed as a
        **fraction of the effective radius**, so it is dimensionless and
        survives cellier's coordinate systems for the same reason ``radius``
        is auto-derived: an absolute bias tuned for learnopengl's few-unit
        scene lets a flat plane self-occlude by 7 percent at a radius of 6,
        and does nothing at all at a radius of 0.0003.  The default is
        learnopengl's own ratio, ``0.025 / 0.5``.
    strength : float
        Lerps between no effect (0) and the full multiply (1).
    power : float
        Contrast exponent applied to the occlusion before the multiply.
        Values above 1 darken only the deepest crevices.

    Examples
    --------
    >>> config = SSAOConfig(enabled=True, n_samples=24, strength=0.8)
    >>> restored = SSAOConfig.model_validate_json(config.model_dump_json())
    >>> restored == config
    True
    """

    enabled: bool = False

    # Structure -- template vars; changing these recompiles the shader.
    n_samples: int = Field(default=16, ge=4, le=64)
    blur_radius: int = Field(default=2, ge=0, le=8)

    # Values -- uniforms; changing these does not recompile.
    radius: float | None = Field(default=None, gt=0.0)
    auto_radius_fraction: float = Field(default=0.02, gt=0.0)
    bias: float = Field(default=0.05, ge=0.0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    power: float = Field(default=1.0, gt=0.0)


class CameraConfig(BaseModel):
    """Configuration for camera-driven automatic reslicing.

    Parameters
    ----------
    reslice_enabled : bool
        When ``False`` camera movement never triggers a reslice.
        Manual calls to ``CellierController.reslice_scene`` still work.
    settle_threshold_s : float
        Seconds of camera stillness required before a reslice is
        triggered. Lower values give more responsive LOD updates;
        higher values reduce redundant I/O during fast panning.
    """

    reslice_enabled: bool = True
    settle_threshold_s: float = Field(default=DEFAULT_CAMERA_SETTLE_THRESHOLD_S, gt=0.0)


class RenderManagerConfig(BaseModel):
    """Top-level rendering performance configuration.

    Pass an instance to ``CellierController(render_config=...)`` at
    construction time. The live state is always accessible and
    serializable via ``render_manager.config``.

    Parameters
    ----------
    slicing : SlicingConfig
        Async chunk-slicing pipeline settings.
    temporal : TemporalAccumulationConfig
        Temporal accumulation pass settings.
    camera : CameraConfig
        Camera-driven reslicing settings.
    outline : OutlineConfig
        Screen-space outline pass settings.  Disabled by default.
    ssao : SSAOConfig
        Screen-space ambient occlusion settings.  Disabled by default.

    Examples
    --------
    Construct with custom settings and serialize:

    >>> config = RenderManagerConfig(
    ...     slicing=SlicingConfig(batch_size=32, render_every=4),
    ...     temporal=TemporalAccumulationConfig(alpha=0.05),
    ...     camera=CameraConfig(settle_threshold_s=0.5),
    ... )
    >>> json_str = config.model_dump_json()
    >>> config2 = RenderManagerConfig.model_validate_json(json_str)
    """

    slicing: SlicingConfig = Field(default_factory=SlicingConfig)
    temporal: TemporalAccumulationConfig = Field(
        default_factory=TemporalAccumulationConfig
    )
    camera: CameraConfig = Field(default_factory=CameraConfig)
    outline: OutlineConfig = Field(default_factory=OutlineConfig)
    ssao: SSAOConfig = Field(default_factory=SSAOConfig)
