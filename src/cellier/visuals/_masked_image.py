"""MaskedImage visual models (masked image design v2, section 4).

A MaskedImage draws an isosurface found in a **mask** image, lit from the mask,
and coloured from a **value** image.  In 3D the colour is the max, mean or min
of the value along the surface normal over ``surface_thickness``; in 2D the same
thickness is the width of a contour band around the iso value, and each pixel
in the band shows the value at that pixel.

The models subclass the multiscale image models, so every site that handles a
multiscale image (level of detail, scheduling, slicing, the hide rule, draw
status) handles a MaskedImage too.  One model serves every storage mix: either
image may be in memory or multiscale (design 4.5, MI-P2).
"""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from cellier.transform import AffineTransform
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageVisual,
)
from cellier.visuals._image_memory import BaseImageSingleAppearance

__all__ = [
    "DEFAULT_MASKED_GPU_BUDGET_BYTES",
    "MaskedImageAppearance",
    "MaskedImageRenderConfig",
    "MaskedImageSingleAppearance",
    "MaskedImageVisual",
]

#: Each side's default GPU budget (G6).  The 3D atlas is a cube of slots,
#: ``int(cbrt(budget / brick bytes))`` a side, so 1,100 MiB holds 17 cubed
#: (4,912 usable 32-voxel bricks, about a 543 cubed in-memory image) where
#: 1 GiB holds 16 cubed (4,095, one short of a 512 cubed image).
DEFAULT_MASKED_GPU_BUDGET_BYTES = 1100 * 1024**2


class MaskedImageAppearance(MultiscaleImageAppearance):
    """Shared appearance of a MaskedImage.

    The level-of-detail fields apply to each image on its own pyramid.  On an
    in-memory image, which is single scale, they have one level to choose and
    do nothing (design 4.5).

    Parameters
    ----------
    interpolation : str
        Texture filter for both images: ``"linear"`` (default) or
        ``"nearest"``.
    target_bricks : str
        Which target bricks a 3D view loads.  ``"visible"`` (default): every
        brick the view shows, as a multiscale image does.  ``"surface"``:
        only those near the surface, found from the coarse mask; a feature
        too thin for the coarse level is never loaded.  Ignored by a 2D view.
    force_level : int or None
        1-based level of the **mask** a view draws, overriding the automatic
        choice (1 is the finest).  Default None.
    value_force_level : int or None
        The same for the **value** image, whose pyramid can differ.  Default
        None.

    Notes
    -----
    ``slab_projection`` must stay ``"none"`` and ``projection_axes`` empty: a
    MaskedImage does not project a slab.  ``projection_steps_per_voxel`` sets
    the samples a value voxel along the surface normal instead.
    """

    interpolation: Literal["linear", "nearest"] = "linear"
    target_bricks: Literal["visible", "surface"] = "visible"
    value_force_level: int | None = None

    @field_validator("slab_projection")
    @classmethod
    def _refuse_slab_projection(cls, value: str) -> str:
        if value != "none":
            raise ValueError(
                f"slab_projection={value!r}: a MaskedImage does not project a "
                "slab; it must stay 'none'."
            )
        return value

    @field_validator("projection_axes")
    @classmethod
    def _refuse_projection_axes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value:
            raise ValueError(
                f"projection_axes={value!r}: a MaskedImage does not project a "
                "slab; it must stay empty."
            )
        return value


class MaskedImageSingleAppearance(BaseImageSingleAppearance):
    """The appearance of a MaskedImage's surface and band.

    ``color_map``, ``clim`` and ``opacity`` apply to the **value** image, in
    its own units.  ``iso_threshold`` is in the **mask's** units and applies
    in 2D as well as 3D.  Both are the arrays' own numbers, not normalised: a
    ``uint8`` binary mask of 0 and 255 takes a threshold of 127.5.

    Parameters
    ----------
    render_mode : Literal["iso"]
        Always ``"iso"``.
    surface_thickness : float
        World units, shared by 2D and 3D (MI-D5).  In 3D the value image is
        reduced along the surface normal over this length; ``0`` reads the
        value at the surface.  In 2D it is the width of the contour band;
        ``0`` is a hairline ``surface_min_width_px`` wide.  Default 0.
    surface_projection : str
        3D only: ``"max"`` (default), ``"mean"`` or ``"min"`` of the value
        along the normal.  A 2D pixel shows its own value (MI-D8).
    surface_placement : str
        Where the thickness lies relative to the surface, in 2D and 3D:
        ``"inward"`` (default, into the mask), ``"centred"`` or
        ``"outward"``.
    surface_min_width_px : float
        2D only: a band is drawn at least this many screen pixels wide, so
        it neither shimmers when zoomed out nor vanishes at thickness 0.
        ``0`` draws the strictly physical width.  Default 1.0.
    band_rays : int
        2D only: rays in the fan that searches for the contour (MI-P15).
        Fewer is faster and loses up to a few percent of the band at its far
        edge on a steep mask.  Default 16.
    band_cap : int
        2D only: the steps a search ray takes before its step widens to
        ``reach / band_cap``, bounding the reads a pixel.  Default 32.
    """

    render_mode: Literal["iso"] = "iso"
    surface_thickness: float = Field(default=0.0, ge=0.0)
    surface_projection: Literal["max", "mean", "min"] = "max"
    surface_placement: Literal["inward", "centred", "outward"] = "inward"
    surface_min_width_px: float = Field(default=1.0, ge=0.0)
    band_rays: int = Field(default=16, ge=1)
    band_cap: int = Field(default=32, ge=1)


class MaskedImageRenderConfig(MultiscaleImageRenderConfig):
    """GPU configuration of a MaskedImage: one budget for each image.

    Each image's 3D atlas is allocated at its budget whatever the image's
    size, so two defaults reserve about 2 GiB of GPU memory even for a small
    pair; lower either field to reserve less.  An in-memory image whose one
    level does not fit its budget is refused when added.

    Parameters
    ----------
    gpu_budget_bytes : int
        The mask's 3D atlas.  Default 1,100 MiB.
    value_gpu_budget_bytes : int
        The value's 3D atlas.  Default 1,100 MiB.

    Notes
    -----
    ``gpu_budget_bytes_2d`` is inherited and unused: every view of a
    MaskedImage, 2D included, is drawn from 3D bricks.
    """

    gpu_budget_bytes: int = DEFAULT_MASKED_GPU_BUDGET_BYTES
    value_gpu_budget_bytes: int = DEFAULT_MASKED_GPU_BUDGET_BYTES


class MaskedImageVisual(MultiscaleImageVisual):
    """Model of a MaskedImage: a mask's isosurface coloured by a value image.

    Build it with :meth:`cellier.controller.CellierController.add_masked_image`,
    which registers both stores and checks that they can be drawn together.

    Parameters
    ----------
    visual_type : Literal["masked_image"]
        Discriminator field.  Always ``"masked_image"``.
    data_store_id : str
        The **mask's** store.
    level_transforms : list[AffineTransform]
        The mask store's levels.
    value_store_id : str
        The value's store.  Frozen.
    value_to_mask : AffineTransform or None
        Value level-0 data coordinates to mask level-0 data coordinates.
        The value image's world transform is this, then ``transform``, so
        moving the visual moves both images together.  ``None`` means the
        two share a grid.  Frozen.
    value_level_transforms : list[AffineTransform]
        The value store's levels.
    appearance : MaskedImageAppearance
        Shared fields, including each image's level of detail.
    single : MaskedImageSingleAppearance
        The surface's appearance.
    render_config : MaskedImageRenderConfig
        Each image's GPU budget.

    Notes
    -----
    Refused at construction: a ``channel_axis``, ``composite=True``, any
    ``channels`` and any ``render_planes``.  A ``c`` axis in the stores is a
    slider axis like ``t``, not a channel axis (design 4.3).
    """

    visual_type: Literal["masked_image"] = "masked_image"
    value_store_id: str = Field(frozen=True)
    value_to_mask: AffineTransform | None = Field(default=None, frozen=True)
    value_level_transforms: list[AffineTransform]
    appearance: MaskedImageAppearance = Field(default_factory=MaskedImageAppearance)
    single: MaskedImageSingleAppearance = Field(
        default_factory=MaskedImageSingleAppearance
    )
    render_config: MaskedImageRenderConfig = Field(
        default_factory=MaskedImageRenderConfig
    )

    @model_validator(mode="after")
    def _refuse_unsupported(self) -> "MaskedImageVisual":
        refused = []
        if self.channel_axis is not None:
            refused.append(f"channel_axis={self.channel_axis}")
        if self.composite:
            refused.append("composite=True")
        if self.channels:
            refused.append(f"channels (keys {sorted(self.channels)})")
        if self.render_planes:
            refused.append(f"{len(self.render_planes)} render_planes")
        if refused:
            raise ValueError(
                f"A MaskedImage has one value channel and no render planes; "
                f"refused: {', '.join(refused)}.  A 'c' axis in the stores is "
                "a slider axis, not a channel axis."
            )
        return self
