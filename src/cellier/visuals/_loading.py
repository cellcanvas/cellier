"""Progressive loading settings for the multiscale visuals.

See ``plans/progressive_loading_design_v3.md`` 5.9 and 5.11.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProgressiveLoadingConfig(BaseModel):
    """How a multiscale visual loads: a coarse backstop, then the target.

    Every setting is explicit, with a fixed default, and never changed by the
    library at runtime.  Replacing a visual's ``render_config`` with one that
    differs only here reslices the visual without reallocating its atlases.

    Parameters
    ----------
    backstop : bool
        Load a coarse *backstop* level ahead of the target level, so the view
        is blurry rather than blank (or showing the previous slice) while
        the target loads.  Backstop reads go before every target read, from
        every visual.  ``False`` loads the target only.  Default ``True``.
    backstop_level : int or None
        1-based level of the backstop, like ``force_level`` (1 is the
        finest).  Clamped to the pyramid.  ``None`` (the default) is the
        coarsest level.
    backstop_extent : {"full", "view"}
        ``"full"`` (default): the whole volume in 3D, so orbiting never
        exposes black, and the whole slice in 2D.  ``"view"``: frustum-culled
        in 3D, and the viewport plus one backstop tile of margin in 2D.
    backstop_max_slot_fraction : float
        At most this share of an atlas's slots holds backstop bricks, filled
        nearest the camera (3D) or canvas centre (2D) first.  Only shallow
        pyramids reach it; when it truncates, one INFO line per visual on
        ``cellier.render.cache`` names the settings that would avoid it.
        Default ``0.1``.
    dims_drag : {"eager", "backstop"}
        What a dims slider tick loads.  ``"eager"`` (default): the full plan,
        so the target starts loading at once; best on local disk.
        ``"backstop"``: the backstop only, and the target once the slider has
        been still for ``SchedulerConfig.dims_settle_s``; this saves most of
        a scrub's reads, and is recommended for remote stores.  Both show the
        slider's slice (blurry) about one read behind it.  Needs
        ``backstop=True``.
    """

    model_config = ConfigDict(frozen=True)

    backstop: bool = True
    backstop_level: int | None = Field(default=None, ge=1)
    backstop_extent: Literal["full", "view"] = "full"
    backstop_max_slot_fraction: float = Field(default=0.1, gt=0, le=0.5)
    dims_drag: Literal["eager", "backstop"] = "eager"

    @model_validator(mode="after")
    def _drag_needs_a_backstop(self) -> ProgressiveLoadingConfig:
        if self.dims_drag == "backstop" and not self.backstop:
            raise ValueError(
                "dims_drag='backstop' needs backstop=True: a backstop-only "
                "slider tick would plan nothing."
            )
        return self
