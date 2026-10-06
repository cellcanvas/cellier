"""Progressive loading settings for the multiscale visuals.

See ``plans/progressive_loading_design_v3.md`` 5.9 and 5.11, and
``plans/mesh_refactor_v3.md`` 5.7 for :class:`GeometryLodConfig`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cellier.visuals._removed import REMOVED_LOADING_FIELDS, refuse_removed_fields


class ProgressiveLoadingConfig(BaseModel):
    """How a multiscale visual loads: a coarse backstop, then the target.

    A coarse *backstop* level is always loaded ahead of the target level, so
    the view is blurry rather than blank (or showing the previous slice)
    while the target loads.  Backstop reads go before every target read,
    from every visual.

    Every setting is explicit, with a fixed default, and never changed by the
    library at runtime.  Replacing a visual's ``render_config`` with one that
    differs only here reslices the visual without reallocating its atlases.

    Parameters
    ----------
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
    """

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _refuse_removed_fields(cls, data: Any) -> Any:
        return refuse_removed_fields(data, REMOVED_LOADING_FIELDS, cls.__name__)

    backstop_level: int | None = Field(default=None, ge=1)
    backstop_extent: Literal["full", "view"] = "full"
    backstop_max_slot_fraction: float = Field(default=0.1, gt=0, le=0.5)


class GeometryLodConfig(BaseModel):
    """How a multiscale geometry visual uses its levels of detail.

    Two levels are kept loaded: the finest, and one coarse level.  Every
    setting is explicit, with a fixed default, and never changed by the
    library at runtime.  The config is frozen: replace it to change it.

    Parameters
    ----------
    coarse_level : int or None
        1-based level kept beside the finest (1 is the finest, so the
        smallest value is 2).  ``None`` (the default) is the coarsest level
        the store has.  A level the store does not have is refused when the
        visual is added.  Level numbers on a pick (``MeshPickInfo.level``)
        and on a store request are 0-based instead.
    dims_drag : {"coarse", "full"}
        What a dims scrub **loads** for each new position.  ``"coarse"``
        (the default) loads the coarse level while the slider moves and the
        finest when it rests; ``"full"`` loads both on every tick.
    dims_drag_draw : {"coarse", "full"}
        What a dims scrub **draws** of a visual the scrub does not change (a
        static mesh next to a time series).  ``"coarse"`` (the default)
        draws the coarse level while the slider moves, so a very large mesh
        does not slow the scrub's frames; the finest stays loaded and
        returns when the scrub ends.  ``"full"`` keeps drawing the finest.
        Applies at once, with no read.
    camera_motion : {"coarse", "full"}
        What is drawn while the camera moves in a 3D view.  ``"coarse"``
        (the default) draws the coarse level on the canvas whose camera is
        moving; ``"full"`` keeps drawing the finest.  Applies at once, with
        no read.
    """

    model_config = ConfigDict(frozen=True)

    coarse_level: int | None = Field(default=None, ge=2)
    dims_drag: Literal["coarse", "full"] = "coarse"
    dims_drag_draw: Literal["coarse", "full"] = "coarse"
    camera_motion: Literal["coarse", "full"] = "coarse"

    def coarse_scale_index(self, level_count: int) -> int | None:
        """The 0-based level kept beside the finest, for a store's levels.

        Parameters
        ----------
        level_count : int
            How many levels the store has.

        Returns
        -------
        int or None
            ``None`` when the store has one level: there is no coarse level.

        Raises
        ------
        ValueError
            If ``coarse_level`` names a level the store does not have.
        """
        if level_count < 2:
            if self.coarse_level is not None:
                raise ValueError(
                    f"coarse_level={self.coarse_level}, but the store has "
                    f"{level_count} level(s)."
                )
            return None
        if self.coarse_level is None:
            return level_count - 1
        if self.coarse_level > level_count:
            raise ValueError(
                f"coarse_level={self.coarse_level}, but the store has "
                f"{level_count} levels (1 is the finest)."
            )
        return self.coarse_level - 1
