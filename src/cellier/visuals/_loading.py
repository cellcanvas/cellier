"""Progressive loading settings for the multiscale visuals.

See ``plans/progressive_loading_design_v3.md`` 5.9 and 5.11, and
``plans/mesh_refactor_v3.md`` 5.7 for :class:`GeometryLodConfig`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cellier.visuals._removed import REMOVED_LOADING_FIELDS, refuse_removed_fields

#: The largest offset: a stage's request class is its offset, in a ``uint8``
#: column whose top value is the backstop's.
_MAX_OFFSET = 254


class ProgressiveLoadingConfig(BaseModel):
    """How a multiscale visual loads: a coarse backstop, then the target.

    A coarse *backstop* level is always loaded ahead of the target level, so
    the view is blurry rather than blank (or showing the previous slice)
    while the target loads.  Backstop reads go before every target read,
    from every visual.

    Optional *intermediate stages* load between the two.  A stage is the
    target moved a fixed number of levels coarser (its *offset*): it covers
    what the view shows and nothing else, with far fewer reads, so a slow
    store shows a medium picture while the target loads.  With no stage
    (the default) loading is coarse, then fine.

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
    intermediate_offsets : tuple of int
        One intermediate stage per offset, given **coarse to fine**: each
        offset at least 1 and smaller than the one before, so ``(4, 2)`` is
        accepted and ``(2, 4)`` and ``(2, 2)`` are refused.  Nothing is
        re-sorted.  An offset is a number of levels coarser than each target
        brick's own level; a stage brick at or beyond the backstop level is
        dropped, so a stage may be empty for a view.  ``()`` (the default)
        is no stage; ``(2,)`` is coarse, medium, fine.
    intermediate_max_slot_fraction : float
        At most this share of an atlas's slots holds the bricks of all the
        stages together, filled coarsest stage first, each nearest the
        camera (3D) or canvas centre (2D) first.  When it truncates, one
        INFO line per visual on ``cellier.render.cache`` says so.  It sits
        beside ``backstop_max_slot_fraction``: the target gets what the
        backstop and stage bricks actually use leave.  Default ``0.1``.
    """

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _refuse_removed_fields(cls, data: Any) -> Any:
        return refuse_removed_fields(data, REMOVED_LOADING_FIELDS, cls.__name__)

    backstop_level: int | None = Field(default=None, ge=1)
    backstop_extent: Literal["full", "view"] = "full"
    backstop_max_slot_fraction: float = Field(default=0.1, gt=0, le=0.5)
    intermediate_offsets: tuple[int, ...] = ()
    intermediate_max_slot_fraction: float = Field(default=0.1, gt=0, le=0.5)

    @field_validator("intermediate_offsets")
    @classmethod
    def _offsets_run_coarse_to_fine(cls, offsets: tuple[int, ...]) -> tuple[int, ...]:
        ordered = all(a > b for a, b in zip(offsets, offsets[1:], strict=False))
        if not ordered or any(offset < 1 for offset in offsets):
            raise ValueError(
                "intermediate_offsets must run coarse to fine: each offset at "
                "least 1 and smaller than the one before, e.g. (4, 2); got "
                f"{offsets}."
            )
        if offsets and offsets[0] > _MAX_OFFSET:
            raise ValueError(
                f"intermediate_offsets may be at most {_MAX_OFFSET}; got {offsets}."
            )
        return offsets


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
