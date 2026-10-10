"""Plane outlines: the line around a render plane or a clipping plane.

Plane outline design (``plans/plane_outline_design.md``), section 4.  An
outline is a field of the plane it is drawn around, so it travels with the
plane: through the controller, the events, the linkers and a saved scene.

Not to be confused with the screen-space outline pass
(:class:`~cellier.visuals.VisualOutline`), which draws the contour of what
a visual drew.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from collections.abc import Iterable

Rgba = tuple[float, float, float, float]


class PlaneOutline(BaseModel):
    """The line along the edge of the polygon a plane makes in its visual.

    For a render plane the polygon is what is drawn of it: the plane cut by
    its extents, by the visual's data box and by the visual's enabled
    clipping planes.  For a clipping plane it is the cut face: the plane
    cut by the data box and by the visual's other enabled clipping planes.

    The model is frozen, as the planes are.  An outline is changed the way
    a plane is moved: ``plane.model_copy(update={"outline": ...})`` in a
    new tuple, which keeps the plane's id.  A change of the outline alone
    redraws; it does not load data again.

    Parameters
    ----------
    enabled : bool
        Whether the outline is drawn.  Default ``False``.  Switching it off
        keeps the colour and the width.
    color : tuple of four float
        RGBA, each in ``[0, 1]``.  Default opaque white.
    width : float
        The line's width in screen pixels.  Default ``2.0``.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    color: Rgba = (1.0, 1.0, 1.0, 1.0)
    width: float = Field(default=2.0, gt=0.0)

    @field_validator("color")
    @classmethod
    def _valid_color(cls, value: Rgba) -> Rgba:
        if not all(np.isfinite(value)) or not all(0.0 <= v <= 1.0 for v in value):
            raise ValueError(
                f"PlaneOutline.color is four numbers in [0, 1] (RGBA); got {value}."
            )
        return value

    @field_validator("width")
    @classmethod
    def _finite_width(cls, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError(f"PlaneOutline.width must be finite; got {value}.")
        return value


#: What every plane carries until it is given an outline: not drawn.
NO_OUTLINE = PlaneOutline()


def without_outlines(planes: Iterable[Any]) -> tuple:
    """*planes* with every outline set aside.

    Two tuples that are equal after this differ in their outlines at most:
    the planes are where they were, so nothing has to be read again
    (plane outline design 4.3).

    Parameters
    ----------
    planes : Iterable[RenderPlane] or Iterable[ClippingPlane]
        A visual's ``render_planes`` or ``clipping_planes``.

    Returns
    -------
    tuple
        The same planes, in order, each with the default outline.
    """
    return tuple(
        plane
        if plane.outline == NO_OUTLINE
        else plane.model_copy(update={"outline": NO_OUTLINE})
        for plane in planes
    )
