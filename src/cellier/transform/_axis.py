"""Axes of a coordinate system (RFC-5 flavoured)."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import UUID4, AliasChoices, ConfigDict, Field

from cellier._model import CellierModel

AxisType = Literal["array", "space", "time", "channel", "coordinate", "displacement"]
"""The closed set of axis types defined by OME-NGFF RFC-5."""

AxisSampling = Literal["discrete", "continuous"]
"""Whether an axis's coordinates are sample indices or measured positions.

``"discrete"`` means the coordinates along this axis are **sample indices**,
so the grid is the integers: a voxel axis, or a points column that holds an
acquisition frame number.  ``"continuous"`` means they are measured
positions that may fall anywhere.

The distinction decides how a position between samples is resolved, and the
two answers are genuinely different operations rather than one being an
approximation of the other.  See `Axis` for why that matters.
"""

AxisRef = str | UUID4
"""A reference to an axis, either by name or by id.

A ``str`` resolves through `CoordinateSystem.axis_by_name`, which
raises when the name is ambiguous within its system.  A ``UUID4``
resolves through `CoordinateSystem.index_of` and is never
ambiguous.
"""


class Axis(CellierModel):
    """A single axis of a coordinate system.

    Parameters
    ----------
    name : str
        Human-readable name, e.g. ``"z"``.  Names are not required to be
        unique within a coordinate system, but querying an ambiguous name
        raises.
    axis_type : AxisType
        The RFC-5 axis type.  Required: there is no default, because a
        channel or time axis silently typed as ``"space"`` fails far from
        where it was constructed.  Serialized as ``type``; both spellings
        parse.
    unit : str or None
        Free-form unit string.  RFC-5 recommends UDUNITS-2, but nothing
        here validates or interprets it.
    sampling : AxisSampling
        Whether this axis's coordinates are sample indices
        (``"discrete"``) or measured positions (``"continuous"``).

        **What it decides.**  A position landing between two samples is
        resolved by *snapping to the nearest sample* on a discrete axis and
        by *containment in a window* on a continuous one.  Those disagree by
        up to half a sampling interval: a snap flips at the midpoint between
        two samples, a window flips when the position reaches the sample.
        With an image (always discrete) and a points or graph store sharing
        one world axis, that difference is directly visible as the markers
        lagging the image by half a frame.

        Only a **data** axis is consulted.  Discreteness is a property of
        how one store samples the world, not of the world itself: the same
        world time axis is sampled at thirteen irregular frames by one store
        and continuously by another, and each needs its own answer.  This is
        the same reasoning that put the coordinate table on the transform
        rather than on the world.

        Unlike ``axis_type`` this **has** a default, and the default is
        ``"continuous"``.  A wrong ``axis_type`` selects the wrong plane
        silently; a wrong ``sampling`` degrades to the behaviour that
        predates the field, which is a visible half-interval lag rather
        than a silent wrong answer.  Gridded stores set it themselves -- a voxel grid is
        sample-indexed by construction -- so the value is only ever authored
        for geometry.
    id : UUID4
        Unique identifier.  Auto-generated.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    axis_type: AxisType = Field(
        serialization_alias="type",
        validation_alias=AliasChoices("axis_type", "type"),
    )
    unit: str | None = None
    sampling: AxisSampling = "continuous"
    id: UUID4 = Field(default_factory=uuid4, frozen=True)
