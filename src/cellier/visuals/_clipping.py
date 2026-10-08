"""Clipping planes: the model a visual carries (clipping planes design v2)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import UUID4, BaseModel, ConfigDict, Field, TypeAdapter

from cellier.transform import Plane
from cellier.visuals._plane_outline import PlaneOutline

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from cellier.transform import AxisRef, CoordinateSystem


class ClippingPlane(BaseModel):
    """One clipping plane of a visual.

    The visual is drawn where ``plane.normal . p >= plane.offset``: the
    half-space the normal points into.  Several planes on one visual keep
    the intersection of their half-spaces.

    The plane is in the visual's level-0 data coordinate system, the system
    its store owns.  Build the store first, then the plane from
    ``store.data_coordinate_systems[0]``, then the visual.  Units are data
    units: voxels for an image.  On anisotropic data the normal is a
    covector in voxel space, so a plane tilted 45 degrees in the sample is
    not ``(1, 1, 0)`` when the voxels are not cubes.

    A plane may be defined on some of the data axes only (see
    :meth:`from_point_normal`).  A ``zyx`` plane on ``tzyx`` data holds at
    every timepoint.

    The model is frozen.  Moving a plane means assigning a new tuple to
    ``visual.clipping_planes``.  Build the moved plane with
    ``item.model_copy(update={"plane": new_plane})`` so it keeps its ``id``:
    the id is what names a plane across replacements of the tuple.  Two
    planes built from scratch have different ids and are not equal, even
    with the same ``plane``.

    Parameters
    ----------
    id : UUID4
        Names this plane.  Unique within a visual's ``clipping_planes``.
        Generated when not given.
    plane : Plane
        The plane, in the visual's level-0 data coordinates.
    enabled : bool
        A disabled plane stays in the list and clips nothing.  Toggling it
        costs no shader compile.  Default ``True``.
    outline : PlaneOutline
        The line around the plane's cut face: the plane where it crosses
        the visual's data, inside the visual's other clipping planes.  Off
        by default.  It is drawn in a 3D view, while the plane is enabled.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID4 = Field(default_factory=uuid4)
    plane: Plane
    enabled: bool = True
    outline: PlaneOutline = Field(default_factory=PlaneOutline)

    @classmethod
    def from_point_normal(
        cls,
        coordinate_system: CoordinateSystem,
        point: ArrayLike,
        normal: ArrayLike,
        axes: Sequence[AxisRef] | None = None,
        *,
        enabled: bool = True,
        outline: PlaneOutline | None = None,
    ) -> ClippingPlane:
        """Build the plane through *point* that keeps the side *normal* points to.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            The visual's level-0 data coordinate system:
            ``store.data_coordinate_systems[0]``.
        point : ArrayLike
            A point on the plane, in data coordinates.
        normal : ArrayLike
            The normal, the same length as *point*.  It points into the
            kept half-space.
        axes : Sequence[AxisRef] or None
            The axes *point* and *normal* are given on, e.g.
            ``("z", "y", "x")``.  The other axes are not constrained.
            ``None`` means every axis of the system.
        enabled : bool
            Whether the plane clips.  Default ``True``.
        outline : PlaneOutline or None
            The line around the plane's cut face.  ``None`` is no outline.

        Returns
        -------
        ClippingPlane
        """
        return cls(
            plane=Plane.from_point_normal(coordinate_system, point, normal, axes),
            enabled=enabled,
            outline=PlaneOutline() if outline is None else outline,
        )


_CLIPPING_PLANES = TypeAdapter(tuple[ClippingPlane, ...])


def validate_clipping_planes(value: object) -> tuple[ClippingPlane, ...]:
    """Coerce *value* to a tuple of :class:`ClippingPlane`, or raise."""
    return _CLIPPING_PLANES.validate_python(value)
