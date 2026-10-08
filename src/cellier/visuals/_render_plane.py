"""Render planes: the planes a visual in ``"plane"`` render mode draws on.

Plane rendering design v3, section 4.  A render plane is a pose in the
scene's world space (an origin and two in-plane axes) with an extent along
each in-plane axis.  It is unlike a ``ClippingPlane``, which is a half-space
in the visual's data coordinates: a frame and a rectangle do not survive an
anisotropic data transform, so they live in world units.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import numpy as np
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from cellier.transform import AxisRef
from cellier.visuals._plane_outline import PlaneOutline

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from cellier.transform import CoordinateSystem

#: The ``render_mode`` value, on image and labels, that draws render planes.
PLANE_RENDER_MODE = "plane"

#: The most render planes a visual may carry: the shader's uniforms hold
#: this many, so adding or removing one is a uniform write, not a compile.
MAX_RENDER_PLANES = 4

#: How far from orthogonal (as a dot product of unit vectors) the in-plane
#: axes may be.
_ORTHOGONAL_TOL = 1e-6

Vector3 = tuple[float, float, float]
Axes3 = tuple[AxisRef, AxisRef, AxisRef]
Extent = tuple[float | None, float | None]


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length == 0.0:
        raise ValueError(f"{name} must be finite and non-zero; got {vector.tolist()}.")
    return vector / length


def _as_vector(value: ArrayLike, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(
            f"{name} must have three entries, one per axis; got shape {vector.shape}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite; got {vector.tolist()}.")
    return vector


def _resolve_axes(
    world: CoordinateSystem, axes: Sequence[AxisRef] | None
) -> tuple[UUID4, UUID4, UUID4]:
    """The ids of the three world axes a plane is given on."""
    if axes is None:
        if world.ndim != 3:
            raise ValueError(
                f"The coordinate system has {world.ndim} axes "
                f"{world.axis_names()}; pass axes= to name the three the "
                "plane is given on."
            )
        return tuple(axis.id for axis in world.axes)
    if len(axes) != 3:
        raise ValueError(f"A render plane is on three axes; got {len(axes)}.")
    return tuple(world.resolve_axis(axis).id for axis in axes)


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Rotation matrix of a quaternion ``(x, y, z, w)``, scalar last."""
    x, y, z, w = _unit(quaternion, "rotation")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class RenderPlane(BaseModel):
    """One plane a visual in ``"plane"`` render mode draws its data on.

    The plane is in the scene's **world** space, on three named world axes.
    ``origin`` is a point on it and ``in_plane_axis_0`` and
    ``in_plane_axis_1`` are two orthogonal unit vectors in it; all three are
    given on ``axes``, in that order.  The plane constrains no other world
    axis: a ``zyx`` plane in a ``tzyx`` scene holds at every time point.

    The plane is drawn where it crosses the visual's data, inside the
    visual's clipping planes, and within ``extent_0`` and ``extent_1``: the
    ``(min, max)`` distances from ``origin`` along the two in-plane axes, in
    world units.  ``None`` is unbounded on that side.

    The model is frozen.  Moving a plane means assigning a new tuple to
    ``visual.render_planes``.  Build the moved plane with
    ``plane.model_copy(update={...})`` so it keeps its ``id``: the id names
    a plane across replacements of the tuple.  Two planes built from scratch
    have different ids and are not equal, even with the same pose.
    (``model_copy`` does not validate: give it unit, orthogonal axes, or use
    a constructor.)

    Parameters
    ----------
    id : UUID4
        Names this plane.  Unique within a visual's ``render_planes``.
        Generated when not given.
    coordinate_system : UUID4
        The id of the scene's world coordinate system.
    axes : tuple of three AxisRef
        The world axes the pose is given on, by id or name.  The
        constructors store ids.
    origin : tuple of three float
        A point on the plane.
    in_plane_axis_0, in_plane_axis_1 : tuple of three float
        The in-plane frame.  Normalised on construction; they must be
        finite, non-zero and orthogonal.
    extent_0, extent_1 : tuple of two (float or None)
        ``(min, max)`` along each in-plane axis; ``None`` is unbounded.  A
        bounded pair needs ``min < max``.
    enabled : bool
        A disabled plane stays in the tuple and is not drawn.  Default
        ``True``.
    outline : PlaneOutline
        The line around what is drawn of the plane.  Off by default.  It is
        drawn while the plane is: in a 3D view of the plane's axes, with
        the visual in the ``"plane"`` render mode.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID4 = Field(default_factory=uuid4)
    coordinate_system: UUID4
    axes: Axes3
    origin: Vector3
    in_plane_axis_0: Vector3
    in_plane_axis_1: Vector3
    extent_0: Extent = (None, None)
    extent_1: Extent = (None, None)
    enabled: bool = True
    outline: PlaneOutline = Field(default_factory=PlaneOutline)

    @field_validator("axes", mode="before")
    @classmethod
    def _axis_ids_from_text(cls, value: Any) -> Any:
        """Read an id written as text (a JSON round trip) back as an id."""
        if not isinstance(value, (list, tuple)):
            return value
        axes = []
        for axis in value:
            if isinstance(axis, str):
                try:
                    axis = UUID(axis)
                except ValueError:
                    pass  # an axis name
            axes.append(axis)
        return tuple(axes)

    @field_validator("origin")
    @classmethod
    def _finite_origin(cls, value: Vector3) -> Vector3:
        if not all(np.isfinite(value)):
            raise ValueError(f"RenderPlane.origin must be finite; got {value}.")
        return value

    @field_validator("extent_0", "extent_1")
    @classmethod
    def _valid_extent(cls, value: Extent) -> Extent:
        low, high = value
        for side in value:
            if side is not None and not np.isfinite(side):
                raise ValueError(
                    "A render plane extent side is a finite number or None "
                    f"(unbounded); got {value}."
                )
        if low is not None and high is not None and not low < high:
            raise ValueError(
                f"A bounded render plane extent needs min < max; got {value}."
            )
        return value

    @model_validator(mode="after")
    def _validate_frame(self) -> Self:
        if len(set(self.axes)) != 3:
            raise ValueError(
                f"RenderPlane.axes must be three distinct axes; got {self.axes!r}."
            )
        axis_0 = _unit(np.asarray(self.in_plane_axis_0), "RenderPlane.in_plane_axis_0")
        axis_1 = _unit(np.asarray(self.in_plane_axis_1), "RenderPlane.in_plane_axis_1")
        if abs(float(axis_0 @ axis_1)) > _ORTHOGONAL_TOL:
            raise ValueError(
                "RenderPlane.in_plane_axis_0 and in_plane_axis_1 must be "
                f"orthogonal; their unit vectors have dot product "
                f"{float(axis_0 @ axis_1):.3g}."
            )
        # Stored normalised.  The model is frozen, so write past the guard.
        object.__setattr__(self, "in_plane_axis_0", tuple(float(v) for v in axis_0))
        object.__setattr__(self, "in_plane_axis_1", tuple(float(v) for v in axis_1))
        return self

    @property
    def normal(self) -> Vector3:
        """The unit normal, ``in_plane_axis_0 x in_plane_axis_1``, on ``axes``.

        Derived, not stored.  Which of the two sides it points to changes
        nothing drawn.
        """
        normal = np.cross(self.in_plane_axis_0, self.in_plane_axis_1)
        return tuple(float(v) for v in normal)

    @classmethod
    def from_point_normal(
        cls,
        world: CoordinateSystem,
        point: ArrayLike,
        normal: ArrayLike,
        axes: Sequence[AxisRef] | None = None,
        up: ArrayLike | None = None,
        **kwargs: Any,
    ) -> RenderPlane:
        """Build the plane through *point* with the given *normal*.

        The in-plane frame is chosen for the caller.  Axis 0 is *up*
        projected into the plane; with no *up* it is the projection of the
        axis (of *axes*) least aligned with the normal.  Axis 1 completes
        the frame so that ``in_plane_axis_0 x in_plane_axis_1`` is the
        normal.

        Parameters
        ----------
        world : CoordinateSystem
            The scene's world coordinate system.
        point : ArrayLike
            A point on the plane, on *axes*.
        normal : ArrayLike
            The plane normal, on *axes*.  Any non-zero length.
        axes : sequence of three AxisRef, or None
            The world axes *point* and *normal* are given on, e.g.
            ``("z", "y", "x")``.  ``None`` means the system's axes, which
            must then be three.
        up : ArrayLike or None
            A direction, on *axes*, whose projection into the plane becomes
            ``in_plane_axis_0``.  It must not be parallel to the normal.
        **kwargs :
            ``extent_0``, ``extent_1``, ``enabled``, ``outline``, ``id``.

        Returns
        -------
        RenderPlane
        """
        axis_ids = _resolve_axes(world, axes)
        origin = _as_vector(point, "point")
        unit_normal = _unit(_as_vector(normal, "normal"), "normal")
        if up is None:
            reference = np.zeros(3)
            reference[int(np.argmin(np.abs(unit_normal)))] = 1.0
        else:
            reference = _as_vector(up, "up")
        in_plane = reference - (reference @ unit_normal) * unit_normal
        if float(np.linalg.norm(in_plane)) < _ORTHOGONAL_TOL:
            raise ValueError(
                "up must not be parallel to the normal: it has no direction "
                "in the plane."
            )
        axis_0 = _unit(in_plane, "up")
        axis_1 = np.cross(unit_normal, axis_0)
        return cls(
            coordinate_system=world.id,
            axes=axis_ids,
            origin=tuple(origin.tolist()),
            in_plane_axis_0=tuple(axis_0.tolist()),
            in_plane_axis_1=tuple(axis_1.tolist()),
            **kwargs,
        )

    @classmethod
    def from_rotation(
        cls,
        world: CoordinateSystem,
        origin: ArrayLike,
        rotation: ArrayLike,
        axes: Sequence[AxisRef] | None = None,
        **kwargs: Any,
    ) -> RenderPlane:
        """Build the plane at *origin* whose frame is a rotation.

        Parameters
        ----------
        world : CoordinateSystem
            The scene's world coordinate system.
        origin : ArrayLike
            A point on the plane, on *axes*.
        rotation : ArrayLike
            A 3 x 3 rotation matrix whose columns are ``in_plane_axis_0``,
            ``in_plane_axis_1`` and the normal, on *axes*; or a quaternion
            ``(x, y, z, w)`` (scalar last) of that rotation, its vector part
            on *axes*.
        axes : sequence of three AxisRef, or None
            The world axes the pose is given on.  ``None`` means the
            system's axes, which must then be three.
        **kwargs :
            ``extent_0``, ``extent_1``, ``enabled``, ``outline``, ``id``.

        Returns
        -------
        RenderPlane

        Raises
        ------
        ValueError
            If *rotation* is not a rotation: not orthonormal, or a
            reflection.
        """
        axis_ids = _resolve_axes(world, axes)
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape == (4,):
            matrix = _quaternion_to_matrix(rotation)
        elif rotation.shape == (3, 3):
            matrix = rotation
        else:
            raise ValueError(
                "rotation must be a 3 x 3 matrix or a quaternion (x, y, z, w); "
                f"got shape {rotation.shape}."
            )
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or (
            np.linalg.det(matrix) < 0
        ):
            raise ValueError(
                "rotation must be a rotation matrix: orthonormal columns and "
                "determinant +1."
            )
        return cls(
            coordinate_system=world.id,
            axes=axis_ids,
            origin=tuple(_as_vector(origin, "origin").tolist()),
            in_plane_axis_0=tuple(matrix[:, 0].tolist()),
            in_plane_axis_1=tuple(matrix[:, 1].tolist()),
            **kwargs,
        )


_RENDER_PLANES = TypeAdapter(tuple[RenderPlane, ...])


def validate_render_planes(value: object) -> tuple[RenderPlane, ...]:
    """Coerce *value* to a tuple of :class:`RenderPlane`, or raise."""
    return _RENDER_PLANES.validate_python(value)
