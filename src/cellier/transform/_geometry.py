"""Geometric value objects: axis-aligned boxes and planes (D29)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from cellier.transform._coordinate_system import AxisRef, CoordinateSystem

from cellier.transform._arrays import (
    coerce_float_array,
    reject_nan,
    reject_non_finite,
    serialize_array,
    serialize_bounds,
)


class AxisAlignedBoundingBox(BaseModel):
    """An axis-aligned box, in one named coordinate system.

    Bounds may be infinite: an unbounded axis is a first-class value
    here (D31), and it is the answer a region gives for a displayed axis
    nothing constrains.  ``nan`` is rejected.

    The coordinate system id is carried so that handing a world-space box
    to a data-space operation is a caught error rather than a silent
    wrong answer -- the same class of mistake the axis-order bugs in this
    repo's history were.

    Parameters
    ----------
    coordinate_system : UUID4
        The id of the coordinate system these bounds are expressed in.
    min_coordinate : np.ndarray
        Lower bounds, one per axis.  May contain ``-inf``.
    max_coordinate : np.ndarray
        Upper bounds, one per axis.  May contain ``+inf``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    coordinate_system: UUID4
    min_coordinate: np.ndarray
    max_coordinate: np.ndarray

    @field_validator("min_coordinate", "max_coordinate", mode="before")
    @classmethod
    def _coerce_bounds(cls, value: Any) -> np.ndarray:
        """Coerce to a float array, accepting the infinity sentinels (D44)."""
        return reject_nan(coerce_float_array(value), "A bounding box bound")

    @field_serializer("min_coordinate", "max_coordinate")
    def _serialize_bounds(self, value: np.ndarray) -> list[float | str]:
        """Emit infinities as sentinel strings so JSON round-trips (D44)."""
        return serialize_bounds(value)

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        """Require equal rank, rank >= 1, and ``min <= max`` on every axis."""
        if self.min_coordinate.ndim != 1 or self.max_coordinate.ndim != 1:
            raise ValueError("Bounding box bounds must be 1-D.")
        if self.min_coordinate.shape != self.max_coordinate.shape:
            raise ValueError(
                f"min_coordinate and max_coordinate must have the same shape, "
                f"got {self.min_coordinate.shape} and {self.max_coordinate.shape}."
            )
        if self.min_coordinate.size == 0:
            raise ValueError("A bounding box must have at least one axis.")
        if np.any(self.min_coordinate > self.max_coordinate):
            raise ValueError(
                f"min_coordinate must not exceed max_coordinate on any axis; got "
                f"{self.min_coordinate.tolist()} and {self.max_coordinate.tolist()}."
            )
        return self

    @property
    def ndim(self) -> int:
        """Number of axes."""
        return int(self.min_coordinate.shape[0])

    def __eq__(self, other: object) -> bool:
        """Compare by coordinate system and bounds (D44)."""
        if not isinstance(other, AxisAlignedBoundingBox):
            return NotImplemented
        return (
            self.coordinate_system == other.coordinate_system
            and bool(np.array_equal(self.min_coordinate, other.min_coordinate))
            and bool(np.array_equal(self.max_coordinate, other.max_coordinate))
        )

    def __hash__(self) -> int:
        """Hash by coordinate system and bounds (D44)."""
        return hash(
            (
                self.coordinate_system,
                self.min_coordinate.tobytes(),
                self.max_coordinate.tobytes(),
            )
        )


class Plane(BaseModel):
    """The set of points where ``normal . p == offset``.

    Point-normal form is deliberately not used: the transform rules of
    design section 4.1 are stated directly on ``(normal, offset)``, and a
    stored point would have to be re-derived on every map anyway.

    The normal is **not** required to be unit length (D28).  It is
    required to be finite and non-zero: a zero normal does not describe a
    plane.

    Parameters
    ----------
    coordinate_system : UUID4
        The id of the coordinate system this plane is expressed in.
    normal : np.ndarray
        The plane normal.  Finite, non-zero, any magnitude.  Read-only: a
        plane is changed by building a new one.
    offset : float
        The plane offset.  Finite (D44).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    coordinate_system: UUID4
    normal: np.ndarray
    offset: float

    @field_validator("normal", mode="before")
    @classmethod
    def _coerce_normal(cls, value: Any) -> np.ndarray:
        """Coerce to a finite float array (D44)."""
        return reject_non_finite(coerce_float_array(value), "Plane.normal")

    @field_validator("offset")
    @classmethod
    def _validate_offset(cls, value: float) -> float:
        """Reject a non-finite offset (D44)."""
        if not np.isfinite(value):
            raise ValueError(f"Plane.offset must be finite; got {value}.")
        return value

    @field_serializer("normal")
    def _serialize_normal(self, value: np.ndarray) -> list[float]:
        """Emit the normal as a list of floats."""
        return serialize_array(value)

    @model_validator(mode="after")
    def _validate_normal(self) -> Self:
        """Require a 1-D, non-zero normal."""
        if self.normal.ndim != 1 or self.normal.size == 0:
            raise ValueError("Plane.normal must be a non-empty 1-D array.")
        if not np.any(self.normal != 0.0):
            raise ValueError(
                "Plane.normal must not be the zero vector: that describes "
                "either all of space or none of it, not a plane."
            )
        # The model is frozen; without this the array could still be edited
        # in place, with no event and a stale hash.
        self.normal.flags.writeable = False
        return self

    @classmethod
    def from_point_normal(
        cls,
        coordinate_system: CoordinateSystem,
        point: ArrayLike,
        normal: ArrayLike,
        axes: Sequence[AxisRef] | None = None,
    ) -> Self:
        """Build the plane through *point* with the given *normal*.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            The system the plane is expressed in.
        point : ArrayLike
            A point on the plane.
        normal : ArrayLike
            The plane normal, the same length as *point*.
        axes : Sequence[AxisRef] or None
            The axes *point* and *normal* are given on, by name or id, in
            the order of their entries.  Every other axis of the system
            gets a zero normal component: the plane does not constrain it.
            ``None`` means every axis of the system, in its order.

        Returns
        -------
        Plane
            With ``offset = normal . point``.
        """
        point = np.asarray(point, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)
        if point.ndim != 1 or point.shape != normal.shape:
            raise ValueError(
                "point and normal must be 1-D and the same length; got "
                f"shapes {point.shape} and {normal.shape}."
            )
        ndim = len(coordinate_system.axes)
        if axes is None:
            if len(normal) != ndim:
                raise ValueError(
                    f"The coordinate system has {ndim} axes but point and "
                    f"normal have {len(normal)} entries; pass axes= to name "
                    "the axes they are given on."
                )
            full = normal
        else:
            indices = [coordinate_system.resolve(axis) for axis in axes]
            if len(indices) != len(normal):
                raise ValueError(
                    f"axes names {len(indices)} axes but point and normal "
                    f"have {len(normal)} entries."
                )
            if len(set(indices)) != len(indices):
                raise ValueError(f"axes names an axis more than once: {axes!r}.")
            full = np.zeros(ndim, dtype=np.float64)
            full[indices] = normal
        return cls(
            coordinate_system=coordinate_system.id,
            normal=full,
            offset=float(normal @ point),
        )

    def signed_distance(self, points: ArrayLike) -> np.ndarray:
        """Distance of each point from the plane, positive along the normal.

        Parameters
        ----------
        points : ArrayLike
            ``(n, ndim)`` points, or one ``(ndim,)`` point.

        Returns
        -------
        np.ndarray
            ``(normal . p - offset) / |normal|``, one value per point.
        """
        points = np.asarray(points, dtype=np.float64)
        return (points @ self.normal - self.offset) / float(np.linalg.norm(self.normal))

    def closest_point_to(self, point: ArrayLike) -> np.ndarray:
        """The point of the plane nearest to *point*.

        Parameters
        ----------
        point : ArrayLike
            One ``(ndim,)`` point.

        Returns
        -------
        np.ndarray
            *point* moved along the normal onto the plane.
        """
        point = np.asarray(point, dtype=np.float64)
        normal = self.normal
        return point - normal * (
            (point @ normal - self.offset) / float(normal @ normal)
        )

    @property
    def ndim(self) -> int:
        """Number of axes."""
        return int(self.normal.shape[0])

    def __eq__(self, other: object) -> bool:
        """Compare by coordinate system, normal and offset (D44)."""
        if not isinstance(other, Plane):
            return NotImplemented
        return (
            self.coordinate_system == other.coordinate_system
            and bool(np.array_equal(self.normal, other.normal))
            and self.offset == other.offset
        )

    def __hash__(self) -> int:
        """Hash by coordinate system, normal and offset (D44)."""
        return hash((self.coordinate_system, self.normal.tobytes(), self.offset))
