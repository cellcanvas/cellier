"""Convex regions as intersections of half-spaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import (
    UUID4,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from cellier._model import CellierModel
from cellier.transform._arrays import (
    coerce_float_array,
    reject_non_finite,
    serialize_array,
)
from cellier.transform._geometry import AxisAlignedBoundingBox
from cellier.transform._geometry_ops import (
    axis_aligned_bounds,
    is_axis_aligned,
    polytope_bounds,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cellier.transform._axis import AxisRef
    from cellier.transform._coordinate_system import CoordinateSystem


class HalfSpace(CellierModel):
    """The set ``{p : normal . p <= offset}``.

    There is no ``inside`` field: the sign of the normal carries the
    side, so a slab is two half-spaces with opposed normals.  There is no
    coordinate system id either -- every constraint in a region is
    necessarily in the same system, so storing it per-constraint would
    only create something that can disagree with itself.

    Unlike `Plane`, a **zero normal is
    allowed**.  That is not an oversight: pulling a world constraint back
    through a transform with no extent along that axis produces exactly
    that, and the honest answers are "vacuous" (offset >= 0) and
    "infeasible" (offset < 0) rather than an error.

    Parameters
    ----------
    normal : np.ndarray
        The constraint normal.  Finite, any magnitude, possibly zero.
    offset : float
        The constraint offset.  Finite.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    normal: np.ndarray
    offset: float

    @field_validator("normal", mode="before")
    @classmethod
    def _coerce_normal(cls, value: Any) -> np.ndarray:
        """Coerce to a finite float array."""
        return reject_non_finite(coerce_float_array(value), "HalfSpace.normal")

    @field_validator("offset")
    @classmethod
    def _validate_offset(cls, value: float) -> float:
        """Reject a non-finite offset."""
        if not np.isfinite(value):
            raise ValueError(f"HalfSpace.offset must be finite; got {value}.")
        return value

    @field_serializer("normal")
    def _serialize_normal(self, value: np.ndarray) -> list[float]:
        """Emit the normal as a list of floats."""
        return serialize_array(value)

    @model_validator(mode="after")
    def _validate_normal(self) -> Self:
        """Require a non-empty 1-D normal."""
        if self.normal.ndim != 1 or self.normal.size == 0:
            raise ValueError("HalfSpace.normal must be a non-empty 1-D array.")
        return self

    @property
    def ndim(self) -> int:
        """Number of axes."""
        return int(self.normal.shape[0])

    def is_vacuous(self) -> bool:
        """Whether this constraint excludes nothing (a zero normal, offset >= 0)."""
        return not np.any(self.normal != 0.0) and self.offset >= 0

    def is_infeasible(self) -> bool:
        """Whether this constraint excludes everything (a zero normal, offset < 0)."""
        return not np.any(self.normal != 0.0) and self.offset < 0

    def __eq__(self, other: object) -> bool:
        """Compare by normal and offset."""
        if not isinstance(other, HalfSpace):
            return NotImplemented
        return (
            bool(np.array_equal(self.normal, other.normal))
            and self.offset == other.offset
        )

    def __hash__(self) -> int:
        """Hash by normal and offset."""
        return hash((self.normal.tobytes(), self.offset))


class ConvexRegion(CellierModel):
    """An intersection of half-spaces.  An empty tuple is all of space.

    This is the *query* type; `AxisAlignedBoundingBox` is the
    *answer* type, and `from_bounding_box` / `bounding_box`
    bridge them.  There is deliberately only one region type: an
    axis-aligned selection, an oblique slab and a camera frustum are the
    same object, and the axis-aligned fast path is an implementation
    detail rather than a kind.

    ``ndim`` is stored rather than derived.  An unbounded region has no
    half-spaces at all, so nothing else in the model carries its rank.
    It is validated against every half-space normal, so it cannot
    disagree with them.

    Parameters
    ----------
    coordinate_system : UUID4
        The id of the coordinate system this region is expressed in.
    ndim : int
        The rank of that coordinate system.
    half_spaces : tuple[HalfSpace, ...]
        The constraints.  Empty means all of space.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    coordinate_system: UUID4
    ndim: int
    half_spaces: tuple[HalfSpace, ...] = ()

    @model_validator(mode="after")
    def _validate_rank(self) -> Self:
        """Require rank >= 1 and every half-space to match it."""
        if self.ndim < 1:
            raise ValueError(f"ndim must be at least 1, got {self.ndim}.")
        mismatched = [
            half_space.ndim
            for half_space in self.half_spaces
            if half_space.ndim != self.ndim
        ]
        if mismatched:
            raise ValueError(
                f"Every half-space normal must have {self.ndim} components; "
                f"got {mismatched}."
            )
        return self

    # -- the constraint arrays ----------------------------------------

    @property
    def normals(self) -> np.ndarray:
        """The ``(M, ndim)`` stack of constraint normals."""
        if not self.half_spaces:
            return np.zeros((0, self.ndim))
        return np.stack([half_space.normal for half_space in self.half_spaces])

    @property
    def offsets(self) -> np.ndarray:
        """The ``(M,)`` stack of constraint offsets."""
        return np.array([half_space.offset for half_space in self.half_spaces])

    # -- queries ------------------------------------------------------

    def _bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the exact bounds, taking the fast path where possible."""
        normals = self.normals
        if is_axis_aligned(normals):
            return axis_aligned_bounds(normals, self.offsets, self.ndim)
        return polytope_bounds(normals, self.offsets, self.ndim)

    def bounding_box(self) -> AxisAlignedBoundingBox:
        """Return the smallest axis-aligned box containing this region.

        Exact, not conservative: the bounds come from ``2 * ndim`` linear
        programs, or -- when every normal lies along a single axis -- are
        read off directly, which is roughly 1000x faster and covers every
        axis-aligned selection, which is what the slicing path sends.

        An unconstrained axis comes back ``-inf`` / ``+inf``.

        A zero-thickness region is not special-cased: the box comes back
        with ``min == max`` on that axis, and converting that to voxel
        indices, with whatever rounding rule applies, is the datastore's
        job.

        Returns
        -------
        AxisAlignedBoundingBox
            The exact bounds, in this region's coordinate system.

        Raises
        ------
        ValueError
            If the region is empty, and so has no bounds.  Check
            `is_empty` first when that is possible.
        """
        bounds = self._bounds()
        if bounds is None:
            raise ValueError(
                "An empty region has no bounding box; its constraints are "
                "mutually infeasible.  Check is_empty() first."
            )
        lower, upper = bounds
        return AxisAlignedBoundingBox(
            coordinate_system=self.coordinate_system,
            min_coordinate=lower,
            max_coordinate=upper,
        )

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Whether each point satisfies every constraint.

        This answers a question about **continuous** points and is *not*
        the voxel selection API.  A zero-thickness region is a
        measure-zero set, so this is float-exact and effectively always
        ``False`` on one.  Voxel selection goes through
        `bounding_box` plus the datastore's rounding rule.

        Parameters
        ----------
        points : np.ndarray
            ``(N, ndim)`` or ``(ndim,)`` points.

        Returns
        -------
        np.ndarray
            ``(N,)`` booleans, or a scalar boolean for 1-D input.
        """
        values = np.asarray(points, dtype=float)
        was_1d = values.ndim == 1
        if was_1d:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.ndim:
            raise ValueError(
                f"points must have shape (N, {self.ndim}) or ({self.ndim},), "
                f"got {np.asarray(points).shape}."
            )
        if not self.half_spaces:
            inside = np.ones(values.shape[0], dtype=bool)
        else:
            inside = (values @ self.normals.T <= self.offsets).all(axis=1)
        return inside[0] if was_1d else inside

    def is_empty(self) -> bool:
        """Whether the constraints are mutually infeasible."""
        return self._bounds() is None

    def simplify(self) -> ConvexRegion:
        """Drop constraints that have become vacuous.

        Vacuous constraints appear under composition: pulling a region
        back through a transform with no extent along some axis turns
        every constraint on that axis into a zero normal, which either
        excludes nothing or excludes everything.

        Constraints that are **infeasible** are kept, not dropped:
        dropping one would change an empty region into a non-empty one.
        Ask `is_empty` for that.

        Returns
        -------
        ConvexRegion
            A region with the same solution set and no vacuous
            constraints.
        """
        kept = tuple(
            half_space for half_space in self.half_spaces if not half_space.is_vacuous()
        )
        if len(kept) == len(self.half_spaces):
            return self
        return type(self)(
            coordinate_system=self.coordinate_system,
            ndim=self.ndim,
            half_spaces=kept,
        )

    # -- constructors -------------------------------------------------

    @classmethod
    def unbounded(cls, coordinate_system: CoordinateSystem) -> Self:
        """Return the region that is all of a coordinate system's space.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            The system the region lives in.  The object, not the id: the
            rank has to come from somewhere, and an empty constraint set
            does not carry it.

        Returns
        -------
        ConvexRegion
            A region with no constraints.
        """
        return cls(
            coordinate_system=coordinate_system.id,
            ndim=coordinate_system.ndim,
        )

    @classmethod
    def from_bounding_box(cls, box: AxisAlignedBoundingBox) -> Self:
        """Build the region equivalent to an axis-aligned box.

        Infinite bounds contribute no constraint, since ``p_i <= inf``
        excludes nothing (and `Plane` rejects an infinite
        offset).

        Parameters
        ----------
        box : AxisAlignedBoundingBox
            The box to convert.

        Returns
        -------
        ConvexRegion
            A region with up to ``2 * ndim`` constraints.
        """
        half_spaces: list[HalfSpace] = []
        for axis in range(box.ndim):
            upper = box.max_coordinate[axis]
            lower = box.min_coordinate[axis]
            if np.isfinite(upper):
                normal = np.zeros(box.ndim)
                normal[axis] = 1.0
                half_spaces.append(HalfSpace(normal=normal, offset=float(upper)))
            if np.isfinite(lower):
                normal = np.zeros(box.ndim)
                normal[axis] = -1.0
                half_spaces.append(HalfSpace(normal=normal, offset=float(-lower)))
        return cls(
            coordinate_system=box.coordinate_system,
            ndim=box.ndim,
            half_spaces=tuple(half_spaces),
        )

    @classmethod
    def from_axis_slabs(
        cls,
        coordinate_system: CoordinateSystem,
        slabs: Mapping[AxisRef, tuple[float, float]],
    ) -> Self:
        """Build a region from per-axis ``(centre, half_thickness)`` slabs.

        An axis absent from ``slabs`` is unbounded.  The region has no
        notion of "displayed" versus "collapsed": any axis may be
        bounded, and a displayed axis is bounded exactly as readily as a
        collapsed one.  That distinction lives in the transform, as the
        difference between a column and a constant column.

        A ``half_thickness`` of zero is allowed and produces a
        zero-thickness slab.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            The system the region lives in.
        slabs : Mapping[AxisRef, tuple[float, float]]
            ``{axis: (centre, half_thickness)}``, keyed by axis name or
            axis id.

        Returns
        -------
        ConvexRegion
            A region with two constraints per named axis.

        Raises
        ------
        ValueError
            If a half thickness is negative.
        """
        ndim = coordinate_system.ndim
        half_spaces: list[HalfSpace] = []
        for reference, (centre, half_thickness) in slabs.items():
            if half_thickness < 0:
                raise ValueError(
                    f"half_thickness must not be negative, got {half_thickness} "
                    f"for axis {reference!r}."
                )
            axis = coordinate_system.resolve(reference)
            upper_normal = np.zeros(ndim)
            upper_normal[axis] = 1.0
            half_spaces.append(
                HalfSpace(normal=upper_normal, offset=float(centre + half_thickness))
            )
            half_spaces.append(
                HalfSpace(
                    normal=-upper_normal, offset=float(-(centre - half_thickness))
                )
            )
        return cls(
            coordinate_system=coordinate_system.id,
            ndim=ndim,
            half_spaces=tuple(half_spaces),
        )

    @classmethod
    def from_plane_slab(
        cls,
        coordinate_system: CoordinateSystem,
        normal: np.ndarray,
        offset: float,
        half_thickness: float,
    ) -> Self:
        """Build an oblique slab of a given thickness about a plane.

        The slab is ``abs(normal . p - offset) <= half_thickness *
        norm(normal)``, so ``half_thickness`` is a distance in the
        coordinate system's units regardless of the normal's magnitude.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            The system the region lives in.
        normal : np.ndarray
            The plane normal.  Need not be unit length.
        offset : float
            The plane offset.
        half_thickness : float
            Half the slab thickness, as a distance.  Zero is allowed.

        Returns
        -------
        ConvexRegion
            A region with two opposed constraints.

        Raises
        ------
        ValueError
            If the normal is zero or the wrong rank, or the half
            thickness is negative.
        """
        values = reject_non_finite(coerce_float_array(normal), "normal")
        if values.shape != (coordinate_system.ndim,):
            raise ValueError(
                f"normal must have {coordinate_system.ndim} components, got "
                f"{values.shape}."
            )
        magnitude = float(np.linalg.norm(values))
        if magnitude == 0.0:
            raise ValueError("normal must not be the zero vector.")
        if half_thickness < 0:
            raise ValueError(
                f"half_thickness must not be negative, got {half_thickness}."
            )
        margin = half_thickness * magnitude
        return cls(
            coordinate_system=coordinate_system.id,
            ndim=coordinate_system.ndim,
            half_spaces=(
                HalfSpace(normal=values, offset=float(offset + margin)),
                HalfSpace(normal=-values, offset=float(-(offset - margin))),
            ),
        )

    @classmethod
    def intersection(cls, *regions: ConvexRegion) -> Self:
        """Intersect regions that share a coordinate system.

        Parameters
        ----------
        *regions : ConvexRegion
            At least one region, all in the same coordinate system.

        Returns
        -------
        ConvexRegion
            A region carrying every constraint of every input.

        Raises
        ------
        ValueError
            If no regions are given, or they do not all share a
            coordinate system and rank.
        """
        if not regions:
            raise ValueError(
                "intersection() needs at least one region; use "
                "ConvexRegion.unbounded(coordinate_system) for all of space."
            )
        first = regions[0]
        for region in regions[1:]:
            if region.coordinate_system != first.coordinate_system:
                raise ValueError(
                    "Cannot intersect regions in different coordinate systems: "
                    f"{first.coordinate_system} and {region.coordinate_system}."
                )
            if region.ndim != first.ndim:
                raise ValueError(
                    f"Cannot intersect regions of different rank: {first.ndim} "
                    f"and {region.ndim}."
                )
        half_spaces: list[HalfSpace] = []
        for region in regions:
            half_spaces.extend(region.half_spaces)
        return cls(
            coordinate_system=first.coordinate_system,
            ndim=first.ndim,
            half_spaces=tuple(half_spaces),
        )


def half_spaces_from_arrays(
    normals: np.ndarray, offsets: np.ndarray
) -> tuple[HalfSpace, ...]:
    """Build a tuple of half-spaces from stacked arrays.

    Parameters
    ----------
    normals : np.ndarray
        ``(M, D)`` constraint normals.
    offsets : np.ndarray
        ``(M,)`` constraint offsets.

    Returns
    -------
    tuple[HalfSpace, ...]
        One half-space per row.
    """
    return tuple(
        HalfSpace(normal=normal, offset=float(offset))
        for normal, offset in zip(
            np.atleast_2d(normals), np.atleast_1d(offsets), strict=True
        )
    )
