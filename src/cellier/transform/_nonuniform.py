"""A single irregularly-sampled, monotonic axis.

The motivating case is a time axis whose frames were not acquired at a
regular interval, but nothing here mentions time: an irregular ``z`` step or
any other monotonic, unevenly sampled axis uses the same class unchanged.
See ``plans/nonuniform_time_axis_transform_design.md``.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    model_validator,
)

# Imported at runtime, not under TYPE_CHECKING: pydantic resolves field
# annotations at class-creation time and cannot see a deferred import.
from transformnd.transforms.bijection import Bijection
from transformnd.transforms.grid import GridInterpolation
from typing_extensions import Self

from cellier._rounding import round_half_up_clamped
from cellier.transform._base import BaseTransform
from cellier.transform._geometry import AxisAlignedBoundingBox
from cellier.transform._geometry_ops import NonAffineTransformError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cellier.transform._axis import AxisRef
    from cellier.transform._coordinate_system import CoordinateSystem
    from cellier.transform._region import ConvexRegion

__all__ = ["AxisCoordinates", "NonUniformAxisTransform"]


class AxisCoordinates(BaseModel):
    """The sample positions of one irregularly-spaced axis.

    ``values`` are the world positions of the sample **centres** -- one per
    data index -- and ``edges`` are the world positions of the axis's two
    outer **edges**.  That is exactly the centre-versus-edge distinction a
    data store already makes: a gridded axis of ``size`` voxels spans
    ``(-0.5, size - 0.5)`` because a voxel is centred on its index, and this
    class is the same statement for an axis whose spacing is irregular.

    Keeping ``edges`` explicit is what lets a non-uniform axis answer the
    question every store answers::

        world extent on an axis = transform.map(store.axis_extents)

    Mapping a store's extent means mapping data index ``-0.5``, which linear
    interpolation cannot extrapolate to from ``values`` alone.

    Parameters
    ----------
    values : tuple[float, ...]
        World position of each sample centre, strictly increasing.  A tuple
        rather than an ndarray so that equality and hashing stay total --
        an ndarray field makes ``__eq__`` return an array, which degrades
        every containing model's comparison.
    edges : tuple[float, float] or None
        World position of the axis's outer edges.  ``None`` (the default)
        extrapolates by half the gap at each end, which is the choice that
        gives the first and last samples a **full** nearest-neighbour
        catchment rather than a half one.  The padding is naturally
        asymmetric when the first and last gaps differ.

        Must contain ``values``: ``edges[0] <= values[0]`` and
        ``edges[1] >= values[-1]``.  A narrower span would make samples
        unreachable, which is silent data loss; shorten ``values`` instead.
    """

    model_config = ConfigDict(frozen=True)

    values: tuple[float, ...]
    edges: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """Require a strictly increasing table and containing edges."""
        if len(self.values) == 0:
            raise ValueError("AxisCoordinates needs at least one value.")

        array = np.asarray(self.values, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Every value must be finite; got {self.values}.")
        if array.size > 1 and not np.all(np.diff(array) > 0):
            # Not sorted-for-you: numpy's interpolation returns a wrong
            # answer silently on a non-monotonic table, so the mistake would
            # surface as a wrong frame far from where it was made.
            raise ValueError(
                f"AxisCoordinates.values must be strictly increasing; got "
                f"{self.values}.  It is not sorted for you: a table given in "
                f"the wrong order would otherwise change which sample a "
                f"position resolves to, silently."
            )

        if self.edges is None:
            if array.size == 1:
                raise ValueError(
                    "A single-value AxisCoordinates has no gap to extrapolate "
                    "from, so `edges` must be given explicitly."
                )
        else:
            low, high = self.edges
            if not (np.isfinite(low) and np.isfinite(high)):
                raise ValueError(f"Both edges must be finite; got {self.edges}.")
            if low > array[0] or high < array[-1]:
                raise ValueError(
                    f"edges {self.edges} must contain values "
                    f"[{array[0]}, {array[-1]}].  A narrower span would make "
                    f"samples unreachable; shorten `values` instead."
                )
        return self

    @cached_property
    def resolved_edges(self) -> tuple[float, float]:
        """The outer edges, extrapolated by half a gap when not given."""
        if self.edges is not None:
            return self.edges
        array = np.asarray(self.values, dtype=float)
        return (
            float(array[0] - (array[1] - array[0]) / 2.0),
            float(array[-1] + (array[-1] - array[-2]) / 2.0),
        )

    @cached_property
    def values_array(self) -> np.ndarray:
        """``values`` as a read-only float array."""
        array = np.asarray(self.values, dtype=float)
        array.flags.writeable = False
        return array

    @property
    def n_samples(self) -> int:
        """Number of samples on the axis."""
        return len(self.values)

    @cached_property
    def index_domain(self) -> tuple[float, float]:
        """The valid span in data-index units, ``(-0.5, n_samples - 0.5)``.

        The same edge convention a gridded store reports, so a leaf and a
        grid answer ``axis_extents`` in the same units.
        """
        return (-0.5, float(self.n_samples) - 0.5)

    @cached_property
    def _forward_table(self) -> tuple[np.ndarray, np.ndarray]:
        """``(index, world)`` including both outer edges.

        The index column is strictly increasing by construction, which is
        all ``numpy.interp`` requires of its ``xp``.
        """
        low_edge, high_edge = self.resolved_edges
        index_low, index_high = self.index_domain
        indices = np.concatenate(
            [[index_low], np.arange(self.n_samples, dtype=float), [index_high]]
        )
        world = np.concatenate([[low_edge], self.values_array, [high_edge]])
        return indices, world

    @cached_property
    def _inverse_table(self) -> tuple[np.ndarray, np.ndarray]:
        """``(world, index)`` with any zero-width edge segment dropped.

        ``edges`` is allowed to sit exactly on the first or last sample,
        which makes that outer half-segment zero-width in world units.
        ``numpy.interp`` needs a strictly increasing ``xp``, so such a point
        is dropped; a query landing on it then resolves to the sample rather
        than to the edge, which is the answer that points at data.
        """
        indices, world = self._forward_table
        keep = np.concatenate([[True], np.diff(world) > 0])
        return world[keep], indices[keep]


class NonUniformAxisTransform(BaseTransform):
    """A 1-D transform between data indices and an irregular world axis.

    Wraps ``Bijection(GridInterpolation, GridInterpolation)`` so the model
    satisfies :class:`~cellier.transform._base.BaseTransform`'s
    ``transform: Transform`` field and reports a ``1 -> 1`` dimensionality
    that ``validate_against`` can check.  The wrapped object is **not** what
    executes: the mapping is done here, because ``GridInterpolation.apply``
    allocates its output with ``zeros_like(coords)`` and so silently
    truncates an integer input array.

    **Out of range does not clamp.**  Outside the axis's span the result is
    ``nan`` -- there is no preimage, and a store has no data there.  Earlier
    revisions of the design clamped to the edge sample, which meant an
    acquisition ending at 9 s re-showed its 9 s frame at 10, 11 and 12 s as
    though it were data.  ``nan`` rather than an exception because a call
    maps a whole array and some of its points may be in range while others
    are not.

    **Point and interval queries differ deliberately.**
    :meth:`map_coordinates` / :meth:`imap_coordinates` are point queries and
    return ``nan`` outside the span.  :meth:`map_bounding_box` /
    :meth:`imap_bounding_box` are interval queries and **intersect** with the
    span, so an interval that merely overhangs an end comes back clipped
    rather than empty.  Both are correct: a slice position outside the data
    selects nothing, while a trail window reaching back past the first frame
    should still start at the first frame.  Do not "simplify" the two into
    one behaviour -- collapsing to point semantics makes every trail window
    vanish near the start of an axis.

    Parameters
    ----------
    coordinates : AxisCoordinates
        The sample centres and outer edges.
    interpolation : Literal["linear", "nearest"]
        How a position between samples is mapped.  Defaults to ``"linear"``,
        which is what makes this a continuous coordinate map.

        Note this differs from cellier's ``"nearest"`` default for *image
        sampling*, and deliberately: that convention is about choosing a
        sample to read, while this is about mapping a position.  The
        pipeline's nearest-ness is applied one layer up, by
        ``round_world_to_voxel`` -- and a linear inverse followed by that
        rounding is exactly nearest-neighbour in world units, verified
        across the whole domain.  Choosing ``"nearest"`` here instead snaps
        positions to sample centres, which also collapses the axis's extent
        onto ``(values[0], values[-1])``; that is self-consistent, but it is
        not the continuous map the slicing path expects.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    transform_type: Literal["nonuniform_axis"] = "nonuniform_axis"
    coordinates: AxisCoordinates
    interpolation: Literal["linear", "nearest"] = "linear"
    transform: Bijection

    # -- construction --------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _derive_transform(cls, data: Any) -> Any:
        """Build the wrapped ``Bijection`` from ``coordinates``.

        The wrapped object is derived, never authored, so a caller never
        has to construct one and a deserialized model rebuilds it rather
        than round-tripping it.
        """
        if not isinstance(data, dict):
            return data
        coordinates = data.get("coordinates")
        if coordinates is None:
            return data
        if not isinstance(coordinates, AxisCoordinates):
            coordinates = AxisCoordinates.model_validate(coordinates)
            data["coordinates"] = coordinates
        interpolation = data.get("interpolation", "linear")

        def index_to_world(values: np.ndarray) -> np.ndarray:
            return _map_index_to_world(coordinates, values, interpolation)

        def world_to_index(values: np.ndarray) -> np.ndarray:
            return _map_world_to_index(coordinates, values, interpolation)

        data["transform"] = Bijection(
            GridInterpolation([index_to_world]),
            GridInterpolation([world_to_index]),
        )
        return data

    @field_serializer("transform")
    def _serialize_transform(self, value: Bijection) -> None:
        """Emit nothing: the wrapped object is rebuilt from ``coordinates``."""
        return None

    # -- points ---------------------------------------------------------

    def map_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Map data indices to world positions.

        Accepts ``(N, 1)`` or ``(1,)`` and returns matching rank.  Positions
        outside ``[-0.5, n_samples - 0.5]`` come back as ``nan``.

        Parameters
        ----------
        coordinates : np.ndarray
            Data indices.

        Returns
        -------
        np.ndarray
            World positions, ``nan`` where there is no preimage.
        """
        values, was_1d = _as_column(coordinates, "coordinates")
        mapped = _map_index_to_world(self.coordinates, values, self.interpolation)
        return _restore(mapped, was_1d)

    def imap_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Map world positions back to data indices.

        Positions outside the axis's world span come back as ``nan``: the
        data does not reach there, and clamping to the edge sample would
        present the last acquired plane as though it were data.

        Parameters
        ----------
        coordinates : np.ndarray
            World positions.

        Returns
        -------
        np.ndarray
            Data indices, ``nan`` where there is no preimage.
        """
        values, was_1d = _as_column(coordinates, "coordinates")
        mapped = _map_world_to_index(self.coordinates, values, self.interpolation)
        return _restore(mapped, was_1d)

    # -- bounding boxes: interval semantics, so they intersect ----------

    def map_bounding_box(
        self,
        box: AxisAlignedBoundingBox,
        output_coordinate_system: CoordinateSystem,
    ) -> AxisAlignedBoundingBox:
        """Map an index interval forward, exactly.

        A monotonic map takes an interval's endpoints to the result's
        endpoints, so this is exact rather than merely conservative.  The
        interval is **intersected** with the axis's span rather than
        rejected when it overhangs -- see the class docstring on point
        versus interval semantics.

        Parameters
        ----------
        box : AxisAlignedBoundingBox
            A 1-D box in the input coordinate system.
        output_coordinate_system : CoordinateSystem
            The system to express the result in.

        Returns
        -------
        AxisAlignedBoundingBox
            The mapped interval, in the output coordinate system.
        """
        self._check_input_coordinate_system(box.coordinate_system)
        low, high = _interval(box)
        index_low, index_high = self.coordinates.index_domain
        mapped_low, mapped_high = _map_interval(
            low,
            high,
            index_low,
            index_high,
            lambda values: _map_index_to_world(
                self.coordinates, values, self.interpolation
            ),
            *self.coordinates.resolved_edges,
        )
        return AxisAlignedBoundingBox(
            coordinate_system=output_coordinate_system.id,
            min_coordinate=np.array([mapped_low]),
            max_coordinate=np.array([mapped_high]),
        )

    def imap_bounding_box(self, box: AxisAlignedBoundingBox) -> AxisAlignedBoundingBox:
        """Map a world interval back, exactly.

        Intersected with the axis's world span, not rejected: this is the
        call a trail window's endpoints go through, and a window reaching
        past the first sample must still start at the first sample.

        Parameters
        ----------
        box : AxisAlignedBoundingBox
            A 1-D box in the output coordinate system.

        Returns
        -------
        AxisAlignedBoundingBox
            The interval in the input coordinate system.
        """
        self._check_output_coordinate_system(box.coordinate_system)
        low, high = _interval(box)
        world_low, world_high = self.coordinates.resolved_edges
        mapped_low, mapped_high = _map_interval(
            low,
            high,
            world_low,
            world_high,
            lambda values: _map_world_to_index(
                self.coordinates, values, self.interpolation
            ),
            *self.coordinates.index_domain,
        )
        return AxisAlignedBoundingBox(
            coordinate_system=self.input_coordinate_system,
            min_coordinate=np.array([mapped_low]),
            max_coordinate=np.array([mapped_high]),
        )

    # -- the operations a non-affine axis cannot answer -----------------

    def map_direction(self, direction: np.ndarray) -> np.ndarray:
        """Not answerable: the Jacobian varies along the axis."""
        raise NonAffineTransformError(self._no_jacobian("map_direction"))

    def imap_direction(self, direction: np.ndarray) -> np.ndarray:
        """Not answerable: the Jacobian varies along the axis."""
        raise NonAffineTransformError(self._no_jacobian("imap_direction"))

    def map_normal(self, normal: np.ndarray) -> np.ndarray:
        """Not answerable: the Jacobian varies along the axis."""
        raise NonAffineTransformError(self._no_jacobian("map_normal"))

    def imap_normal(self, normal: np.ndarray) -> np.ndarray:
        """Not answerable: the Jacobian varies along the axis."""
        raise NonAffineTransformError(self._no_jacobian("imap_normal"))

    def map_plane(self, plane):
        """Not answerable: a plane needs a spatially constant Jacobian."""
        raise NonAffineTransformError(self._no_jacobian("map_plane"))

    def imap_plane(self, plane):
        """Not answerable: a plane needs a spatially constant Jacobian."""
        raise NonAffineTransformError(self._no_jacobian("imap_plane"))

    def map_region(self, region: ConvexRegion) -> ConvexRegion:
        """Not answerable: a half-space needs a constant normal."""
        raise NonAffineTransformError(self._no_jacobian("map_region"))

    def imap_region(
        self,
        region: ConvexRegion,
        output_coordinate_system: CoordinateSystem,
    ) -> ConvexRegion:
        """Not answerable: a half-space needs a constant normal."""
        raise NonAffineTransformError(self._no_jacobian("imap_region"))

    def _no_jacobian(self, operation: str) -> str:
        """The message every direction/plane/region refusal shares."""
        return (
            f"{operation} is not defined for a non-uniform axis: the spacing "
            f"between samples varies, so there is no single Jacobian to map "
            f"a direction, normal or half-space by.  Use the bounding-box "
            f"methods, which are exact for a monotonic axis, or put this "
            f"axis inside a ByDimensionTransform where the affine axes keep "
            f"their own answers."
        )

    def input_domain(self) -> dict[int, tuple[float, float]]:
        """The table's index span, ``(-0.5, n_samples - 0.5)``.

        The same edge convention a gridded store reports: sample ``i`` is
        centred on integer ``i`` and its cell reaches half a unit either
        side, so the axis as a whole spans half a unit past its end samples.

        Returns
        -------
        dict[int, tuple[float, float]]
            ``{0: (low, high)}``.
        """
        return {0: self.coordinates.index_domain}

    def axis_coupling(self) -> np.ndarray:
        """The leaf is 1-D: its one input axis reaches its one output axis.

        Returns
        -------
        np.ndarray
            ``[[True]]``.
        """
        return np.ones((1, 1), dtype=bool)

    def axis_correspondence(self) -> dict[int, int]:
        """The leaf is 1-D, so its one input axis reaches its one output axis.

        Structural, with no matrix involved -- which is the point: a
        non-affine transform can still say which axis becomes which.

        Returns
        -------
        dict[int, int]
            Always ``{0: 0}``.
        """
        return {0: 0}

    # -- inversion, restriction, affine-ness ----------------------------

    def inverse(self) -> BaseTransform | None:
        """No inverse *as a transform of this class*.

        The map is invertible -- :meth:`imap_coordinates` performs the
        inversion -- but its inverse is not itself a table of world
        positions per data index, which is what this class models.  Rather
        than invent a second class nothing yet consumes, this returns
        ``None`` and callers use the ``imap_*`` methods.

        Returns
        -------
        None
            Always.
        """
        return None

    def to_affine(self) -> None:
        """Never affine.

        Returns
        -------
        None
            Always.  An irregular table has no matrix, which is the whole
            reason this class exists.
        """
        return None

    def restrict(
        self,
        fixed: Mapping[AxisRef | int, float],
        input_coordinate_system: CoordinateSystem | None = None,
    ) -> BaseTransform:
        """Always raises; a 1-D leaf has nothing left after being fixed.

        Fixing this transform's only axis leaves a zero-dimensional
        transform, which is not modelled.  Collapsing a fully-fixed axis to
        a constant is the *container's* job:
        ``ByDimensionTransform.restrict`` evaluates such a block directly
        rather than delegating here.

        Parameters
        ----------
        fixed : Mapping[AxisRef | int, float]
            Ignored.
        input_coordinate_system : CoordinateSystem or None
            Ignored.

        Raises
        ------
        NonAffineTransformError
            Always.
        """
        raise NonAffineTransformError(
            "A NonUniformAxisTransform cannot be restricted on its own: "
            "fixing its only axis would leave a zero-dimensional transform, "
            "which is not modelled.  Put it in a ByDimensionTransform, which "
            "evaluates a fully-fixed block to a constant and folds it into "
            "the remaining affine axes."
        )

    # -- equality (mirrors AffineTransform) -----------------------------

    def __eq__(self, other: object) -> bool:
        """Compare by endpoints, table and interpolation, not by id.

        Defined explicitly because the wrapped ``transformnd`` classes
        define no ``__eq__`` at all: two independently built leaves over
        the same table would otherwise compare unequal by object identity.
        """
        if not isinstance(other, NonUniformAxisTransform):
            return NotImplemented
        return (
            self.input_coordinate_system == other.input_coordinate_system
            and self.output_coordinate_system == other.output_coordinate_system
            and self.coordinates == other.coordinates
            and self.interpolation == other.interpolation
        )

    def __hash__(self) -> int:
        """Hash by endpoints, table and interpolation."""
        return hash(
            (
                self.input_coordinate_system,
                self.output_coordinate_system,
                self.coordinates.values,
                self.coordinates.resolved_edges,
                self.interpolation,
            )
        )


# ---------------------------------------------------------------------
# the mapping itself
# ---------------------------------------------------------------------


def _map_index_to_world(
    coordinates: AxisCoordinates,
    values: np.ndarray,
    interpolation: str,
) -> np.ndarray:
    """Data indices to world positions, ``nan`` outside the axis's span."""
    indices = np.asarray(values, dtype=float)
    low, high = coordinates.index_domain
    inside = (indices >= low) & (indices <= high)

    if interpolation == "nearest":
        table = coordinates.values_array
        snapped = np.array(
            [
                table[round_half_up_clamped(float(value), coordinates.n_samples)]
                for value in np.ravel(indices)
            ],
            dtype=float,
        ).reshape(indices.shape)
        return np.where(inside, snapped, np.nan)

    table_index, table_world = coordinates._forward_table
    return np.where(inside, np.interp(indices, table_index, table_world), np.nan)


def _map_world_to_index(
    coordinates: AxisCoordinates,
    values: np.ndarray,
    interpolation: str,
) -> np.ndarray:
    """World positions to data indices, ``nan`` outside the axis's span."""
    world = np.asarray(values, dtype=float)
    low, high = coordinates.resolved_edges
    inside = (world >= low) & (world <= high)

    if interpolation == "nearest":
        table = coordinates.values_array
        # Ties go to the higher index, matching round-half-up.
        distances = np.abs(table[np.newaxis, :] - np.ravel(world)[:, np.newaxis])
        nearest = (
            (table.size - 1 - np.argmin(distances[:, ::-1], axis=1))
            .astype(float)
            .reshape(world.shape)
        )
        return np.where(inside, nearest, np.nan)

    table_world, table_index = coordinates._inverse_table
    return np.where(inside, np.interp(world, table_world, table_index), np.nan)


def _map_interval(
    low: float,
    high: float,
    domain_low: float,
    domain_high: float,
    mapper,
    image_low: float,
    image_high: float,
) -> tuple[float, float]:
    """Map an interval, intersecting it with the map's domain first.

    An infinite bound is not clipped to the domain but carried through to
    the corresponding end of the image, because an unbounded axis is a
    first-class value here: it means "this region does not constrain the
    axis", not "this region reaches to the edge of the data".
    """
    mapped_low = (
        image_low
        if np.isneginf(low)
        else float(mapper(np.array([min(max(low, domain_low), domain_high)]))[0])
    )
    mapped_high = (
        image_high
        if np.isposinf(high)
        else float(mapper(np.array([min(max(high, domain_low), domain_high)]))[0])
    )
    return mapped_low, mapped_high


def _interval(box: AxisAlignedBoundingBox) -> tuple[float, float]:
    """Read a 1-D box's single interval, rejecting any other rank."""
    if box.min_coordinate.shape != (1,):
        raise ValueError(
            f"A NonUniformAxisTransform maps one axis, so it needs a 1-D "
            f"bounding box; got rank {box.min_coordinate.shape[0]}."
        )
    return float(box.min_coordinate[0]), float(box.max_coordinate[0])


def _as_column(array: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    """Normalize ``(N, 1)`` or ``(1,)`` input to ``(N,)``, tracking rank."""
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        if values.shape[0] != 1:
            raise ValueError(
                f"{name} must have one component per point for a 1-D "
                f"transform; got shape {values.shape}."
            )
        return values, True
    if values.ndim == 2:
        if values.shape[1] != 1:
            raise ValueError(
                f"{name} must have exactly one column for a 1-D transform; "
                f"got shape {values.shape}."
            )
        return values[:, 0], False
    raise ValueError(f"{name} must be 1-D or 2-D; got shape {values.shape}.")


def _restore(values: np.ndarray, was_1d: bool) -> np.ndarray:
    """Put a mapped column back into the rank its input had."""
    return values if was_1d else values[:, np.newaxis]


def coordinates_from_sequence(
    values: Sequence[float], edges: tuple[float, float] | None = None
) -> AxisCoordinates:
    """Build :class:`AxisCoordinates` from any sequence of positions.

    Parameters
    ----------
    values : Sequence[float]
        World position of each sample centre, strictly increasing.
    edges : tuple[float, float] or None
        The axis's outer edges, or ``None`` to extrapolate by half a gap.

    Returns
    -------
    AxisCoordinates
        The validated table.
    """
    return AxisCoordinates(values=tuple(float(value) for value in values), edges=edges)
