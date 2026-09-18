"""The affine transform model (design section 9)."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import UUID4, field_serializer, field_validator

# Affine is imported at runtime, not under TYPE_CHECKING: pydantic resolves
# field annotations at class-creation time and cannot see a deferred import.
from transformnd.transforms.affine import Affine
from typing_extensions import Self

from cellier.transform import _geometry_ops as ops
from cellier.transform._base import BaseTransform
from cellier.transform._geometry import AxisAlignedBoundingBox, Plane
from cellier.transform._geometry_ops import NonInvertibleTransformError
from cellier.transform._region import ConvexRegion, half_spaces_from_arrays

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cellier.transform._axis import AxisRef
    from cellier.transform._coordinate_system import CoordinateSystem


class AffineTransform(BaseTransform):
    """An affine transform between two coordinate systems.

    The matrix is ``(D_out + 1, D_in + 1)``; non-square is normal, not an
    edge case.  It is stored at float64 (D20): narrowing to float32 is
    the renderer's job at upload, and chained affines across several
    coordinate systems are exactly where float32 error accumulates.

    Composition is :meth:`then`.  There is no ``__matmul__`` and no
    ``compose``: ``transformnd.Affine.__matmul__`` applies its *right*
    operand first, the wrapped affine is reachable through
    ``.transform``, and having ``a.transform @ b.transform`` and a
    cellier-level composition mean opposite things in the same file is a
    trap worth closing with a ``TypeError``.

    Parameters
    ----------
    transform_type : Literal["affine"]
        Discriminator field.
    transform : Affine
        The wrapped affine.  A matrix-like is accepted and coerced.
    broadcast_axes : frozenset[UUID4]
        Ids of output axes the input has no extent along -- a dataset
        broadcast over a channel axis, say.  A zero matrix row is
        indistinguishable from "constant zero", and the two differ for
        extents: a broadcast dataset occupies *all* of that axis in the
        output space, not the point ``{0}``.  Recording it is not
        reconstructible later (D25).
    """

    transform_type: Literal["affine"] = "affine"
    transform: Affine
    broadcast_axes: frozenset[UUID4] = frozenset()

    @field_validator("transform", mode="before")
    @classmethod
    def _coerce_affine(cls, value: Any) -> Affine:
        """Accept an ``Affine`` or any ``(D_out + 1, D_in + 1)`` matrix-like."""
        matrix = value.matrix if isinstance(value, Affine) else value
        matrix = np.asarray(matrix, dtype=float)
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"An affine matrix must be finite; got {matrix.tolist()}.")
        return value if isinstance(value, Affine) else Affine(matrix)

    @field_serializer("transform")
    def _serialize_affine(self, value: Affine) -> list[list[float]]:
        """Emit the matrix as a nested list of floats (D17)."""
        return value.matrix.tolist()

    # -- matrix views --------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        """The full ``(D_out + 1, D_in + 1)`` augmented matrix."""
        return self.transform.matrix

    @property
    def linear(self) -> np.ndarray:
        """The ``(D_out, D_in)`` linear block."""
        return self.transform.matrix[:-1, :-1]

    @property
    def translation(self) -> np.ndarray:
        """The ``(D_out,)`` translation column."""
        return self.transform.matrix[:-1, -1]

    # -- the inverse (D24) ---------------------------------------------

    @cached_property
    def _inverse_blocks(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The inverse linear block and translation, or ``None`` (D24).

        Cached so that repeated inverse mapping does not re-invert the
        matrix.
        """
        n_out, n_in = self.linear.shape
        if n_out == n_in:
            inverted = self.transform.invert()
            if inverted is None:
                return None
            return inverted.matrix[:-1, :-1], inverted.matrix[:-1, -1]
        return ops.pseudo_inverse_affine(self.linear, self.translation)

    def _require_inverse(self, operation: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the inverse blocks or raise.

        Parameters
        ----------
        operation : str
            The method name, used in the error message.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The inverse linear block and translation.

        Raises
        ------
        NonInvertibleTransformError
            If no left inverse exists.
        """
        blocks = self._inverse_blocks
        if blocks is None:
            raise NonInvertibleTransformError(
                f"{operation} requires an inverse, but this "
                f"{self.input_ndim}D -> {self.output_ndim}D transform has "
                f"none: it is rank deficient or loses a dimension, so no left "
                f"inverse exists (D24 case 3)."
            )
        return blocks

    def inverse(self) -> AffineTransform | None:
        """Return the inverse transform, or ``None``.

        Three cases (D24): exact for a non-singular square matrix; the
        pseudo-inverse of the linear block for a full-column-rank
        embedding, which is an exact **left** inverse; ``None``
        otherwise.  A dimension-reducing transform is not silently handed
        a right inverse, which would return a different point than the
        caller put in.

        **The coordinate system ids are swapped.**  That is the whole
        point, and it is where ``transformnd.Spaced.invert()`` goes
        wrong: it inverts the inner transform but returns the original
        space order, so the result claims a direction it does not
        perform.

        ``broadcast_axes`` is empty on the result: those name axes of the
        forward transform's *output* system, which the inverse maps from,
        not to.

        Returns
        -------
        AffineTransform or None
            The inverse, or ``None`` when there is no left inverse.
        """
        blocks = self._inverse_blocks
        if blocks is None:
            return None
        inverse_linear, inverse_translation = blocks
        n_in, n_out = inverse_linear.shape
        matrix = np.zeros((n_in + 1, n_out + 1))
        matrix[:n_in, :n_out] = inverse_linear
        matrix[:n_in, n_out] = inverse_translation
        matrix[n_in, n_out] = 1.0
        return type(self)(
            name=self.name,
            transform=Affine(matrix),
            input_coordinate_system=self.output_coordinate_system,
            output_coordinate_system=self.input_coordinate_system,
        )

    # -- points --------------------------------------------------------

    def map_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Map points from the input system to the output system.

        Accepts ``(N, D_in)`` or ``(D_in,)`` and returns matching rank.
        Homogeneous ``(N, D_in + 1)`` input is rejected (D12).  The
        result is always a distinct array, even for an identity affine,
        where ``transformnd`` would return the input object (D11).

        Parameters
        ----------
        coordinates : np.ndarray
            Points in the input coordinate system.

        Returns
        -------
        np.ndarray
            Points in the output coordinate system.
        """
        return ops.map_points(coordinates, self.linear, self.translation)

    def imap_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Map points from the output system back to the input system.

        Parameters
        ----------
        coordinates : np.ndarray
            Points in the output coordinate system.

        Returns
        -------
        np.ndarray
            Points in the input coordinate system.

        Raises
        ------
        NonInvertibleTransformError
            If this transform has no left inverse (D24 case 3).
        """
        inverse_linear, _ = self._require_inverse("imap_coordinates")
        return ops.imap_points(coordinates, inverse_linear, self.translation)

    # -- vectors (D27, D28) --------------------------------------------

    def map_direction(self, direction: np.ndarray) -> np.ndarray:
        """Map displacement vectors forward by ``A``.

        Needs no inverse, so it works on a transform whose
        :meth:`inverse` is ``None``.  Nothing is normalized (D28).

        Parameters
        ----------
        direction : np.ndarray
            Vectors in the input coordinate system.

        Returns
        -------
        np.ndarray
            Vectors in the output coordinate system.
        """
        return ops.map_directions(direction, self.linear)

    def imap_direction(self, direction: np.ndarray) -> np.ndarray:
        """Map displacement vectors back by ``A+``.

        Parameters
        ----------
        direction : np.ndarray
            Vectors in the output coordinate system.

        Returns
        -------
        np.ndarray
            Vectors in the input coordinate system.

        Raises
        ------
        NonInvertibleTransformError
            If this transform has no left inverse.
        """
        inverse_linear, _ = self._require_inverse("imap_direction")
        return ops.imap_directions(direction, inverse_linear)

    def map_normal(self, normal: np.ndarray) -> np.ndarray:
        """Map plane normals forward by ``(A+)^T`` (covariant, D27).

        A normal is not a displacement: it transforms by the inverse
        transpose, and using ``A`` is correct only for a rigid transform.
        Nothing is normalized (D28).

        Parameters
        ----------
        normal : np.ndarray
            Normals in the input coordinate system.

        Returns
        -------
        np.ndarray
            Normals in the output coordinate system.

        Raises
        ------
        NonInvertibleTransformError
            If this transform has no left inverse.
        DegenerateNormalError
            If a normal maps to the zero vector (D32).
        """
        inverse_linear, _ = self._require_inverse("map_normal")
        return ops.map_normals(normal, inverse_linear)

    def imap_normal(self, normal: np.ndarray) -> np.ndarray:
        """Map plane normals back by ``A^T`` (covariant, D27).

        Needs no inverse, so it works on a transform whose
        :meth:`inverse` is ``None``.  Not injective: a normal tilted
        along an axis this transform has no extent in comes back with
        that tilt discarded.

        Parameters
        ----------
        normal : np.ndarray
            Normals in the output coordinate system.

        Returns
        -------
        np.ndarray
            Normals in the input coordinate system.

        Raises
        ------
        DegenerateNormalError
            If a normal maps to the zero vector (D32).
        """
        return ops.imap_normals(normal, self.linear)

    # -- planes --------------------------------------------------------

    def map_plane(self, plane: Plane) -> Plane:
        """Map a plane from the input system to the output system.

        Parameters
        ----------
        plane : Plane
            A plane in the input coordinate system.

        Returns
        -------
        Plane
            The mapped plane, in the output coordinate system.

        Raises
        ------
        ValueError
            If the plane is not in this transform's input system.
        NonInvertibleTransformError
            If this transform has no left inverse.
        DegenerateNormalError
            If the normal maps to the zero vector (D32).
        """
        self._check_input_coordinate_system(plane.coordinate_system)
        inverse_linear, _ = self._require_inverse("map_plane")
        normal, offset = ops.map_plane(
            plane.normal, plane.offset, inverse_linear, self.translation
        )
        return Plane(
            coordinate_system=self.output_coordinate_system,
            normal=normal,
            offset=offset,
        )

    def imap_plane(self, plane: Plane) -> Plane:
        """Map a plane from the output system back to the input system.

        Needs no inverse.  Not injective (see :meth:`imap_normal`), so
        ``map_plane(imap_plane(p)) == p`` does not hold in general.

        Parameters
        ----------
        plane : Plane
            A plane in the output coordinate system.

        Returns
        -------
        Plane
            The pulled-back plane, in the input coordinate system.

        Raises
        ------
        ValueError
            If the plane is not in this transform's output system.
        DegenerateNormalError
            If the normal maps to the zero vector (D32).
        """
        self._check_output_coordinate_system(plane.coordinate_system)
        normal, offset = ops.imap_plane(
            plane.normal, plane.offset, self.linear, self.translation
        )
        return Plane(
            coordinate_system=self.input_coordinate_system,
            normal=normal,
            offset=offset,
        )

    # -- bounding boxes (D30, D31) -------------------------------------

    def map_bounding_box(
        self,
        box: AxisAlignedBoundingBox,
        output_coordinate_system: CoordinateSystem,
    ) -> AxisAlignedBoundingBox:
        """Map a box forward, conservatively.

        The result is the smallest axis-aligned box *enclosing* the
        image, which is a superset unless the linear block is diagonal or
        a signed permutation.  Every axis in ``broadcast_axes`` comes back
        ``(-inf, +inf)`` (D31).

        The output coordinate system is required because
        ``broadcast_axes`` holds axis **ids** and the arithmetic needs
        their indices, which only the system can supply.

        Parameters
        ----------
        box : AxisAlignedBoundingBox
            A box in the input coordinate system.
        output_coordinate_system : CoordinateSystem
            This transform's output system, used to resolve
            ``broadcast_axes`` to axis indices.

        Returns
        -------
        AxisAlignedBoundingBox
            The enclosing box, in the output coordinate system.

        Raises
        ------
        ValueError
            If the box is not in this transform's input system, or the
            given system is not this transform's output system.
        """
        self._check_input_coordinate_system(box.coordinate_system)
        if output_coordinate_system.id != self.output_coordinate_system:
            raise ValueError(
                f"output_coordinate_system must be this transform's output "
                f"system {self.output_coordinate_system}, got "
                f"{output_coordinate_system.id}."
            )
        unbounded = tuple(
            sorted(
                output_coordinate_system.index_of(axis_id)
                for axis_id in self.broadcast_axes
            )
        )
        lower, upper = ops.affine_bounding_box(
            box.min_coordinate,
            box.max_coordinate,
            self.linear,
            self.translation,
            unbounded,
        )
        return AxisAlignedBoundingBox(
            coordinate_system=self.output_coordinate_system,
            min_coordinate=lower,
            max_coordinate=upper,
        )

    def imap_bounding_box(self, box: AxisAlignedBoundingBox) -> AxisAlignedBoundingBox:
        """Map a box back, conservatively.

        Needs no broadcast handling (D31): the pseudo-inverse has an
        all-zero column for a broadcast axis, so that axis's extent is
        dropped whether it is finite or infinite.

        Parameters
        ----------
        box : AxisAlignedBoundingBox
            A box in the output coordinate system.

        Returns
        -------
        AxisAlignedBoundingBox
            The enclosing box, in the input coordinate system.

        Raises
        ------
        ValueError
            If the box is not in this transform's output system.
        NonInvertibleTransformError
            If this transform has no left inverse.
        """
        self._check_output_coordinate_system(box.coordinate_system)
        inverse_linear, inverse_translation = self._require_inverse("imap_bounding_box")
        lower, upper = ops.affine_bounding_box(
            box.min_coordinate,
            box.max_coordinate,
            inverse_linear,
            inverse_translation,
        )
        return AxisAlignedBoundingBox(
            coordinate_system=self.input_coordinate_system,
            min_coordinate=lower,
            max_coordinate=upper,
        )

    # -- regions (D39, D41) --------------------------------------------

    def map_region(self, region: ConvexRegion) -> ConvexRegion:
        """Map a convex region forward.

        Parameters
        ----------
        region : ConvexRegion
            A region in the input coordinate system.

        Returns
        -------
        ConvexRegion
            The mapped region, in the output coordinate system.

        Raises
        ------
        ValueError
            If the region is not in this transform's input system.
        NonInvertibleTransformError
            If this transform has no left inverse.
        """
        self._check_input_coordinate_system(region.coordinate_system)
        inverse_linear, _ = self._require_inverse("map_region")
        normals, offsets = ops.map_half_spaces(
            region.normals, region.offsets, inverse_linear, self.translation
        )
        return ConvexRegion(
            coordinate_system=self.output_coordinate_system,
            ndim=self.output_ndim,
            half_spaces=half_spaces_from_arrays(normals, offsets),
        )

    def imap_region(
        self,
        region: ConvexRegion,
        output_coordinate_system: CoordinateSystem,
    ) -> ConvexRegion:
        """Map a convex region back into the input coordinate system.

        This is the operation the slicer runs, and it needs nothing but
        ``A^T`` -- exact, cheap, and available on a transform whose
        :meth:`inverse` is ``None`` (D39).

        The result is a region, **never** a collapsed bounding box
        (D41).  Reducing to a box at the world level costs about 6x the
        voxels for a 45-degree slab, and the datastore wants both: the
        box for chunk selection and the constraints for the per-voxel
        mask.  Callers that want the box ask for it themselves.

        Constraints on a **broadcast** axis are dropped before ``A^T`` is
        applied (D8).  A broadcast axis is a free variable -- the source
        exists at *every* position on it -- so such a constraint can
        always be satisfied by moving along that axis.  This is exactly
        Fourier-Motzkin elimination of the broadcast axes specialised to
        the axis-aligned case: the single ``+b`` / ``-b`` pair combines
        to ``0 <= 2 * half_thickness``, which is trivially true.  The
        general pairwise combination is deferred until a consumer exists
        that tilts an oblique plane *through* a broadcast axis **and**
        bounds that same axis; nothing shipping does.

        Without this step, selecting ``C = 2`` against a source declared
        broadcast over ``C`` pulls back through an all-zero row to
        ``0 <= -2`` and the region comes back **empty** -- the visual
        disappears the moment the channel slider leaves zero.

        A constraint on an axis this transform has no extent along and
        which is *not* broadcast still comes back with a zero normal,
        which is vacuous or infeasible rather than an error.  Call
        :meth:`ConvexRegion.simplify` to drop the vacuous ones.

        ``map_region`` is deliberately **not** given the symmetric
        treatment: the asymmetry is D31's, and ``broadcast_axes``
        affects the forward direction as unboundedness rather than as a
        dropped constraint.

        Parameters
        ----------
        region : ConvexRegion
            A region in the output coordinate system.
        output_coordinate_system : CoordinateSystem
            This transform's output system, used to resolve
            ``broadcast_axes`` to axis indices.  Required for the same
            reason it is on :meth:`map_bounding_box` and :meth:`then`:
            the field holds axis **ids** and the arithmetic needs
            **indices**, and a transform stores only ids.

        Returns
        -------
        ConvexRegion
            The pulled-back region, in the input coordinate system.

        Raises
        ------
        ValueError
            If the region is not in this transform's output system, or
            the given system is not this transform's output system.
        """
        if output_coordinate_system.id != self.output_coordinate_system:
            raise ValueError(
                f"output_coordinate_system must be this transform's output "
                f"system {self.output_coordinate_system}, got "
                f"{output_coordinate_system.id}."
            )
        broadcast_indices = tuple(
            sorted(
                output_coordinate_system.index_of(axis_id)
                for axis_id in self.broadcast_axes
            )
        )
        return self._imap_region(region, broadcast_indices)

    def _imap_region(
        self, region: ConvexRegion, broadcast_axes: tuple[int, ...]
    ) -> ConvexRegion:
        """Pull a region back with the broadcast axes already resolved.

        Split out of :meth:`imap_region` for the one caller that knows
        the indices without holding the coordinate system object:
        :class:`~cellier.transform.RegionSelection`, whose validator
        runs on a ``rendered -> world`` embedding that carries no
        broadcast axes at all (D34/D35).

        Parameters
        ----------
        region : ConvexRegion
            A region in the output coordinate system.
        broadcast_axes : tuple[int, ...]
            Output-space indices of the broadcast axes.

        Returns
        -------
        ConvexRegion
            The pulled-back region, in the input coordinate system.
        """
        self._check_output_coordinate_system(region.coordinate_system)
        normals, offsets = ops.imap_half_spaces(
            region.normals,
            region.offsets,
            self.linear,
            self.translation,
            broadcast_axes,
        )
        return ConvexRegion(
            coordinate_system=self.input_coordinate_system,
            ndim=self.input_ndim,
            half_spaces=half_spaces_from_arrays(normals, offsets),
        )

    # -- composition (D10, D19) ----------------------------------------

    def then(
        self,
        other: AffineTransform,
        intermediate_coordinate_system: CoordinateSystem,
        output_coordinate_system: CoordinateSystem,
    ) -> AffineTransform:
        """Return the transform that applies ``self``, then ``other``.

        Both coordinate systems are required on every call, broadcast or
        not.  They are not decoration: ``broadcast_axes`` holds axis
        **ids**, and propagating them through ``other`` needs the
        intermediate system to resolve those ids to matrix columns and
        the output system to name the resulting axes.  Requiring them
        unconditionally keeps one rule rather than two.

        An output axis of the result is broadcast if ``other`` declares
        it so, or if its row has a non-zero coefficient on any axis that
        ``self`` broadcasts over -- anything downstream of an unbounded
        axis is itself unbounded.

        There is no ``|`` operator and no ``__matmul__``.

        Parameters
        ----------
        other : AffineTransform
            The transform to apply second.
        intermediate_coordinate_system : CoordinateSystem
            The system both transforms meet in: this transform's output
            and ``other``'s input.
        output_coordinate_system : CoordinateSystem
            ``other``'s output system.

        Returns
        -------
        AffineTransform
            A transform from this one's input to ``other``'s output.

        Raises
        ------
        ValueError
            If the coordinate systems do not line up.  A mismatch raises
            even when the ranks happen to agree (D19).
        """
        if self.output_coordinate_system != other.input_coordinate_system:
            raise ValueError(
                f"Cannot chain transforms across a coordinate system "
                f"mismatch: this transform outputs "
                f"{self.output_coordinate_system} and the next takes "
                f"{other.input_coordinate_system}."
            )
        if intermediate_coordinate_system.id != self.output_coordinate_system:
            raise ValueError(
                f"intermediate_coordinate_system must be "
                f"{self.output_coordinate_system}, got "
                f"{intermediate_coordinate_system.id}."
            )
        if intermediate_coordinate_system.ndim != self.output_ndim:
            raise ValueError(
                f"intermediate_coordinate_system has "
                f"{intermediate_coordinate_system.ndim} axes, but this "
                f"transform produces {self.output_ndim} dimensions."
            )
        other.validate_against(intermediate_coordinate_system, output_coordinate_system)

        broadcast = set(other.broadcast_axes)
        if self.broadcast_axes:
            columns = [
                intermediate_coordinate_system.index_of(axis_id)
                for axis_id in self.broadcast_axes
            ]
            for row in range(other.output_ndim):
                if np.any(other.linear[row, columns] != 0.0):
                    broadcast.add(output_coordinate_system.axes[row].id)

        return type(self)(
            name=self.name,
            transform=Affine(other.matrix @ self.matrix),
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=other.output_coordinate_system,
            broadcast_axes=frozenset(broadcast),
        )

    # -- constructors (D18, D22, D23, D26, D35) -------------------------

    @classmethod
    def from_axis_map(
        cls,
        input_coordinate_system: CoordinateSystem,
        output_coordinate_system: CoordinateSystem,
        axis_map: Mapping[AxisRef, AxisRef],
        scale: Mapping[AxisRef, float] | None = None,
        translation: Mapping[AxisRef, float] | None = None,
        broadcast_output_axes: Sequence[AxisRef] = (),
        constant_output_axes: Mapping[AxisRef, float] | None = None,
        name: str | None = None,
    ) -> Self:
        """Build a transform by stating which axis corresponds to which.

        This is the primary constructor, and ``axis_map`` is always
        required.  There is no positional default and no name-matching
        default: a positional default is exactly wrong when the two
        systems have equal rank but different order, and it fails
        *silently*.  Stating the correspondence costs one dict literal
        and puts it where a reader can check it against the two systems.

        ``scale`` and ``translation`` are keyed by **input** axis, so
        every per-axis quantity in the call uses one key space.  Omitted
        axes get scale 1 and translation 0.

        Parameters
        ----------
        input_coordinate_system : CoordinateSystem
            The system to map from.  The object, not the id: building the
            matrix needs its axes (D23).
        output_coordinate_system : CoordinateSystem
            The system to map to.
        axis_map : Mapping[AxisRef, AxisRef]
            ``{input axis: output axis}``.  Every input axis must appear.
            Mapped pairs must share an ``axis_type`` (D26); units are not
            checked (D16).
        scale : Mapping[AxisRef, float] or None
            Per-input-axis scale factors.  Default 1.
        translation : Mapping[AxisRef, float] or None
            Per-input-axis translations.  Default 0.
        broadcast_output_axes : Sequence[AxisRef]
            Output axes the input has no extent along.  These get a zero
            row and are recorded in ``broadcast_axes``.
        constant_output_axes : Mapping[AxisRef, float] or None
            Output axes the input has no extent along that sit at a
            fixed value -- a slice index.  These get a zero row and that
            value in the translation column, and are **not** recorded in
            ``broadcast_axes``: a broadcast axis is unbounded, a slice
            index is not (D35).
        name : str or None
            Optional name for the transform.

        Returns
        -------
        AffineTransform
            A transform of shape ``(D_out + 1, D_in + 1)``.

        Raises
        ------
        ValueError
            If an input axis is unmapped, an output axis is neither
            mapped nor declared, an output axis is claimed twice, or a
            mapped pair disagrees on ``axis_type``.
        """
        n_in = input_coordinate_system.ndim
        n_out = output_coordinate_system.ndim

        mapped: dict[int, int] = {}
        for input_ref, output_ref in axis_map.items():
            input_index = input_coordinate_system.resolve(input_ref)
            output_index = output_coordinate_system.resolve(output_ref)
            input_axis = input_coordinate_system.axes[input_index]
            output_axis = output_coordinate_system.axes[output_index]
            if input_axis.axis_type != output_axis.axis_type:
                raise ValueError(
                    f"Mapped axes must share an axis_type: input axis "
                    f"'{input_axis.name}' is {input_axis.axis_type!r} but "
                    f"output axis '{output_axis.name}' is "
                    f"{output_axis.axis_type!r}.  A mismatch here is usually "
                    f"a transposed axis map."
                )
            if input_index in mapped:
                raise ValueError(
                    f"Input axis '{input_axis.name}' appears twice in axis_map."
                )
            if output_index in set(mapped.values()):
                raise ValueError(
                    f"Output axis '{output_axis.name}' is claimed by more than "
                    f"one input axis."
                )
            mapped[input_index] = output_index

        unmapped_inputs = [
            axis.name
            for index, axis in enumerate(input_coordinate_system.axes)
            if index not in mapped
        ]
        if unmapped_inputs:
            raise ValueError(
                f"Every input axis must appear in axis_map; {unmapped_inputs} do not."
            )

        broadcast_indices = {
            output_coordinate_system.resolve(ref) for ref in broadcast_output_axes
        }
        constant_indices = {
            output_coordinate_system.resolve(ref): float(value)
            for ref, value in (constant_output_axes or {}).items()
        }
        claimed = set(mapped.values())
        for label, indices in (
            ("broadcast_output_axes", broadcast_indices),
            ("constant_output_axes", set(constant_indices)),
        ):
            overlap = claimed & indices
            if overlap:
                raise ValueError(
                    f"Output axes "
                    f"{[output_coordinate_system.axes[i].name for i in overlap]} "
                    f"are both mapped and listed in {label}."
                )
        both = broadcast_indices & set(constant_indices)
        if both:
            raise ValueError(
                f"Output axes "
                f"{[output_coordinate_system.axes[i].name for i in both]} are "
                f"listed as both broadcast and constant."
            )

        undeclared = [
            axis.name
            for index, axis in enumerate(output_coordinate_system.axes)
            if index not in claimed
            and index not in broadcast_indices
            and index not in constant_indices
        ]
        if undeclared:
            raise ValueError(
                f"Output axes {undeclared} are neither mapped nor declared as "
                f"broadcast_output_axes or constant_output_axes.  There is no "
                f"silent default for an unmapped output axis."
            )

        scale_by_index = {
            input_coordinate_system.resolve(ref): float(value)
            for ref, value in (scale or {}).items()
        }
        translation_by_index = {
            input_coordinate_system.resolve(ref): float(value)
            for ref, value in (translation or {}).items()
        }

        matrix = np.zeros((n_out + 1, n_in + 1))
        matrix[n_out, n_in] = 1.0
        for input_index, output_index in mapped.items():
            matrix[output_index, input_index] = scale_by_index.get(input_index, 1.0)
            matrix[output_index, n_in] = translation_by_index.get(input_index, 0.0)
        for output_index, value in constant_indices.items():
            matrix[output_index, n_in] = value

        return cls(
            name=name,
            transform=Affine(matrix),
            input_coordinate_system=input_coordinate_system.id,
            output_coordinate_system=output_coordinate_system.id,
            broadcast_axes=frozenset(
                output_coordinate_system.axes[index].id for index in broadcast_indices
            ),
        )

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        input_coordinate_system: CoordinateSystem,
        output_coordinate_system: CoordinateSystem,
        broadcast_output_axes: Sequence[AxisRef] = (),
        name: str | None = None,
    ) -> Self:
        """Build a transform from a matrix, the escape hatch.

        Both coordinate systems are still required (D18): a transform
        without both endpoints is meaningless in this model, and there is
        no ``identity(ndim)`` that invents or omits them.

        Parameters
        ----------
        matrix : np.ndarray
            The ``(D_out + 1, D_in + 1)`` augmented matrix.
        input_coordinate_system : CoordinateSystem
            The system to map from.
        output_coordinate_system : CoordinateSystem
            The system to map to.
        broadcast_output_axes : Sequence[AxisRef]
            Output axes the input has no extent along (D25).
        name : str or None
            Optional name for the transform.

        Returns
        -------
        AffineTransform
            The transform.

        Raises
        ------
        ValueError
            If the matrix shape disagrees with the two systems' ranks.
        """
        values = np.asarray(matrix, dtype=float)
        expected = (
            output_coordinate_system.ndim + 1,
            input_coordinate_system.ndim + 1,
        )
        if values.shape != expected:
            raise ValueError(
                f"matrix must have shape {expected} for a "
                f"{input_coordinate_system.ndim}D -> "
                f"{output_coordinate_system.ndim}D transform, got "
                f"{values.shape}."
            )
        return cls(
            name=name,
            transform=Affine(values),
            input_coordinate_system=input_coordinate_system.id,
            output_coordinate_system=output_coordinate_system.id,
            broadcast_axes=frozenset(
                output_coordinate_system.axes[output_coordinate_system.resolve(ref)].id
                for ref in broadcast_output_axes
            ),
        )

    # -- structure -------------------------------------------------------

    def input_domain(self) -> dict[int, tuple[float, float]]:
        """An affine transform has no intrinsic domain, so this is empty.

        A matrix maps every real coordinate to another; whatever bounds the
        data has come from the store's extent, not from the transform.

        Returns
        -------
        dict[int, tuple[float, float]]
            Always empty.
        """
        return {}

    def broadcast_output_axes(self) -> frozenset[UUID4]:
        """Return :attr:`broadcast_axes`, the recorded broadcast output axes.

        Returns
        -------
        frozenset[UUID4]
            Output axis ids.
        """
        return self.broadcast_axes

    def axis_correspondence(self) -> dict[int, int]:
        """Read ``{input axis: output axis}`` off the matrix.

        Returns
        -------
        dict[int, int]
            Input axis index to output axis index, for every input axis that
            reaches an output axis.

        Raises
        ------
        ValueError
            If any input axis feeds more than one output axis, or any output
            axis is fed by more than one input axis -- a shear or a rotation,
            which the axis-aligned slicing path cannot express.  See
            ``_check_transform_no_rotation``, which imposes the same
            restriction on the multiscale brick shader.
        """
        linear = np.asarray(self.linear)
        correspondence: dict[int, int] = {}
        for column in range(linear.shape[1]):
            rows = np.flatnonzero(linear[:, column])
            if rows.size == 0:
                continue
            if rows.size > 1:
                raise ValueError(
                    f"Input axis {column} of this transform feeds output axes "
                    f"{rows.tolist()}.  An axis-aligned slicing path needs at "
                    f"most one output axis per input axis; a shear or rotation "
                    f"is not yet supported."
                )
            correspondence[column] = int(rows[0])
        claimed: dict[int, int] = {}
        for column, row in correspondence.items():
            if row in claimed:
                raise ValueError(
                    f"Output axis {row} of this transform is fed by input axes "
                    f"{claimed[row]} and {column}.  An axis-aligned slicing "
                    f"path needs at most one input axis per output axis."
                )
            claimed[row] = column
        return correspondence

    # -- restriction and affine-ness ------------------------------------

    def restrict(
        self,
        fixed: Mapping[AxisRef | int, float],
        input_coordinate_system: CoordinateSystem | None = None,
    ) -> AffineTransform:
        """Pin input axes to fixed values, folding them into the translation.

        Always succeeds: for an affine transform this is ordinary matrix
        algebra.  Each fixed axis's column is multiplied by its value and
        added to the translation, then dropped from the linear block --
        the input-side mirror of ``constant_output_axes``.

        ``broadcast_axes`` carries through unchanged: it names *output*
        axes, which restricting the input does not touch.

        **The result is an intermediate, not a registrable transform.** Its
        ``input_coordinate_system`` is kept as provenance, but the
        restricted domain is a strict subset of that system's axes, so the
        rank no longer matches and :meth:`validate_against` will reject it.
        That rejection is deliberate and loud; the intended use is to call
        :meth:`to_affine` on the result and upload the matrix.

        Parameters
        ----------
        fixed : Mapping[AxisRef | int, float]
            ``{input axis: value}``.  Values are exact -- nothing here
            rounds or clamps them.
        input_coordinate_system : CoordinateSystem or None
            Needed only to resolve named axes.

        Returns
        -------
        AffineTransform
            A transform over the remaining input axes, in their original
            relative order.

        Raises
        ------
        ValueError
            If an axis is named twice, is out of range, or a name is given
            with no coordinate system to resolve it against.
        """
        indices = _resolve_fixed_axes(
            fixed, self.input_ndim, input_coordinate_system, "restrict"
        )
        if not indices:
            return self

        linear = self.linear
        free = [axis for axis in range(self.input_ndim) if axis not in indices]
        fixed_axes = sorted(indices)
        fixed_values = np.array([indices[axis] for axis in fixed_axes], dtype=float)

        translation = self.translation + linear[:, fixed_axes] @ fixed_values
        reduced = linear[:, free]

        matrix = np.zeros((reduced.shape[0] + 1, len(free) + 1), dtype=float)
        matrix[:-1, :-1] = reduced
        matrix[:-1, -1] = translation
        matrix[-1, -1] = 1.0

        return AffineTransform(
            name=self.name,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
            transform=Affine(matrix),
            broadcast_axes=self.broadcast_axes,
        )

    def to_affine(self) -> AffineTransform:
        """Return ``self``: an affine transform already is one.

        Returns
        -------
        AffineTransform
            This transform.
        """
        return self

    # -- equality (D21) -------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Compare by endpoints, matrix and broadcast axes, not by id.

        Two independently constructed but identical transforms compare
        equal, which is why ``id`` is excluded.  ``broadcast_axes`` is
        included: two transforms with the same matrix but different
        broadcast axes give different bounding boxes, so they are not
        the same value.
        """
        if not isinstance(other, AffineTransform):
            return NotImplemented
        return (
            self.input_coordinate_system == other.input_coordinate_system
            and self.output_coordinate_system == other.output_coordinate_system
            and self.broadcast_axes == other.broadcast_axes
            and bool(np.array_equal(self.matrix, other.matrix))
        )

    def __hash__(self) -> int:
        """Hash by endpoints, matrix and broadcast axes.

        Defined alongside ``__eq__`` because ``transformnd.Affine``
        defines ``__eq__`` without ``__hash__`` and is therefore
        unhashable, which would make a frozen model containing one raise
        on ``hash()``.
        """
        return hash(
            (
                self.input_coordinate_system,
                self.output_coordinate_system,
                self.broadcast_axes,
                self.matrix.tobytes(),
            )
        )


def _resolve_fixed_axes(
    fixed: Mapping[AxisRef | int, float],
    ndim: int,
    coordinate_system: CoordinateSystem | None,
    operation: str,
) -> dict[int, float]:
    """Resolve a ``restrict`` mapping to ``{axis index: value}``.

    Integer keys are indices and need nothing else.  A name or id needs a
    coordinate system, because a transform stores its endpoints as ids and
    has no axis metadata of its own.

    Parameters
    ----------
    fixed : Mapping[AxisRef | int, float]
        The mapping to resolve.
    ndim : int
        The transform's input rank, for range checking.
    coordinate_system : CoordinateSystem or None
        The system to resolve names against.
    operation : str
        The calling method's name, for error messages.

    Returns
    -------
    dict[int, float]
        ``{axis index: value}``.

    Raises
    ------
    ValueError
        If a name is given without a coordinate system, an index is out of
        range, or one axis is named twice.
    """
    resolved: dict[int, float] = {}
    for reference, value in fixed.items():
        if isinstance(reference, (int, np.integer)) and not isinstance(reference, bool):
            index = int(reference)
        else:
            if coordinate_system is None:
                raise ValueError(
                    f"{operation} was given the axis name {reference!r}, but "
                    f"no input_coordinate_system to resolve it against.  A "
                    f"transform stores its endpoints as ids and has no axis "
                    f"metadata of its own -- pass the system, or key `fixed` "
                    f"by axis index."
                )
            index = coordinate_system.resolve(reference)
        if not 0 <= index < ndim:
            raise ValueError(
                f"{operation} got axis index {index}, which is out of range "
                f"for a transform with {ndim} input axes."
            )
        if index in resolved:
            raise ValueError(f"{operation} got axis {index} twice in `fixed`.")
        resolved[index] = float(value)
    return resolved
