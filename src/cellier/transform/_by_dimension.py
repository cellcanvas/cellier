"""A transform assembled from independent per-axis blocks.

The container that lets one axis be irregular while the rest stay affine.
Its blocks act on disjoint subsets of the input and output axes, which is
what makes every operation decompose: a question about the affine axes is
answered by the affine block and never reaches the non-uniform one.

Mirrors RFC-5's ``byDimension``.  See
``plans/nonuniform_time_axis_transform_design.md``.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator

# Imported at runtime, not under TYPE_CHECKING: pydantic resolves field
# annotations at class-creation time and cannot see a deferred import.
from transformnd.transforms.affine import Affine
from transformnd.transforms.by_dimension import ByDimension, SubTransform
from typing_extensions import Self

from cellier.transform._affine import AffineTransform, _resolve_fixed_axes
from cellier.transform._base import BaseTransform
from cellier.transform._coordinate_system import CoordinateSystem
from cellier.transform._geometry import AxisAlignedBoundingBox
from cellier.transform._geometry_ops import NonAffineTransformError, is_axis_aligned
from cellier.transform._region import ConvexRegion

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cellier.transform._axis import AxisRef

__all__ = ["ByDimensionTransform", "TransformBlock"]


_DERIVED_SYSTEM_NAMESPACE = uuid.UUID("6f2a1f3c-0b1d-4a6e-9c5d-7e8f90a1b2c3")


def _derived_system_id(*parts: object) -> uuid.UUID:
    """A deterministic id for an internally-minted sub-coordinate-system.

    ``from_axis_map`` derives a sub-system per block rather than making
    callers mint one.  A fresh ``uuid4`` each time would make two
    structurally identical transforms compare unequal -- their blocks would
    differ only by an id nobody chose -- and would make serialization
    unstable across runs.  Deriving the id from the parent system and the
    axes it covers makes the same construction produce the same id.

    The result is stamped as version 4 because the models annotate their
    ids as ``UUID4`` and pydantic checks the version.  It is a hash, not
    randomness, which is the point.

    Parameters
    ----------
    *parts : object
        Whatever identifies this sub-system: the parent id and its axes.

    Returns
    -------
    uuid.UUID
        A deterministic, version-4-shaped id.
    """
    digest = uuid.uuid5(
        _DERIVED_SYSTEM_NAMESPACE, "|".join(str(part) for part in parts)
    ).bytes
    forced = bytearray(digest)
    forced[6] = (forced[6] & 0x0F) | 0x40  # version 4
    forced[8] = (forced[8] & 0x3F) | 0x80  # RFC 4122 variant
    return uuid.UUID(bytes=bytes(forced))


class TransformBlock(BaseModel):
    """One block of a :class:`ByDimensionTransform`.

    Parameters
    ----------
    transform : BaseTransform
        The transform this block applies.
    input_axes : tuple[int, ...]
        Which input axes of the container it consumes, in order.
    output_axes : tuple[int, ...]
        Which output axes of the container it produces, in order.  May be
        longer than ``input_axes``: a block that broadcasts over an output
        axis is non-square, which is normal here rather than an edge case.
    input_coordinate_system : CoordinateSystem
        The block's own input system.  Carried so bounding-box delegation
        can stamp a sub-box with the right id; its ``id`` matches
        ``transform.input_coordinate_system``.
    output_coordinate_system : CoordinateSystem
        The block's own output system, likewise.
    """

    model_config = ConfigDict(frozen=True)

    transform: BaseTransform
    input_axes: tuple[int, ...]
    output_axes: tuple[int, ...]
    input_coordinate_system: CoordinateSystem
    output_coordinate_system: CoordinateSystem

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """Check the declared axes against the wrapped transform's rank."""
        if len(self.input_axes) != self.transform.input_ndim:
            raise ValueError(
                f"Block declares {len(self.input_axes)} input axes but its "
                f"transform takes {self.transform.input_ndim}."
            )
        if len(self.output_axes) != self.transform.output_ndim:
            raise ValueError(
                f"Block declares {len(self.output_axes)} output axes but its "
                f"transform produces {self.transform.output_ndim}."
            )
        if self.input_coordinate_system.id != self.transform.input_coordinate_system:
            raise ValueError("Block input system id does not match its transform.")
        if self.output_coordinate_system.id != self.transform.output_coordinate_system:
            raise ValueError("Block output system id does not match its transform.")
        return self

    @property
    def is_affine(self) -> bool:
        """Whether this block can be expressed as a matrix."""
        return self.transform.to_affine() is not None


class ByDimensionTransform(BaseTransform):
    """A transform whose axes are handled by independent blocks.

    Blocks partition both the input and the output axes, so every operation
    decomposes per block.  That is what keeps a non-uniform axis from
    infecting the rest: a question confined to the affine axes is answered
    by the affine block, and the irregular one is never consulted.

    **This class routes its own blocks rather than delegating to
    ``transformnd.ByDimension.apply``.**  That method allocates its output
    with ``empty_like(coords)`` -- an array shaped *and typed* like its
    **input** -- so it is wrong for every non-square block, which includes
    this class's motivating case.  The expanding direction raises
    ``IndexError``; the contracting direction, which is the pull-back the
    whole slicing path runs on, returns an input-shaped array with an
    uninitialised trailing column and **does not raise**.  The wrapped
    ``ByDimension`` is still the object in the ``transform`` field -- it is
    the right RFC-5 object and it reports the dimensionality
    ``validate_against`` checks -- but it never executes.

    Parameters
    ----------
    blocks : tuple[TransformBlock, ...]
        The blocks, whose input and output axis sets must each partition
        the container's axes exactly.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    transform_type: Literal["by_dimension"] = "by_dimension"
    blocks: tuple[TransformBlock, ...]
    transform: ByDimension

    # -- construction ---------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _derive_transform(cls, data: Any) -> Any:
        """Build the wrapped ``ByDimension`` from the blocks."""
        if not isinstance(data, dict):
            return data
        blocks = data.get("blocks")
        if not blocks:
            return data
        blocks = tuple(
            block
            if isinstance(block, TransformBlock)
            else TransformBlock.model_validate(block)
            for block in blocks
        )
        data["blocks"] = blocks
        data["transform"] = ByDimension(
            [
                SubTransform(
                    block.transform.transform,
                    input_axes=list(block.input_axes),
                    output_axes=list(block.output_axes),
                )
                for block in blocks
            ]
        )
        return data

    @field_serializer("transform")
    def _serialize_transform(self, value: ByDimension) -> None:
        """Emit nothing: the wrapped object is rebuilt from ``blocks``."""
        return None

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
        axis_transforms: Mapping[AxisRef, BaseTransform] | None = None,
        name: str | None = None,
    ) -> Self:
        """Build a transform by stating which axis corresponds to which.

        Every argument but ``axis_transforms`` keeps the meaning it has in
        :meth:`AffineTransform.from_axis_map`, so a caller who needs no
        per-axis block writes exactly today's call and the non-uniform case
        is a one-key diff rather than a different idiom.

        Blocks are assembled as one 1-D block per ``axis_transforms`` entry,
        plus one affine block covering every remaining input axis -- which
        is also where ``broadcast_output_axes`` and ``constant_output_axes``
        ride.  Sub-coordinate-systems are derived internally; callers never
        mint them.

        Parameters
        ----------
        input_coordinate_system : CoordinateSystem
            The system to map from.
        output_coordinate_system : CoordinateSystem
            The system to map to.
        axis_map : Mapping[AxisRef, AxisRef]
            ``{input axis: output axis}``.  Every input axis must appear,
            including those in ``axis_transforms``.
        scale : Mapping[AxisRef, float] or None
            Per-input-axis scale factors, for the affine axes only.
        translation : Mapping[AxisRef, float] or None
            Per-input-axis translations, for the affine axes only.
        broadcast_output_axes : Sequence[AxisRef]
            Output axes the input has no extent along.
        constant_output_axes : Mapping[AxisRef, float] or None
            Output axes pinned to a fixed value.
        axis_transforms : Mapping[AxisRef, BaseTransform] or None
            ``{input axis: 1-D transform}``.  The transform carries that
            axis's whole mapping.
        name : str or None
            Optional name.

        Returns
        -------
        ByDimensionTransform
            The assembled transform.

        Raises
        ------
        ValueError
            If an axis appears in both ``axis_transforms`` and ``scale`` or
            ``translation``, if a per-axis transform is not 1-D, or for any
            reason ``AffineTransform.from_axis_map`` would raise.
        """
        axis_transforms = dict(axis_transforms or {})
        scale = dict(scale or {})
        translation = dict(translation or {})

        leaf_indices: dict[int, BaseTransform] = {}
        for reference, block_transform in axis_transforms.items():
            index = input_coordinate_system.resolve(reference)
            axis_name = input_coordinate_system.axes[index].name
            for label, mapping in (("scale", scale), ("translation", translation)):
                if any(
                    input_coordinate_system.resolve(key) == index for key in mapping
                ):
                    raise ValueError(
                        f"Axis '{axis_name}' appears in both axis_transforms "
                        f"and {label}.  The per-axis transform carries that "
                        f"axis's whole mapping, so a {label} beside it is two "
                        f"answers to one question -- even a no-op value like "
                        f"1.0, because what is wrong is the shape of the call."
                    )
            if block_transform.input_ndim != 1 or block_transform.output_ndim != 1:
                raise ValueError(
                    f"axis_transforms['{axis_name}'] must be 1-D in and 1-D "
                    f"out; got {block_transform.input_ndim} -> "
                    f"{block_transform.output_ndim}."
                )
            leaf_indices[index] = block_transform

        blocks: list[TransformBlock] = []
        for index, block_transform in sorted(leaf_indices.items()):
            output_index = output_coordinate_system.resolve(
                axis_map[_key_for(axis_map, input_coordinate_system, index)]
            )
            blocks.append(
                TransformBlock(
                    transform=block_transform,
                    input_axes=(index,),
                    output_axes=(output_index,),
                    input_coordinate_system=CoordinateSystem(
                        name=input_coordinate_system.axes[index].name,
                        axes=(input_coordinate_system.axes[index],),
                        id=block_transform.input_coordinate_system,
                    ),
                    output_coordinate_system=CoordinateSystem(
                        name=output_coordinate_system.axes[output_index].name,
                        axes=(output_coordinate_system.axes[output_index],),
                        id=block_transform.output_coordinate_system,
                    ),
                )
            )

        remaining = [
            index
            for index in range(input_coordinate_system.ndim)
            if index not in leaf_indices
        ]
        declared = {
            output_coordinate_system.resolve(reference)
            for reference in broadcast_output_axes
        } | {
            output_coordinate_system.resolve(reference)
            for reference in (constant_output_axes or {})
        }
        mapped_outputs = {
            output_coordinate_system.resolve(axis_map[key])
            for key in axis_map
            if input_coordinate_system.resolve(key) in remaining
        }
        affine_outputs = sorted(mapped_outputs | declared)

        if remaining or affine_outputs:
            sub_input = CoordinateSystem(
                name=f"{input_coordinate_system.name}[affine]",
                axes=tuple(input_coordinate_system.axes[i] for i in remaining),
                id=_derived_system_id(
                    input_coordinate_system.id, "in", tuple(remaining)
                ),
            )
            sub_output = CoordinateSystem(
                name=f"{output_coordinate_system.name}[affine]",
                axes=tuple(output_coordinate_system.axes[i] for i in affine_outputs),
                id=_derived_system_id(
                    output_coordinate_system.id, "out", tuple(affine_outputs)
                ),
            )
            sub_axis_map = {
                input_coordinate_system.axes[i].id: output_coordinate_system.axes[
                    output_coordinate_system.resolve(
                        axis_map[_key_for(axis_map, input_coordinate_system, i)]
                    )
                ].id
                for i in remaining
            }
            affine = AffineTransform.from_axis_map(
                sub_input,
                sub_output,
                axis_map=sub_axis_map,
                scale={
                    input_coordinate_system.axes[
                        input_coordinate_system.resolve(key)
                    ].id: value
                    for key, value in scale.items()
                },
                translation={
                    input_coordinate_system.axes[
                        input_coordinate_system.resolve(key)
                    ].id: value
                    for key, value in translation.items()
                },
                broadcast_output_axes=tuple(
                    output_coordinate_system.axes[
                        output_coordinate_system.resolve(reference)
                    ].id
                    for reference in broadcast_output_axes
                ),
                constant_output_axes={
                    output_coordinate_system.axes[
                        output_coordinate_system.resolve(reference)
                    ].id: value
                    for reference, value in (constant_output_axes or {}).items()
                },
            )
            blocks.append(
                TransformBlock(
                    transform=affine,
                    input_axes=tuple(remaining),
                    output_axes=tuple(affine_outputs),
                    input_coordinate_system=sub_input,
                    output_coordinate_system=sub_output,
                )
            )

        return cls(
            name=name,
            blocks=tuple(blocks),
            input_coordinate_system=input_coordinate_system.id,
            output_coordinate_system=output_coordinate_system.id,
        )

    # -- points: routed here, not by ByDimension.apply ------------------

    def map_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Map points forward, block by block.

        Parameters
        ----------
        coordinates : np.ndarray
            ``(N, D_in)`` or ``(D_in,)`` points.

        Returns
        -------
        np.ndarray
            ``(N, D_out)`` or ``(D_out,)``, matching the input's rank.
        """
        return self._route(coordinates, forward=True)

    def imap_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Map points back, block by block.

        Parameters
        ----------
        coordinates : np.ndarray
            ``(N, D_out)`` or ``(D_out,)`` points.

        Returns
        -------
        np.ndarray
            ``(N, D_in)`` or ``(D_in,)``, matching the input's rank.
        """
        return self._route(coordinates, forward=False)

    def _route(self, coordinates: np.ndarray, forward: bool) -> np.ndarray:
        """Apply every block and scatter into its declared output axes.

        Allocates at float64 rather than with ``empty_like``: an integer or
        float32 input would otherwise truncate every mapped value, which is
        exactly the upstream defect this method exists to avoid.
        """
        source_ndim = self.input_ndim if forward else self.output_ndim
        target_ndim = self.output_ndim if forward else self.input_ndim
        values, was_1d = _as_2d(coordinates, source_ndim)

        result = np.zeros((values.shape[0], target_ndim), dtype=np.float64)
        for block in self.blocks:
            take = block.input_axes if forward else block.output_axes
            put = block.output_axes if forward else block.input_axes
            columns = values[:, list(take)]
            mapped = (
                block.transform.map_coordinates(columns)
                if forward
                else block.transform.imap_coordinates(columns)
            )
            result[:, list(put)] = mapped
        return result[0] if was_1d else result

    # -- bounding boxes: per block, exact -------------------------------

    def map_bounding_box(
        self,
        box: AxisAlignedBoundingBox,
        output_coordinate_system: CoordinateSystem,
    ) -> AxisAlignedBoundingBox:
        """Map a box forward, exactly, block by block.

        Parameters
        ----------
        box : AxisAlignedBoundingBox
            A box in the input coordinate system.
        output_coordinate_system : CoordinateSystem
            The system to express the result in.

        Returns
        -------
        AxisAlignedBoundingBox
            The mapped box.
        """
        self._check_input_coordinate_system(box.coordinate_system)
        return self._route_box(box, output_coordinate_system.id, forward=True)

    def imap_bounding_box(self, box: AxisAlignedBoundingBox) -> AxisAlignedBoundingBox:
        """Map a box back, exactly, block by block.

        A monotonic 1-D block maps an interval's endpoints through its own
        monotonic function, which is order-preserving and therefore exact
        rather than merely conservative.

        Parameters
        ----------
        box : AxisAlignedBoundingBox
            A box in the output coordinate system.

        Returns
        -------
        AxisAlignedBoundingBox
            The box in the input coordinate system.
        """
        self._check_output_coordinate_system(box.coordinate_system)
        return self._route_box(box, self.input_coordinate_system, forward=False)

    def _route_box(
        self,
        box: AxisAlignedBoundingBox,
        result_system: Any,
        forward: bool,
    ) -> AxisAlignedBoundingBox:
        """Delegate an axis-aligned box to each block and reassemble."""
        target_ndim = self.output_ndim if forward else self.input_ndim
        lower = np.full(target_ndim, np.nan)
        upper = np.full(target_ndim, np.nan)

        for block in self.blocks:
            take = block.input_axes if forward else block.output_axes
            put = block.output_axes if forward else block.input_axes
            sub_system = (
                block.input_coordinate_system
                if forward
                else block.output_coordinate_system
            )
            sub_box = AxisAlignedBoundingBox(
                coordinate_system=sub_system.id,
                min_coordinate=box.min_coordinate[list(take)],
                max_coordinate=box.max_coordinate[list(take)],
            )
            mapped = (
                block.transform.map_bounding_box(
                    sub_box, block.output_coordinate_system
                )
                if forward
                else block.transform.imap_bounding_box(sub_box)
            )
            lower[list(put)] = mapped.min_coordinate
            upper[list(put)] = mapped.max_coordinate

        return AxisAlignedBoundingBox(
            coordinate_system=result_system,
            min_coordinate=lower,
            max_coordinate=upper,
        )

    # -- structure -------------------------------------------------------

    def input_domain(self) -> dict[int, tuple[float, float]]:
        """Collect each block's own domain, in container axis indices.

        Blocks partition the input axes, so the union needs no reconciling:
        an affine block contributes nothing and a table-backed one
        contributes its own span.

        Returns
        -------
        dict[int, tuple[float, float]]
            Input axis index to ``(low, high)``, for bounded axes only.
        """
        domain: dict[int, tuple[float, float]] = {}
        for block in self.blocks:
            for local_axis, bounds in block.transform.input_domain().items():
                domain[block.input_axes[local_axis]] = bounds
        return domain

    def broadcast_output_axes(self) -> frozenset[uuid.UUID]:
        """Union of every block's broadcast output axes.

        Blocks keep the container's axis ids on their own sub-systems, so a
        block's answer needs no translation.

        Returns
        -------
        frozenset[uuid.UUID]
            Output axis ids.
        """
        return frozenset().union(
            *(block.transform.broadcast_output_axes() for block in self.blocks)
        )

    def axis_correspondence(self) -> dict[int, int]:
        """Read the correspondence off the block declarations, structurally.

        **The reason this is a transform method rather than a matrix read.**
        Each block already states which container axes it consumes and which
        it produces, so the answer needs no matrix and is available even when
        a block has none.  A block is asked for its own correspondence and
        its answer is translated into container axis indices.

        Returns
        -------
        dict[int, int]
            Input axis index to output axis index.

        Raises
        ------
        ValueError
            If two blocks claim the same output axis, or a block is itself a
            shear.  Blocks partition the axes by construction, so the former
            can only happen if the blocks were built by hand.
        """
        correspondence: dict[int, int] = {}
        claimed: dict[int, int] = {}
        for block in self.blocks:
            for local_in, local_out in block.transform.axis_correspondence().items():
                axis_in = block.input_axes[local_in]
                axis_out = block.output_axes[local_out]
                if axis_out in claimed:
                    raise ValueError(
                        f"Output axis {axis_out} of this transform is fed by "
                        f"input axes {claimed[axis_out]} and {axis_in}.  An "
                        f"axis-aligned slicing path needs at most one input "
                        f"axis per output axis."
                    )
                claimed[axis_out] = axis_in
                correspondence[axis_in] = axis_out
        return correspondence

    # -- restriction and affine-ness ------------------------------------

    def restrict(
        self,
        fixed: Mapping[AxisRef | int, float],
        input_coordinate_system: CoordinateSystem | None = None,
    ) -> BaseTransform:
        """Pin input axes, deciding per block what that means.

        Three cases, per the design:

        * **Block entirely fixed** -- evaluated at the fixed value, giving a
          constant that is folded into the remaining translation.  Legal
          whether or not the block is affine, which is why a sliced
          non-uniform axis costs nothing.  Note this *evaluates* the block
          rather than recursing into its own ``restrict``: there is nothing
          left of its domain afterwards.
        * **Block entirely free** -- passed through unchanged.
        * **Block straddling both** -- legal only if the block is affine,
          which is ordinary matrix algebra.  A straddling non-affine block
          has no closed form for "fix one input, what is left as a function
          of the others", and raises.

        Parameters
        ----------
        fixed : Mapping[AxisRef | int, float]
            ``{input axis: value}``.  Exact; nothing here rounds or clamps.
        input_coordinate_system : CoordinateSystem or None
            Needed only to resolve named axes.

        Returns
        -------
        BaseTransform
            A transform over the free input axes.

        Raises
        ------
        NonAffineTransformError
            If a non-affine block straddles the fixed/free split.
        """
        pinned = _resolve_fixed_axes(
            fixed, self.input_ndim, input_coordinate_system, "restrict"
        )
        if not pinned:
            return self

        free_axes = [axis for axis in range(self.input_ndim) if axis not in pinned]
        # The restricted transform's matrix, assembled block-diagonally.
        linear = np.zeros((self.output_ndim, len(free_axes)))
        translation = np.zeros(self.output_ndim)
        free_position = {axis: index for index, axis in enumerate(free_axes)}

        for block in self.blocks:
            block_fixed = [axis for axis in block.input_axes if axis in pinned]
            block_free = [axis for axis in block.input_axes if axis not in pinned]

            if not block_free:
                # Entirely fixed: evaluate it, and keep the constant.
                point = np.array(
                    [[pinned[axis] for axis in block.input_axes]], dtype=float
                )
                constant = block.transform.map_coordinates(point)[0]
                translation[list(block.output_axes)] = constant
                continue

            affine = block.transform.to_affine()
            if affine is None:
                if block_fixed:
                    raise NonAffineTransformError(
                        f"Cannot restrict this transform: the block on input "
                        f"axes {block.input_axes} is not affine and is only "
                        f"partly fixed (fixed {block_fixed}, free "
                        f"{block_free}).  There is no closed form for fixing "
                        f"one input of a non-affine block and keeping the "
                        f"rest as a function of the others."
                    )
                raise NonAffineTransformError(
                    f"Cannot restrict this transform to an affine result: the "
                    f"block on input axes {block.input_axes} is not affine "
                    f"and is not fixed by this request.  A non-uniform axis "
                    f"can be sliced but not displayed -- the fix is to stop "
                    f"displaying axis "
                    f"{block.input_coordinate_system.axes[0].name!r}."
                )

            reduced = affine.restrict(
                {block.input_axes.index(axis): pinned[axis] for axis in block_fixed}
            )
            for column, axis in enumerate(block_free):
                linear[list(block.output_axes), free_position[axis]] = reduced.linear[
                    :, column
                ]
            translation[list(block.output_axes)] = reduced.translation

        matrix = np.zeros((self.output_ndim + 1, len(free_axes) + 1))
        matrix[:-1, :-1] = linear
        matrix[:-1, -1] = translation
        matrix[-1, -1] = 1.0
        return AffineTransform(
            name=self.name,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
            transform=Affine(matrix),
        )

    def to_affine(self) -> AffineTransform | None:
        """Assemble a block-diagonal affine, or ``None`` if any block is not.

        The axes are disjoint by construction, so this is direct assembly
        rather than general linear algebra.

        Returns
        -------
        AffineTransform or None
            The equivalent affine, or ``None`` when some block has no matrix.
        """
        linear = np.zeros((self.output_ndim, self.input_ndim))
        translation = np.zeros(self.output_ndim)
        for block in self.blocks:
            affine = block.transform.to_affine()
            if affine is None:
                return None
            rows = list(block.output_axes)
            for column, axis in enumerate(block.input_axes):
                linear[rows, axis] = affine.linear[:, column]
            translation[rows] = affine.translation

        matrix = np.zeros((self.output_ndim + 1, self.input_ndim + 1))
        matrix[:-1, :-1] = linear
        matrix[:-1, -1] = translation
        matrix[-1, -1] = 1.0
        return AffineTransform(
            name=self.name,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
            transform=Affine(matrix),
            broadcast_axes=frozenset().union(
                *(
                    block.transform.broadcast_axes
                    for block in self.blocks
                    if isinstance(block.transform, AffineTransform)
                ),
                frozenset(),
            ),
        )

    def inverse(self) -> BaseTransform | None:
        """Invert every block, or return ``None`` if any block cannot.

        Returns
        -------
        BaseTransform or None
            The inverse, or ``None``.
        """
        inverted: list[TransformBlock] = []
        for block in self.blocks:
            block_inverse = block.transform.inverse()
            if block_inverse is None:
                return None
            inverted.append(
                TransformBlock(
                    transform=block_inverse,
                    input_axes=block.output_axes,
                    output_axes=block.input_axes,
                    input_coordinate_system=block.output_coordinate_system,
                    output_coordinate_system=block.input_coordinate_system,
                )
            )
        return type(self)(
            name=self.name,
            blocks=tuple(inverted),
            input_coordinate_system=self.output_coordinate_system,
            output_coordinate_system=self.input_coordinate_system,
        )

    # -- operations that need a constant Jacobian -----------------------

    def map_direction(self, direction: np.ndarray) -> np.ndarray:
        """Map a displacement vector, if every block is affine."""
        return self._affine_only("map_direction").map_direction(direction)

    def imap_direction(self, direction: np.ndarray) -> np.ndarray:
        """Map a displacement vector back, if every block is affine."""
        return self._affine_only("imap_direction").imap_direction(direction)

    def map_normal(self, normal: np.ndarray) -> np.ndarray:
        """Map a normal, if every block is affine."""
        return self._affine_only("map_normal").map_normal(normal)

    def imap_normal(self, normal: np.ndarray) -> np.ndarray:
        """Map a normal back, if every block is affine."""
        return self._affine_only("imap_normal").imap_normal(normal)

    def map_plane(self, plane):
        """Map a plane, if every block is affine."""
        return self._affine_only("map_plane").map_plane(plane)

    def imap_plane(self, plane):
        """Map a plane back, if every block is affine."""
        return self._affine_only("imap_plane").imap_plane(plane)

    def map_region(self, region: ConvexRegion) -> ConvexRegion:
        """Map a convex region, if every block is affine."""
        return self._affine_only("map_region").map_region(region)

    def imap_region(
        self,
        region: ConvexRegion,
        output_coordinate_system: CoordinateSystem,
    ) -> ConvexRegion:
        """Map a convex region back, exactly.

        Two cases, and the split is the design's:

        * **Axis-aligned** -- every half-space normal lies along a single
          axis, which is what ``AxisAlignedSelection`` produces and so what
          the whole slicing path actually sends.  Each axis is an interval,
          and an interval through a monotonic block maps by its endpoints,
          so the pull-back is **exact** even across a non-affine block.
        * **Anything else** -- handled by the equivalent affine when every
          block has one, and otherwise refused.

        **The refusal is a permanent, documented restriction, not a TODO.**
        A half-space whose normal mixes a non-affine block's axis with
        another has no exact preimage, and no approximation is offered.
        Nothing produces that case today -- ``PlaneSelection`` is an
        unimplemented stub -- but an oblique selection over a non-uniform
        axis would.

        Parameters
        ----------
        region : ConvexRegion
            A region in the output coordinate system.
        output_coordinate_system : CoordinateSystem
            This transform's **input** system, which the result is
            expressed in.  Named for the abstract signature, which reads
            from the caller's side.

        Returns
        -------
        ConvexRegion
            The region in the input coordinate system.
        """
        self._check_output_coordinate_system(region.coordinate_system)
        if not is_axis_aligned(region.normals):
            return self._affine_only("imap_region").imap_region(
                region, output_coordinate_system
            )

        box = region.bounding_box()
        pulled = self._route_box(box, self.input_coordinate_system, forward=False)

        # Broadcast axes lose their constraint (D8): a block the input has
        # no extent along cannot bound anything.
        broadcast = set()
        for block in self.blocks:
            affine = (
                block.transform
                if isinstance(block.transform, AffineTransform)
                else None
            )
            if affine is None:
                continue
            for column, axis in enumerate(block.input_axes):
                if not np.any(affine.linear[:, column]):
                    broadcast.add(axis)

        lower = np.asarray(pulled.min_coordinate, dtype=float).copy()
        upper = np.asarray(pulled.max_coordinate, dtype=float).copy()
        for axis in broadcast:
            lower[axis] = -np.inf
            upper[axis] = np.inf

        return ConvexRegion.from_bounding_box(
            AxisAlignedBoundingBox(
                coordinate_system=self.input_coordinate_system,
                min_coordinate=lower,
                max_coordinate=upper,
            )
        )

    def _affine_only(self, operation: str) -> AffineTransform:
        """Return the equivalent affine, or raise naming the offending block."""
        affine = self.to_affine()
        if affine is not None:
            return affine
        offending = [
            block.input_coordinate_system.axes[0].name
            for block in self.blocks
            if not block.is_affine
        ]
        raise NonAffineTransformError(
            f"{operation} needs a spatially constant Jacobian, which this "
            f"transform does not have: the block(s) on axis/axes {offending} "
            f"are not affine.  Axis-aligned bounding boxes are exact for a "
            f"monotonic block and are the supported path; a direction, "
            f"normal, plane or half-space that mixes such an axis with "
            f"another has no exact answer and none is approximated."
        )

    # -- equality -------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Compare by endpoints and blocks, not by id."""
        if not isinstance(other, ByDimensionTransform):
            return NotImplemented
        return (
            self.input_coordinate_system == other.input_coordinate_system
            and self.output_coordinate_system == other.output_coordinate_system
            and self.blocks == other.blocks
        )

    def __hash__(self) -> int:
        """Hash by endpoints and block structure."""
        return hash(
            (
                self.input_coordinate_system,
                self.output_coordinate_system,
                tuple(
                    (block.input_axes, block.output_axes, block.transform)
                    for block in self.blocks
                ),
            )
        )


def _key_for(
    axis_map: Mapping[AxisRef, AxisRef],
    system: CoordinateSystem,
    index: int,
) -> AxisRef:
    """Find the ``axis_map`` key naming a given input axis index."""
    for key in axis_map:
        if system.resolve(key) == index:
            return key
    raise ValueError(
        f"Input axis '{system.axes[index].name}' does not appear in axis_map; "
        f"every input axis must, including those in axis_transforms."
    )


def _as_2d(array: np.ndarray, ndim: int) -> tuple[np.ndarray, bool]:
    """Normalize ``(N, ndim)`` or ``(ndim,)`` input to 2-D at float64."""
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        if values.shape[0] != ndim:
            raise ValueError(
                f"Expected {ndim} components per point; got shape {values.shape}."
            )
        return values[np.newaxis, :], True
    if values.ndim == 2:
        if values.shape[1] != ndim:
            raise ValueError(f"Expected {ndim} columns; got shape {values.shape}.")
        return values, False
    raise ValueError(f"Points must be 1-D or 2-D; got shape {values.shape}.")
