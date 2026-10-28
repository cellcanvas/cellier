"""One canvas's view of the world: where the slice is, and how thick (D43)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

# Both models are imported at runtime, not under TYPE_CHECKING: pydantic
# resolves field annotations at class-creation time and cannot see a
# deferred import.
from cellier.transform._affine import AffineTransform  # noqa: TC001
from cellier.transform._region import ConvexRegion, HalfSpace

if TYPE_CHECKING:
    from collections.abc import Collection


class RegionSelection(BaseModel):
    """A slice position and its extent, as one frozen artifact.

    This is the authoritative artifact the slicer consumes.  A
    ``DimsManager`` becomes an *editor* that holds the ergonomic
    parameterization -- an index and a thickness, the thing a GUI binds
    to -- and emits one of these; an oblique editor holds a plane and a
    thickness and emits the same type.  The slicer consumes one thing and
    does not know which editor produced it (R6).

    **The region is absolute and the translation is a point inside it.**
    The transform's constant column holds the slice **position**, exactly
    as the editor stored it.  The region holds the selected set as absolute
    world bounds, so it encodes the position too: a slab of half thickness
    ``h`` around ``p`` is stored as the offsets ``p + h`` and ``-(p - h)``,
    which round.  Both halves come from the editor's index and thickness,
    and the validator checks that the translation reaches the region.  So
    neither is ever rebuilt from the other: a consumer that needs the
    position reads :attr:`slice_position`, and one that needs the extent
    reads the region.  Recovering the position as the midpoint of the
    region's bounds can miss it by an ulp, and a zero-thickness plane put
    there (D42) excludes the translation -- use
    :meth:`with_planes_at_slice` to narrow a slab to a plane.

    A region relative to the translation (an extent around the position,
    stored once) would make that mistake impossible to write; it is to be
    revisited with the oblique raster editor (full affine design v2,
    section 12), whose plane and thickness are that form already.

    The region is in **world** coordinates, not rendered.  Rendered space
    has only the two or three displayed axes and so cannot express a
    thickness on a collapsed one; world space expresses both, and
    obliquity is carried by the half-space normals rather than by another
    coordinate system.

    Parameters
    ----------
    transform : AffineTransform
        The rendered -> world embedding (D34).  Its translation column
        carries the slice indices.
    region : ConvexRegion
        The selected region, in the transform's **output** (world)
        coordinate system.
    """

    model_config = ConfigDict(frozen=True)

    transform: AffineTransform
    region: ConvexRegion

    @model_validator(mode="after")
    def _validate_consistency(self) -> Self:
        """Check that the slice the transform names selects something.

        Three checks.  The region must be in the transform's output
        system and of matching rank -- otherwise the two halves describe
        different spaces.  And the invariant that actually connects them:
        **some rendered point lands inside the region**, i.e. pulling the
        region back through the transform leaves a non-empty set.  A
        selection whose slice sits outside its own region is incoherent,
        and this is what catches it.

        D43 states that invariant as "the transform's translation
        satisfies every half-space".  That is the same test whenever the
        region constrains only *collapsed* axes, because the translation
        is exactly the slice position on those.  It is too strong as
        soon as a **displayed** axis is bounded, which R3 explicitly
        permits: the translation is zero on a displayed axis, so
        ``Z in [11, 13]`` on a displayed ``Z`` would be rejected even
        though every rendered point with ``Z in [11, 13]`` is in the
        region.  The pull-back accepts exactly the selections that select
        something and still rejects the case D43 names.

        This is a consistency check on a hand-built selection, not a
        synchronisation mechanism.  Nothing here updates one half when
        the other changes; that is the editor's job.

        Returns
        -------
        RegionSelection
            The validated selection.

        Raises
        ------
        ValueError
            If the region is in the wrong coordinate system, has the
            wrong rank, declares broadcast axes, or excludes every point
            the transform can reach.
        """
        if self.region.coordinate_system != self.transform.output_coordinate_system:
            raise ValueError(
                f"The region must be in the transform's output coordinate "
                f"system {self.transform.output_coordinate_system}, but it is "
                f"in {self.region.coordinate_system}."
            )
        if self.region.ndim != self.transform.output_ndim:
            raise ValueError(
                f"The region has rank {self.region.ndim} but the transform "
                f"produces {self.transform.output_ndim} dimensions."
            )
        if self.transform.broadcast_axes:
            raise ValueError(
                "A RegionSelection's transform is the rendered -> world "
                "embedding of D34, which records where the slice sits as "
                "constant_output_axes.  It must not declare broadcast_axes: a "
                "world axis the canvas does not display is at a definite "
                "position, not free.  Rebuild the embedding with "
                "constant_output_axes instead of broadcast_output_axes."
            )
        # The embedding has no broadcast axes -- checked immediately above --
        # so there is nothing for D8's drop rule to remove and the indices
        # are empty.  This is the one caller that has the ids but not the
        # world CoordinateSystem object needed to resolve them.
        if self.transform._imap_region(self.region, ()).is_empty():
            raise ValueError(
                f"No point this transform can reach lies inside the selected "
                f"region: the selection's own slice position "
                f"{self.transform.translation.tolist()} would select nothing."
            )
        return self

    @property
    def slice_position(self) -> np.ndarray:
        """The world point the rendered origin maps to.

        This is the transform's translation column, which is where the
        slice indices live (D35).
        """
        return self.transform.translation

    def with_planes_at_slice(self, axes: Collection[int]) -> RegionSelection:
        """Narrow each of *axes* to the plane at its slice position.

        For every world axis in *axes*, the region's constraints on that axis
        alone (a half-space whose normal is non-zero only there) are replaced
        by the zero-thickness plane ``x = p``, with ``p`` read from
        :attr:`slice_position`.  Every other constraint is kept as it is.
        The plane is built from the translation itself, not from the
        region's bounds, so the result always passes the validator (see the
        class docstring).

        On a sliced axis the embedding is constant at ``p``, and validation
        has already shown ``p`` satisfies the region, so the result selects
        a subset of what this selection does.

        Parameters
        ----------
        axes : Collection[int]
            World axis indices.  Each must be sliced: the embedding's row
            for it must be zero.

        Returns
        -------
        RegionSelection
            The narrowed selection; ``self`` when *axes* is empty.

        Raises
        ------
        ValueError
            If an axis is out of range or displayed.
        """
        pinned = sorted({int(axis) for axis in axes})
        if not pinned:
            return self
        ndim = self.region.ndim
        linear = self.transform.linear
        for axis in pinned:
            if not 0 <= axis < ndim:
                raise ValueError(
                    f"World axis {axis} is out of range for a selection of rank {ndim}."
                )
            if np.any(linear[axis] != 0):
                raise ValueError(
                    f"World axis {axis} is displayed: the embedding moves along "
                    f"it, so it has no slice position to put a plane at."
                )
        on_pinned = set(pinned)
        kept: list[HalfSpace] = []
        for half_space in self.region.half_spaces:
            reads = np.flatnonzero(half_space.normal)
            if reads.size == 1 and int(reads[0]) in on_pinned:
                continue
            kept.append(half_space)
        position = self.slice_position
        for axis in pinned:
            normal = np.zeros(ndim)
            normal[axis] = 1.0
            value = float(position[axis])
            kept.append(HalfSpace(normal=normal, offset=value))
            kept.append(HalfSpace(normal=-normal, offset=-value))
        return RegionSelection(
            transform=self.transform,
            region=ConvexRegion(
                coordinate_system=self.region.coordinate_system,
                ndim=ndim,
                half_spaces=tuple(kept),
            ),
        )
