"""The transform base class (design section 8)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import UUID4, BaseModel, ConfigDict, Field

# Transform is imported at runtime, not under TYPE_CHECKING: pydantic
# resolves field annotations at class-creation time and cannot see a
# deferred import.
from transformnd.base import Transform  # noqa: TC002

from cellier.transform._geometry_ops import (
    DegenerateNormalError,
    NonInvertibleTransformError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

    from cellier.transform._affine import AffineTransform
    from cellier.transform._axis import AxisRef
    from cellier.transform._coordinate_system import CoordinateSystem
    from cellier.transform._geometry import AxisAlignedBoundingBox, Plane
    from cellier.transform._region import ConvexRegion

__all__ = [
    "BaseTransform",
    "DegenerateNormalError",
    "NonInvertibleTransformError",
]


class BaseTransform(BaseModel, ABC):
    """A transform between two coordinate systems.

    Wraps a :class:`transformnd.base.Transform` rather than
    reimplementing one.  ``transformnd.Spaced`` is deliberately not used:
    it carries no axis metadata, does not serialize, and its ``invert()``
    returns the original space order while performing the inverted
    transform.  Owning the source/target pairing here is what avoids
    inheriting that bug.

    Transforms are frozen (Q13).  Dimensionality is **not** stored: it
    lives on the wrapped transform and is read through, so the model
    cannot disagree with itself (D9).  The cross-check against the
    coordinate systems' axis counts cannot happen here, because the model
    holds ids rather than systems; it is factored into
    :meth:`validate_against`.

    Parameters
    ----------
    name : str or None
        Optional human-readable name.  RFC-5 gives transformations one.
    input_coordinate_system : UUID4
        The id of the coordinate system this transform maps from.
    output_coordinate_system : UUID4
        The id of the coordinate system this transform maps to.
    transform : Transform
        The wrapped ``transformnd`` transform.
    id : UUID4
        Unique identifier.  Auto-generated.  Not ``(input, output)``,
        because two coordinate systems may be joined by more than one
        transform (D14).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str | None = None
    input_coordinate_system: UUID4
    output_coordinate_system: UUID4
    transform: Transform
    id: UUID4 = Field(default_factory=uuid4, frozen=True)

    # -- dimensionality, read through to the wrapped transform (D9) ----

    @property
    def input_ndim(self) -> int:
        """Number of input dimensions, read from the wrapped transform."""
        return self.transform.ndims.source

    @property
    def output_ndim(self) -> int:
        """Number of output dimensions, read from the wrapped transform."""
        return self.transform.ndims.target

    # -- validation ----------------------------------------------------

    def validate_against(
        self,
        input_coordinate_system: CoordinateSystem,
        output_coordinate_system: CoordinateSystem,
    ) -> None:
        """Check this transform against the two systems it claims to join.

        The four checks of D9: both ids match, and both axis counts match
        the wrapped transform's dimensionality.  A future registry calls
        this on insertion; defining it now means the registry inherits a
        decided rule rather than inventing one.

        Parameters
        ----------
        input_coordinate_system : CoordinateSystem
            The system this transform should map from.
        output_coordinate_system : CoordinateSystem
            The system this transform should map to.

        Raises
        ------
        ValueError
            If any of the four checks fails.
        """
        if input_coordinate_system.id != self.input_coordinate_system:
            raise ValueError(
                f"Input coordinate system id mismatch: transform expects "
                f"{self.input_coordinate_system}, got "
                f"{input_coordinate_system.id}."
            )
        if output_coordinate_system.id != self.output_coordinate_system:
            raise ValueError(
                f"Output coordinate system id mismatch: transform expects "
                f"{self.output_coordinate_system}, got "
                f"{output_coordinate_system.id}."
            )
        if input_coordinate_system.ndim != self.input_ndim:
            raise ValueError(
                f"Input rank mismatch: transform takes {self.input_ndim} "
                f"dimensions, coordinate system "
                f"'{input_coordinate_system.name}' has "
                f"{input_coordinate_system.ndim} axes."
            )
        if output_coordinate_system.ndim != self.output_ndim:
            raise ValueError(
                f"Output rank mismatch: transform produces "
                f"{self.output_ndim} dimensions, coordinate system "
                f"'{output_coordinate_system.name}' has "
                f"{output_coordinate_system.ndim} axes."
            )

    def _check_input_coordinate_system(self, coordinate_system: UUID4) -> None:
        """Raise if an argument is not in this transform's input system.

        Parameters
        ----------
        coordinate_system : UUID4
            The id carried by the argument.

        Raises
        ------
        ValueError
            If it is not the input coordinate system.
        """
        if coordinate_system != self.input_coordinate_system:
            raise ValueError(
                f"Expected an object in coordinate system "
                f"{self.input_coordinate_system} (this transform's input), "
                f"got one in {coordinate_system}."
            )

    def _check_output_coordinate_system(self, coordinate_system: UUID4) -> None:
        """Raise if an argument is not in this transform's output system.

        Parameters
        ----------
        coordinate_system : UUID4
            The id carried by the argument.

        Raises
        ------
        ValueError
            If it is not the output coordinate system.
        """
        if coordinate_system != self.output_coordinate_system:
            raise ValueError(
                f"Expected an object in coordinate system "
                f"{self.output_coordinate_system} (this transform's output), "
                f"got one in {coordinate_system}."
            )

    # -- points --------------------------------------------------------

    @abstractmethod
    def map_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Map points from the input system to the output system."""

    @abstractmethod
    def imap_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Map points from the output system back to the input system."""

    # -- vectors, two kinds and never one (D27) ------------------------

    @abstractmethod
    def map_direction(self, direction: np.ndarray) -> np.ndarray:
        """Map displacement vectors forward (contravariant)."""

    @abstractmethod
    def imap_direction(self, direction: np.ndarray) -> np.ndarray:
        """Map displacement vectors back (contravariant)."""

    @abstractmethod
    def map_normal(self, normal: np.ndarray) -> np.ndarray:
        """Map plane normals forward (covariant)."""

    @abstractmethod
    def imap_normal(self, normal: np.ndarray) -> np.ndarray:
        """Map plane normals back (covariant)."""

    # -- regions -------------------------------------------------------

    @abstractmethod
    def map_bounding_box(
        self,
        box: AxisAlignedBoundingBox,
        output_coordinate_system: CoordinateSystem,
    ) -> AxisAlignedBoundingBox:
        """Map an axis-aligned box forward, conservatively."""

    @abstractmethod
    def imap_bounding_box(self, box: AxisAlignedBoundingBox) -> AxisAlignedBoundingBox:
        """Map an axis-aligned box back, conservatively."""

    @abstractmethod
    def map_plane(self, plane: Plane) -> Plane:
        """Map a plane forward."""

    @abstractmethod
    def imap_plane(self, plane: Plane) -> Plane:
        """Map a plane back."""

    @abstractmethod
    def map_region(self, region: ConvexRegion) -> ConvexRegion:
        """Map a convex region forward."""

    @abstractmethod
    def imap_region(
        self,
        region: ConvexRegion,
        output_coordinate_system: CoordinateSystem,
    ) -> ConvexRegion:
        """Map a convex region back, dropping broadcast constraints (D8)."""

    # -- structure ------------------------------------------------------

    @abstractmethod
    def input_domain(self) -> dict[int, tuple[float, float]]:
        """Return ``{input axis: (low, high)}`` for every **bounded** axis.

        The span of input coordinates this transform can actually map.  An
        axis absent from the result is unbounded -- an affine transform has
        no intrinsic domain at all, so its result is empty.

        **What it is for.**  A caller that rounds a position to a whole
        sample needs to know which samples exist, or rounding can leave the
        domain: the last sample's cell ends half a unit past its centre, and
        round-half-up sends that boundary *upward* to a sample that is not
        there.  Structural, like :meth:`axis_correspondence`, and available
        without a matrix -- a non-affine block answers from its own table.

        Returns
        -------
        dict[int, tuple[float, float]]
            Input axis index to ``(low, high)``, in input coordinates.
            Bounds are inclusive.
        """

    @abstractmethod
    def axis_correspondence(self) -> dict[int, int]:
        """Return ``{input axis: output axis}`` for every axis that reaches one.

        Answers "which output axis does each input axis become", which the
        render layer needs in order to line a data axis up with the world
        axis a slider moves.

        The correspondence is **not stored** -- keeping it beside the
        transform that already encodes it would be a second source of truth
        (D23) -- so each transform reads it back from whatever it does hold.
        An affine reads it off its matrix; a block container reads it off
        its block declarations, structurally, with no matrix involved, which
        is what lets a non-affine transform answer at all.

        Returns
        -------
        dict[int, int]
            Input axis index to output axis index.  An input axis that
            reaches no output axis is absent.

        Raises
        ------
        ValueError
            If any input axis feeds more than one output axis, or any output
            axis is fed by more than one input axis.  That is a shear or a
            rotation, which the axis-aligned slicing path cannot express.
        """

    def broadcast_output_axes(self) -> frozenset[UUID4]:
        """Return the ids of the output axes the input has no extent along.

        A dataset broadcast over an axis occupies *all* of it, not the point
        its zero matrix row maps to, so a caller measuring where the data is
        -- a world bounding box, say -- must skip these axes rather than read
        a mapped coordinate off them.  Structural, like
        :meth:`axis_correspondence`: a block container answers from its
        blocks, with no matrix involved.

        Returns
        -------
        frozenset[UUID4]
            Output axis ids.  Empty unless a subclass records broadcasts.
        """
        return frozenset()

    # -- restriction and affine-ness -----------------------------------

    @abstractmethod
    def restrict(
        self,
        fixed: Mapping[AxisRef | int, float],
        input_coordinate_system: CoordinateSystem | None = None,
    ) -> BaseTransform:
        """Pin some input axes to fixed values, dropping them from the domain.

        The question the render layer actually needs answered is not "is
        this transform affine" but "with every collapsed axis pinned to
        *this request's* value, is what remains affine".  That is strictly
        weaker, because evaluating any transform at a fixed input produces
        a constant, and folding a constant into a translation is something
        affine algebra already does.  So a non-uniform axis that is sliced
        rather than displayed costs nothing at the GPU boundary.

        The values in *fixed* are **exact and already resolved**.
        ``restrict`` must not round or clamp them: that already happened one
        layer up, in ``round_world_to_voxel``, and doing it twice risks the
        two disagreeing about which plane a position selects.

        This is the input-side, post-hoc mirror of what
        ``AffineTransform.from_axis_map(..., constant_output_axes=...)``
        does output-side at construction time.  Named ``restrict`` rather
        than ``slice`` because ``slice`` is claimed, hard, by the request
        pipeline, which rounds and clamps -- precisely what this must not
        do.

        Parameters
        ----------
        fixed : Mapping[AxisRef | int, float]
            ``{input axis: value}``.  An ``int`` key is an axis index and
            needs nothing else; a name or id key needs
            *input_coordinate_system* to resolve against.
        input_coordinate_system : CoordinateSystem or None
            The system to resolve named axes against.  Required only when
            *fixed* has non-integer keys: a transform stores its endpoints
            as ids and cannot resolve a name on its own.

        Returns
        -------
        BaseTransform
            A transform over the remaining (free) input axes.

        Raises
        ------
        NonAffineTransformError
            If some part of the transform straddles the fixed/free split
            and is not affine, so there is no closed form for "fix one
            input, what is left as a function of the others".
        """

    @abstractmethod
    def to_affine(self) -> AffineTransform | None:
        """Return this transform as an affine, or ``None`` if it is not one.

        ``None`` is an answer, not a failure: callers that need a matrix
        pair this with :meth:`restrict` and raise
        ``NonAffineTransformError`` themselves, with a message naming the
        axis they cannot express.

        Returns
        -------
        AffineTransform or None
            ``self`` for an affine transform.
        """

    # -- inversion -----------------------------------------------------

    @abstractmethod
    def inverse(self) -> BaseTransform | None:
        """Return the inverse transform, or ``None`` if there is none."""
