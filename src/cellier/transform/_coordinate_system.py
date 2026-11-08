"""Coordinate systems: ordered sets of axes, addressed by id."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Union
from uuid import UUID, uuid4

from pydantic import UUID4, BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from cellier.transform._axis import Axis, AxisRef

if TYPE_CHECKING:
    from collections.abc import Sequence


class CoordinateSystem(BaseModel):
    """An ordered set of axes.

    Coordinate systems are frozen.  Changing the axes of a system means
    constructing a replacement, so that a transform already validated
    against a system cannot silently face a different one.

    Parameters
    ----------
    coordinate_system_type : Literal["coordinate_system"]
        Discriminator field.
    name : str
        Human-readable name.
    axes : tuple[Axis, ...]
        The axes, in order.  At least one is required and their ids must
        be unique.
    id : UUID4
        Unique identifier.  Auto-generated.
    """

    model_config = ConfigDict(frozen=True)

    coordinate_system_type: Literal["coordinate_system"] = "coordinate_system"
    name: str
    axes: tuple[Axis, ...]
    id: UUID4 = Field(default_factory=uuid4, frozen=True)

    @model_validator(mode="after")
    def _validate_axes(self) -> Self:
        """Require at least one axis and unique axis ids."""
        if not self.axes:
            raise ValueError("A coordinate system must have at least one axis.")
        ids = [axis.id for axis in self.axes]
        if len(set(ids)) != len(ids):
            raise ValueError(
                "Axis ids must be unique within a coordinate system; "
                f"got {len(ids)} axes with {len(set(ids))} distinct ids."
            )
        return self

    @property
    def ndim(self) -> int:
        """Number of axes."""
        return len(self.axes)

    def axis_names(self) -> tuple[str, ...]:
        """Return the axis names, in order."""
        return tuple(axis.name for axis in self.axes)

    def index_of(self, axis_id: UUID4) -> int:
        """Return the index of the axis with the given id.

        Parameters
        ----------
        axis_id : UUID4
            The id to look up.

        Returns
        -------
        int
            The index of the axis in ``axes``.

        Raises
        ------
        KeyError
            If no axis in this system has that id.
        """
        for index, axis in enumerate(self.axes):
            if axis.id == axis_id:
                return index
        raise KeyError(
            f"No axis with id {axis_id} in coordinate system '{self.name}' "
            f"(axes: {self.axis_names()})."
        )

    def axis_by_name(self, name: str) -> Axis:
        """Return the single axis with the given name.

        Axis names are not required to be unique, so ambiguity is a
        property of the query rather than of the model: this raises
        instead of returning the first match.

        Parameters
        ----------
        name : str
            The axis name to look up.

        Returns
        -------
        Axis
            The matching axis.

        Raises
        ------
        KeyError
            If no axis has that name.
        ValueError
            If more than one axis has that name.
        """
        matches = [axis for axis in self.axes if axis.name == name]
        if not matches:
            raise KeyError(
                f"No axis named '{name}' in coordinate system '{self.name}' "
                f"(axes: {self.axis_names()})."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Axis name '{name}' is ambiguous in coordinate system "
                f"'{self.name}': {len(matches)} axes share it.  Refer to the "
                f"axis by id instead."
            )
        return matches[0]

    def resolve(self, axis: AxisRef) -> int:
        """Resolve an axis reference to an axis index in this system.

        Parameters
        ----------
        axis : AxisRef
            An axis name or an axis id.

        Returns
        -------
        int
            The index of the referenced axis.

        Raises
        ------
        KeyError
            If the reference does not name an axis of this system.
        ValueError
            If a name is ambiguous within this system.
        """
        if isinstance(axis, UUID):
            return self.index_of(axis)
        return self.index_of(self.axis_by_name(axis).id)

    def resolve_axis(self, axis: AxisRef) -> Axis:
        """Resolve an axis reference to the :class:`Axis` itself.

        Parameters
        ----------
        axis : AxisRef
            An axis name or an axis id.

        Returns
        -------
        Axis
            The referenced axis.
        """
        return self.axes[self.resolve(axis)]


class DataCoordinateSystem(CoordinateSystem):
    """The intrinsic coordinate system of a datastore (e.g. voxel indices).

    Parameters
    ----------
    coordinate_system_type : Literal["data"]
        Discriminator field.
    datastore_id : UUID4
        The id of the datastore this system belongs to.  ``name`` is the
        datastore's name by convention, but the model has no access to
        the datastore and does not validate that.
    """

    coordinate_system_type: Literal["data"] = "data"
    datastore_id: UUID4


class VisualCoordinateSystem(CoordinateSystem):
    """The space one visual's GPU geometry is indexed in.

    This is the space ``node.local.matrix`` maps *from*: the index space
    of the array a datastore returned, or the normalized proxy-box space
    a multiscale volume's vertex shader emits.  It is **not** the data
    coordinate system -- a request drops collapsed axes (numpy applies an
    integer index and the axis disappears) and may start at a non-zero
    origin.

    There is **one per visual per render mode**, not one per visual: a
    multiscale visual's 3D node is indexed in normalized space while its
    2D node is in level-0 pixel coordinates.  Everything that varies per
    request -- the collapsed voxel indices and the window origin -- lives
    on the paired ``visual -> data`` transform, not here, so a system is
    rebuilt only when the displayed axes change.

    Deliberately **not** per chunk: a multiscale brick's position never
    exists as a CPU-side transform, and modelling one per brick would
    produce objects no consumer reads.

    Unlike :class:`RenderedCoordinateSystem` there is no rank check.  A
    rendered system is 2D or 3D because that is what a camera draws; a
    visual system is whatever rank the request retained.

    Parameters
    ----------
    coordinate_system_type : Literal["visual"]
        Discriminator field.
    name : str
        Human-readable name.  Defaults to ``"visual"``.
    visual_id : UUID4
        The id of the visual whose geometry is indexed in this space.
    """

    coordinate_system_type: Literal["visual"] = "visual"
    name: str = "visual"
    visual_id: UUID4

    @classmethod
    def from_data(
        cls,
        data_coordinate_system: DataCoordinateSystem,
        retained_axes: Sequence[AxisRef],
        visual_id: UUID4,
        name: str = "visual",
    ) -> Self:
        """Build a visual system from the data axes a request retains.

        Mirrors :meth:`RenderedCoordinateSystem.from_world` exactly: each
        visual axis corresponds to one data axis, so ``name``,
        ``axis_type`` and ``unit`` are inherited from it rather than
        re-specified, and the axis ids are **fresh** because these are
        distinct axes of a distinct system.

        Parameters
        ----------
        data_coordinate_system : DataCoordinateSystem
            The data system the array is drawn from.
        retained_axes : Sequence[AxisRef]
            The data axes the request keeps, **in the order the returned
            array carries them**.  Collapsed axes are absent; they are
            pinned on the paired ``visual -> data`` transform as
            ``constant_output_axes``.
        visual_id : UUID4
            The id of the visual this space belongs to.
        name : str
            Human-readable name for the new system.

        Returns
        -------
        VisualCoordinateSystem
            A system of the same rank as ``retained_axes``.

        Raises
        ------
        ValueError
            If ``retained_axes`` names the same data axis twice.
        """
        indices = [data_coordinate_system.resolve(ref) for ref in retained_axes]
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"retained_axes must name distinct data axes, got "
                f"{list(retained_axes)}."
            )
        axes = tuple(
            Axis(
                name=data_coordinate_system.axes[index].name,
                axis_type=data_coordinate_system.axes[index].axis_type,
                unit=data_coordinate_system.axes[index].unit,
            )
            for index in indices
        )
        return cls(name=name, axes=axes, visual_id=visual_id)


class WorldCoordinateSystem(CoordinateSystem):
    """A coordinate system that data can be transformed into for rendering.

    Parameters
    ----------
    coordinate_system_type : Literal["world"]
        Discriminator field.
    name : str
        Human-readable name.  Defaults to ``"world"``.
    """

    coordinate_system_type: Literal["world"] = "world"
    name: str = "world"


class RenderedCoordinateSystem(CoordinateSystem):
    """The 2D or 3D scene drawn on one canvas.

    This is scene space in world units; the camera transform from scene
    space to normalized device coordinates is a separate concern and is
    not modelled here.

    There is deliberately no ``world_coordinate_system`` backref: the link
    to the world system is carried by the transform between them, which
    already stores both ids.

    Parameters
    ----------
    coordinate_system_type : Literal["rendered"]
        Discriminator field.
    name : str
        Human-readable name.  Defaults to ``"rendered"``.
    canvas_id : UUID4
        The id of the canvas this system is drawn on.  One system per
        canvas.
    """

    coordinate_system_type: Literal["rendered"] = "rendered"
    name: str = "rendered"
    canvas_id: UUID4

    @model_validator(mode="after")
    def _validate_rank(self) -> Self:
        """Require a rank of 2 or 3, which is what a camera can draw."""
        if self.ndim not in (2, 3):
            raise ValueError(
                f"A rendered coordinate system must have 2 or 3 axes, got "
                f"{self.ndim} ({self.axis_names()})."
            )
        return self

    @classmethod
    def from_world(
        cls,
        world_coordinate_system: WorldCoordinateSystem,
        displayed_axes: Sequence[AxisRef],
        canvas_id: UUID4,
        name: str = "rendered",
    ) -> Self:
        """Build a rendered system from the world axes it displays.

        Each rendered axis corresponds to exactly one world axis, so
        ``name``, ``axis_type`` and ``unit`` are inherited from that world
        axis rather than re-specified.  The axis ids are **fresh**: these
        are distinct axes of a distinct coordinate system, and reusing the
        world ids would make ``index_of`` ambiguous across systems.

        Parameters
        ----------
        world_coordinate_system : WorldCoordinateSystem
            The world system being displayed.
        displayed_axes : Sequence[AxisRef]
            The world axes to display, **in rendered axis order**.  The
            order is the point: it is where the renderer's axis
            convention stops being an unwritten rule.
        canvas_id : UUID4
            The id of the canvas this system is drawn on.
        name : str
            Human-readable name for the new system.

        Returns
        -------
        RenderedCoordinateSystem
            A system of the same rank as ``displayed_axes``.

        Raises
        ------
        ValueError
            If ``displayed_axes`` names the same world axis twice, or has
            a rank other than 2 or 3.
        """
        indices = [world_coordinate_system.resolve(ref) for ref in displayed_axes]
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"displayed_axes must name distinct world axes, got "
                f"{list(displayed_axes)}."
            )
        axes = tuple(
            Axis(
                name=world_coordinate_system.axes[index].name,
                axis_type=world_coordinate_system.axes[index].axis_type,
                unit=world_coordinate_system.axes[index].unit,
            )
            for index in indices
        )
        return cls(name=name, axes=axes, canvas_id=canvas_id)


CoordinateSystemType = Annotated[
    Union[
        CoordinateSystem,
        DataCoordinateSystem,
        VisualCoordinateSystem,
        WorldCoordinateSystem,
        RenderedCoordinateSystem,
    ],
    Field(discriminator="coordinate_system_type"),
]
"""Discriminated union over every coordinate system kind."""
