"""Dimension and coordinate system models for cellier v2."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union
from uuid import uuid4

from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, Field, model_validator

from cellier._state import (
    AxisAlignedSelectionState,
    DimsState,
    PlaneSelectionState,
)
from cellier.transform import (
    Axis,
    ConvexRegion,
    RegionSelection,
    WorldCoordinateSystem,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cellier.transform import (
        AffineTransform,
        AxisType,
        RenderedCoordinateSystem,
    )


#: One axis of a world coordinate system, stated explicitly.
#:
#: Either a built :class:`~cellier.transform.Axis` or a
#: ``(name, axis_type)`` pair.  There is no bare-string form: ``Axis.axis_type``
#: has no default, and a channel or time axis silently typed ``"space"`` is
#: rejected far from where it was written -- by ``from_axis_map``, which
#: refuses to map axes whose types disagree.
AxisSpec = Union["Axis", tuple[str, "AxisType"]]

#: What every public entry point accepts where a world used to take labels.
WorldAxesLike = Union["WorldCoordinateSystem", "Sequence[AxisSpec]"]


def spatial_axes(*names: str) -> tuple[Axis, ...]:
    """Build spatial axes from their names.

    The shorthand for the common all-spatial world -- ``spatial_axes("z", "y",
    "x")`` -- where writing ``("z", "space")`` three times says nothing extra.
    Mixed worlds spell the non-spatial axes out::

        axes = [("t", "time"), ("c", "channel"), *spatial_axes("z", "y", "x")]

    Parameters
    ----------
    *names : str
        The axis names, in order.

    Returns
    -------
    tuple[Axis, ...]
        One ``axis_type="space"`` axis per name, each with a fresh id.
    """
    return tuple(Axis(name=name, axis_type="space") for name in names)


def world_coordinate_system(
    axes: WorldAxesLike, name: str = "world"
) -> WorldCoordinateSystem:
    """Coerce an axis specification into a world coordinate system.

    A :class:`WorldCoordinateSystem` passes through unchanged, so its axis
    **ids** survive -- which matters, because every stored transform names its
    endpoints by id.  Anything else builds a new system with fresh ids.

    Parameters
    ----------
    axes : WorldAxesLike
        A ``WorldCoordinateSystem``, or a sequence of ``Axis`` objects and/or
        ``(name, axis_type)`` pairs.
    name : str
        Name for the system when one is built.  Ignored when *axes* is
        already a ``WorldCoordinateSystem``.

    Returns
    -------
    WorldCoordinateSystem
        The world system.

    Raises
    ------
    TypeError
        If an entry is neither an ``Axis`` nor a ``(name, axis_type)`` pair.
        A bare string raises here, naming ``spatial_axes`` as the fix.
    """
    if isinstance(axes, WorldCoordinateSystem):
        return axes
    built: list[Axis] = []
    for entry in axes:
        if isinstance(entry, Axis):
            built.append(entry)
            continue
        if isinstance(entry, str):
            raise TypeError(
                f"World axes must state their type: got the bare name "
                f"{entry!r}.  Use spatial_axes({entry!r}, ...) for spatial "
                f"axes, or pass ({entry!r}, 'time') / ({entry!r}, 'channel') "
                f"pairs."
            )
        try:
            axis_name, axis_type = entry
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Each world axis must be an Axis or a (name, axis_type) "
                f"pair; got {entry!r}."
            ) from error
        built.append(Axis(name=axis_name, axis_type=axis_type))
    return WorldCoordinateSystem(name=name, axes=tuple(built))


#: Half-thickness used for an axis with no entry in ``thickness``.
#:
#: Matches the number the geometry request builders hardcoded before D4, so
#: an identity transform slices exactly as it did.
DEFAULT_HALF_THICKNESS = 0.5


class AxisAlignedSelection(EventedModel):
    """Mutable selection model for axis-aligned slicing.

    Parameters
    ----------
    selector_type : Literal["axis_aligned"]
        Discriminator field.
    displayed_axes : tuple[int, ...]
        Indices into the coordinate system that are rendered.
        Length 2 -> 2D; length 3 -> 3D.
    slice_indices : dict[int, float]
        Mapping of axis index -> **world-space slice position**, for **every**
        world axis, displayed or not (unified image design D36).  A displayed
        axis keeps the position it slices at once it stops being displayed,
        so a 3D -> 2D switch needs no bookkeeping.  A world position, not a
        voxel index: an integer-valued slider on a 0.5 world-unit-per-voxel
        axis cannot address the odd-numbered planes (D3).
    thickness : dict[int, float]
        Mapping of axis index -> **half**-thickness in world units.  An axis
        absent from the mapping uses :data:`DEFAULT_HALF_THICKNESS`.  Per axis
        rather than scalar because one number means three frames on a
        0.5 s/frame time axis and a quarter of a voxel on a 2 um/voxel spatial
        one (D4).
    """

    selector_type: Literal["axis_aligned"] = "axis_aligned"
    displayed_axes: tuple[int, ...]
    slice_indices: dict[int, float] = Field(default_factory=dict)
    thickness: dict[int, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_displayed_rank(self) -> AxisAlignedSelection:
        """Display rank must be 2 or 3 (the GPU/camera constraint)."""
        n = len(self.displayed_axes)
        if n not in (2, 3):
            raise ValueError(
                f"displayed_axes must have length 2 or 3, got {n} "
                f"(displayed_axes={self.displayed_axes})"
            )
        if len(set(self.displayed_axes)) != n:
            raise ValueError(
                f"displayed_axes must contain distinct indices, "
                f"got {self.displayed_axes}"
            )
        for axis, half_thickness in self.thickness.items():
            if half_thickness < 0:
                raise ValueError(
                    f"thickness is a half-thickness in world units and must "
                    f"not be negative; got {half_thickness} on axis {axis}."
                )
        return self

    def half_thickness(self, axis: int) -> float:
        """Return the world-unit half-thickness for *axis*.

        Parameters
        ----------
        axis : int
            World axis index.

        Returns
        -------
        float
            The stored half-thickness, or :data:`DEFAULT_HALF_THICKNESS` when
            the axis has none.
        """
        return float(self.thickness.get(axis, DEFAULT_HALF_THICKNESS))

    def to_state(self) -> AxisAlignedSelectionState:
        """Return an immutable snapshot of this selection."""
        return AxisAlignedSelectionState(
            displayed_axes=self.displayed_axes,
            thickness=dict(self.thickness),
        )


class PlaneSelection(EventedModel):
    """Stub -- not yet implemented.

    Parameters
    ----------
    selector_type : Literal["plane"]
        Discriminator field.
    """

    selector_type: Literal["plane"] = "plane"

    def to_state(self) -> PlaneSelectionState:
        """Return an immutable snapshot of this selection."""
        raise NotImplementedError("PlaneSelection is not yet implemented.")


SelectionType = Annotated[
    Union[AxisAlignedSelection, PlaneSelection],
    Field(discriminator="selector_type"),
]


class DimsManager(EventedModel):
    """Tracks which axes are displayed and the slice index for non-displayed axes.

    Single source of truth for render dimensionality, and -- since D2 -- the
    owner of the scene's world coordinate system.  One ``DimsManager`` per
    ``Scene``, so there is no second claimant for the world to arbitrate
    against.

    Parameters
    ----------
    id : UUID4
        Unique identifier. Auto-generated.
    world_coordinate_system : WorldCoordinateSystem
        The scene's world axes.  Frozen: changing the axes means assigning a
        replacement (D7), which is why there is no event relay onto it.
    selection : SelectionType
        The current axis selection (axis-aligned or plane).
    slider_overrides : dict[int, bool]
        Per world axis, whether to force its slider shown (``True``) or
        hidden (``False``).  An axis absent from the mapping is automatic:
        it gets a slider when a visual slices it.  Only the overrides are
        stored; the effective set is ``Scene.slider_axes``, which is derived
        from the visuals and never serialized (D23).
    """

    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    world_coordinate_system: WorldCoordinateSystem
    selection: SelectionType
    slider_overrides: dict[int, bool] = Field(default_factory=dict)

    @property
    def axis_labels(self) -> tuple[str, ...]:
        """The world axis names, in order."""
        return self.world_coordinate_system.axis_names()

    @property
    def ndim(self) -> int:
        """The number of world axes."""
        return self.world_coordinate_system.ndim

    @model_validator(mode="after")
    def _validate_axis_coverage(self) -> DimsManager:
        """Every world axis has a slice position, and nothing else does (D36).

        A displayed axis keeps its entry, so ``slice_indices`` alone covers
        the world.  Overrides must name world axes too.
        """
        ndim = self.world_coordinate_system.ndim
        expected = set(range(ndim))
        if isinstance(self.selection, AxisAlignedSelection):
            covered = set(self.selection.slice_indices.keys())
            if covered != expected:
                raise ValueError(
                    f"Axis coverage mismatch: slice_indices.keys() = "
                    f"{sorted(covered)}, expected every world axis "
                    f"{sorted(expected)} for ndim={ndim}.  A displayed axis "
                    f"keeps a slice position too."
                )
            if not set(self.selection.displayed_axes) <= expected:
                raise ValueError(
                    f"displayed_axes {self.selection.displayed_axes} names an "
                    f"axis outside the world (ndim={ndim})."
                )
        unknown = set(self.slider_overrides) - expected
        if unknown:
            raise ValueError(
                f"slider_overrides names axes {sorted(unknown)} outside the "
                f"world (ndim={ndim})."
            )
        return self

    def model_post_init(self, __context: Any) -> None:
        """Wire event relays after model initialization.

        Only the selection needs one.  ``world_coordinate_system`` is a frozen
        ``BaseModel`` with no ``events`` group: it cannot change in place, and
        assigning a replacement already emits this model's own field signal.
        """
        self.selection.events.all.connect(self._on_selection_updated)

    def _on_selection_updated(self, info: Any) -> None:
        self.events.selection.emit(self.selection)

    def to_state(self) -> DimsState:
        """Return an immutable snapshot of the current dims state."""
        return DimsState(
            axis_labels=self.axis_labels,
            selection=self.selection.to_state(),
        )

    def to_selection(
        self,
        rendered_coordinate_system: RenderedCoordinateSystem,
        rendered_to_world: AffineTransform,
    ) -> RegionSelection:
        """Emit the region one canvas is showing, as the artifact the slicer takes.

        This model is the **editor** -- it holds an index and a thickness, the
        thing a GUI binds to -- and the slicer consumes one type regardless of
        which editor produced it (D43, R6).  An oblique editor would hold a
        plane and emit the same type.

        The rendered system arrives as an argument rather than being reached
        for: it is per canvas and is not model state, and this way the model
        layer never reaches into the render layer.

        **Thickness comes from the stored mapping, and an axis without an
        entry is a plane.**  ``half_thickness`` defaults to 0.5 because that
        is the number the geometry request builders hardcoded, but a region
        built at that default would give every collapsed axis a one-unit slab
        -- and an image visual, which draws a single plane, would start
        fetching more than it can show.  So the region says exactly what the
        user asked for, and nothing when they asked for nothing.

        Displayed axes are left unbounded, although they keep a stored
        position (D36).  Bounding one is the viewport-crop follow-up; the
        region type already expresses it (R3).

        Parameters
        ----------
        rendered_coordinate_system : RenderedCoordinateSystem
            The system the canvas draws in, in cellier displayed order.
        rendered_to_world : AffineTransform
            The D34 embedding.  Its constant column carries the slice
            positions.

        Returns
        -------
        RegionSelection
            The transform and the region, in world coordinates.

        Raises
        ------
        NotImplementedError
            If the selection is not axis aligned.
        """
        if not isinstance(self.selection, AxisAlignedSelection):
            raise NotImplementedError(
                f"{type(self.selection).__name__} cannot emit a RegionSelection "
                f"yet; only AxisAlignedSelection can."
            )
        world = self.world_coordinate_system
        slabs = {
            world.axes[axis].id: (
                float(position),
                float(self.selection.thickness.get(axis, 0.0)),
            )
            for axis, position in self.selection.slice_indices.items()
            if axis not in self.selection.displayed_axes
        }
        return RegionSelection(
            transform=rendered_to_world,
            region=ConvexRegion.from_axis_slabs(world, slabs),
        )
