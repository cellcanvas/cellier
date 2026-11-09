"""Naming the endpoints of a transform, for tests that drive the render layer.

Most of the suite reaches the render layer through ``CellierController``,
which builds a ``data -> world`` transform between the store's coordinate
system and the scene's world.  The render-layer unit tests construct a GFX
visual directly, with no controller and no scene, so they have to say what the
two spaces are themselves.

These helpers build synthetic ones.  They are deliberately positional -- data
axis *i* is world axis *i* -- which is what the tests that use them assume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from cellier.render._spaces import build_render_spaces
from cellier.transform import (
    AffineTransform,
    Axis,
    ConvexRegion,
    DataCoordinateSystem,
    RegionSelection,
    RenderedCoordinateSystem,
    VisualCoordinateSystem,
    WorldCoordinateSystem,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cellier.transform import AxisSampling

_AXIS_TYPE = {"t": "time", "c": "channel"}


def _labels(ndim: int) -> tuple[str, ...]:
    """Conventional axis names for a rank: ``(t, c, z, y, x)`` from the right."""
    conventional = ("t", "c", "z", "y", "x")
    if ndim <= len(conventional):
        return conventional[len(conventional) - ndim :]
    return tuple(f"axis_{index}" for index in range(ndim))


def axes(labels: Sequence[str]) -> tuple[Axis, ...]:
    """Build axes, typing ``t`` and ``c`` and calling everything else spatial."""
    return tuple(
        Axis(name=label, axis_type=_AXIS_TYPE.get(label, "space")) for label in labels
    )


def data_system(
    labels: Sequence[str],
    *,
    sampling: AxisSampling = "continuous",
    name: str = "data",
) -> DataCoordinateSystem:
    """A store's coordinate system, for ``data_coordinate_systems=[...]``.

    Axes are typed as :func:`axes` types them.  The ``datastore_id`` is fresh,
    and the store the system is passed to adopts it as its own id.
    """
    return DataCoordinateSystem(
        name=name,
        axes=tuple(
            Axis(
                name=label,
                axis_type=_AXIS_TYPE.get(label, "space"),
                sampling=sampling,
            )
            for label in labels
        ),
        datastore_id=uuid4(),
    )


def systems(
    ndim: int, labels: Sequence[str] | None = None
) -> tuple[DataCoordinateSystem, WorldCoordinateSystem]:
    """A matching ``(data, world)`` pair of rank *ndim*."""
    names = tuple(labels) if labels is not None else _labels(ndim)
    return (
        DataCoordinateSystem(name="data", axes=axes(names), datastore_id=uuid4()),
        WorldCoordinateSystem(name="world", axes=axes(names)),
    )


def scale_and_translation(
    scale: Sequence[float],
    translation: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
) -> AffineTransform:
    """A diagonal ``data -> world`` transform between synthetic systems."""
    ndim = len(scale)
    data, world = systems(ndim, labels)
    offsets = tuple(translation) if translation is not None else (0.0,) * ndim
    return AffineTransform.from_axis_map(
        data,
        world,
        axis_map={data.axes[i].id: world.axes[i].id for i in range(ndim)},
        scale={data.axes[i].id: float(scale[i]) for i in range(ndim)},
        translation={data.axes[i].id: float(offsets[i]) for i in range(ndim)},
        name="data_to_world",
    )


def bound(controller, scene_id, store, scale, translation=None) -> AffineTransform:
    """A ``data -> world`` transform between the systems a live viewer uses.

    A transform names its endpoints by id, so one built against synthetic
    systems cannot be handed to a real visual -- composing it would join two
    different worlds, and ``then`` refuses.  This builds one against the
    store's and scene's actual systems, which is what a caller who wants to
    place a visual precisely has to do.

    Parameters
    ----------
    controller : CellierController
        The live controller.
    scene_id : UUID
        The scene whose world the transform maps into.
    store : BaseDataStore
        The store whose voxel space it maps from.
    scale : Sequence[float]
        Per-data-axis scale.
    translation : Sequence[float] or None
        Per-data-axis translation.

    Returns
    -------
    AffineTransform
        The transform, ready to hand to ``add_*`` or assign to
        ``visual.transform``.
    """
    controller._ensure_data_coordinate_systems(scene_id, store)
    data = store.data_coordinate_system
    world = controller._model.scenes[scene_id].dims.world_coordinate_system
    ndim = data.ndim
    offsets = tuple(translation) if translation is not None else (0.0,) * ndim
    return AffineTransform.from_axis_map(
        data,
        world,
        axis_map={data.axes[i].id: world.axes[i].id for i in range(ndim)},
        scale={data.axes[i].id: float(scale[i]) for i in range(ndim)},
        translation={data.axes[i].id: float(offsets[i]) for i in range(ndim)},
        name="data_to_world",
    )


def identity(ndim: int, labels=None) -> AffineTransform:
    """The identity ``data -> world`` transform of a given rank.

    Data axis *i* maps to world axis *i*, between synthetic systems from
    :func:`systems`.
    """
    data, world = systems(ndim, labels)
    return AffineTransform.from_axis_map(
        data,
        world,
        axis_map={data.axes[i].id: world.axes[i].id for i in range(ndim)},
        name="data_to_world",
    )


def pyramid_levels(scales, translations=None) -> dict:
    """Store keyword arguments for a pyramid, as the raw per-level numbers.

    A store states its pyramid as ``level_scales`` and
    ``level_translations`` rather than as transforms: a transform names the
    two coordinate systems it sits between, and a store has none until its
    axes are known.  ``install_level_transforms`` builds the transforms once
    they are.

    Parameters
    ----------
    scales : Sequence[Sequence[float]]
        Per-level, per-axis scale of level-k voxels in level-0 voxels.
    translations : Sequence[Sequence[float]] or None
        The offset half.  ``None`` means all zeros.

    Returns
    -------
    dict
        ``{"level_scales": ..., "level_translations": ...}``.
    """
    scale_rows = [tuple(float(v) for v in row) for row in scales]
    if translations is None:
        offset_rows = [(0.0,) * len(row) for row in scale_rows]
    else:
        offset_rows = [tuple(float(v) for v in row) for row in translations]
    return {"level_scales": scale_rows, "level_translations": offset_rows}


def level_transforms(scales, translations=None, labels=None):
    """Per-level ``level k -> level 0`` transforms, between synthetic systems.

    For a visual model built without a controller.  Through the controller
    the model's list is copied from the store, so it names the store's own
    level systems; a headless test has no store systems to name and these
    stand in.  Only the matrices are read downstream -- the brick grid's
    per-axis scale and translation -- so the substitution is invisible.
    """
    ndim = len(scales[0])
    names = tuple(labels) if labels is not None else _labels(ndim)
    store_id = uuid4()
    systems = [
        DataCoordinateSystem(
            name=f"level{level}", axes=axes(names), datastore_id=store_id
        )
        for level in range(len(scales))
    ]
    offsets = (
        translations if translations is not None else [(0.0,) * ndim] * len(scales)
    )
    return [
        AffineTransform.from_axis_map(
            systems[level],
            systems[0],
            axis_map={
                systems[level].axes[i].id: systems[0].axes[i].id for i in range(ndim)
            },
            scale={
                systems[level].axes[i].id: float(scales[level][i]) for i in range(ndim)
            },
            translation={
                systems[level].axes[i].id: float(offsets[level][i]) for i in range(ndim)
            },
            name=f"level{level}_to_level0",
        )
        for level in range(len(scales))
    ]


def data_region(ndim: int, slabs=None, labels=None) -> ConvexRegion:
    """A region in a store's own **data** space, from per-axis slabs.

    What a geometry slice request carries: the store's whole filter, already
    pulled back, so a world position is never compared against data
    coordinates.

    Parameters
    ----------
    ndim : int
        Rank of the store's positions.
    slabs : Mapping[int, tuple[float, float]] or None
        Data axis to ``(centre, half_thickness)``.  Axes left out are
        unbounded, which is what a displayed axis looks like.
    labels : Sequence[str] or None
        Axis names, for a system that has to match another one by rank.

    Returns
    -------
    ConvexRegion
        Ready to hand to a geometry slice request.
    """
    data, _ = systems(ndim, labels)
    return ConvexRegion.from_axis_slabs(
        data,
        {
            data.axes[axis].id: (float(centre), float(half))
            for axis, (centre, half) in dict(slabs or {}).items()
        },
    )


class Context:
    """A ``data -> world`` transform plus the systems a visual is placed with.

    The render-layer unit tests construct a GFX visual with no controller
    behind it, so nothing pushes it the coordinate systems its node matrix is
    composed through.  This builds the same ones the controller would: the
    world in cellier displayed order, the rendered system from
    ``displayed_axes`` in that order, and the visual system from the retained
    data axes **ascending**.
    """

    def __init__(
        self,
        ndim: int,
        scale: Sequence[float] | None = None,
        translation: Sequence[float] | None = None,
        *,
        displayed_axes: Sequence[int],
        slice_indices: dict[int, float] | None = None,
        labels: Sequence[str] | None = None,
        data: DataCoordinateSystem | None = None,
        world: WorldCoordinateSystem | None = None,
    ) -> None:
        # *data* and *world* let a replacement transform be built against the
        # systems a visual is already placed with; a transform names its
        # endpoints, so a fresh pair would describe a different space.
        if data is None or world is None:
            data, world = systems(ndim, labels)
        self.data, self.world = data, world
        factors = tuple(scale) if scale is not None else (1.0,) * ndim
        offsets = tuple(translation) if translation is not None else (0.0,) * ndim
        self.transform = AffineTransform.from_axis_map(
            self.data,
            self.world,
            axis_map={self.data.axes[i].id: self.world.axes[i].id for i in range(ndim)},
            scale={self.data.axes[i].id: float(factors[i]) for i in range(ndim)},
            translation={self.data.axes[i].id: float(offsets[i]) for i in range(ndim)},
            name="data_to_world",
        )
        self.displayed_axes = tuple(displayed_axes)
        self._canvas_id = uuid4()
        self.rendered = RenderedCoordinateSystem.from_world(
            self.world,
            [self.world.axes[axis].id for axis in self.displayed_axes],
            self._canvas_id,
        )
        positions = dict(slice_indices or {})
        self.slice_indices = positions
        self.rendered_to_world = AffineTransform.from_axis_map(
            self.rendered,
            self.world,
            axis_map={
                self.rendered.axes[index].id: self.world.axes[axis].id
                for index, axis in enumerate(self.displayed_axes)
            },
            constant_output_axes={
                self.world.axes[axis].id: float(positions.get(axis, 0.0))
                for axis in range(ndim)
                if axis not in self.displayed_axes
            },
            name="rendered_to_world",
        )

    @property
    def selection(self) -> RegionSelection:
        """The region one canvas is showing, as the controller would emit it.

        Every image and label family plans from this and nothing else, so a
        render-layer test that builds a slice request has to supply the region
        as well as the systems.

        Each collapsed axis gets a zero-thickness slab -- a plane -- which is
        what ``DimsManager.to_selection`` emits for an axis nobody gave a
        thickness.
        """
        return RegionSelection(
            transform=self.rendered_to_world,
            region=ConvexRegion.from_axis_slabs(
                self.world,
                {
                    self.world.axes[axis].id: (float(position), 0.0)
                    for axis, position in self.slice_indices.items()
                },
            ),
        )

    def place(self, visual) -> None:
        """Hand *visual* the systems the controller would have pushed."""
        retained = sorted(self.displayed_axes)
        visual_system = VisualCoordinateSystem.from_data(
            self.data,
            [self.data.axes[axis].id for axis in retained],
            visual.visual_model_id,
        )
        visual.set_render_spaces(
            build_render_spaces(
                self.data,
                visual_system,
                self.world,
                self.rendered,
                self.rendered_to_world,
                self.transform,
                retained,
            )
        )
