"""Geometry utilities for computing world-space extents from a Viewer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from cellier.gui._axis_values import ContinuousAxisValues, DiscreteAxisValues
from cellier.scene._bounds import scene_world_bounds

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from cellier.controller import CellierController
    from cellier.convenience._ortho_viewer import OrthoViewer
    from cellier.convenience._viewer import Viewer
    from cellier.scene.scene import Scene
    from cellier.transform import BaseTransform, WorldCoordinateSystem

MAX_DISCRETE_VALUES = 10_000
"""The most sample positions a discrete slider is built from.

Past this an axis falls back to a continuous slider: a discrete one would be
a list no one could step through, and building it would cost a transform per
sample.
"""

_SAME_POSITION_TOLERANCE = 1e-9
"""Relative tolerance under which two stores' sample positions are one value."""


def _axis_values_from_scene(
    controller: CellierController, scene: Scene
) -> dict[int, ContinuousAxisValues | DiscreteAxisValues]:
    """Compute the slider values of every world axis from a single scene.

    Maps each visual's backing store's per-axis extents from data space to
    world space and returns the per-axis union.

    Every store contributes, gridded or geometry alike, because every store
    answers :attr:`~cellier.data._base_data_store.BaseDataStore.axis_extents`.
    That was not true before: this function used to read ``level_shapes``,
    which only image and label stores have, so a scene built solely from
    points, graph, mesh or lines visuals raised rather than returning a
    range.

    The extents use the **edge** convention -- a gridded axis of ``size``
    voxels spans ``[-0.5, size - 0.5]`` -- matching
    :func:`~cellier.render.visuals._slicing.round_world_to_voxel` and the
    rest of the render layer.  Reading the corner *centres* ``(0, size - 1)``,
    as this function used to, understated every gridded axis by half a voxel
    at each end.

    The union across visuals is what makes a mixed-extent scene work: one
    visual's data may end before another's, and the slider still has to
    cover both.

    **Discrete axes.**  A world axis that is not a ``"space"`` axis gets a
    :class:`DiscreteAxisValues` when every visual reaching it maps a
    ``sampling="discrete"`` data axis onto it -- an OME-Zarr channel or time
    axis, or a tracking graph's frame column.  Its values are the union of
    those samples' world positions, so an irregular time axis steps through
    the frames it has.  When every store that names its channels agrees on a
    position's name (``OMEZarrImageDataStore.channel_labels``), the values are
    labelled with them.  One continuous contributor, or more than
    :data:`MAX_DISCRETE_VALUES` samples, keeps the axis continuous.  Space
    axes are always continuous: every voxel grid is discrete, and a z slider
    that stepped voxel by voxel through a volume is not what anyone wants.

    Parameters
    ----------
    controller : CellierController
        Controller owning the data stores referenced by the scene's visuals.
    scene : Scene
        The scene whose visuals are inspected.

    Returns
    -------
    dict[int, ContinuousAxisValues | DiscreteAxisValues]
        Mapping of axis index to that axis's slider values.

    Raises
    ------
    ValueError
        If no visual in the scene has any extent -- an empty scene, or one
        whose stores all hold zero vertices.
    """
    ndim = len(scene.dims.axis_labels)
    world = scene.dims.world_coordinate_system

    # The continuous range is the scene's world bounding box -- the same rule
    # the scene bounding-box overlay draws (cellier.scene._bounds).
    bounds = scene_world_bounds(scene, controller.get_data_store)
    if bounds is None:
        raise ValueError(
            "No visuals with extents found.  Every visual's data store "
            "reported no data at all, or the scene has no visuals."
        )
    # An axis no visual reaches -- every contributor broadcasts over it --
    # has no extent.  It still needs a slider range, and the origin is where
    # a broadcast dataset's zero row put it before broadcasts were skipped.
    world_mins, world_maxs = (np.nan_to_num(edge, nan=0.0) for edge in bounds)

    # world axis -> the sample positions discrete contributors put there, or
    # None once any contributor rules a discrete slider out.
    samples: dict[int, list[float] | None] = {}
    # world axis -> (position, channel name) from stores that name channels.
    named: dict[int, list[tuple[float, str]]] = {}

    for visual_model in scene.visuals:
        store = controller.get_data_store(UUID(str(visual_model.data_store_id)))
        extents = store.axis_extents
        if extents is None:
            continue
        _collect_samples(store, visual_model.transform, extents, world, samples, named)

    values: dict[int, ContinuousAxisValues | DiscreteAxisValues] = {
        i: ContinuousAxisValues(min=float(world_mins[i]), max=float(world_maxs[i]))
        for i in range(ndim)
    }
    for axis, positions in samples.items():
        discrete = _discrete_values(positions, named.get(axis))
        if discrete is not None:
            values[axis] = discrete
    return values


def _collect_samples(
    store: object,
    transform: BaseTransform,
    extents: tuple[tuple[float, float], ...],
    world: WorldCoordinateSystem,
    samples: dict[int, list[float] | None],
    named: dict[int, list[tuple[float, str]]],
) -> None:
    """Record where one store's samples land on the non-space world axes.

    A data axis contributes its integer sample indices inside its extent,
    mapped through *transform* onto the world axis it reaches.  The other data
    axes are held at the centre of their extents, which changes nothing for
    an axis-aligned transform and keeps a lookup-table axis inside its table.
    A world axis the store reaches through a continuous data axis is marked
    ``None``, which no later store can undo.
    """
    try:
        correspondence = transform.axis_correspondence()
    except ValueError:
        # A shear or rotation: which world axis each data axis becomes is not
        # known, so no non-space axis this store might reach can be discrete.
        for index, axis in enumerate(world.axes):
            if axis.axis_type != "space":
                samples[index] = None
        return

    systems = getattr(store, "data_coordinate_systems", None) or []
    data_axes = systems[0].axes if systems else None
    channel_labels = getattr(store, "channel_labels", None)
    centre = np.array([(low + high) / 2.0 for low, high in extents], dtype=np.float64)

    for data_axis, world_axis in correspondence.items():
        if world.axes[world_axis].axis_type == "space":
            continue
        if world_axis in samples and samples[world_axis] is None:
            continue
        low, high = extents[data_axis]
        indices = np.arange(math.ceil(low), math.floor(high) + 1)
        axis = data_axes[data_axis] if data_axes is not None else None
        if (
            axis is None
            or axis.sampling != "discrete"
            or not 0 < indices.size <= MAX_DISCRETE_VALUES
        ):
            samples[world_axis] = None
            continue

        points = np.tile(centre, (indices.size, 1))
        points[:, data_axis] = indices
        positions = transform.map_coordinates(points)[:, world_axis]
        if not np.all(np.isfinite(positions)):
            samples[world_axis] = None
            continue
        samples.setdefault(world_axis, []).extend(float(p) for p in positions)

        if (
            channel_labels
            and axis.axis_type == "channel"
            and indices[0] == 0
            and len(channel_labels) == indices.size
        ):
            named.setdefault(world_axis, []).extend(
                zip((float(p) for p in positions), channel_labels)
            )


def _discrete_values(
    positions: list[float] | None, named: list[tuple[float, str]] | None
) -> DiscreteAxisValues | None:
    """Merge sample positions into slider values, labelled when names agree.

    Positions within :data:`_SAME_POSITION_TOLERANCE` of each other are one
    value -- two stores sharing a transform put the same frame in the same
    place.  Labels are used only when every value has exactly one name.
    """
    if not positions:
        return None
    merged: list[float] = []
    for position in sorted(positions):
        if merged and _same_position(position, merged[-1]):
            continue
        merged.append(position)
    if len(merged) > MAX_DISCRETE_VALUES:
        return None

    labels: list[str] | None = None
    if named:
        labels = []
        for value in merged:
            names = {
                name for position, name in named if _same_position(position, value)
            }
            if len(names) != 1:
                labels = None
                break
            labels.append(names.pop())
    return DiscreteAxisValues(
        values=tuple(merged), labels=tuple(labels) if labels is not None else None
    )


def _same_position(a: float, b: float) -> bool:
    return abs(a - b) <= _SAME_POSITION_TOLERANCE * max(1.0, abs(a), abs(b))


def _resolve_tick_axes(
    draw_ticks: Iterable[str], axis_labels: Sequence[str]
) -> list[int]:
    """Resolve the ``draw_ticks`` axis names to world axis indices.

    Checked before any extent is measured, so a misspelt name fails on its
    own rather than behind an unrelated error.  Names only: an axis is chosen
    by what it is called, and the returned mapping's integer keys are an
    artefact of the world's axis order.
    """
    if isinstance(draw_ticks, str):
        raise TypeError(
            f"draw_ticks takes a list of axis names, not a single string; "
            f"write draw_ticks=[{draw_ticks!r}]."
        )
    names = list(draw_ticks)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"draw_ticks takes world axis names; got {name!r} "
                f"({type(name).__name__})."
            )
        if name not in axis_labels:
            raise ValueError(
                f"draw_ticks names {name!r}, which is not a world axis. "
                f"World axes: {list(axis_labels)}."
            )
    return [axis_labels.index(name) for name in dict.fromkeys(names)]


def _apply_ticks(
    values: dict[int, ContinuousAxisValues | DiscreteAxisValues],
    tick_axes: list[int],
    axis_labels: Sequence[str],
) -> dict[int, ContinuousAxisValues | DiscreteAxisValues]:
    """Turn ticks on for *tick_axes*, which must all have come out discrete.

    Whether an axis is discrete is decided by the data, not the caller, so a
    requested axis can come out continuous; that raises rather than being
    skipped, because a silent no-op would hide exactly the case where the
    request cannot be met.
    """
    for axis in tick_axes:
        entry = values[axis]
        if not isinstance(entry, DiscreteAxisValues):
            raise ValueError(
                f"draw_ticks names {axis_labels[axis]!r}, but that axis has a "
                f"continuous slider, which has no ticks.  An axis is discrete "
                f"only when it is not a space axis, every visual reaching it "
                f"samples it discretely (sampling='discrete' on the data axis), "
                f"and it has at most {MAX_DISCRETE_VALUES} samples."
            )
        # The model is frozen; the copy also re-runs the dense-ticks warning.
        values[axis] = DiscreteAxisValues(**{**entry.model_dump(), "draw_ticks": True})
    return values


def axis_values_from_viewer(
    viewer: Viewer,
    *,
    draw_ticks: Iterable[str] = (),
) -> dict[int, ContinuousAxisValues | DiscreteAxisValues]:
    """Compute every world axis's slider values from the viewer's visuals.

    Transforms each visual's data extent into world space and takes the union
    per world axis.  A non-space axis that every visual reaching it samples
    discretely -- an OME-Zarr channel or time axis, say -- gets a
    :class:`DiscreteAxisValues` stepping through those samples, labelled with
    the channel names when the stores provide them.  See
    :func:`_axis_values_from_scene` for the exact rule.

    Parameters
    ----------
    viewer : Viewer
        The viewer to inspect.  Must have at least one visual whose store
        holds data.
    draw_ticks : Iterable[str]
        Names of world axes whose sliders mark each value with a tick, e.g.
        ``["c"]``.  Empty (default) draws none.  Each named axis must come out
        discrete.  Ticks cost per mark on every repaint, so name short axes
        such as a channel axis; see
        :data:`~cellier.gui._axis_values.TICK_WARNING_LIMIT`.

    Returns
    -------
    dict[int, ContinuousAxisValues | DiscreteAxisValues]
        Mapping of axis index to that axis's slider values.

    Raises
    ------
    ValueError
        If no visual holds any data, if *draw_ticks* names an axis the world
        does not have, or if a named axis comes out continuous.
    TypeError
        If *draw_ticks* is a single string or holds something other than
        axis names.
    """
    axis_labels = viewer.scene.dims.axis_labels
    tick_axes = _resolve_tick_axes(draw_ticks, axis_labels)
    values = _axis_values_from_scene(viewer.controller, viewer.scene)
    return _apply_ticks(values, tick_axes, axis_labels)


def axis_values_from_ortho(
    ortho: OrthoViewer,
    *,
    draw_ticks: Iterable[str] = (),
) -> dict[int, ContinuousAxisValues | DiscreteAxisValues]:
    """Compute every world axis's slider values for an :class:`OrthoViewer`.

    The four panels share their data stores, so the values are identical
    across panels.  The first panel that has visuals with data is used.

    Parameters
    ----------
    ortho : OrthoViewer
        The orthoviewer to inspect.  Must have at least one visual whose
        store holds data on some panel.
    draw_ticks : Iterable[str]
        Names of world axes whose sliders mark each value with a tick.  See
        :func:`axis_values_from_viewer`.

    Returns
    -------
    dict[int, ContinuousAxisValues | DiscreteAxisValues]
        Mapping of axis index to that axis's slider values.

    Raises
    ------
    ValueError
        If no panel has a visual with data, if *draw_ticks* names an axis the
        world does not have, or if a named axis comes out continuous.
    TypeError
        If *draw_ticks* is a single string or holds something other than
        axis names.
    """
    scenes = list(ortho.scenes.values())
    # The panels share one world, so any panel's labels resolve the names.
    # Resolved outside the loop below, whose ValueError means "no data here,
    # try the next panel" and would otherwise swallow a bad name.
    axis_labels = scenes[0].dims.axis_labels if scenes else ()
    tick_axes = _resolve_tick_axes(draw_ticks, axis_labels)
    for scene in scenes:
        try:
            values = _axis_values_from_scene(ortho.controller, scene)
        except ValueError:
            continue
        return _apply_ticks(values, tick_axes, axis_labels)
    raise ValueError(
        "No visuals with known shapes found on any orthoviewer panel. "
        "Add an image or label visual before computing axis ranges."
    )
