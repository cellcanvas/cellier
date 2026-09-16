# src/cellier/v2/render/visuals/_slicing.py
"""Shared slice-index to voxel-index conversion for render-layer visuals.

All image and label visuals -- in-memory and multiscale, single- and
multi-channel, 2D and 3D -- snap a world-space slice position to an integer
voxel index using a single rule defined here.  Keeping it in one place
guarantees that the same physical world position selects the same data plane
regardless of which visual family renders it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cellier._rounding import round_half_up_clamped
from cellier.transform import ConvexRegion, RegionSelection

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    from cellier.render._spaces import RenderSpaces
    from cellier.transform import AxisAlignedBoundingBox, BaseTransform


def round_world_to_voxel(raw: float, size: int) -> int:
    """Snap a level-k voxel-space position to the nearest voxel index.

    Uses round-half-up: ``floor(raw + 0.5)``.  This is the unique rounding
    rule consistent with the center-at-integer convention used everywhere
    else in the render layer -- voxel ``i`` is centered at integer ``i`` and
    spans ``[i - 0.5, i + 0.5)``, the geometry extent is ``[-0.5, N - 0.5]``,
    and the shader samples at ``pos = local + 0.5``.  Picking the nearest
    voxel center (rather than ``floor(raw)``, which would adopt a
    corner-at-integer convention) keeps the sliced axes on the same grid as
    the displayed axes.

    Ties (exact half-integer positions) round toward ``+inf``, which is
    deterministic and monotonic for slider scrubbing.  The result is clamped
    to the valid index range ``[0, size - 1]``.

    The arithmetic itself lives in
    :func:`cellier._rounding.round_half_up_clamped`, shared with the
    non-uniform axis transform's nearest-neighbour lookup so the two cannot
    drift apart.  This function keeps its name, its voxel-specific meaning
    and every call site.

    Parameters
    ----------
    raw : float
        Slice position already mapped into level-k voxel space.
    size : int
        Extent of the axis at this level (``store_shape[axis]`` for in-memory
        data, ``level_shape[axis]`` for multiscale data).  Used for clamping.

    Returns
    -------
    int
        Voxel index in ``[0, size - 1]``.
    """
    return round_half_up_clamped(raw, size)


def axis_selections_from_box(
    box: AxisAlignedBoundingBox,
    level_shape: tuple[int, ...],
    windows: Mapping[int, tuple[int, int]] | None = None,
) -> tuple[int | tuple[int, int], ...]:
    """Turn a voxel-space bounding box into a datastore's per-axis selection.

    The one assembler for every image and label family, replacing the
    per-family ``_build_axis_selections_*``.  Its input is a box already
    pulled back into level-k voxel space, and design 3.7 step 4 is its three
    branches, per data axis:

    * **unbounded** -- the region does not constrain this axis, so the whole
      extent is fetched.  This is what a displayed axis looks like until the
      viewport crop lands, and it reproduces today's ``(0, shape[axis])``
      exactly.
    * **collapsed** (``lo == hi``) -- one plane.  The position is snapped with
      :func:`round_world_to_voxel`, the same half-up rule and the same
      function as before this migration, and the axis is dropped from the
      returned array by ``get_data``.
    * **a real slab** -- the third branch, reached when a thickness is asked
      for.  The bounds are widened to whole voxels: any voxel the slab touches
      is fetched.

    An axis that comes back **half** bounded on a collapsed axis is a shear:
    the region's preimage is a slanted hyperplane with no single voxel index
    on it, and there is no honest answer.  See D7.

    Parameters
    ----------
    box : AxisAlignedBoundingBox
        The region's bounds in level-k voxel space, one entry per data axis.
    level_shape : tuple[int, ...]
        Extent of the data at this level, for clamping.
    windows : Mapping[int, tuple[int, int]] or None
        Per-axis ``(start, stop)`` overrides for axes whose window comes from
        somewhere other than the region -- a multiscale brick's padded grid
        window, say.  These win over the box.

    Returns
    -------
    tuple[int | tuple[int, int], ...]
        One entry per data axis, in ascending data-axis order.

    Raises
    ------
    ValueError
        If an axis is bounded on one side only.
    """
    windows = dict(windows or {})
    result: list[int | tuple[int, int]] = []
    for axis, size in enumerate(level_shape):
        if axis in windows:
            result.append(tuple(int(value) for value in windows[axis]))
            continue
        low = float(box.min_coordinate[axis])
        high = float(box.max_coordinate[axis])
        low_unbounded = np.isneginf(low)
        high_unbounded = np.isposinf(high)
        if low_unbounded and high_unbounded:
            result.append((0, int(size)))
        elif low_unbounded or high_unbounded:
            raise ValueError(
                f"Axis {axis} of this region is bounded on one side only "
                f"([{low}, {high}]).  That happens when the data -> world "
                f"transform has a cross-term on a collapsed axis: the "
                f"preimage of the slice plane is a slanted hyperplane and no "
                f"single voxel index lies on it.  Remove the shear -- "
                f"_check_transform_no_rotation imposes the same restriction "
                f"on the multiscale brick shader -- or wait for the oblique "
                f"path."
            )
        elif low == high:
            result.append(round_world_to_voxel(low, int(size)))
        else:
            # Both ends are clamped into [0, size], so a slab that misses the
            # data entirely comes back as an in-range empty window rather than
            # an out-of-range or inverted one.
            start = min(max(0, int(np.floor(low + 0.5))), int(size))
            stop = min(int(size), int(np.floor(high + 0.5)) + 1)
            result.append((start, max(start, stop)))
    return tuple(result)


def select_plane_within_slab(
    position: float,
    half_thickness: float,
    size: int,
    index_to_world: Callable[[np.ndarray], np.ndarray],
    world_to_index: Callable[[np.ndarray], np.ndarray],
) -> int | None:
    """Choose the one sample an image draws on a sliced axis (design 3.2).

    Sample ``i`` spans ``[i - 0.5, i + 0.5)`` in data units.  It is a
    candidate when that extent, mapped to world, overlaps the band
    ``[position - half_thickness, position + half_thickness]``.  Among the
    candidates the one whose centre maps closest to ``position`` wins.  A tie
    goes to the higher data index, whatever the sign of the world scale, which
    is how :func:`round_world_to_voxel` breaks it (D40).  No candidate means
    the image draws nothing.

    Overlap and distance are measured in world units, so an axis whose samples
    are unevenly spaced in world picks the nearest *world* position, not the
    nearest index.

    The nearest sample to ``position`` is found by pulling the position (held
    inside the data's world extent) back to a data index and rounding it half
    up.  Every candidate lies on the same side of that sample as the band, so
    if the nearest sample does not overlap the band, nothing does.  Its two
    neighbours are checked too, so that a rounding error of one unit in the last
    place at a tie cannot turn a hit into a miss.

    Parameters
    ----------
    position : float
        The slice position, in world units.
    half_thickness : float
        Half the band's width, in world units.  ``0`` is a plane.
    size : int
        The number of samples on the data axis.
    index_to_world : Callable[[np.ndarray], np.ndarray]
        Maps a 1-D array of (possibly fractional) data indices to world
        positions on this axis.  Must be monotonic over
        ``[-0.5, size - 0.5]``.
    world_to_index : Callable[[np.ndarray], np.ndarray]
        The inverse of ``index_to_world`` over that range.

    Returns
    -------
    int or None
        The chosen data index, or ``None`` when no sample overlaps the band.
    """
    if size <= 0:
        return None
    band_low = position - half_thickness
    band_high = position + half_thickness

    edges = np.asarray(index_to_world(np.array([-0.5, size - 0.5])), dtype=float)
    increasing = bool(edges[1] >= edges[0])
    held = min(max(position, float(edges.min())), float(edges.max()))
    raw = float(np.asarray(world_to_index(np.array([held])), dtype=float)[0])
    if not np.isfinite(raw):
        return None
    nearest = round_half_up_clamped(raw, size)

    # Descending index order, so a strict ``<`` below keeps the higher index
    # on a tie.
    candidates = np.array(
        [i for i in (nearest + 1, nearest, nearest - 1) if 0 <= i < size],
        dtype=float,
    )
    lower = np.asarray(index_to_world(candidates - 0.5), dtype=float)
    upper = np.asarray(index_to_world(candidates + 0.5), dtype=float)
    centres = np.asarray(index_to_world(candidates), dtype=float)

    chosen: int | None = None
    best_distance = np.inf
    for index, low_edge, high_edge, centre in zip(
        candidates, lower, upper, centres, strict=True
    ):
        # ``[i - 0.5, i + 0.5)`` is half open in data units.  A negative world
        # scale flips it, so the open end lands on the lower world edge.
        if increasing:
            overlaps = band_low < high_edge and band_high >= low_edge
        else:
            overlaps = band_low <= low_edge and band_high > high_edge
        if not overlaps:
            continue
        distance = abs(float(centre) - position)
        if distance < best_distance:
            chosen = int(index)
            best_distance = distance
    return chosen


def image_plane_selection(
    selection: RegionSelection,
    transform: BaseTransform,
    spaces: RenderSpaces,
    shape: tuple[int, ...],
    exempt_data_axes: Collection[int] = (),
) -> RegionSelection | None:
    """Apply the image slicing rule to a selection (design 3.2).

    Image visuals draw one plane per sliced axis and draw nothing when the
    slice misses the data; labels keep the clamping assembler.  For every world
    axis the region bounds and this visual's data maps to, the band's sample is
    chosen with :func:`select_plane_within_slab`.  If any axis has no sample,
    the visual draws nothing and ``None`` is returned.

    Otherwise the returned selection replaces each of those slabs with a plane
    at the slice position.  :func:`axis_selections_from_box` then rounds that
    position to exactly the sample chosen here -- the nearest sample is the one
    that contains the position, or the end sample when the position lies
    past the data but within the band -- and a multiscale visual's coarser
    levels keep their own rounding of the same position.

    Parameters
    ----------
    selection : RegionSelection
        The canvas's selection, in world coordinates.
    transform : BaseTransform
        The visual's level-0 ``data -> world`` transform.  Must be block
        diagonal: each sliced world axis is fed by one data axis alone.
    spaces : RenderSpaces
        Supplies the world system and the data -> world axis correspondence.
    shape : tuple[int, ...]
        The level-0 data shape.
    exempt_data_axes : Collection[int]
        Data axes the rule skips, whose slabs pass through unchanged.  The
        multichannel visuals exempt their channel axis, which each request
        overwrites.

    Returns
    -------
    RegionSelection or None
        The selection to plan from, or ``None`` when the visual draws nothing.
    """
    world = spaces.world
    box = selection.region.bounding_box()
    world_to_data = {int(w): int(d) for d, w in spaces.data_to_world_axes.items()}
    data_ndim = len(shape)
    base_world = np.asarray(
        transform.map_coordinates(np.zeros((1, data_ndim))), dtype=float
    )

    slabs: dict = {}
    changed = False
    for world_axis in range(world.ndim):
        low = float(box.min_coordinate[world_axis])
        high = float(box.max_coordinate[world_axis])
        low_unbounded, high_unbounded = np.isneginf(low), np.isposinf(high)
        if low_unbounded and high_unbounded:
            continue
        if low_unbounded or high_unbounded:
            # A shear; the assembler rejects it with the explanation.
            return selection
        position = (low + high) / 2.0
        half_thickness = (high - low) / 2.0
        axis_id = world.axes[world_axis].id
        data_axis = world_to_data.get(world_axis)
        if data_axis is None or data_axis in exempt_data_axes:
            slabs[axis_id] = (position, half_thickness)
            continue

        def index_to_world(indices, d=data_axis, w=world_axis):
            points = np.zeros((indices.size, data_ndim))
            points[:, d] = indices
            return np.asarray(transform.map_coordinates(points))[:, w]

        def world_to_index(values, d=data_axis, w=world_axis):
            points = np.repeat(base_world, values.size, axis=0)
            points[:, w] = values
            return np.asarray(transform.imap_coordinates(points))[:, d]

        index = select_plane_within_slab(
            position,
            half_thickness,
            int(shape[data_axis]),
            index_to_world,
            world_to_index,
        )
        if index is None:
            return None
        slabs[axis_id] = (position, 0.0)
        changed = changed or half_thickness != 0.0

    if not changed:
        return selection
    return RegionSelection(
        transform=selection.transform,
        region=ConvexRegion.from_axis_slabs(world, slabs),
    )
