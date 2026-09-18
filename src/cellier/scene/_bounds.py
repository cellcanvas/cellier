"""The world-space extent of a scene's visuals.

One rule, used by the slider ranges (``convenience._geometry``) and by the
scene bounding-box overlay: map every corner of each store's level-0 extent
box through the visual's ``data -> world`` transform, drop the world axes the
visual broadcasts over, and take the per-axis union across visuals.

**All corners, not two.**  Two opposite corners are exact only for a
transform whose linear block is diagonal or a signed permutation.  The image
of a box under any affine is the convex hull of its mapped corners, so the
``2**m`` corners give the exact world AABB for rotations and shears too, and
for a per-axis monotone block such as a lookup-table time axis.

**Broadcast axes contribute nothing.**  A broadcast axis maps through a zero
matrix row to the translation, which is a number but not an extent: the data
occupies the whole axis.  Reading that number would pin the axis's range to
it.

**NaN means "nothing reaches this axis".**  A lookup-table axis maps positions
outside its span to NaN, and a world axis that no visual reaches keeps its NaN
seed; ``fmin`` / ``fmax`` skip NaN, so neither poisons the union.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cellier.scene.scene import Scene
    from cellier.transform import BaseTransform, CoordinateSystem


def extent_corners(extents: Sequence[tuple[float, float]]) -> np.ndarray:
    """Return every corner of an axis-aligned box.

    Parameters
    ----------
    extents : Sequence[tuple[float, float]]
        Per-axis ``(low, high)``.

    Returns
    -------
    np.ndarray
        ``(2**m, m)`` float64 corners.
    """
    return np.array(list(itertools.product(*extents)), dtype=np.float64)


def visual_world_bounds(
    transform: BaseTransform,
    extents: Sequence[tuple[float, float]],
    world: CoordinateSystem,
) -> tuple[np.ndarray, np.ndarray]:
    """World AABB of one store's extent box mapped through *transform*.

    Parameters
    ----------
    transform : BaseTransform
        The visual's ``data -> world`` transform.
    extents : Sequence[tuple[float, float]]
        The store's per-axis level-0 ``(low, high)``.
    world : CoordinateSystem
        The transform's output system, used to resolve broadcast axis ids.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(low, high)``, one entry per world axis; NaN on an axis the visual
        broadcasts over or has no finite position on.
    """
    mapped = np.array(
        transform.map_coordinates(extent_corners(extents)), dtype=np.float64
    )
    for axis_id in transform.broadcast_output_axes():
        mapped[:, world.index_of(axis_id)] = np.nan
    return np.fmin.reduce(mapped, axis=0), np.fmax.reduce(mapped, axis=0)


def scene_world_bounds(
    scene: Scene,
    get_store: Callable[[UUID], object],
) -> tuple[np.ndarray, np.ndarray] | None:
    """World AABB of every visual in *scene*, hidden ones included.

    Parameters
    ----------
    scene : Scene
        The scene whose visuals are measured.
    get_store : Callable[[UUID], object]
        Looks a visual's data store up by id -- typically
        ``CellierController.get_data_store``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or None
        ``(low, high)`` per world axis, NaN on an axis no visual reaches; or
        ``None`` when no visual has any extent (an empty scene, or only empty
        stores).
    """
    world = scene.dims.world_coordinate_system
    low = np.full(world.ndim, np.nan)
    high = np.full(world.ndim, np.nan)
    found = False
    for visual in scene.visuals:
        transform = getattr(visual, "transform", None)
        if transform is None:
            continue
        store = get_store(UUID(str(visual.data_store_id)))
        extents = store.axis_extents
        if extents is None:
            # An empty store occupies nothing, so it must not pull the union
            # anywhere.
            continue
        visual_low, visual_high = visual_world_bounds(transform, extents, world)
        low = np.fmin(low, visual_low)
        high = np.fmax(high, visual_high)
        found = True
    if not found:
        return None
    return low, high
