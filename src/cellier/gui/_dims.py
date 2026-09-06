"""Shared dims-panel logic, used by both GUI backends.

The 2D/3D toggle has to hand the slicer an index for every axis it is about
to hide.  Where those indices come from is one question with one answer, and
answering it twice is what let the two front ends drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def initial_slice_indices(
    selection: object, axis_ranges: Mapping[int, tuple[float, float]]
) -> dict[int, int]:
    """Return a slice index for **every** axis, not only the hidden ones.

    A dims panel needs a position for each axis, including the ones currently
    displayed: switching a 3D scene to 2D hides an axis that had no index a
    moment earlier, and the slicer needs one for it immediately.

    The scene's own value wins where it has one.  A displayed axis has none --
    a scene showing all three axes carries an empty ``slice_indices`` -- so
    those are seeded to the middle of the axis, matching what
    ``OrthoViewer.center_slices`` picks and what a reader expects the first
    slice of a volume to be.

    Seeding to the axis *minimum* instead, which is what a Qt slider does when
    nothing sets it, lands the first 2D view on the edge of the volume: often
    blank, and never the slice anyone wanted (``plans/gui_backend_seam.md``
    D16).

    Parameters
    ----------
    selection :
        The scene's ``AxisAlignedSelection``; its ``slice_indices`` are used
        where present.
    axis_ranges :
        Axis index to ``(world_min, world_max)``.  Its keys define which axes
        get an index.

    Returns
    -------
    dict[int, int]
        One entry per axis in *axis_ranges*.
    """
    known = {
        int(axis): int(value)
        for axis, value in getattr(selection, "slice_indices", {}).items()
    }
    seeded: dict[int, int] = {}
    for axis, bounds in axis_ranges.items():
        axis = int(axis)
        if axis in known:
            seeded[axis] = known[axis]
            continue
        low, high = float(bounds[0]), float(bounds[1])
        seeded[axis] = round((low + high) / 2.0)
    return seeded
