"""Shared dims-panel logic, used by both GUI backends.

The 2D/3D toggle has to hand the slicer an index for every axis it is about
to hide.  Where those indices come from is one question with one answer, and
answering it twice is what let the two front ends drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellier.gui._axis_values import DiscreteAxisValues, nearest_value_index

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cellier.gui._axis_values import AxisValues


def initial_slice_indices(
    selection: object, axis_values: Mapping[int, AxisValues]
) -> dict[int, float]:
    """Return a slider value for **every** axis in *axis_values*.

    The scene's own value wins.  A scene holds a position for every axis,
    displayed ones included (D36), so the fallback below only matters for a
    selection built without one -- a panel constructed by hand.  Such an axis
    is seeded to the middle of its range, matching what
    ``OrthoViewer.center_slices`` picks and what a reader expects the first
    slice of a volume to be.

    Seeding to the axis *minimum* instead, which is what a Qt slider does when
    nothing sets it, lands the first 2D view on the edge of the volume: often
    blank, and never the slice anyone wanted (``plans/gui_backend_seam.md``
    D16).

    A discrete axis is different: it has no meaningful middle, and a
    channel axis's midpoint is a tie between two channels.  It starts on its
    **first** value, and a value the scene already holds is moved to the
    nearest listed value so the slider and the renderer agree.

    Parameters
    ----------
    selection :
        The scene's ``AxisAlignedSelection``; its ``slice_indices`` are used
        where present.
    axis_values :
        Axis index to that axis's slider values.  Its keys define which axes
        get an index.

    Returns
    -------
    dict[int, float]
        One entry per axis in *axis_values*.  World positions, not voxel
        indices: since D3 a slice position is a float, so the midpoint is no
        longer rounded and a fine axis's odd-numbered planes are reachable.
    """
    known = {
        int(axis): float(value)
        for axis, value in getattr(selection, "slice_indices", {}).items()
    }
    seeded: dict[int, float] = {}
    for axis, spec in axis_values.items():
        axis = int(axis)
        if isinstance(spec, DiscreteAxisValues):
            if axis in known:
                seeded[axis] = spec.values[
                    nearest_value_index(spec.values, known[axis])
                ]
            else:
                seeded[axis] = spec.values[0]
            continue
        if axis in known:
            seeded[axis] = known[axis]
            continue
        seeded[axis] = (spec.min + spec.max) / 2.0
    return seeded
