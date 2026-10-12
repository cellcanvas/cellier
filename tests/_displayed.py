"""Displayed axes for test helpers that take a world and a mode.

``Viewer`` and ``CellierController.add_scene`` take the axes a scene displays
by name.  A helper that builds a scene from whatever world its caller hands
it, in whatever mode, uses this to state "the trailing axes", which is what
such helpers displayed when the constructors took ``dim=``.
"""

from __future__ import annotations

from cellier.scene.dims import world_coordinate_system


def trailing_axes(axes, dim: str) -> tuple[str, ...]:
    """The names of the last two (``"2d"``) or three (``"3d"``) world axes.

    Parameters
    ----------
    axes : WorldAxesLike
        The world axes, as the constructors accept them.
    dim : "2d" or "3d"
        The mode to display.

    Returns
    -------
    tuple[str, ...]
        Axis names, in world order.  Fewer than asked for when the world has
        fewer axes, so the constructor reports the mismatch.
    """
    names = world_coordinate_system(axes).axis_names()
    return tuple(names[-(3 if dim == "3d" else 2) :])
