"""Shared dims-panel logic, used by both GUI backends.

The 2D/3D toggle has to hand the slicer an index for every axis it is about
to hide.  Where those indices come from is one question with one answer, and
answering it twice is what let the two front ends drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cellier.gui._axis_values import DiscreteAxisValues, nearest_value_index

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cellier.gui._axis_values import AxisValues


class DisplayedAxesMemory:
    """The displayed axes the 2D/3D toggle goes back to.

    The toggle used to switch between two fixed sets, the last two and the
    last three world axes, whatever the scene had been showing.  This records
    the last 2D set and the last 3D set a dims widget has seen, so a round
    trip returns to where it started.  The two sets given at construction are
    used for a mode the scene has not shown yet.

    The memory belongs to the widget, not to the dims model: nothing is
    saved with the scene.

    Parameters
    ----------
    axes_2d : tuple of int
        The 2D displayed axes to use until a 2D set has been seen.
    axes_3d : tuple of int
        The 3D displayed axes to use until a 3D set has been seen.
    """

    def __init__(self, axes_2d: Sequence[int], axes_3d: Sequence[int]) -> None:
        self._default_2d = tuple(int(axis) for axis in axes_2d)
        self._default_3d = tuple(int(axis) for axis in axes_3d)
        self._seen_2d: tuple[int, ...] | None = None
        self._seen_3d: tuple[int, ...] | None = None

    @property
    def axes_2d(self) -> tuple[int, ...]:
        """The last 2D set seen, or the default."""
        return self._default_2d if self._seen_2d is None else self._seen_2d

    @property
    def axes_3d(self) -> tuple[int, ...]:
        """The last 3D set seen, or the default."""
        return self._default_3d if self._seen_3d is None else self._seen_3d

    def observe(self, displayed: Sequence[int]) -> None:
        """Record *displayed* as the last set seen of its rank.

        The order is kept: ``(0, 1)`` and ``(1, 0)`` display transposed, and
        the toggle returns to the one the scene showed.  A set that is
        neither 2D nor 3D is ignored.
        """
        axes = tuple(int(axis) for axis in displayed)
        if len(axes) == 2:
            self._seen_2d = axes
        elif len(axes) == 3:
            self._seen_3d = axes

    def toggle_target(self, current: Sequence[int]) -> tuple[int, ...]:
        """The displayed axes the toggle moves to from *current*.

        From 2D it is the remembered 3D set.  From 3D it is the remembered
        2D set when every one of its axes is in *current*; otherwise the
        default 2D set under the same condition, and otherwise the last two
        of *current* in world order.  The condition keeps an axis that the
        3D view is not showing (time, in a ``(t, z, y, x)`` scene) from
        coming back on screen by way of the toggle.
        """
        axes = tuple(int(axis) for axis in current)
        if len(axes) != 3:
            return self.axes_3d
        for candidate in (self._seen_2d, self._default_2d):
            if candidate is not None and set(candidate) <= set(axes):
                return candidate
        return tuple(sorted(axes)[-2:])


def default_toggle_axes(
    displayed: Sequence[int], axis_types: Sequence[str]
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """The 2D and 3D sets a toggle starts with, given what a scene displays.

    The set of the scene's own rank is *displayed*.  The other is a default
    and nothing more: it is replaced as soon as the scene shows that mode
    (:class:`DisplayedAxesMemory`), so a user who wants something else states
    it by starting in that mode or by setting the displayed axes while in it.

    - 2D displayed: the 3D set is those axes plus one more, in world order.
      The added axis is the last remaining ``"space"`` axis in world order,
      and when no space axis remains, the last remaining axis of any type.
      The type decides between candidates and never rejects an axis.
    - 3D displayed: the 2D set is the last two of those axes in world order.

    Parameters
    ----------
    displayed : sequence of int
        The scene's displayed axes, in display order.
    axis_types : sequence of str
        The ``axis_type`` of every world axis, in world order.

    Returns
    -------
    tuple or None
        ``(axes_2d, axes_3d)``.  ``None`` when there is no other mode to
        offer: 2D displayed with no axis left to add, or a displayed set
        that is neither 2D nor 3D.
    """
    axes = tuple(int(axis) for axis in displayed)
    if len(axes) == 3:
        return tuple(sorted(axes)[-2:]), axes
    if len(axes) != 2:
        return None
    remaining = [axis for axis in range(len(axis_types)) if axis not in axes]
    if not remaining:
        return None
    spatial = [axis for axis in remaining if axis_types[axis] == "space"]
    added = spatial[-1] if spatial else remaining[-1]
    return axes, tuple(sorted((*axes, added)))


def scene_toggle_axes(
    scene: object,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """``(axes_2d, axes_3d)`` for a dims widget built from *scene*.

    Both are ``None`` when the widget offers no toggle.  The toggle is
    offered only when the *scene* says it can render both ways.  An
    OrthoViewer panel declares exactly one mode -- three slice views and one
    volume -- so offering to switch it put the scene into a mode it was
    never configured for: the reslice produced no geometry, the scene had no
    bounds, and the panel went blank (``plans/gui_backend_seam.md`` D18).  A
    plain ``Viewer`` declares both, so its toggle is unaffected.
    """
    modes = {str(mode) for mode in getattr(scene, "render_modes", ())}
    if not {"2d", "3d"} <= modes:
        return None, None
    axes = scene.dims.world_coordinate_system.axes
    displayed = getattr(scene.dims.selection, "displayed_axes", ())
    pair = default_toggle_axes(displayed, [axis.axis_type for axis in axes])
    return (None, None) if pair is None else pair


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


def thickness_axes(scene: object) -> tuple[int, ...]:
    """The world axes of *scene* whose slider row gets a thickness box.

    Every axis except a channel axis: a thickness is a depth in world units,
    and a channel axis has none.

    Parameters
    ----------
    scene :
        The live ``Scene``.

    Returns
    -------
    tuple[int, ...]
        World axis indices, ascending.
    """
    axes = scene.dims.world_coordinate_system.axes
    return tuple(
        index for index, axis in enumerate(axes) if axis.axis_type != "channel"
    )
