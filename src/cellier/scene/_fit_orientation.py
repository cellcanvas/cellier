"""The orientation a 3D camera is fitted with, given on named world axes.

A camera fit needs a view direction and an up vector.  Both are stated as a
mapping of world axis to component, e.g. ``{"z": -1.0}``, so there is no
axis order to get wrong: the caller names the axes and cellier places the
components.  An axis left out has component 0.

The vectors are stored on world axes but a fit happens in a 3D view of three
*displayed* axes, which can change at run time.  :func:`resolve_fit_orientation`
turns the stored vectors into ones on the displayed axes and falls back to the
default orientation when a vector cannot be expressed on them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    from cellier.transform import AxisRef, CoordinateSystem

# The default orientation, on the three displayed axes in displayed order: the
# camera looks along minus all three, and up is the first displayed axis.
DEFAULT_FIT_VIEW_DIRECTION_3D = (-1.0, -1.0, -1.0)
DEFAULT_FIT_UP_3D = (1.0, 0.0, 0.0)

# Two unit vectors whose cross product is shorter than this are parallel.
_PARALLEL_TOLERANCE = 1e-6

AxisVector = dict[UUID, float]


def normalize_axis_vector(
    world: CoordinateSystem,
    vector: Mapping[AxisRef, float] | None,
    name: str,
) -> AxisVector | None:
    """Check an axis-keyed vector and key it by world axis id.

    Parameters
    ----------
    world : CoordinateSystem
        The world coordinate system the vector is given on.
    vector : Mapping[AxisRef, float] or None
        The component along each named axis, by axis name or id.  An axis
        left out has component 0.  ``None`` is passed through.
    name : str
        What the vector is called in an error message.

    Returns
    -------
    dict[UUID, float] or None
        The non-zero components, by world axis id.

    Raises
    ------
    TypeError
        If *vector* is not a mapping.
    KeyError
        If a key does not name an axis of *world*.
    ValueError
        If a name is ambiguous in *world*, an axis is given twice, a
        component is not finite, or every component is 0.
    """
    if vector is None:
        return None
    if not isinstance(vector, Mapping):
        raise TypeError(
            f"{name} must map world axes to components, e.g. "
            f'{{"{world.axes[0].name}": 1.0}}; got {type(vector).__name__}.  '
            "A bare sequence has no axis order."
        )
    components: AxisVector = {}
    for axis, value in vector.items():
        axis_id = world.resolve_axis(axis).id
        if axis_id in components:
            raise ValueError(f"{name} gives the axis {axis!r} more than once.")
        component = float(value)
        if not np.isfinite(component):
            raise ValueError(f"{name} must be finite; got {component} on {axis!r}.")
        components[axis_id] = component
    components = {axis: value for axis, value in components.items() if value != 0.0}
    if not components:
        raise ValueError(f"{name} must have a non-zero component.")
    return components


def check_not_parallel(
    view_direction: AxisVector | None, up: AxisVector | None
) -> None:
    """Refuse a view direction and an up vector that are parallel.

    Parameters
    ----------
    view_direction, up : dict[UUID, float] or None
        Vectors from :func:`normalize_axis_vector`.  Nothing is checked
        unless both are given.

    Raises
    ------
    ValueError
        If the two are parallel, in either sense.
    """
    if view_direction is None or up is None:
        return
    axes = sorted(set(view_direction) | set(up), key=str)
    if _parallel(
        [view_direction.get(axis, 0.0) for axis in axes],
        [up.get(axis, 0.0) for axis in axes],
    ):
        raise ValueError(
            "The view direction and the up vector are parallel, which leaves "
            "the camera's roll undefined."
        )


def resolve_fit_orientation(
    view_directions: Sequence[AxisVector | None],
    ups: Sequence[AxisVector | None],
    displayed_axis_ids: Sequence[UUID],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The view direction and up vector of a fit, on the displayed axes.

    Each vector is the first of its candidates that can be expressed on the
    displayed axes, and the default otherwise.  A candidate can be expressed
    when every axis it has a component on is displayed.  It is never
    projected: dropping the components on axes that are not displayed would
    give an orientation nobody asked for.

    When the two chosen vectors are parallel both are replaced by the
    defaults.  That can happen when only one of the two was given.

    Parameters
    ----------
    view_directions, ups : sequence of dict[UUID, float] or None
        The candidates for each vector, most preferred first.  ``None``
        entries are skipped.
    displayed_axis_ids : sequence of UUID
        The world axis ids of the three displayed axes, in displayed order.

    Returns
    -------
    view_direction, up : tuple[float, float, float]
        On the displayed axes, in displayed order.  Not normalized.

    Raises
    ------
    ValueError
        If there are not three displayed axes.
    """
    if len(displayed_axis_ids) != 3:
        raise ValueError(
            f"A 3D fit is on three displayed axes; got {len(displayed_axis_ids)}."
        )
    view_direction = _first_displayed(
        view_directions, displayed_axis_ids, DEFAULT_FIT_VIEW_DIRECTION_3D
    )
    up = _first_displayed(ups, displayed_axis_ids, DEFAULT_FIT_UP_3D)
    if _parallel(view_direction, up):
        return DEFAULT_FIT_VIEW_DIRECTION_3D, DEFAULT_FIT_UP_3D
    return view_direction, up


def _first_displayed(
    candidates: Sequence[AxisVector | None],
    displayed_axis_ids: Sequence[UUID],
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    """The first candidate that lies on the displayed axes, or *default*."""
    displayed = set(displayed_axis_ids)
    for candidate in candidates:
        if candidate is None or not set(candidate) <= displayed:
            continue
        return tuple(candidate.get(axis, 0.0) for axis in displayed_axis_ids)
    return default


def _parallel(a: Sequence[float], b: Sequence[float]) -> bool:
    """Whether two non-zero vectors are parallel, in either sense."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cosine = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 - cosine < _PARALLEL_TOLERANCE
