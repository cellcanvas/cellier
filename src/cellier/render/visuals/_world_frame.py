"""World-frame placement for geometry in ``OBLIQUE`` views (design v2, Phase 6).

A 2D view whose slice is tilted through a geometry visual's data -- a
rotation about an axis that is not the view's normal -- selects a tilted
slab of elements.  The node matrix that places everything else maps the
retained data axes to the view, and here no set of data axes spans the
view: the map from data to the screen is a projection, and a node matrix
holding it is singular (pygfx inverts it for picking and sizes).  So these
visuals fall back to D4 as written: each element's **rendered** position is
computed on the CPU from its full data coordinate and uploaded, and the
node matrix is a translation only (the upload is relative to a rounded
origin, so float32 keeps its precision far from zero).

Selection is unchanged: ``geometry_data_region`` pulls the tilted world
slab back exactly (K12).  Clipping planes are applied by the read on the
elements' true positions, since the node frame is no longer the data's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cellier.transform import SliceAlignment

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cellier.render._spaces import RenderSpaces
    from cellier.transform import BaseTransform


def world_frame_active(spaces: RenderSpaces | None) -> bool:
    """Whether a geometry visual is drawn in the world frame in this view."""
    return (
        spaces is not None
        and spaces.classification is not None
        and spaces.classification.alignment is SliceAlignment.OBLIQUE
    )


def full_data_points(
    n_data: int,
    columns: Mapping[int, np.ndarray],
    constants: Mapping[int, float],
) -> np.ndarray:
    """``(n, n_data)`` float64 data points from per-axis columns.

    Parameters
    ----------
    n_data : int
        The data rank.
    columns : Mapping[int, np.ndarray]
        ``{data axis: (n,) values}`` for the axes the read returned.
    constants : Mapping[int, float]
        Values for every other axis (the slice positions).

    Returns
    -------
    np.ndarray
    """
    n = len(next(iter(columns.values()))) if columns else 0
    points = np.zeros((n, n_data), dtype=np.float64)
    for axis in range(n_data):
        if axis in columns:
            points[:, axis] = columns[axis]
        else:
            points[:, axis] = float(constants.get(axis, 0.0))
    return points


def rendered_positions(
    spaces: RenderSpaces,
    data_to_world: BaseTransform,
    data_points: np.ndarray,
    origin: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Upload-ready pygfx ``(x, y, z)`` positions of *data_points*, and origin.

    Parameters
    ----------
    spaces : RenderSpaces
        The visual's systems, for ``world -> rendered``.
    data_to_world : BaseTransform
        The visual's (effective) transform.
    data_points : np.ndarray
        ``(n, n_data)``.
    origin : np.ndarray or None
        The pygfx-order origin to upload relative to; ``None`` picks the
        rounded mean of these points (or zero for none).

    Returns
    -------
    positions : np.ndarray
        ``(n, 3)`` float32, relative to *origin*.
    origin : np.ndarray
        ``(3,)`` float64, for :func:`translation_matrix`.
    """
    if len(data_points) == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, np.zeros(3) if origin is None else origin
    world = np.asarray(data_to_world.map_coordinates(data_points), dtype=np.float64)
    rendered = np.asarray(
        spaces.world_to_rendered.map_coordinates(world), dtype=np.float64
    )
    pygfx = np.zeros((len(rendered), 3), dtype=np.float64)
    pygfx[:, : rendered.shape[1]] = rendered[:, ::-1]
    if origin is None:
        origin = np.round(pygfx.mean(axis=0))
    return np.ascontiguousarray(pygfx - origin, dtype=np.float32), origin


def translation_matrix(origin: Sequence[float]) -> np.ndarray:
    """The ``(4, 4)`` float32 node matrix of a world-frame upload."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.asarray(origin, dtype=np.float32)
    return matrix


def world_frame_positions(
    spaces: RenderSpaces,
    data_to_world: BaseTransform,
    retained: np.ndarray,
    collapsed: np.ndarray,
    origin: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rendered positions from a read's retained and collapsed columns.

    Parameters
    ----------
    spaces : RenderSpaces
        The visual's systems: ``retained_axes`` name *retained*'s columns
        and ``collapsed_axes`` name *collapsed*'s.
    data_to_world : BaseTransform
        The visual's (effective) transform.
    retained, collapsed : np.ndarray
        ``(n, len(retained_axes))`` and ``(n, len(collapsed_axes))``.
    origin : np.ndarray or None
        As :func:`rendered_positions`.

    Returns
    -------
    positions, origin
        As :func:`rendered_positions`.
    """
    columns = {axis: retained[:, k] for k, axis in enumerate(spaces.retained_axes)}
    columns.update(
        {axis: collapsed[:, k] for k, axis in enumerate(spaces.collapsed_axes)}
    )
    data_points = full_data_points(spaces.data.ndim, columns, {})
    return rendered_positions(spaces, data_to_world, data_points, origin)


def collapsed_columns(
    retained: np.ndarray, fold_positions: np.ndarray | None
) -> np.ndarray:
    """A read's collapsed columns, ``(n, 0)`` when it returned none."""
    if fold_positions is None:
        return np.zeros((len(retained), 0), dtype=np.float64)
    return fold_positions
