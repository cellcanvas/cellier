"""CPU clipping of geometry against half-spaces (clipping planes design 5.2).

Used only where a shader cannot do it: geometry flattened into a 2D slab,
which has lost the coordinate a clipping plane needs.  The kernels work in
whatever space the points are given in; a plane is ``(normal, offset)`` and
keeps ``normal . p >= offset``.

Triangles are clipped by ``cellier.data.mesh._section`` (``_clip``, with
``cut_plane`` and ``_cap`` for capped cross-sections); segments and points
here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

#: One clipping plane as plain data: ``(normal, offset)``, hashable, so it
#: can sit in a request and in a request key.
PlaneTuple = tuple[tuple[float, ...], float]


def plane_tuples(planes: Sequence[object]) -> tuple[PlaneTuple, ...]:
    """The enabled planes of *planes* as hashable ``(normal, offset)`` pairs.

    Parameters
    ----------
    planes : Sequence[ClippingPlane]
        A visual's clipping planes.

    Returns
    -------
    tuple[PlaneTuple, ...]
        One entry per enabled plane, in level-0 data coordinates, one
        normal entry per data axis.
    """
    return tuple(
        (tuple(float(v) for v in item.plane.normal), float(item.plane.offset))
        for item in planes
        if item.enabled
    )


def kept_points(points: np.ndarray, planes: Sequence[PlaneTuple]) -> np.ndarray:
    """Boolean mask of the points on the kept side of every plane.

    Parameters
    ----------
    points : np.ndarray
        ``(n, ndim)``.
    planes : Sequence[PlaneTuple]
        ``(normal, offset)`` pairs with ``ndim`` normal entries.

    Returns
    -------
    np.ndarray
        ``(n,)`` bool.  A point on a plane is kept.
    """
    mask = np.ones(len(points), dtype=bool)
    for normal, offset in planes:
        mask &= points @ np.asarray(normal, dtype=points.dtype) >= offset
    return mask


def clip_segments(
    start: np.ndarray,
    end: np.ndarray,
    planes: Sequence[PlaneTuple],
    n_position_columns: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut segments at the planes, keeping the part on the kept side.

    A segment wholly on the clipped side of any plane is dropped; one that
    crosses is cut at the plane.  Every column is interpolated, so
    attributes (a colour, an alpha) ride along as extra columns after the
    position.

    Parameters
    ----------
    start, end : np.ndarray
        ``(n, c)``: the two ends of each segment.
    planes : Sequence[PlaneTuple]
        ``(normal, offset)`` pairs.
    n_position_columns : int or None
        How many leading columns are the position the planes test.  ``None``
        means the length of the first plane's normal.

    Returns
    -------
    start, end : np.ndarray
        The surviving segments, cut.
    kept : np.ndarray
        ``(m,)`` indices into the input of the surviving segments.
    """
    kept = np.arange(len(start))
    for normal, offset in planes:
        n = np.asarray(normal, dtype=np.float64)
        width = len(n) if n_position_columns is None else n_position_columns
        d_start = start[:, :width] @ n[:width] - offset
        d_end = end[:, :width] @ n[:width] - offset
        keep = (d_start >= 0) | (d_end >= 0)
        start, end, kept = start[keep], end[keep], kept[keep]
        d_start, d_end = d_start[keep], d_end[keep]
        cut_start, cut_end = d_start < 0, d_end < 0
        if not (cut_start.any() or cut_end.any()):
            continue
        # Only the rows that cross use ``t``; a segment that does not cross
        # may have equal distances, which must not divide.
        span = d_start - d_end
        t = np.divide(d_start, span, out=np.zeros_like(d_start), where=span != 0)[
            :, None
        ]
        crossing = (start + t * (end - start)).astype(start.dtype, copy=False)
        start = np.where(cut_start[:, None], crossing, start)
        end = np.where(cut_end[:, None], crossing, end)
    return start, end, kept
