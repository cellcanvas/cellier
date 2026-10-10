"""The polygon a plane makes in a visual (plane outline design, section 5).

A plane's outline is the edge of a convex polygon: the plane, cut by the
visual's data box and by its clipping planes, and for a render plane by
the plane's own extents.  This module computes that polygon.  It knows
nothing of pygfx objects or of the GPU.

Everything is in the scene's rendered space, in pygfx ``(x, y, z)`` order
and float64.  The box and the clipping planes are **half-space rows**
``(a, b, c, d)``, kept where ``a*x + b*y + c*z >= d``: the form
:func:`~cellier.render._clipping.reduce_clipping_planes` returns.  The box
is written as six data-space planes and reduced by that same function
(:func:`box_frame`), so it follows whatever transform a clipping plane
follows.

The box is closed: a plane lying in one of its faces has that face as its
polygon.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from cellier.render._clipping import reduce_clipping_planes
from cellier.render._spaces import affine_for_node

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cellier.render._spaces import RenderSpaces
    from cellier.transform import BaseTransform

#: Lengths under this fraction of the box's diagonal are zero: a vertex
#: that close to a cutting plane is on it, and two vertices that close
#: together are one.
TOLERANCE = 1.0e-6

#: How far outside the box an unbounded side of a plane is put, in box
#: diagonals from the box's centre.
_REACH = 2.0

_EMPTY = np.zeros((0, 3), dtype=np.float64)


def voxel_edge_box(
    shape: Sequence[int], retained_axes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """The box a volume shader cuts a plane with, in level-0 data coordinates.

    The edges of the voxels, not their centres: ``-0.5`` to ``n - 0.5`` on
    each axis, whatever level is drawn (design 5.3, measured).

    Parameters
    ----------
    shape : Sequence[int]
        The store's level-0 shape, one entry per data axis.
    retained_axes : Sequence[int]
        The three data axes the view keeps.

    Returns
    -------
    low, high : np.ndarray
        One entry per retained axis, in ascending axis order: what
        :func:`box_frame` takes.
    """
    sizes = np.array([float(shape[axis]) for axis in sorted(retained_axes)])
    return np.full(len(sizes), -0.5), sizes - 0.5


def polygons_equal(first: np.ndarray, second: np.ndarray) -> bool:
    """Whether two polygons are the same vertices in the same order.

    Two visuals with the same box and the same planes go through the same
    steps and give the same order, so no rotation of the cycle is tried
    (design 5.4).  The tolerance is :data:`TOLERANCE` of the polygon's
    extent.
    """
    if first.shape != second.shape:
        return False
    if len(first) == 0:
        return True
    extent = float(np.linalg.norm(first.max(axis=0) - first.min(axis=0)))
    return bool(np.abs(first - second).max() <= TOLERANCE * extent)


def box_frame(
    spaces: RenderSpaces,
    data_to_world: BaseTransform,
    constants: Mapping[int, float],
    low: Sequence[float],
    high: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """A data-space box as half-space rows and corners in rendered space.

    Takes the same ``constants`` as
    :func:`~cellier.render._clipping.reduce_clipping_planes` and must be
    given the same values, so the box and the clipping planes are cut by
    the same slice.

    Parameters
    ----------
    spaces : RenderSpaces
        The systems the visual is placed with, for a 3D view.
    data_to_world : BaseTransform
        The visual's own transform.
    constants : Mapping[int, float]
        ``{collapsed data axis: data position}``.
    low, high : Sequence[float]
        The box's corners in level-0 data coordinates, one entry per
        retained data axis, in ascending axis order.

    Returns
    -------
    rows : np.ndarray
        ``(6, 4)``: the faces, kept where ``dot(row[:3], p) >= row[3]``.
    corners : np.ndarray
        ``(8, 3)``: the corners.

    Raises
    ------
    ValueError
        If the view does not keep three data axes.
    """
    rendered = affine_for_node(data_to_world, constants).then(
        spaces.world_to_rendered, spaces.world, spaces.rendered
    )
    ndim = rendered.linear.shape[1]
    retained = [axis for axis in range(ndim) if axis not in constants]
    if len(retained) != 3:
        raise ValueError(
            "A plane's outline is computed for a 3D view only; this view "
            f"keeps {len(retained)} data axes."
        )
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)

    faces = []
    for k, axis in enumerate(retained):
        for sign, bound in ((1.0, low[k]), (-1.0, high[k])):
            normal = np.zeros(ndim, dtype=np.float64)
            normal[axis] = sign
            faces.append(
                SimpleNamespace(
                    enabled=True,
                    plane=SimpleNamespace(normal=normal, offset=sign * float(bound)),
                )
            )
    rows = np.asarray(
        reduce_clipping_planes(spaces, data_to_world, constants, faces),
        dtype=np.float64,
    )

    linear = np.asarray(rendered.linear, dtype=np.float64)[:, retained]
    tau = np.asarray(rendered.translation, dtype=np.float64)
    corners = np.empty((8, 3), dtype=np.float64)
    for index in range(8):
        pick = np.array([(index >> k) & 1 for k in range(3)], dtype=bool)
        # Rendered order is pygfx order reversed.
        corners[index] = (linear @ np.where(pick, high, low) + tau)[::-1]
    return rows, corners


def node_box_frame(
    matrix: np.ndarray, low: Sequence[float], high: Sequence[float]
) -> tuple[np.ndarray, np.ndarray] | None:
    """A box in a node's own frame as half-space rows and corners.

    For a visual whose box is known where its geometry is: the bounding
    box of the vertices a node draws, in that node's coordinates.

    Parameters
    ----------
    matrix : np.ndarray
        The node's 4 x 4 world matrix: its frame to rendered space.
    low, high : Sequence[float]
        The box's corners in the node's ``(x, y, z)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or None
        ``(rows, corners)`` as :func:`box_frame` returns them; ``None``
        when the matrix cannot be inverted (a node scaled to nothing).
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    linear, tau = matrix[:3, :3], matrix[:3, 3]
    try:
        inverse_transpose = np.linalg.inv(linear).T
    except np.linalg.LinAlgError:
        return None
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    rows = np.empty((6, 4), dtype=np.float64)
    for axis in range(3):
        for k, (sign, bound) in enumerate(((1.0, low[axis]), (-1.0, high[axis]))):
            # The plane sign * p[axis] >= sign * bound, carried through
            # r = L p + tau: n' = L^-T n, d' = d + n' . tau.
            normal = sign * inverse_transpose[:, axis]
            rows[2 * axis + k, :3] = normal
            rows[2 * axis + k, 3] = sign * bound + float(normal @ tau)
    corners = np.empty((8, 3), dtype=np.float64)
    for index in range(8):
        pick = np.array([(index >> k) & 1 for k in range(3)], dtype=bool)
        corners[index] = linear @ np.where(pick, high, low) + tau
    return rows, corners


def box_diagonal(corners: np.ndarray) -> float:
    """The length the tolerance is a fraction of: the box's extent."""
    corners = np.asarray(corners, dtype=np.float64)
    return float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)))


def clip_polygon(vertices: np.ndarray, rows: np.ndarray, scale: float) -> np.ndarray:
    """Cut a convex, planar polygon by half-spaces.

    Sutherland-Hodgman, one half-space at a time; each cut adds at most
    one vertex.  A vertex within the tolerance of a cutting plane is on
    it, which is what makes the box closed.

    Parameters
    ----------
    vertices : np.ndarray
        ``(n, 3)``, in order around the polygon.
    rows : np.ndarray
        ``(m, 4)``: kept where ``dot(row[:3], p) >= row[3]``.  A row with
        a zero normal keeps everything when ``row[3] <= 0`` and nothing
        otherwise (a clipping plane parallel to the slice; a disabled
        one).
    scale : float
        The box's diagonal (:func:`box_diagonal`).

    Returns
    -------
    np.ndarray
        ``(k, 3)`` float64, in order around the polygon; ``k`` is 0 when
        nothing is left, and never 1 or 2.
    """
    polygon = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tolerance = TOLERANCE * float(scale)
    for row in np.asarray(rows, dtype=np.float64).reshape(-1, 4):
        if len(polygon) == 0:
            return _EMPTY
        normal, offset = row[:3], row[3]
        length = float(np.linalg.norm(normal))
        if length == 0.0:
            if offset > 0.0:
                return _EMPTY
            continue
        distance = (polygon @ normal - offset) / length
        distance[np.abs(distance) <= tolerance] = 0.0
        kept = []
        count = len(polygon)
        for i in range(count):
            j = (i + 1) % count
            here, there = distance[i], distance[j]
            if here >= 0.0:
                kept.append(polygon[i])
            if (here > 0.0 and there < 0.0) or (here < 0.0 and there > 0.0):
                kept.append(
                    polygon[i] + (polygon[j] - polygon[i]) * (here / (here - there))
                )
        polygon = np.asarray(kept, dtype=np.float64).reshape(-1, 3)
    return _tidy(polygon, tolerance, float(scale))


def _tidy(polygon: np.ndarray, tolerance: float, scale: float) -> np.ndarray:
    """Drop repeated vertices; nothing when no area is left."""
    if len(polygon) < 3:
        return _EMPTY
    keep = [0]
    for i in range(1, len(polygon)):
        if np.linalg.norm(polygon[i] - polygon[keep[-1]]) > tolerance:
            keep.append(i)
    if len(keep) > 1 and np.linalg.norm(polygon[keep[-1]] - polygon[keep[0]]) <= (
        tolerance
    ):
        keep.pop()
    polygon = polygon[keep]
    if len(polygon) < 3:
        return _EMPTY
    fan = np.cross(polygon[1:-1] - polygon[0], polygon[2:] - polygon[0])
    area = 0.5 * float(np.linalg.norm(fan.sum(axis=0)))
    if area <= tolerance * scale:
        return _EMPTY
    return polygon


def _quad(
    origin: np.ndarray,
    axis_0: np.ndarray,
    axis_1: np.ndarray,
    extent: Sequence[float],
    bounded: Sequence[bool],
    corners: np.ndarray,
) -> np.ndarray:
    """A plane's rectangle, each unbounded side put outside the box."""
    centre = corners.mean(axis=0)
    reach = _REACH * box_diagonal(corners)
    c0 = float((centre - origin) @ axis_0)
    c1 = float((centre - origin) @ axis_1)
    min_0 = float(extent[0]) if bounded[0] else c0 - reach
    max_0 = float(extent[1]) if bounded[1] else c0 + reach
    min_1 = float(extent[2]) if bounded[2] else c1 - reach
    max_1 = float(extent[3]) if bounded[3] else c1 + reach
    return np.asarray(
        [
            origin + min_0 * axis_0 + min_1 * axis_1,
            origin + max_0 * axis_0 + min_1 * axis_1,
            origin + max_0 * axis_0 + max_1 * axis_1,
            origin + min_0 * axis_0 + max_1 * axis_1,
        ],
        dtype=np.float64,
    )


def render_plane_polygon(
    origin: Sequence[float],
    axis_0: Sequence[float],
    axis_1: Sequence[float],
    extent: Sequence[float],
    bounded: Sequence[bool],
    box_rows: np.ndarray,
    corners: np.ndarray,
    clipping_rows: np.ndarray | Sequence = (),
) -> np.ndarray:
    """What a visual draws of one render plane, as a polygon (design 3.1).

    The plane's rectangle, cut by the data box and by the clipping planes.

    Parameters
    ----------
    origin, axis_0, axis_1 : Sequence[float]
        The plane's frame in rendered ``(x, y, z)``: a point on it and two
        orthogonal unit vectors in it (one row of a ``ReducedPlanes``).
    extent : Sequence[float]
        ``(min_0, max_0, min_1, max_1)`` along the two axes, from the
        origin.  A side that is not *bounded* is not read.
    bounded : Sequence[bool]
        Which of the four sides are finite.
    box_rows, corners : np.ndarray
        The visual's data box, from :func:`box_frame`.
    clipping_rows : array-like
        ``(n, 4)``: the visual's clipping planes, as
        :func:`~cellier.render._clipping.reduce_clipping_planes` returns
        them (a disabled plane keeps everything).

    Returns
    -------
    np.ndarray
        ``(k, 3)`` float64 vertices in order around the polygon; ``k`` is
        0 when the plane draws nothing, otherwise from 3 to
        ``10 + n``.
    """
    corners = np.asarray(corners, dtype=np.float64)
    quad = _quad(
        np.asarray(origin, dtype=np.float64),
        np.asarray(axis_0, dtype=np.float64),
        np.asarray(axis_1, dtype=np.float64),
        extent,
        bounded,
        corners,
    )
    rows = np.vstack(
        [
            np.asarray(box_rows, dtype=np.float64).reshape(-1, 4),
            np.asarray(clipping_rows, dtype=np.float64).reshape(-1, 4),
        ]
    )
    return clip_polygon(quad, rows, box_diagonal(corners))


def clipping_plane_polygon(
    index: int,
    clipping_rows: np.ndarray | Sequence,
    box_rows: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """The cut face of one clipping plane, as a polygon (design 3.2).

    The plane itself, cut by the data box and by the visual's other
    clipping planes.

    Parameters
    ----------
    index : int
        Which row of *clipping_rows* the plane is.
    clipping_rows : array-like
        ``(n, 4)``: all of the visual's clipping planes, as
        :func:`~cellier.render._clipping.reduce_clipping_planes` returns
        them.
    box_rows, corners : np.ndarray
        The visual's data box, from :func:`box_frame`.

    Returns
    -------
    np.ndarray
        ``(k, 3)`` float64 vertices in order around the polygon; ``k`` is
        0 when the plane does not cross what is left of the box, or has
        no normal in this view (a disabled plane, or one parallel to the
        slice).
    """
    clipping_rows = np.asarray(clipping_rows, dtype=np.float64).reshape(-1, 4)
    corners = np.asarray(corners, dtype=np.float64)
    normal, offset = clipping_rows[index, :3], clipping_rows[index, 3]
    length = float(np.linalg.norm(normal))
    if length == 0.0:
        return _EMPTY
    unit = normal / length
    # The point of the plane nearest the box's centre, and a frame there.
    centre = corners.mean(axis=0)
    origin = centre - (float(centre @ unit) - offset / length) * unit
    seed = np.zeros(3, dtype=np.float64)
    seed[int(np.argmin(np.abs(unit)))] = 1.0
    axis_0 = seed - float(seed @ unit) * unit
    axis_0 /= np.linalg.norm(axis_0)
    axis_1 = np.cross(unit, axis_0)
    quad = _quad(origin, axis_0, axis_1, (0.0,) * 4, (False,) * 4, corners)
    rows = np.vstack(
        [
            np.asarray(box_rows, dtype=np.float64).reshape(-1, 4),
            np.delete(clipping_rows, index, axis=0),
        ]
    )
    return clip_polygon(quad, rows, box_diagonal(corners))
