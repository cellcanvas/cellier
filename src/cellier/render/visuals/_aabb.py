"""Axis-aligned bounding-box wireframes for the geometry visuals.

The image and label visuals build their box from a known array *shape*, which
they can do at construction and refresh once on first data.  A mesh, points,
lines or graph visual has no shape -- its extent is whatever vertices arrived
-- so the box has to be recomputed from the committed geometry each time.
That is the only real difference, and it is why these helpers exist rather
than the image path being reused directly.

The line is added as a **child of the data node**, so it inherits that node's
transform for free and is hidden along with it when ``visible`` goes False.
Its positions come from ``get_geometry_bounding_box()``, which reads the
node's own geometry and ignores its children -- without that the box would
measure itself and grow on every refresh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygfx as gfx

if TYPE_CHECKING:
    from collections.abc import Iterable


def box_wireframe_positions(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """Return ``(24, 3)`` float32 positions for a box, as 12 line segments.

    Paired positions, for a ``LineSegmentMaterial``: bottom face, top face,
    then the four verticals.
    """
    x0, y0, z0 = (float(v) for v in box_min)
    x1, y1, z1 = (float(v) for v in box_max)
    return np.array(
        [
            # bottom face
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y1, z0],
            [x0, y0, z0],
            # top face
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x1, y1, z1],
            [x0, y1, z1],
            [x0, y1, z1],
            [x0, y0, z1],
            # verticals
            [x0, y0, z0],
            [x0, y0, z1],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y1, z1],
            [x0, y1, z0],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )


def make_aabb_line(color: str, line_width: float) -> gfx.Line:
    """Create the wireframe line, degenerate and hidden until data arrives.

    pygfx rejects a zero-length buffer, so the placeholder is a real box at
    the origin rather than an empty geometry; it is never seen, because the
    line stays invisible until `refresh_aabb_line` has both real bounds
    and an enabled flag.
    """
    line = gfx.Line(
        gfx.Geometry(positions=box_wireframe_positions(np.zeros(3), np.zeros(3))),
        gfx.LineSegmentMaterial(color=color, thickness=line_width),
    )
    line.visible = False
    return line


def geometry_bounds(
    nodes: Iterable[gfx.WorldObject],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Union of *nodes*' own geometry bounding boxes, or ``None`` if empty.

    ``get_geometry_bounding_box`` deliberately, not ``get_bounding_box``: the
    latter includes children, and the AABB line is a child.
    """
    boxes = []
    for node in nodes:
        if node is None:
            continue
        box = node.get_geometry_bounding_box()
        if box is not None:
            boxes.append(np.asarray(box, dtype=np.float64))
    if not boxes:
        return None
    stacked = np.stack(boxes)  # (n, 2, 3)
    return stacked[:, 0, :].min(axis=0), stacked[:, 1, :].max(axis=0)


def refresh_aabb_line(
    line: gfx.Line | None,
    data_nodes: Iterable[gfx.WorldObject],
    *,
    enabled: bool,
) -> bool:
    """Resize *line* to the data's current extent and apply *enabled*.

    Call after every commit: unlike an image, a geometry visual's extent is
    whatever vertices arrived, so it changes with the data.

    Returns whether real bounds were found.  A visual with no geometry yet
    keeps the box hidden regardless of *enabled* -- a box around nothing is a
    box around the origin, which reads as a bug -- and the caller keeps the
    answer so that enabling the box *before* the first commit stays pending
    rather than drawing that degenerate box.  The image visuals gate on
    ``_data_ready_2d`` / ``_data_ready_3d`` for the same reason.
    """
    if line is None:
        return False
    bounds = geometry_bounds(data_nodes)
    if bounds is None:
        line.visible = False
        return False
    line.geometry = gfx.Geometry(positions=box_wireframe_positions(*bounds))
    line.visible = bool(enabled)
    return True
