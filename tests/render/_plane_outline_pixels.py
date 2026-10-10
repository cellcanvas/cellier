"""Where a plane's outline should be on the screen, for the pixel tests.

A polygon in rendered ``(x, y, z)`` is projected with a camera's matrix and
every pixel centre is given its distance to the polygon's edge.  Nothing
here is shared with the render layer.
"""

from __future__ import annotations

import numpy as np


def to_pixels(points: np.ndarray, camera, size: tuple[int, int]) -> np.ndarray:
    """Project rendered ``(x, y, z)`` points to pixel ``(col, row)`` floats."""
    width, height = size
    matrix = np.asarray(camera.camera_matrix, dtype=np.float64)
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ matrix.T
    ndc = homogeneous[:, :3] / homogeneous[:, 3:4]
    return np.column_stack(
        [(ndc[:, 0] + 1.0) / 2.0 * width, (1.0 - ndc[:, 1]) / 2.0 * height]
    )


def signed_distance(polygon_px: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Per pixel centre: its distance to the polygon's edge, positive inside.

    ``(H, W)`` float.  The polygon is convex.
    """
    width, height = size
    cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    pixels = np.stack([cols, rows], axis=-1)
    count = len(polygon_px)
    turn = sum(
        polygon_px[i][0] * polygon_px[(i + 1) % count][1]
        - polygon_px[(i + 1) % count][0] * polygon_px[i][1]
        for i in range(count)
    )
    sign = 1.0 if turn > 0 else -1.0
    inside = np.full((height, width), np.inf)
    nearest = np.full((height, width), np.inf)
    for i in range(count):
        a, b = polygon_px[i], polygon_px[(i + 1) % count]
        edge = b - a
        length = np.linalg.norm(edge)
        rel = pixels - a
        across = sign * (edge[0] * rel[..., 1] - edge[1] * rel[..., 0]) / length
        inside = np.minimum(inside, across)
        along = np.clip((rel @ edge) / length**2, 0.0, 1.0)
        closest = a + along[..., None] * edge
        nearest = np.minimum(nearest, np.linalg.norm(pixels - closest, axis=-1))
    return np.where(inside >= 0.0, nearest, -nearest)


def edge_distance(polygon: np.ndarray, camera, size: tuple[int, int]) -> np.ndarray:
    """:func:`signed_distance` of a polygon given in rendered space."""
    return signed_distance(to_pixels(polygon, camera, size), size)


def is_red(frame: np.ndarray) -> np.ndarray:
    """Pixels of a pure red line (the tests' outline colour)."""
    return (frame[..., 0] > 200) & (frame[..., 1] < 70) & (frame[..., 2] < 70)


def line_shares(frame: np.ndarray, distance: np.ndarray, width: float) -> dict:
    """How much of a red line of *width* pixels along the edge is drawn.

    The line's two halves are counted apart: the half inside the polygon is
    over the plane, which is where a plane could hide its own outline.

    Returns
    -------
    dict
        ``inner`` and ``outer``: the share of each half's pixels that are
        red (its core, clear of the anti-aliased rim); ``stray``: red
        pixels more than a pixel outside the line.
    """
    red = is_red(frame)
    core = width / 2.0 - 0.75
    inner = (distance > 0.1) & (distance < core)
    outer = (distance < -0.1) & (distance > -core)
    return {
        "inner": float(red[inner].mean()),
        "outer": float(red[outer].mean()),
        "stray": int((red & (np.abs(distance) > width / 2.0 + 1.0)).sum()),
        "pixels": int(inner.sum() + outer.sum()),
    }
