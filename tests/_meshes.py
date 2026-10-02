"""Small closed meshes for tests, built with numpy only."""

from __future__ import annotations

import numpy as np


def uv_sphere(
    radius: float = 1.0,
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_lat: int = 12,
    n_lon: int = 24,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """A watertight latitude/longitude sphere, or ellipsoid.

    Two pole vertices and ``n_lat - 1`` rings of ``n_lon`` vertices; every
    edge is shared by exactly two faces.

    Returns
    -------
    positions : np.ndarray
        ``(V, 3)`` float32, columns ``(z, y, x)``; the poles are on ``z``.
    indices : np.ndarray
        ``(F, 3)`` int32, wound outward.
    """
    theta = np.linspace(0.0, np.pi, n_lat + 1)[1:-1]
    phi = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    ring_z = np.repeat(np.cos(theta), n_lon)
    ring_y = np.outer(np.sin(theta), np.sin(phi)).ravel()
    ring_x = np.outer(np.sin(theta), np.cos(phi)).ravel()
    unit = np.concatenate(
        [
            [[1.0, 0.0, 0.0]],
            np.stack([ring_z, ring_y, ring_x], axis=1),
            [[-1.0, 0.0, 0.0]],
        ]
    )
    positions = radius * unit * np.asarray(scale) + np.asarray(centre)

    # Built with array operations, so a sphere of a million faces is quick.
    south = len(unit) - 1
    j = np.arange(n_lon)
    k = (j + 1) % n_lon
    last = 1 + (n_lat - 2) * n_lon
    north = np.stack([np.zeros(n_lon, dtype=np.int64), 1 + j, 1 + k], axis=1)
    south_fan = np.stack([np.full(n_lon, south), last + k, last + j], axis=1)
    caps = np.stack([north, south_fan], axis=1).reshape(-1, 3)
    row = 1 + np.arange(n_lat - 2)[:, None] * n_lon
    a, b = row + j, row + k
    c, d = a + n_lon, b + n_lon
    body = np.stack(
        [np.stack([a, c, b], axis=-1), np.stack([b, c, d], axis=-1)], axis=2
    ).reshape(-1, 3)
    faces = np.concatenate([caps, body])
    return positions.astype(np.float32), np.array(faces, dtype=np.int32)
