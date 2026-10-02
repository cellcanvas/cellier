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

    faces = []
    south = len(unit) - 1
    for j in range(n_lon):
        k = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + k])
        last = 1 + (n_lat - 2) * n_lon
        faces.append([south, last + k, last + j])
    for i in range(n_lat - 2):
        for j in range(n_lon):
            k = (j + 1) % n_lon
            a, b = 1 + i * n_lon + j, 1 + i * n_lon + k
            c, d = a + n_lon, b + n_lon
            faces += [[a, c, b], [b, c, d]]
    return positions.astype(np.float32), np.array(faces, dtype=np.int32)
