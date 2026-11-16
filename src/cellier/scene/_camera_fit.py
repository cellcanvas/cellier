"""Fit a camera to a displayed box, and the far plane that keeps the box in view.

The box is the scene's store-derived extent on the displayed axes
(``scene_world_bounds`` then ``project_bounds``), in cellier displayed order:
``(z, y, x)`` in 3D, ``(y, x)`` in 2D.  pygfx works in ``(x, y, z)``, so the
sphere returned here is already reversed, and a 2D box sits at ``z = 0``.

**Where the fit puts the camera.**  pygfx's ``show_object((x, y, z, r))`` puts
the camera ``fov_distance_factor(fov) * 2r`` from the centre, back along the
view direction, whatever the aspect.  A radius of 0 is fitted as 1.  So the
fitted distance ``d`` is known from the box alone and no camera is read.

**The far plane.**  A wheel zoom on the 3D camera is a dolly towards the
target, so zooming out ``m`` times puts the camera ``m * d`` from the centre.
The box reaches ``o`` beyond its centre along the view direction (the largest
``dot(corner - centre, F)``), so

    far = m * d + o

keeps the whole box inside the far plane until the user has zoomed out ``m``
times the fitted distance.  Near is never changed: depth precision is set by
near, not far.  The range only grows (`grow_far`), so a small scene keeps the
configured default and an explicit range is respected unless it is too short
for the data.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from cellier.scene._bounds import extent_corners

if TYPE_CHECKING:
    from collections.abc import Sequence

#: How many times the fitted distance the camera can zoom out before the far
#: end of the box starts to clip.
FAR_MARGIN = 4.0


def fov_distance_factor(fov: float) -> float:
    """Return the camera distance per unit of fitted extent at *fov*.

    The same formula as pygfx's ``fov_distance_factor``: ``0.5 / tan(fov / 2)``,
    and 0 for an orthographic camera (``fov <= 0``), which pygfx places at the
    centre of what it fits.

    Parameters
    ----------
    fov : float
        The vertical field of view, in degrees.

    Returns
    -------
    float
        The factor; the fitted camera is ``factor * extent`` from the centre.
    """
    if fov <= 0:
        return 0.0
    return max(0.5 / math.tan(math.radians(fov) / 2.0), 0.0)


def _box_xyz(
    box: tuple[Sequence[float], Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """*box* reversed to pygfx ``(x, y, z)`` order, a 2D box at ``z = 0``."""
    low = np.asarray(box[0], dtype=np.float64)[::-1]
    high = np.asarray(box[1], dtype=np.float64)[::-1]
    if low.shape != high.shape or low.size not in (2, 3):
        raise ValueError(
            f"A fit box has 2 or 3 axes on each side, got {low.size} and {high.size}."
        )
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ValueError("A fit box must be finite on every axis.")
    if low.size == 2:
        low = np.append(low, 0.0)
        high = np.append(high, 0.0)
    return low, high


def fit_sphere(
    box: tuple[Sequence[float], Sequence[float]],
) -> tuple[tuple[float, float, float], float]:
    """Return the sphere a camera is fitted to for *box*.

    Parameters
    ----------
    box : tuple[Sequence[float], Sequence[float]]
        ``(low, high)`` on the displayed axes, in cellier displayed order.

    Returns
    -------
    centre : tuple[float, float, float]
        The box centre, in pygfx ``(x, y, z)`` order; ``z = 0`` for a 2D box.
    radius : float
        Half the box diagonal.

    Raises
    ------
    ValueError
        If the box does not have 2 or 3 axes, or is not finite.
    """
    low, high = _box_xyz(box)
    centre = (low + high) / 2.0
    radius = float(np.linalg.norm(high - low) / 2.0)
    return (float(centre[0]), float(centre[1]), float(centre[2])), radius


def fit_distance(radius: float, fov: float) -> float:
    """Return how far pygfx's fit puts the camera from the sphere's centre.

    Parameters
    ----------
    radius : float
        The fitted sphere's radius; 0 is fitted as 1, as pygfx does.
    fov : float
        The camera's field of view, in degrees.

    Returns
    -------
    float
        The distance; 0 for an orthographic camera.
    """
    return fov_distance_factor(fov) * 2.0 * (max(radius, 0.0) or 1.0)


def fitted_far(
    box: tuple[Sequence[float], Sequence[float]],
    fov: float,
    view_direction: Sequence[float],
    *,
    margin: float = FAR_MARGIN,
) -> float:
    """Return the far plane that keeps *box* in view, ``margin * d + o``.

    Parameters
    ----------
    box : tuple[Sequence[float], Sequence[float]]
        ``(low, high)`` on the three displayed axes, in cellier displayed
        order.
    fov : float
        The 3D camera's field of view, in degrees.
    view_direction : Sequence[float]
        The direction the camera is fitted looking along, in pygfx
        ``(x, y, z)`` order.  Need not be normalized.
    margin : float
        How many times the fitted distance the camera can zoom out before
        the far end of the box clips.

    Returns
    -------
    float
        The far plane distance.

    Raises
    ------
    ValueError
        If the box does not have 3 finite axes, or *view_direction* is zero
        or not finite.
    """
    if len(box[0]) != 3:
        raise ValueError(f"The far plane needs a 3D box, got {len(box[0])} axes.")
    low, high = _box_xyz(box)
    direction = np.asarray(view_direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if direction.shape != (3,) or not math.isfinite(norm) or norm == 0.0:
        raise ValueError(
            f"view_direction must be a finite non-zero 3-vector, got "
            f"{tuple(view_direction)}."
        )
    direction /= norm
    centre = (low + high) / 2.0
    radius = float(np.linalg.norm(high - low) / 2.0)
    corners = extent_corners(list(zip(low, high, strict=True)))
    reach = float(((corners - centre) @ direction).max())
    return margin * fit_distance(radius, fov) + reach


def grow_far(
    current: tuple[float, float] | None, far: float
) -> tuple[float, float] | None:
    """Return *current* with its far plane raised to at least *far*.

    Near is kept.  A range that already reaches *far* comes back unchanged,
    and so does an automatic range (``None``), which pygfx sizes to the scene
    on its own.

    Parameters
    ----------
    current : tuple[float, float] or None
        The camera's configured ``(near, far)``, or ``None`` for automatic.
    far : float
        The far plane the scene needs (`fitted_far`).

    Returns
    -------
    tuple[float, float] or None
        The range to set.
    """
    if current is None:
        return None
    near, current_far = float(current[0]), float(current[1])
    return near, max(current_far, float(far))
