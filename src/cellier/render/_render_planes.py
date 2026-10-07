"""Render planes in the render layer (plane rendering design v3, 5.3, 6.2).

A visual's render planes are authored in the scene's world space, on three
named world axes.  ``reduce_render_planes`` turns them into what one 3D
view draws: frames in the scene's rendered space, in pygfx ``(x, y, z)``
order.  Only enabled planes whose axes are the displayed ones are kept.

The planes reach the shaders through one uniform buffer per render visual,
bound by every material of the visual that is built for the ``"plane"``
render mode.  A change of planes is therefore one buffer write, whatever
the number of channels, and a material created later needs nothing.

``RenderPlanesMixin`` gives a render visual that buffer and the one seam
through which planes reach it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx as gfx
from pygfx.utils import array_from_shadertype

from cellier.visuals._render_plane import MAX_RENDER_PLANES

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from cellier.render._spaces import RenderSpaces

#: What an unbounded extent side is uploaded as.  Not infinity: an infinity
#: times zero in the shader is NaN.
UNBOUNDED = 1.0e30

#: The uniform block ``u_render_planes`` (design 5.3).
RENDER_PLANES_SHADERTYPE: dict[str, str] = {
    "origin": f"{MAX_RENDER_PLANES}*4xf4",
    "axis_0": f"{MAX_RENDER_PLANES}*4xf4",
    "axis_1": f"{MAX_RENDER_PLANES}*4xf4",
    "extent": f"{MAX_RENDER_PLANES}*4xf4",
    "count": "i4",
}

#: The WGSL struct name of the block.
RENDER_PLANES_STRUCT = "RenderPlanes"


@dataclass(frozen=True)
class ReducedPlanes:
    """The planes one 3D view draws, in pygfx ``(x, y, z)`` order.

    Parameters
    ----------
    origin, axis_0, axis_1 : np.ndarray
        ``(n, 3)`` float64.
    extent : np.ndarray
        ``(n, 4)`` float64, ``(min_0, max_0, min_1, max_1)``; an unbounded
        side is ``-+UNBOUNDED``.
    bounded : np.ndarray
        ``(n, 4)`` bool: which sides of ``extent`` are finite.
    """

    origin: np.ndarray
    axis_0: np.ndarray
    axis_1: np.ndarray
    extent: np.ndarray
    bounded: np.ndarray

    def __len__(self) -> int:
        return len(self.origin)

    @property
    def normal(self) -> np.ndarray:
        """``(n, 3)``: ``axis_0 x axis_1``."""
        return np.cross(self.axis_0, self.axis_1).reshape(-1, 3)


def _no_planes() -> ReducedPlanes:
    return ReducedPlanes(
        origin=np.zeros((0, 3)),
        axis_0=np.zeros((0, 3)),
        axis_1=np.zeros((0, 3)),
        extent=np.zeros((0, 4)),
        bounded=np.zeros((0, 4), dtype=bool),
    )


def reduce_render_planes(
    spaces: RenderSpaces | None, planes: Sequence[Any]
) -> ReducedPlanes:
    """Reduce world-space render planes to one view's rendered space.

    A plane is kept when it is enabled and its three axes are the world
    axes the view displays.  Its origin and in-plane axes are carried
    through ``world -> rendered`` (a reorder of the displayed axes) and
    reversed to pygfx order.

    Parameters
    ----------
    spaces : RenderSpaces or None
        The systems the visual is placed with.  ``None``, or a 2D view,
        keeps no plane.
    planes : Sequence[RenderPlane]
        The visual's planes.

    Returns
    -------
    ReducedPlanes
        The planes the view draws, in the order given.
    """
    if spaces is None or not planes:
        return _no_planes()
    linear = np.asarray(spaces.world_to_rendered.linear, dtype=np.float64)
    if linear.shape[0] != 3:
        return _no_planes()
    translation = np.asarray(spaces.world_to_rendered.translation, dtype=np.float64)
    displayed = set(np.flatnonzero(np.any(linear != 0.0, axis=0)).tolist())
    signs = np.array([-1.0, 1.0, -1.0, 1.0])
    origins, axes_0, axes_1, extents, bounded = [], [], [], [], []
    for plane in planes:
        if not plane.enabled:
            continue
        try:
            axes = [spaces.world.resolve(axis) for axis in plane.axes]
        except (KeyError, ValueError):
            continue
        if set(axes) != displayed:
            continue
        block = linear[:, axes]
        # Rendered order is pygfx order reversed.
        origins.append((block @ np.asarray(plane.origin) + translation)[::-1])
        axes_0.append((block @ np.asarray(plane.in_plane_axis_0))[::-1])
        axes_1.append((block @ np.asarray(plane.in_plane_axis_1))[::-1])
        sides = (*plane.extent_0, *plane.extent_1)
        bounded.append([side is not None for side in sides])
        extents.append(
            [
                sign * UNBOUNDED if side is None else float(side)
                for side, sign in zip(sides, signs, strict=True)
            ]
        )
    if not origins:
        return _no_planes()
    return ReducedPlanes(
        origin=np.asarray(origins, dtype=np.float64),
        axis_0=np.asarray(axes_0, dtype=np.float64),
        axis_1=np.asarray(axes_1, dtype=np.float64),
        extent=np.asarray(extents, dtype=np.float64),
        bounded=np.asarray(bounded, dtype=bool),
    )


def make_render_planes_buffer() -> gfx.Buffer:
    """A ``u_render_planes`` uniform buffer holding no plane."""
    return gfx.Buffer(
        array_from_shadertype(RENDER_PLANES_SHADERTYPE), force_contiguous=True
    )


def write_render_planes(buffer: gfx.Buffer, reduced: ReducedPlanes) -> bool:
    """Write *reduced* to a ``u_render_planes`` buffer.

    Parameters
    ----------
    buffer : gfx.Buffer
        From :func:`make_render_planes_buffer`.
    reduced : ReducedPlanes
        At most ``MAX_RENDER_PLANES`` planes.

    Returns
    -------
    bool
        Whether the buffer changed.
    """
    count = len(reduced)
    if count > MAX_RENDER_PLANES:
        raise ValueError(
            f"A visual draws at most {MAX_RENDER_PLANES} render planes; got {count}."
        )
    new = np.zeros_like(buffer.data)
    new["origin"][:count, :3] = reduced.origin
    new["origin"][:count, 3] = 1.0
    new["axis_0"][:count, :3] = reduced.axis_0
    new["axis_1"][:count, :3] = reduced.axis_1
    new["extent"][:count] = reduced.extent
    new["count"] = count
    if new.tobytes() == buffer.data.tobytes():
        return False
    buffer.data[...] = new
    buffer.update_full()
    return True


class RenderPlanesMixin:
    """The one way render planes reach a render visual's shaders.

    A host class:

    - gives ``self.render_planes_buffer`` to every material it builds that
      can draw the ``"plane"`` render mode;
    - calls :meth:`_apply_render_planes` wherever it is given its render
      spaces (the displayed axes may have changed).

    It reads ``self._spaces``.
    """

    _render_planes: tuple = ()
    _render_planes_buffer: gfx.Buffer | None = None
    _reduced_render_planes: ReducedPlanes | None = None

    @property
    def render_planes(self) -> tuple:
        """The planes last given to this visual."""
        return self._render_planes

    @property
    def render_planes_buffer(self) -> gfx.Buffer:
        """The visual's ``u_render_planes`` buffer, shared by its materials."""
        if self._render_planes_buffer is None:
            self._render_planes_buffer = make_render_planes_buffer()
        return self._render_planes_buffer

    @property
    def drawn_render_planes(self) -> ReducedPlanes:
        """The planes the current view draws, in pygfx order."""
        if self._reduced_render_planes is None:
            self._reduced_render_planes = reduce_render_planes(
                getattr(self, "_spaces", None), self._render_planes
            )
        return self._reduced_render_planes

    def set_render_planes(self, planes: Iterable[Any]) -> None:
        """Adopt *planes* and write the drawn ones to the uniform buffer."""
        self._render_planes = tuple(planes)
        self._apply_render_planes()

    def on_render_planes_changed(self, event: Any) -> None:
        """Bus handler for ``RenderPlanesChangedEvent``."""
        self.set_render_planes(event.render_planes)

    def _apply_render_planes(self) -> None:
        """Reduce the stored planes for the view and upload them."""
        self._reduced_render_planes = None
        write_render_planes(self.render_planes_buffer, self.drawn_render_planes)
