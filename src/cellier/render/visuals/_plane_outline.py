"""Plane outlines in a scene (plane outline design, sections 5.4 and 6).

``visual_plane_outlines`` asks one render visual for the polygons of its
outlined planes.  ``GFXPlaneOutlines`` holds the lines of one scene: it is
given every visual's polygons, drops the duplicates a shared plane makes,
and keeps one ``gfx.Line`` per outline drawn.

The lines are in the scene's rendered space, so they need no node matrix.
They are not visuals: never sliced, never picked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx as gfx

from cellier.render._clipping import reduce_clipping_planes
from cellier.render._plane_outline import (
    clipping_plane_polygon,
    polygons_equal,
    render_plane_polygon,
)
from cellier.render._render_planes import reduce_render_planes
from cellier.render.shaders._plane_outline import PlaneOutlineMaterial

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable
    from uuid import UUID

#: The kinds of plane a visual can show outlines of.
RENDER = "render"
CLIPPING = "clipping"


@dataclass(frozen=True)
class PlaneOutlinePolygon:
    """One visual's polygon of one outlined plane.

    Parameters
    ----------
    kind : str
        ``"render"`` or ``"clipping"``.
    plane_id : UUID
        The plane's id.
    vertices : np.ndarray
        ``(k, 3)`` float64, in rendered ``(x, y, z)``, in order around the
        polygon.
    color : tuple of four float
        The outline's RGBA.
    width : float
        The outline's width in screen pixels.
    """

    kind: str
    plane_id: UUID
    vertices: np.ndarray
    color: tuple[float, float, float, float]
    width: float


def visual_plane_outlines(
    gfx_visual: Any, kinds: Collection[str]
) -> list[PlaneOutlinePolygon]:
    """The polygons of a render visual's outlined planes, for the view drawn.

    Parameters
    ----------
    gfx_visual :
        A render visual.  One that takes no planes, or is not placed in a
        3D view, gives nothing.
    kinds : Collection[str]
        Which of the visual's planes are shown now (design 3.3):
        ``"render"`` when it draws its render planes, ``"clipping"`` when
        its clipping planes cut what it draws.

    Returns
    -------
    list[PlaneOutlinePolygon]
        One per plane with an enabled outline and a polygon: the render
        planes in their order, then the clipping planes in theirs.
    """
    render_planes = []
    if RENDER in kinds:
        render_planes = [
            plane
            for plane in getattr(gfx_visual, "render_planes", ())
            if plane.enabled and plane.outline.enabled
        ]
    clipping = list(getattr(gfx_visual, "clipping_planes", ()))
    outlined_cuts = []
    if CLIPPING in kinds:
        outlined_cuts = [
            index
            for index, item in enumerate(clipping)
            if item.enabled and item.outline.enabled
        ]
    if not render_planes and not outlined_cuts:
        return []
    frame = getattr(gfx_visual, "clip_frame", lambda: None)()
    box = getattr(gfx_visual, "outline_frame", lambda: None)()
    if frame is None or box is None:
        return []
    spaces, transform, constants = frame
    box_rows, corners = box
    # Every clipping plane, as the materials are given them: a disabled
    # one keeps everything.
    clipping_rows = (
        reduce_clipping_planes(spaces, transform, constants, clipping)
        if clipping
        else ()
    )
    found = []
    for plane in render_planes:
        # Nothing when the view does not display the plane's axes.
        reduced = reduce_render_planes(spaces, [plane])
        if len(reduced) == 0:
            continue
        vertices = render_plane_polygon(
            reduced.origin[0],
            reduced.axis_0[0],
            reduced.axis_1[0],
            reduced.extent[0],
            reduced.bounded[0],
            box_rows,
            corners,
            clipping_rows,
        )
        if len(vertices):
            found.append(_polygon(RENDER, plane, vertices))
    for index in outlined_cuts:
        vertices = clipping_plane_polygon(index, clipping_rows, box_rows, corners)
        if len(vertices):
            found.append(_polygon(CLIPPING, clipping[index], vertices))
    return found


def _polygon(kind: str, plane: Any, vertices: np.ndarray) -> PlaneOutlinePolygon:
    return PlaneOutlinePolygon(
        kind=kind,
        plane_id=plane.id,
        vertices=vertices,
        color=tuple(plane.outline.color),
        width=float(plane.outline.width),
    )


def drop_duplicates(
    polygons: Iterable[PlaneOutlinePolygon],
) -> dict[tuple[str, UUID, int], PlaneOutlinePolygon]:
    """The outlines to draw: equal polygons of one plane, once (design 5.4).

    Parameters
    ----------
    polygons : Iterable[PlaneOutlinePolygon]
        Every visual's polygons, the visuals in the scene's order.

    Returns
    -------
    dict
        ``(kind, plane id, n) -> polygon``: the ``n``-th different polygon
        of that plane.  The first visual to give a polygon gives its style.
    """
    kept: dict[tuple[str, UUID, int], PlaneOutlinePolygon] = {}
    counts: dict[tuple[str, UUID], int] = {}
    for polygon in polygons:
        plane = (polygon.kind, polygon.plane_id)
        count = counts.get(plane, 0)
        if any(
            polygons_equal(kept[(*plane, n)].vertices, polygon.vertices)
            for n in range(count)
        ):
            continue
        kept[(*plane, count)] = polygon
        counts[plane] = count + 1
    return kept


class GFXPlaneOutlines:
    """The plane outlines of one scene, as lines under one group."""

    def __init__(self) -> None:
        self._group = gfx.Group()
        self._lines: dict[tuple[str, UUID, int], gfx.Line] = {}
        self._drawn: dict[tuple[str, UUID, int], PlaneOutlinePolygon] = {}

    @property
    def node(self) -> gfx.Group:
        """The group holding the lines; a child of the scene."""
        return self._group

    @property
    def drawn(self) -> dict[tuple[str, UUID, int], PlaneOutlinePolygon]:
        """What is drawn now, by ``(kind, plane id, n)``."""
        return dict(self._drawn)

    def line(self, key: tuple[str, UUID, int]) -> gfx.Line | None:
        """The line of one outline, if it is drawn."""
        return self._lines.get(key)

    def update(self, polygons: Iterable[PlaneOutlinePolygon]) -> bool:
        """Draw exactly the outlines of *polygons*, duplicates dropped.

        A line is made when its polygon first exists and removed when it no
        longer does: there is never a placeholder in the scene.

        Returns
        -------
        bool
            Whether anything drawn changed.
        """
        wanted = drop_duplicates(polygons)
        changed = False
        for key in [key for key in self._lines if key not in wanted]:
            self._group.remove(self._lines.pop(key))
            del self._drawn[key]
            changed = True
        for key, polygon in wanted.items():
            before = self._drawn.get(key)
            line = self._lines.get(key)
            if line is None:
                line = gfx.Line(
                    _geometry(polygon.vertices),
                    PlaneOutlineMaterial(color=polygon.color, thickness=polygon.width),
                )
                self._lines[key] = line
                self._group.add(line)
                changed = True
            else:
                if not np.array_equal(before.vertices, polygon.vertices):
                    line.geometry = _geometry(polygon.vertices)
                    changed = True
                if before.color != polygon.color:
                    line.material.color = polygon.color
                    changed = True
                if before.width != polygon.width:
                    line.material.thickness = polygon.width
                    changed = True
            self._drawn[key] = polygon
        return changed

    def clear(self) -> bool:
        """Remove every line."""
        return self.update(())


def _geometry(vertices: np.ndarray) -> gfx.Geometry:
    return gfx.Geometry(positions=np.ascontiguousarray(vertices, dtype=np.float32))
