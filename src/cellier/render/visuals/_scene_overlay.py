"""Render-layer scene overlays for cellier v2.

A scene overlay's node lives in the scene's own ``gfx.Scene`` -- beside the
background and the visuals -- and is drawn by the scene camera in the main
pass.  Its vertices are written directly in the **rendered** coordinate system
(the displayed world axes, reversed to pygfx ``(x, y, z)``), so its node
matrix is the identity and it never goes through slicing.

The controller owns the world-space computation (it has the data stores) and
pushes the result in through :meth:`GFXSceneOverlay.update_scene_extent`;
appearance changes arrive through :meth:`GFXSceneOverlay.apply`.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx as gfx

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cellier.visuals._scene_overlay import (
        SceneBoundingBox,
        SceneBoundingBoxAppearance,
    )


def project_bounds(
    bounds: tuple[np.ndarray, np.ndarray] | None,
    displayed_axes: Sequence[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Project a world AABB onto the displayed axes (world -> rendered).

    For an axis-aligned selection the rendered system is the displayed world
    axes in ``displayed_axes`` order, so the projection is a selection.

    Parameters
    ----------
    bounds : tuple[np.ndarray, np.ndarray] or None
        World ``(low, high)``, one entry per world axis, NaN where nothing
        reaches.
    displayed_axes : Sequence[int]
        World axis indices, in rendered (cellier displayed) order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or None
        Rendered ``(low, high)``, or ``None`` when there are no bounds or a
        displayed axis has none -- a box with an unknown side cannot be drawn.
    """
    if bounds is None:
        return None
    axes = list(displayed_axes)
    low = np.asarray(bounds[0], dtype=np.float64)[axes]
    high = np.asarray(bounds[1], dtype=np.float64)[axes]
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        return None
    return low, high


def box_edge_positions(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Return the edges of an axis-aligned box as pygfx segment pairs.

    An edge joins two corners that differ along exactly one axis: 4 edges for
    a 2D rectangle, 12 for a 3D box.  *low* and *high* are in cellier
    displayed order; the result is reversed to pygfx ``(x, y, z)`` and, in
    2D, placed at ``z = 0``.

    Parameters
    ----------
    low, high : np.ndarray
        Rendered-space corners, 2 or 3 entries each.

    Returns
    -------
    np.ndarray
        ``(2 * n_edges, 3)`` float32; rows ``2k`` and ``2k + 1`` are the ends
        of edge ``k``.
    """
    ndim = len(low)
    corners = list(itertools.product((0, 1), repeat=ndim))
    vertices = []
    for a, b in itertools.combinations(corners, 2):
        if sum(i != j for i, j in zip(a, b)) == 1:
            vertices.append(np.where(a, high, low))
            vertices.append(np.where(b, high, low))
    positions = np.asarray(vertices, dtype=np.float64)[:, ::-1]
    if ndim == 2:
        positions = np.column_stack([positions, np.zeros(len(positions))])
    return positions.astype(np.float32)


class GFXSceneOverlay(ABC):
    """Base class for render-layer scene overlays.

    Subclasses build :attr:`node` at construction; the scene manager adds it
    to the scene's ``gfx.Scene``.  Every material must set
    ``pick_write=False``: an overlay is not pickable, and a pick id would also
    cut into the outlines the screen-space outline pass derives from the pick
    buffer.
    """

    @property
    @abstractmethod
    def node(self) -> gfx.WorldObject:
        """The node added to the scene's ``gfx.Scene``."""

    @abstractmethod
    def update_scene_extent(
        self,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        displayed_axes: tuple[int, ...],
    ) -> None:
        """Rebuild the geometry for the scene's current world extent.

        Parameters
        ----------
        bounds : tuple[np.ndarray, np.ndarray] or None
            World ``(low, high)`` over every visual, NaN on an axis no visual
            reaches; ``None`` for a scene with no extent at all.
        displayed_axes : tuple[int, ...]
            The scene's displayed world axes, in rendered order.
        """

    @abstractmethod
    def apply(self, field_name: str, value: Any) -> None:
        """Apply one model change.

        Parameters
        ----------
        field_name : str
            Dotted path of the changed field: ``"visible"``,
            ``"appearance.<field>"``, or ``"appearance"`` when the whole
            appearance model was replaced (*value* is then the new model).
        value : Any
            The new value.
        """


class GFXSceneBoundingBox(GFXSceneOverlay):
    """A wireframe box around the scene's world extent.

    Parameters
    ----------
    model : SceneBoundingBox
        The model-layer overlay.  Read at construction and on
        ``"appearance"`` replacement; later field changes arrive through
        :meth:`apply`.
    """

    def __init__(self, model: SceneBoundingBox) -> None:
        self._model = model
        self._visible = model.visible
        # False until update_scene_extent has real geometry: a box around
        # nothing is a box around the origin, which reads as a bug.
        self._has_geometry = False
        self._displayed_axes: tuple[int, ...] = ()

        appearance = model.appearance
        # pygfx rejects a zero-length buffer, so the placeholder is a
        # degenerate segment that stays hidden until real bounds arrive.
        self._line = gfx.Line(
            gfx.Geometry(positions=np.zeros((2, 3), dtype=np.float32)),
            gfx.LineSegmentMaterial(
                color=appearance.color,
                thickness=appearance.thickness,
                thickness_space="screen",
                depth_test=True,
                # Thin lines writing depth would put speckles into the depth
                # buffer the ambient occlusion pass reads, and would occlude
                # translucent visuals drawn after the box.
                depth_write=False,
                pick_write=False,
            ),
        )
        self._line.render_order = appearance.render_order
        self._sync_visibility()

    # ------------------------------------------------------------------
    # GFXSceneOverlay interface
    # ------------------------------------------------------------------

    @property
    def node(self) -> gfx.Line:
        """The wireframe line."""
        return self._line

    @property
    def model(self) -> SceneBoundingBox:
        """The model-layer overlay this node draws."""
        return self._model

    def update_scene_extent(
        self,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        displayed_axes: tuple[int, ...],
    ) -> None:
        """Rebuild the box for *bounds* projected onto *displayed_axes*."""
        self._displayed_axes = tuple(displayed_axes)
        rendered = project_bounds(bounds, displayed_axes)
        if rendered is None:
            self._has_geometry = False
        else:
            self._line.geometry = gfx.Geometry(positions=box_edge_positions(*rendered))
            self._has_geometry = True
        # In 2D every visual sits at z = 0 and most write depth there, so
        # with the default "<" comparison the box would lose every depth test
        # against an image.  render_order alone decides what draws on top.
        self._line.material.depth_test = len(self._displayed_axes) == 3
        self._sync_visibility()

    def apply(self, field_name: str, value: Any) -> None:
        """Apply one model change to the line."""
        if field_name == "visible":
            self._visible = bool(value)
            self._sync_visibility()
        elif field_name == "appearance":
            self._apply_appearance(value)
        elif field_name == "appearance.color":
            self._line.material.color = value
        elif field_name == "appearance.thickness":
            self._line.material.thickness = float(value)
        elif field_name == "appearance.render_order":
            self._line.render_order = int(value)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_appearance(self, appearance: SceneBoundingBoxAppearance) -> None:
        self._line.material.color = appearance.color
        self._line.material.thickness = float(appearance.thickness)
        self._line.render_order = int(appearance.render_order)

    def _sync_visibility(self) -> None:
        self._line.visible = self._visible and self._has_geometry
