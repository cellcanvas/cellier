"""Immutable data-transfer objects for the render layer request pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from cellier._state import DimsState

__all__ = ["DimsState", "ReslicingRequest"]

if TYPE_CHECKING:
    from uuid import UUID

    import numpy as np

    from cellier.transform import RegionSelection


class ReslicingRequest(NamedTuple):
    """Frozen snapshot driving a complete reslicing cycle.

    Constructed by ``CanvasView.capture_reslicing_request()`` and consumed
    by ``SliceCoordinator``.  All array fields must be ``.copy()``d by the
    caller — the ``NamedTuple`` does not enforce this.

    Parameters
    ----------
    camera_type : str
        ``"perspective"`` or ``"orthographic"``.
    camera_pos : ndarray, shape (3,)
        World-space camera position, copy.
    frustum_corners : ndarray, shape (2, 4, 3)
        World-space frustum corners, copy.  For orthographic cameras this
        is set to ``np.zeros((2, 4, 3))``.
    fov_y_rad : float
        Vertical field of view in radians.  ``0.0`` for orthographic.
    screen_size_px : tuple[float, float]
        Logical ``(width, height)`` in pixels, baked in at snapshot time.
    world_extent : tuple[float, float]
        Visible ``(width, height)`` in world units for orthographic cameras,
        a 3D camera at a field of view of 0 included.  ``(0.0, 0.0)`` for
        perspective cameras.
    dims_state : DimsState
        Current dimension display state.
    selection : RegionSelection or None
        Where this canvas's slice is and how thick, as one frozen artifact
        (D43).  The transform is the ``rendered -> world`` embedding whose
        constant column carries the slice positions; the region is in
        **world** coordinates.  ``None`` when the controller could not build
        one -- a scene with no rendered system yet -- in which case the
        planning phase falls back to ``dims_state``.
    request_id : UUID
        Unique identifier per trigger; used for cancellation.
    scene_id : UUID
        Which scene this camera belongs to.
    canvas_id : UUID
        Which canvas produced this request.  Used by ``SliceCoordinator``
        to key cancellation entries so that requests from different canvases
        rendering the same scene can be tracked and cancelled independently.
    target_visual_ids : frozenset[UUID] or None
        ``None`` means reslice all visuals in the scene (camera-moved case).
        A non-None set means reslice only those specific visuals
        (data-updated case).
    """

    camera_type: str
    camera_pos: np.ndarray
    frustum_corners: np.ndarray
    fov_y_rad: float
    screen_size_px: tuple[float, float]
    world_extent: tuple[float, float]
    dims_state: DimsState
    selection: RegionSelection | None
    request_id: UUID
    scene_id: UUID
    canvas_id: UUID
    target_visual_ids: frozenset[UUID] | None
