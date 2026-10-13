"""Canvas model for cellier v2."""

from __future__ import annotations

import uuid
from typing import Annotated, Any
from uuid import uuid4

from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, Field

from cellier.scene.cameras import CameraType
from cellier.visuals._overlay_types import CanvasOverlayType


class Canvas(EventedModel):
    """One rendered canvas, holding one or more cameras.

    Parameters
    ----------
    id : UUID4
        Unique identifier. Auto-generated.
    cameras : dict[str, CameraType]
        Mapping of ``"2d"`` and/or ``"3d"`` to camera models.
        At least one entry is required.
    overlays : list[CanvasOverlayType]
        Screen-space overlays attached to this canvas.  Rendered as
        post-passes on top of the main scene.  Default empty list.
    fit_view_direction_3d : dict[UUID4, float] or None
        The direction the 3D camera looks along when it is fitted to the
        scene, as the component along each world axis, by axis id.  An axis
        left out has component 0.  ``None`` (default) looks along minus all
        three displayed axes.  Used only while every axis it has a component
        on is displayed; see ``cellier.scene._fit_orientation``.
    fit_up_3d : dict[UUID4, float] or None
        The up direction of a fitted 3D camera, given like
        *fit_view_direction_3d*.  ``None`` (default) is the first displayed
        axis.
    """

    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    cameras: dict[str, CameraType]
    overlays: list[CanvasOverlayType] = Field(default_factory=list)
    size: tuple[int, int] | None = None
    fit_view_direction_3d: dict[UUID4, float] | None = None
    fit_up_3d: dict[UUID4, float] | None = None

    def model_post_init(self, __context: Any) -> None:
        """Wire camera event relays after model initialization."""
        for camera in self.cameras.values():
            camera.events.all.connect(
                lambda info, cam=camera: self.events.cameras.emit(self.cameras)
            )
