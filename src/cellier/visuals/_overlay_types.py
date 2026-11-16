"""Discriminated unions of the overlay model types."""

from __future__ import annotations

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.visuals._canvas_overlay import CenteredAxes2D, DrawStatusOverlay
from cellier.visuals._scene_overlay import SceneBoundingBox

CanvasOverlayType = Annotated[
    Union[CenteredAxes2D, DrawStatusOverlay],
    Field(discriminator="overlay_type"),
]
"""Any canvas overlay model, told apart by its ``overlay_type`` field."""

SceneOverlayType = Annotated[
    Union[SceneBoundingBox,],
    Field(discriminator="overlay_type"),
]
"""Any scene overlay model, told apart by its ``overlay_type`` field."""
