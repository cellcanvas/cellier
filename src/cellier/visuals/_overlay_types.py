"""Discriminated unions of the overlay model types for cellier v2."""

from __future__ import annotations

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from cellier.visuals._canvas_overlay import CenteredAxes2D
from cellier.visuals._scene_overlay import SceneBoundingBox

CanvasOverlayType = Annotated[
    Union[CenteredAxes2D,],
    Field(discriminator="overlay_type"),
]

SceneOverlayType = Annotated[
    Union[SceneBoundingBox,],
    Field(discriminator="overlay_type"),
]
