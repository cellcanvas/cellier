"""Model-layer types for scene overlays.

A scene overlay is a decoration that belongs to one ``Scene`` and is placed in
that scene's world coordinate system.  It is drawn in the main render pass by
the scene camera -- so, unlike a canvas overlay, it depth-tests against the
visuals in 3D -- but it is not a visual: it has no data store, no transform and
no slicing, and it does not contribute to ``Scene.slider_axes`` or to the
world extent a slider is sized from.

The controller recomputes a scene overlay's geometry whenever the scene's
contents or displayed axes change (see
``CellierController._refresh_scene_overlays``).
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal
from uuid import uuid4

from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, ConfigDict, Field


class SceneOverlay(EventedModel):
    """Base model for all scene overlays.

    Parameters
    ----------
    id : UUID4
        Unique identifier.  Auto-generated.
    name : str
        Human-readable label.
    visible : bool
        Whether the overlay is rendered.  Default ``True``.
    """

    id: UUID4 | Annotated[str, AfterValidator(lambda x: uuid.UUID(x, version=4))] = (
        Field(frozen=True, default_factory=lambda: uuid4())
    )
    name: str
    visible: bool = True


class SceneBoundingBoxAppearance(EventedModel):
    """Appearance model for a :class:`SceneBoundingBox`.

    Parameters
    ----------
    color : tuple[float, float, float, float]
        RGBA line colour.  Default mid gray ``(0.5, 0.5, 0.5, 1.0)``.
    thickness : float
        Line thickness in screen pixels.  Default ``1.5``.
    render_order : int
        pygfx render order.  Default ``1``, after visuals at the default
        ``0``: a MIP image depth-tests but writes no depth, so drawn after the
        box it would paint over every edge behind the volume's front faces.
    """

    model_config = ConfigDict(validate_assignment=True)

    color: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    thickness: float = Field(default=1.5, gt=0.0)
    render_order: int = 1


class SceneBoundingBox(SceneOverlay):
    """Wireframe around the world bounding box of every visual in a scene.

    The box is the union over all visuals -- hidden ones included -- of each
    store's level-0 extent mapped through the visual's ``data -> world``
    transform, taken over **every** world axis and then projected onto the
    displayed ones.  It therefore spans the whole dataset rather than the
    current slice: moving a slider does not move it.  Displayed in 3D it is
    the 12 edges of a box; in 2D, the 4 edges of a rectangle.

    Parameters
    ----------
    overlay_type : str
        Discriminator literal ``"scene_bounding_box"``.  Do not set manually.
    appearance : SceneBoundingBoxAppearance
        Visual style for the box.
    """

    overlay_type: Literal["scene_bounding_box"] = "scene_bounding_box"
    appearance: SceneBoundingBoxAppearance = Field(
        default_factory=SceneBoundingBoxAppearance
    )
