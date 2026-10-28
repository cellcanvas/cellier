"""Model-layer types for canvas-space overlays in cellier v2."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal
from uuid import uuid4

from psygnal import EventedModel
from pydantic import UUID4, AfterValidator, Field


class CanvasOverlay(EventedModel):
    """Base model for all canvas-space overlays.

    Overlays are rendered as a post-pass on top of the main scene using a
    ``gfx.ScreenCoordsCamera``.  They have no world-space transform and do
    not participate in the reslicing pipeline.

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


class CenteredAxes2DAppearance(EventedModel):
    """Appearance model for a :class:`CenteredAxes2D` overlay.

    Parameters
    ----------
    axis_a_color : tuple[float, float, float, float]
        RGBA color for the first axis line segment.
        Default green ``(0.23, 0.67, 0.23, 1.0)``.
    axis_b_color : tuple[float, float, float, float]
        RGBA color for the second axis line segment.
        Default orange ``(0.80, 0.40, 0.00, 1.0)``.
    line_thickness_px : float
        Line thickness in screen pixels.  Default ``2.0``.
    length_px : float
        Length of each axis segment in screen pixels.  Default ``60.0``.
    corner : {"bottom_left", "bottom_right", "top_left", "top_right", "center"}
        Canvas corner to anchor to, or ``"center"`` to place the origin at the
        canvas centre.  Default ``"center"``.
    corner_offset_px : tuple[float, float]
        ``(x, y)`` offset in pixels from the anchor corner, where ``(0, 0)``
        is the corner itself.  The offset moves the origin inward (right for
        left corners, left for right corners; down for top corners, up for
        bottom corners).  Default ``(20.0, 20.0)``.
    show_labels : bool
        Whether to render text labels at the arrow tips.  Default ``True``.
    font_size_px : float
        Label font size in screen pixels.  Default ``12.0``.
    label_color : tuple[float, float, float, float]
        RGBA color for axis text labels.  Default white ``(1.0, 1.0, 1.0, 1.0)``.
    """

    axis_a_color: tuple[float, float, float, float] = (0.23, 0.67, 0.23, 1.0)
    axis_b_color: tuple[float, float, float, float] = (0.80, 0.40, 0.00, 1.0)
    line_thickness_px: float = 2.0
    length_px: float = 60.0
    corner: Literal[
        "bottom_left", "bottom_right", "top_left", "top_right", "center"
    ] = "center"
    corner_offset_px: tuple[float, float] = (20.0, 20.0)
    show_labels: bool = True
    font_size_px: float = 12.0
    label_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


class CenteredAxes2D(CanvasOverlay):
    """2D axis indicator rendered in screen space as a corner overlay.

    Displays two line segments anchored to a fixed canvas corner, one for
    each displayed axis.  The screen-space directions of the segments are
    computed from the world-space direction vectors supplied here and the
    live camera matrix, so the indicator correctly reflects any camera
    orientation.  Geometry is recomputed only when the canvas resizes or
    the camera matrix changes — never on pan or zoom.

    Parameters
    ----------
    overlay_type : str
        Discriminator literal ``"centered_axes_2d"``.  Do not set manually.
    axis_a_direction : tuple[float, float, float]
        Unit vector in pygfx world-space XYZ defining the direction of
        axis A.  Default ``(0.0, 1.0, 0.0)`` (world +Y, screen up for
        standard 2D cameras).
    axis_a_label : str
        Text label drawn at the tip of the axis-A segment.  Default ``"A"``.
    axis_b_direction : tuple[float, float, float]
        Unit vector in pygfx world-space XYZ defining the direction of
        axis B.  Default ``(1.0, 0.0, 0.0)`` (world +X, screen right for
        standard 2D cameras).
    axis_b_label : str
        Text label drawn at the tip of the axis-B segment.  Default ``"B"``.
    appearance : CenteredAxes2DAppearance
        Visual style for the overlay.
    """

    overlay_type: Literal["centered_axes_2d"] = "centered_axes_2d"
    axis_a_direction: tuple[float, float, float] = (0.0, 1.0, 0.0)
    axis_a_label: str = "A"
    axis_b_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    axis_b_label: str = "B"
    appearance: CenteredAxes2DAppearance = Field(
        default_factory=CenteredAxes2DAppearance
    )


class DrawStatusOverlay(CanvasOverlay):
    """Lists, in a canvas corner, the visuals its view does not draw.

    Full affine design v2, decision D8 (variant B).  A 2D view whose slice
    cuts a rotated image at an angle draws nothing of it rather than
    something wrong (D3); this overlay says so where the user is looking,
    one line per visual: ``Not shown: <name> (OBLIQUE)``.  It draws nothing
    while every visual draws.  The full reason is on
    :meth:`~cellier.CellierController.draw_status`.

    The convenience viewers add one to every canvas by default
    (``draw_status_overlay=True``).

    Parameters
    ----------
    overlay_type : str
        Discriminator literal ``"draw_status"``.  Do not set manually.
    name : str
        Human-readable label.  Default ``"draw status"``.
    font_size_px : float
        Text size in screen pixels.  Default ``13.0``.
    color : tuple[float, float, float, float]
        RGBA text colour.  Default amber ``(1.0, 0.85, 0.35, 1.0)``.
    corner_offset_px : tuple[float, float]
        ``(x, y)`` offset of the text's top-left corner from the canvas's
        top-left corner, in pixels.  Default ``(8.0, 8.0)``.
    """

    overlay_type: Literal["draw_status"] = "draw_status"
    name: str = "draw status"
    font_size_px: float = 13.0
    color: tuple[float, float, float, float] = (1.0, 0.85, 0.35, 1.0)
    corner_offset_px: tuple[float, float] = (8.0, 8.0)
