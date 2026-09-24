"""Background appearance model for cellier v2 scenes."""

from __future__ import annotations

from typing import Literal

from psygnal import EventedModel

RGBA = tuple[float, float, float, float]

# The gradient cellier has always drawn: a mid gray at the bottom fading to a
# lighter gray at the top.  Kept as the default so existing scenes look the
# same as they did when the colors were hard coded in SceneManager.
DEFAULT_BOTTOM_COLOR: RGBA = (100 / 255, 100 / 255, 100 / 255, 1.0)
DEFAULT_TOP_COLOR: RGBA = (169 / 255, 167 / 255, 168 / 255, 1.0)

# What the convenience viewers (``Viewer``, ``OrthoViewer``) fill the canvas
# with by default.
VIEWER_BACKGROUND_COLOR: RGBA = (0.0, 0.0, 0.0, 1.0)


class BackgroundAppearance(EventedModel):
    """Appearance of the background drawn behind a scene's visuals.

    The background belongs to the scene rather than the canvas: pygfx draws
    it as a member of the scene graph, so every canvas viewing a scene shares
    it.  Use one ``Scene`` per panel when panels need different backgrounds
    (this is what ``OrthoViewer`` already does).

    Which color fields are used depends on ``mode``; the others stay set but
    inert, so switching modes back and forth does not lose a color.

    Parameters
    ----------
    visible : bool
        Whether the background is drawn at all.  When ``False`` the canvas
        shows the renderer's clear color.  Default ``True``.
    mode : {"uniform", "vertical_gradient"}
        ``"uniform"`` fills the canvas with ``color``.
        ``"vertical_gradient"`` (default) blends ``bottom_color`` into
        ``top_color`` from the bottom of the canvas to the top.
    color : tuple[float, float, float, float]
        RGBA fill color used when ``mode`` is ``"uniform"``.
    bottom_color : tuple[float, float, float, float]
        RGBA color at the bottom of the canvas when ``mode`` is
        ``"vertical_gradient"``.
    top_color : tuple[float, float, float, float]
        RGBA color at the top of the canvas when ``mode`` is
        ``"vertical_gradient"``.
    """

    visible: bool = True
    mode: Literal["uniform", "vertical_gradient"] = "vertical_gradient"
    color: RGBA = DEFAULT_BOTTOM_COLOR
    bottom_color: RGBA = DEFAULT_BOTTOM_COLOR
    top_color: RGBA = DEFAULT_TOP_COLOR

    def to_colors(self) -> tuple[RGBA, ...]:
        """Return the colors to pass to ``pygfx.BackgroundMaterial.set_colors``.

        One color for ``"uniform"``, two (bottom, top) for
        ``"vertical_gradient"`` -- the argument order pygfx expects.
        """
        if self.mode == "uniform":
            return (self.color,)
        return (self.bottom_color, self.top_color)


def viewer_background() -> BackgroundAppearance:
    """The background ``Viewer`` and ``OrthoViewer`` start with.

    A uniform :data:`VIEWER_BACKGROUND_COLOR` (black).  The gradient colors
    keep their defaults, so switching ``mode`` to ``"vertical_gradient"``
    shows the usual gray gradient.  A scene built directly (``add_scene``,
    ``Scene``) keeps ``BackgroundAppearance``'s own default, the gradient.

    Returns
    -------
    BackgroundAppearance
        A new model on every call, so no two scenes share one.
    """
    return BackgroundAppearance(mode="uniform", color=VIEWER_BACKGROUND_COLOR)
