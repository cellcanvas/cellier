"""The ``"plane"`` render mode in the render layer.

Plane rendering design v3: the model and the controller know the mode from
Phase 5; the shaders that draw a plane arrive with Phases 6 (in-memory) and
7 (multiscale).  Until then a material asked for ``"plane"`` is given its
family's default volume mode, so a visual in plane mode builds and draws
without a shader that has no branch for it.  Each caller of
:func:`volume_mode_until_planes_draw` is a place one of those phases
replaces.
"""

from __future__ import annotations

from cellier.visuals._render_plane import PLANE_RENDER_MODE


def volume_mode_until_planes_draw(render_mode: str, fallback: str) -> str:
    """*render_mode*, or *fallback* in its place when it is ``"plane"``.

    Parameters
    ----------
    render_mode : str
        A visual's render mode.
    fallback : str
        The volume mode the material draws until it can draw a plane.

    Returns
    -------
    str
        A mode the material's shader has a branch for.
    """
    return fallback if render_mode == PLANE_RENDER_MODE else render_mode
