"""The line material of a plane outline (plane outline design 6.1, 6.2).

A plane's outline lies in the plane it is drawn around, and a plane drawn
by a volume shader writes depth there.  Two things keep the plane from
hiding its own outline, and both were measured (design 13.2):

- **A late render queue.**  The volume materials are in pygfx's
  transparent queue, and pygfx sorts by queue before ``render_order``, so a
  line in a line's usual queue is drawn first and painted over.  The
  outline is drawn after the visuals and writes no depth.
- **A depth offset sized in pixels.**  pygfx gives the whole width of a
  line the depth of its centre, while a tilted plane's depth changes across
  that width; the half of the line inside the polygon would be behind the
  plane.  The line's depth is therefore moved toward the camera, in view
  space, by the world size of :data:`OUTLINE_DEPTH_OFFSET_PX` pixels at its
  own depth.  Only the depth changes: the line stays where it is on the
  screen.  A mesh closer than that in front of the line does not hide it.

Coupling: the substitution is anchored to one line of pygfx's ``line.wgsl``.
The anchor is asserted, as in ``_alpha_modulated.py``, so a pygfx bump that
moves it raises at shader-build time and names the anchor.

Verified against pygfx 0.17.0.
"""

from __future__ import annotations

from typing import Any

import pygfx as gfx
from pygfx.renderers.wgpu import register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.lineshader import LineShader

from cellier.render.shaders._alpha_modulated import ShaderAnchorError

__all__ = [
    "OUTLINE_DEPTH_OFFSET_PX",
    "OUTLINE_RENDER_QUEUE",
    "PlaneOutlineMaterial",
    "PlaneOutlineShader",
]

#: How far the line's depth is moved toward the camera, in logical pixels'
#: worth of world distance at the line's depth (design B-D7).
OUTLINE_DEPTH_OFFSET_PX = 8.0

#: After every visual: pygfx's opaque queues end at 2600 and the volume
#: materials are at 3000.
OUTLINE_RENDER_QUEUE = 3500

#: Where pygfx's line vertex shader writes the clip-space position.
_POSITION_ANCHOR = "varyings.position = vec4<f32>(the_pos_n);"

_POSITION_WITH_OFFSET = f"""
    // cellier: the depth of a plane outline, moved toward the camera.
    let outline_p = u_stdinfo.projection_transform;
    // The world size of one logical pixel at this depth.
    let outline_px = abs(
        2.0 * the_pos_n.w / (outline_p[1][1] * u_stdinfo.logical_size.y));
    let outline_d = {OUTLINE_DEPTH_OFFSET_PX!r} * outline_px;
    // The clip-space depth of the point outline_d nearer the camera.
    let outline_z = the_pos_n.z + outline_d * outline_p[2][2];
    let outline_w = the_pos_n.w + outline_d * outline_p[2][3];
    varyings.position = vec4<f32>(
        the_pos_n.xy, outline_z / outline_w * the_pos_n.w, the_pos_n.w);
"""


class PlaneOutlineMaterial(gfx.LineMaterial):
    """A closed line in screen pixels, drawn after the visuals.

    Depth tested, so a mesh in front hides it; it writes no depth and is
    not picked.

    Parameters
    ----------
    color : tuple of four float
        RGBA.
    thickness : float
        The line's width in logical screen pixels.
    """

    def __init__(self, *, color: Any, thickness: float) -> None:
        super().__init__(
            color=color,
            thickness=thickness,
            thickness_space="screen",
            loop=True,
            aa=True,
            pick_write=False,
            depth_test=True,
            depth_write=False,
            render_queue=OUTLINE_RENDER_QUEUE,
        )


@register_wgpu_render_function(gfx.Line, PlaneOutlineMaterial)
class PlaneOutlineShader(LineShader):
    """pygfx's line shader with the line's depth moved toward the camera."""

    type = "render"

    def get_code(self) -> str:
        """Pygfx's line wgsl with the position write replaced.

        Raises
        ------
        ShaderAnchorError
            If the line of pygfx's shader this replaces is no longer there.
        """
        code = super().get_code()
        if _POSITION_ANCHOR not in code:
            raise ShaderAnchorError(
                "pygfx shader anchor missing -- line.wgsl changed under us. "
                "Rerun scripts/plane_border/v2_depth.py and re-anchor:\n"
                f"{_POSITION_ANCHOR}"
            )
        return code.replace(_POSITION_ANCHOR, _POSITION_WITH_OFFSET, 1)
