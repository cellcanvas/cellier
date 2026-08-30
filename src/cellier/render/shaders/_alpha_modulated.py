"""Point and line-segment materials carrying a per-element alpha buffer.

The trail fade and the caller's ``color_mode`` used to contend for one
channel: in stock pygfx the only per-element colour input is the vertex
colour buffer, and it *replaces* the uniform colour rather than
multiplying it.  Rendering a fade therefore meant forcing
``color_mode="vertex"`` behind the caller's back.

These materials give alpha its own channel (D19).  Each adds one storage
binding, ``s_alphas``, fed from ``geometry.alphas``, and multiplies it into
the final alpha *after* whatever ``color_mode`` composed the colour.  So a
fade composes with ``"uniform"`` and ``"vertex"`` identically, and
``color_mode`` is never written by the render layer (D20).

Coupling
--------
The substitutions are anchored to exact strings in pygfx's own shader
source, so a pygfx bump can move them.  That is the same coupling the
screen-space outline work accepted, and it is handled the same way: every
anchor is asserted, so a moved anchor raises at shader-build time with the
anchor in the message rather than silently rendering without the fade.

``scripts/v2/graph_alpha_spike.py`` is the runnable regression check after
any pygfx upgrade; ``tests/render/test_alpha_modulated_materials.py`` is
the GPU-free version of the same check that runs in CI.

Three things the spike established that are not obvious from the docs:

1. ``Shader.get_bindings`` returns ``{bindgroup: {slot: Binding}}`` -- a
   dict of dicts, not a list.  A subclass must copy group 0, add its slot,
   and re-call ``define_bindings`` for the whole group.
2. ``color_mode="auto"`` does *not* detect vertex colours.  For points it
   resolves to ``vertex_map`` when ``material.map`` is set and ``uniform``
   otherwise, so there is no stock path to "uniform RGB times per-vertex
   alpha".  That is what forced a custom material rather than a
   configuration.
3. In ``line.wgsl`` the ``varyings`` struct is not in scope where the
   per-vertex colour is interpolated.  The mixed alpha has to land in a
   local and be published later, at the varyings block.

Verified against pygfx 0.17.0 / wgpu 0.31.0.
"""

from __future__ import annotations

from typing import Any

import pygfx as gfx
from pygfx.renderers.wgpu import Binding, register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.lineshader import LineShader
from pygfx.renderers.wgpu.shaders.pointsshader import PointsShader

__all__ = [
    "AlphaLineSegmentMaterial",
    "AlphaLineSegmentShader",
    "AlphaPointsMaterial",
    "AlphaPointsShader",
]


class ShaderAnchorError(RuntimeError):
    """Raised when a pygfx wgsl anchor this module patches has moved.

    The pygfx-bump canary.  It names the anchor so the fix is a search in
    pygfx's shader source rather than a bisect.
    """


def _with_alpha_binding(
    shader: Any, wobject: Any, super_bindings: dict
) -> dict[int, dict]:
    """Append an ``s_alphas`` storage binding to bind group 0.

    See note 1 in the module docstring for why this is not a list append.
    """
    bindings = dict(super_bindings[0])
    bindings[len(bindings)] = Binding(
        "s_alphas", "buffer/read_only_storage", wobject.geometry.alphas, "VERTEX"
    )
    shader.define_bindings(0, bindings)
    return {0: bindings}


def _substitute(code: str, subs: list[tuple[str, str]]) -> str:
    """Apply anchored substitutions, asserting each anchor is still present.

    Parameters
    ----------
    code : str
        The pygfx-generated wgsl.
    subs : list[tuple[str, str]]
        ``(anchor, replacement)`` pairs, applied once each in order.

    Returns
    -------
    str
        The patched wgsl.

    Raises
    ------
    ShaderAnchorError
        When an anchor is no longer present in the pygfx source.
    """
    for old, new in subs:
        if old not in code:
            raise ShaderAnchorError(
                "pygfx shader anchor missing -- the wgsl changed under us. "
                "Rerun scripts/v2/graph_alpha_spike.py and re-anchor:\n"
                f"{old[:160]}"
            )
        code = code.replace(old, new, 1)
    return code


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------


class AlphaPointsMaterial(gfx.PointsMaterial):
    """``PointsMaterial`` whose final alpha is multiplied by ``geometry.alphas``.

    The geometry must carry an ``alphas`` buffer of one float per point.
    Upload a buffer of ones when no fade is active rather than swapping
    back to a stock material, which would force a pipeline rebuild on
    every fade toggle.
    """


@register_wgpu_render_function(gfx.Points, AlphaPointsMaterial)
class AlphaPointsShader(PointsShader):
    """Points shader with one extra storage binding and two substitutions."""

    type = "render"

    def get_bindings(self, wobject, shared, scene):
        """Add the ``s_alphas`` binding to bind group 0."""
        return _with_alpha_binding(
            self, wobject, super().get_bindings(wobject, shared, scene)
        )

    def get_code(self) -> str:
        """Return pygfx's points wgsl with the alpha multiply injected."""
        anchor_vertex = "    varyings.pick_idx = u32(node_index);"
        anchor_fragment = (
            "    var face_color = vec4<f32>(sampled_face_color.rgb, "
            "clamp(sampled_face_color.a, 0.0, 1.0) * face_alpha);"
        )
        return _substitute(
            super().get_code(),
            [
                (
                    anchor_vertex,
                    anchor_vertex
                    + "\n    varyings.trail_alpha = f32(load_s_alphas(node_index));",
                ),
                # One multiply, sited after every color_mode branch has
                # already composed its colour -- so one line covers them all.
                (
                    anchor_fragment,
                    anchor_fragment + "\n    face_color = vec4<f32>(face_color.rgb, "
                    "face_color.a * varyings.trail_alpha);",
                ),
            ],
        )


# ---------------------------------------------------------------------------
# Line segments
# ---------------------------------------------------------------------------


class AlphaLineSegmentMaterial(gfx.LineSegmentMaterial):
    """``LineSegmentMaterial`` whose alpha is multiplied by ``geometry.alphas``.

    The geometry must carry an ``alphas`` buffer of one float per *vertex*,
    so a segment's two ends can fade independently and the ramp runs along
    the segment.
    """


@register_wgpu_render_function(gfx.Line, AlphaLineSegmentMaterial)
class AlphaLineSegmentShader(LineShader):
    """Line shader carrying an alpha that ramps *along* each segment.

    Five substitutions rather than the points shader's two, because all six
    of the quad's vertices share one ``node_index``: relying on rasterizer
    varying interpolation alone would give a flat alpha per segment.  That
    is exactly why pygfx itself does ``mix(color_node, color_other,
    ratio_interp)`` for colours, and the alpha mirrors it.
    """

    type = "render"

    def get_bindings(self, wobject, shared, scene):
        """Add the ``s_alphas`` binding to bind group 0."""
        return _with_alpha_binding(
            self, wobject, super().get_bindings(wobject, shared, scene)
        )

    def get_code(self) -> str:
        """Return pygfx's line wgsl with the alpha ramp injected."""
        load_other = (
            "load_s_alphas(select(node_index_prev, node_index_next, vertex_num >= 4))"
        )
        return _substitute(
            super().get_code(),
            [
                # Seed this node's alpha and its neighbour unconditionally --
                # the colour equivalents sit inside a color_mode branch, and
                # the fade must not depend on which branch is live.
                (
                    "    $$ if color_mode == 'vertex'\n"
                    "        let color_node = load_s_colors(node_index);\n"
                    "        var color_other = color_node;",
                    "    let alpha_node = load_s_alphas(node_index);\n"
                    "    var alpha_other = alpha_node;\n"
                    "    $$ if color_mode == 'vertex'\n"
                    "        let color_node = load_s_colors(node_index);\n"
                    "        var color_other = color_node;",
                ),
                (
                    "        $$ if color_mode == 'vertex'\n"
                    "        color_other = load_s_colors(select(node_index_prev, "
                    "node_index_next, vertex_num >= 4));",
                    f"        alpha_other = {load_other};\n"
                    "        $$ if color_mode == 'vertex'\n"
                    "        color_other = load_s_colors(select(node_index_prev, "
                    "node_index_next, vertex_num >= 4));",
                ),
                # Interpolate along the segment exactly as pygfx does for
                # colour.  Lands in a local: see note 3 in the module
                # docstring -- varyings is not in scope here.
                (
                    "    $$ if color_mode == 'vertex'\n"
                    "        let color_vert = mix(color_node, color_other, "
                    "ratio_interp);",
                    "    let alpha_vert = mix(alpha_node, alpha_other, ratio_interp);\n"
                    "    $$ if color_mode == 'vertex'\n"
                    "        let color_vert = mix(color_node, color_other, "
                    "ratio_interp);",
                ),
                (
                    "    varyings.pick_idx = u32(node_index);",
                    "    varyings.pick_idx = u32(node_index);\n"
                    "    varyings.trail_alpha = f32(alpha_vert);",
                ),
                # Single fragment multiply, independent of color_mode.
                (
                    "    let opacity = min(1.0, color.a) * alpha * u_material.opacity;",
                    "    let opacity = min(1.0, color.a) * alpha * u_material.opacity "
                    "* varyings.trail_alpha;",
                ),
            ],
        )
