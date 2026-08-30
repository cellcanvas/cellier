"""The per-vertex alpha materials (D19/D20).

``test_pygfx_shader_anchors_still_valid`` is the pygfx-bump canary and
needs no GPU: it builds both shaders' wgsl and fails loudly if an anchored
substring has moved in pygfx's own shader source.  Everything else renders
through the shared ``offscreen_renderer`` fixture, which skips cleanly on a
machine with no wgpu adapter.

Sampled values are **sRGB-encoded**, so these tests assert monotonicity and
endpoints, never linearity: alpha 0.25 lands near 138, not near 64.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import pytest

from cellier.render.shaders._alpha_modulated import (
    AlphaLineSegmentMaterial,
    AlphaLineSegmentShader,
    AlphaPointsMaterial,
    AlphaPointsShader,
    ShaderAnchorError,
    _substitute,
)

_SIZE = (320, 80)


def _scene() -> gfx.Scene:
    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#000000"))
    return scene


def _camera() -> gfx.Camera:
    return gfx.OrthographicCamera(*_SIZE)


def _points_object(alphas, color=(1.0, 0.0, 0.0, 1.0)):
    positions = np.zeros((len(alphas), 3), dtype=np.float32)
    positions[:, 0] = np.linspace(-120, 120, len(alphas))
    material = AlphaPointsMaterial(size=40, color=color)
    material.color_mode = "uniform"
    geometry = gfx.Geometry(
        positions=positions, alphas=np.asarray(alphas, dtype=np.float32)
    )
    return gfx.Points(geometry, material), positions


def _line_object(alphas, color=(1.0, 0.0, 0.0, 1.0)):
    positions = np.array([[-140, 0, 0], [140, 0, 0]], dtype=np.float32)
    material = AlphaLineSegmentMaterial(thickness=20, color=color)
    material.color_mode = "uniform"
    geometry = gfx.Geometry(
        positions=positions, alphas=np.asarray(alphas, dtype=np.float32)
    )
    return gfx.Line(geometry, material)


def _point_peaks(image, positions, channel):
    """Peak channel value in a band around each point's screen column."""
    peaks = []
    for i in range(positions.shape[0]):
        cx = int(image.shape[1] / 2 + positions[i, 0])
        band = image[:, max(0, cx - 6) : cx + 6, channel].astype(np.float32)
        peaks.append(float(band.max()))
    return peaks


# ── The canary: no GPU required ────────────────────────────────────────────


def test_pygfx_shader_anchors_still_valid():
    """Build both shaders' wgsl and assert every anchor was found.

    This is the cheap unit-level version of
    ``scripts/v2/graph_alpha_spike.py`` and the thing that catches a pygfx
    bump moving a substitution site.  It runs without a wgpu adapter, so it
    executes everywhere the suite does.
    """
    points, _ = _points_object([0.0, 1.0])
    points_code = AlphaPointsShader(points).get_code()
    assert "varyings.trail_alpha" in points_code
    assert "face_color.a * varyings.trail_alpha" in points_code

    line = _line_object([0.0, 1.0])
    line_code = AlphaLineSegmentShader(line).get_code()
    assert "let alpha_node = load_s_alphas(node_index);" in line_code
    assert "mix(alpha_node, alpha_other, ratio_interp)" in line_code
    assert "* varyings.trail_alpha;" in line_code


def test_missing_anchor_raises_naming_the_anchor():
    """A moved anchor fails at shader-build time, not silently at render."""
    with pytest.raises(ShaderAnchorError, match="anchor missing"):
        _substitute("some unrelated wgsl", [("let x = 1;", "let x = 2;")])


# ── Rendered output ────────────────────────────────────────────────────────


def test_points_alpha_ramp_renders(offscreen_renderer):
    """A known alpha ramp produces a monotonic sampled ramp."""
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    points, positions = _points_object(alphas)
    scene = _scene()
    scene.add(points)

    image = offscreen_renderer(scene, _camera(), _SIZE)
    peaks = _point_peaks(image, positions, channel=0)

    assert peaks[0] < 5, f"alpha=0 should be invisible: {peaks}"
    assert peaks[-1] > 200, f"alpha=1 should be full: {peaks}"
    for i in range(len(peaks) - 1):
        assert peaks[i] <= peaks[i + 1] + 2, f"not monotonic: {peaks}"


def test_line_alpha_ramps_along_segment(offscreen_renderer):
    """The ramp varies *along* the segment, not flat per segment.

    All six of the quad's vertices share one ``node_index``, so without the
    ``mix(alpha_node, alpha_other, ratio_interp)`` injection this would be
    a single flat value.
    """
    scene = _scene()
    scene.add(_line_object([0.0, 1.0]))

    image = offscreen_renderer(scene, _camera(), _SIZE)
    row = image[40, :, 0].astype(np.float32)
    samples = [float(row[x]) for x in (30, 90, 160, 230, 289)]

    for i in range(len(samples) - 1):
        assert samples[i] <= samples[i + 1] + 3, f"not monotonic: {samples}"
    assert samples[0] < 60, f"ramp start too bright: {samples}"
    assert samples[-1] > 200, f"ramp end too dim: {samples}"
    assert samples[-1] - samples[0] > 100, f"ramp is flat: {samples}"


def test_color_change_preserves_alpha_ramp(offscreen_renderer):
    """Changing the uniform colour leaves the ramp identical in the new channel.

    This is what makes a colour edit under an active fade a plain uniform
    write, with no colour-buffer rebuild.
    """
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    red_points, positions = _points_object(alphas, color=(1.0, 0.0, 0.0, 1.0))
    red_scene = _scene()
    red_scene.add(red_points)
    red_peaks = _point_peaks(
        offscreen_renderer(red_scene, _camera(), _SIZE), positions, channel=0
    )

    green_points, _ = _points_object(alphas, color=(0.0, 1.0, 0.0, 1.0))
    green_scene = _scene()
    green_scene.add(green_points)
    green_peaks = _point_peaks(
        offscreen_renderer(green_scene, _camera(), _SIZE), positions, channel=1
    )

    assert red_peaks == green_peaks


def test_color_mode_untouched(offscreen_renderer):
    """``color_mode`` survives a full render with a fade (D19/D20).

    The direct regression for the contention this design removes: rendering
    a fade used to require forcing ``color_mode="vertex"``.
    """
    points, _ = _points_object([0.0, 0.5, 1.0])
    line = _line_object([0.0, 1.0])
    scene = _scene()
    scene.add(points)
    scene.add(line)

    offscreen_renderer(scene, _camera(), _SIZE)

    assert points.material.color_mode == "uniform"
    assert line.material.color_mode == "uniform"


def test_alpha_survives_buffer_replacement(offscreen_renderer):
    """Replacing the geometry keeps the binding live, including a length change.

    This is the per-frame path: every reslice replaces the geometry, and a
    binding that went stale here would show as a fade that stops updating.
    """
    points, _ = _points_object([1.0, 1.0, 1.0, 1.0, 1.0])
    scene = _scene()
    scene.add(points)
    offscreen_renderer(scene, _camera(), _SIZE)

    # Shorter buffer, and the first point now fully transparent.
    new_positions = np.zeros((3, 3), dtype=np.float32)
    new_positions[:, 0] = np.linspace(-120, 120, 3)
    points.geometry = gfx.Geometry(
        positions=new_positions, alphas=np.array([0.0, 1.0, 1.0], dtype=np.float32)
    )
    image = offscreen_renderer(scene, _camera(), _SIZE)

    peaks = _point_peaks(image, new_positions, channel=0)
    assert peaks[0] < 5, f"alpha=0 point should be invisible after replacement: {peaks}"
    assert peaks[1] > 200, f"alpha=1 point should be full: {peaks}"
