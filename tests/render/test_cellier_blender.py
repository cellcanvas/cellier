"""Tests for cellier's extra render targets.

Two targets share one ``Blender`` subclass: ``outline_id`` (the per-pixel
label key the outline pass reads) and ``normal`` (the per-pixel view-space
surface normal the ambient occlusion pass prefers over reconstruction).

The important test is ``test_cellier_blender_still_valid``.  ``CellierBlender``
extends three public ``Blender`` methods, and one of them --
``get_shader_kwargs`` -- returns the ``FragmentOutput`` struct as a **WGSL
source string** that this module edits.  If pygfx rewords that string, the
result is a shader compile error inside the draw callback, which
``rendercanvas`` swallows without logging: a black canvas and an empty
console.  Rendering a real frame here turns that into a CI failure.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import pytest
import wgpu
from pygfx.renderers.wgpu.engine.blender import Blender
from pygfx.renderers.wgpu.engine.shared import get_shared

from cellier.render._cellier_blender import (
    EXTRA_TARGETS,
    OUTLINE_ID_TARGET,
    CellierBlender,
    get_extra_target_view,
    install_cellier_blender,
)

_ALPHA_CONFIGS = {
    "opaque": {"method": "opaque", "premultiply_alpha": False},
    "blended": {
        "method": "blended",
        "color_src": "src-alpha",
        "color_dst": "one-minus-src-alpha",
        "alpha_src": "one",
        "alpha_dst": "one-minus-src-alpha",
    },
    "weighted": {"method": "weighted"},
}


def _render(scene, camera, *, with_target: bool, size=(64, 64)):
    """Draw *scene* offscreen; return ``(renderer, rgba frame)``."""
    from rendercanvas.offscreen import RenderCanvas

    canvas = RenderCanvas(size=size, pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    if with_target:
        assert install_cellier_blender(renderer, [OUTLINE_ID_TARGET]) is True

    errors: list[BaseException] = []

    def _draw() -> None:
        try:
            renderer.render(scene, camera)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)
            raise

    canvas.request_draw(_draw)
    image = canvas.draw()
    if errors:  # pragma: no cover - failure path
        raise RuntimeError(
            f"draw failed -- {type(errors[0]).__name__}: {errors[0]}"
        ) from errors[0]
    return renderer, np.asarray(image)


def _read_outline_id(renderer) -> np.ndarray:
    """Read the ``outline_id`` target back as an ``(h, w)`` uint32 array."""
    texture = renderer._blender.get_texture(OUTLINE_ID_TARGET)
    width, height = texture.size[:2]
    raw = get_shared().device.queue.read_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": 4 * width, "rows_per_image": height},
        (width, height, 1),
    )
    return np.frombuffer(raw, np.uint32).reshape(height, width)


def _sphere_scene(*, pick_write: bool = True):
    scene = gfx.Scene()
    scene.add(
        gfx.Mesh(
            gfx.sphere_geometry(1.0, 32, 16),
            gfx.MeshBasicMaterial(color="#ff8800", pick_write=pick_write),
        )
    )
    camera = gfx.OrthographicCamera()
    camera.show_object(scene)
    return scene, camera


# ---------------------------------------------------------------------------
# The canary
# ---------------------------------------------------------------------------


def test_cellier_blender_still_valid(offscreen_renderer):
    """A frame renders through ``CellierBlender`` and the target reads back.

    This is the Stage 7 counterpart to ``test_pygfx_coupling_still_valid``.
    If it fails after a pygfx upgrade, check
    ``cellier.render._cellier_blender`` against the new ``Blender`` --
    specifically the generated ``FragmentOutput`` text.
    """
    scene, camera = _sphere_scene()
    renderer, _frame = _render(scene, camera, with_target=True)

    assert get_extra_target_view(renderer, OUTLINE_ID_TARGET) is not None
    values = _read_outline_id(renderer)
    assert values.shape == (64, 64)
    # Nothing writes the field yet (that is Stage 7b), and WGSL
    # zero-initialises function-scope ``var``, so every pixel must be 0.
    # A nonzero value here means a shader is writing garbage into it.
    assert np.array_equal(np.unique(values), np.array([0], dtype=np.uint32))


def test_cellier_blender_leaves_the_colour_output_unchanged(offscreen_renderer):
    """Adding the target must not perturb the rendered image at all."""
    scene, camera = _sphere_scene()
    _stock_renderer, stock = _render(scene, camera, with_target=False)
    _extra_renderer, extended = _render(scene, camera, with_target=True)

    np.testing.assert_array_equal(stock, extended)


def test_renders_with_pick_write_disabled(offscreen_renderer):
    """An object that writes no pick also declares no ``outline_id``.

    Both fields are commented out of the struct together, which keeps the
    declared locations contiguous; the pipeline still carries both targets
    with a zero write mask.  This is the path that would break if the two
    were gated independently.
    """
    scene, camera = _sphere_scene(pick_write=False)
    renderer, frame = _render(scene, camera, with_target=True)

    assert np.count_nonzero(frame[..., 3]) > 0
    assert np.array_equal(np.unique(_read_outline_id(renderer)), np.array([0]))


# ---------------------------------------------------------------------------
# The struct surgery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "expected_location"),
    [("opaque", 2), ("blended", 2), ("weighted", 3)],
)
def test_field_lands_on_the_right_location(method, expected_location):
    """The location index follows the alpha method, not a hardcoded number.

    Weighted blending spends locations 0 and 1 on accum and reveal, so the
    field lands at 3 there and at 2 everywhere else.  Deriving it from the
    struct rather than hardcoding is what keeps those in step.
    """
    stock = Blender()
    code = stock.get_shader_kwargs(True, _ALPHA_CONFIGS[method])["fragment_output_code"]
    out = CellierBlender._add_field(
        code, EXTRA_TARGETS[OUTLINE_ID_TARGET], enabled=True
    )

    assert f"@location({expected_location}) outline_id: u32," in out
    # Inserted inside the struct, not after it.
    struct = out[out.index("struct FragmentOutput") : out.index("};")]
    assert "outline_id" in struct


def test_commented_out_field_when_disabled():
    """Disabled means the line is commented, mirroring how pygfx hides pick."""
    stock = Blender()
    code = stock.get_shader_kwargs(True, _ALPHA_CONFIGS["opaque"])[
        "fragment_output_code"
    ]
    out = CellierBlender._add_field(
        code, EXTRA_TARGETS[OUTLINE_ID_TARGET], enabled=False
    )

    assert "// @location(2) outline_id: u32," in out


def test_commented_pick_still_reserves_its_location():
    """A commented-out pick line still holds location 1.

    pygfx disables pick by commenting the line out while the pipeline keeps
    the target, so the number stays reserved.  Skipping commented lines
    when picking the next index would collide with pick's slot.
    """
    stock = Blender()
    code = stock.get_shader_kwargs(False, _ALPHA_CONFIGS["opaque"])[
        "fragment_output_code"
    ]
    assert "// @location(1) pick" in code
    out = CellierBlender._add_field(
        code, EXTRA_TARGETS[OUTLINE_ID_TARGET], enabled=False
    )
    assert "// @location(2) outline_id: u32," in out


def test_unrecognised_generated_code_fails_loudly():
    """A pygfx rewording must raise here, not compile-fail inside a draw."""
    with pytest.raises(RuntimeError, match="FragmentOutput"):
        CellierBlender._add_field(
            "no struct here", EXTRA_TARGETS[OUTLINE_ID_TARGET], enabled=True
        )

    with pytest.raises(RuntimeError, match="@location"):
        CellierBlender._add_field(
            "struct FragmentOutput {\n    stub: u32,\n};",
            EXTRA_TARGETS[OUTLINE_ID_TARGET],
            enabled=True,
        )


# ---------------------------------------------------------------------------
# Blender behaviour
# ---------------------------------------------------------------------------


def test_target_is_declared_and_changes_the_pipeline_cache_hash(offscreen_renderer):
    """``hash`` derives from the target list, so pipelines never mix.

    ``BlendRenderState`` keys on this hash.  If it did not change, a
    pipeline built for the stock blender could be reused with an attachment
    list that has one more entry.
    """
    stock = Blender()
    outline = CellierBlender([OUTLINE_ID_TARGET])

    assert OUTLINE_ID_TARGET not in stock.texture_info
    assert OUTLINE_ID_TARGET in outline.texture_info
    assert outline.texture_info[OUTLINE_ID_TARGET]["format"] == (
        wgpu.TextureFormat.r32uint
    )
    assert stock.hash != outline.hash
    assert OUTLINE_ID_TARGET in outline.hash


def test_target_states_and_attachments_stay_aligned(offscreen_renderer):
    """The new entry is appended last in both, so ordering matches.

    Target-state order, attachment order and ``@location(N)`` all have to
    agree; appending last in every override is what guarantees it.
    """
    outline = CellierBlender([OUTLINE_ID_TARGET])
    outline.ensure_target_size((8, 8))

    states = outline.get_color_descriptors(True, _ALPHA_CONFIGS["opaque"])
    attachments = outline.get_color_attachments("normal")

    assert len(states) == len(attachments)
    assert states[-1]["format"] == wgpu.TextureFormat.r32uint
    assert attachments[-1].view.texture.format == wgpu.TextureFormat.r32uint


def test_write_mask_follows_pick_write(offscreen_renderer):
    """An object that writes no pick writes no outline key either."""
    outline = CellierBlender([OUTLINE_ID_TARGET])
    with_pick = outline.get_color_descriptors(True, _ALPHA_CONFIGS["opaque"])
    without = outline.get_color_descriptors(False, _ALPHA_CONFIGS["opaque"])

    assert with_pick[-1]["write_mask"] == wgpu.ColorWrite.ALL
    assert without[-1]["write_mask"] == 0


def test_shader_kwargs_expose_the_template_var(offscreen_renderer):
    """``write_outline_id`` lets label shaders compile the write away."""
    outline = CellierBlender([OUTLINE_ID_TARGET])
    assert (
        outline.get_shader_kwargs(True, _ALPHA_CONFIGS["opaque"])["write_outline_id"]
        is True
    )
    assert (
        outline.get_shader_kwargs(False, _ALPHA_CONFIGS["opaque"])["write_outline_id"]
        is False
    )


def test_pick_and_depth_flags_are_preserved(offscreen_renderer):
    """Installing must not silently re-enable a target the renderer lacked."""
    from rendercanvas.offscreen import RenderCanvas

    canvas = RenderCanvas(size=(16, 16), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer._blender = Blender(enable_pick=False)

    assert install_cellier_blender(renderer, [OUTLINE_ID_TARGET]) is True
    assert "pick" not in renderer._blender.texture_info
    assert OUTLINE_ID_TARGET in renderer._blender.texture_info


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def test_install_degrades_on_unexpected_objects():
    """A pygfx rename disables label keys rather than breaking the canvas."""

    class _NoBlender:
        pass

    assert install_cellier_blender(_NoBlender(), [OUTLINE_ID_TARGET]) is False
    assert get_extra_target_view(_NoBlender(), OUTLINE_ID_TARGET) is None


def test_install_is_idempotent(offscreen_renderer):
    """A second call keeps the blender it already installed."""
    from rendercanvas.offscreen import RenderCanvas

    canvas = RenderCanvas(size=(16, 16), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)

    assert install_cellier_blender(renderer, [OUTLINE_ID_TARGET]) is True
    first = renderer._blender
    assert install_cellier_blender(renderer, [OUTLINE_ID_TARGET]) is True
    assert renderer._blender is first


def test_get_extra_target_view_returns_none_on_the_stock_blender(offscreen_renderer):
    """No target means the composite pass falls back to ``global_id``."""
    scene, camera = _sphere_scene()
    renderer, _frame = _render(scene, camera, with_target=False)
    assert get_extra_target_view(renderer, OUTLINE_ID_TARGET) is None


# ---------------------------------------------------------------------------
# CanvasView wiring
# ---------------------------------------------------------------------------


def test_canvas_installs_the_blender_only_when_outlines_are_enabled(
    qtbot, offscreen_renderer
):
    """Outlines off is the default, and it must cost nothing.

    The target is worth 4 bytes per pixel, and the decision is
    construction-time because the target list keys the pipeline cache.
    """
    from cellier.controller import CellierController
    from cellier.render._config import OutlineConfig, RenderManagerConfig

    def _canvas_blender(enabled: bool):
        controller = CellierController(
            render_config=RenderManagerConfig(outline=OutlineConfig(enabled=enabled))
        )
        controller.camera_reslice_enabled = False
        scene = controller.add_scene(dim="3d", name="scene")
        controller.add_canvas(scene_id=scene.id)
        canvas = next(iter(controller._render_manager._canvases.values()))
        return canvas

    off = _canvas_blender(False)
    assert off._outline_id_available is False
    assert not isinstance(off._renderer._blender, CellierBlender)
    assert OUTLINE_ID_TARGET not in off._renderer._blender.texture_info

    on = _canvas_blender(True)
    assert on._outline_id_available is True
    assert isinstance(on._renderer._blender, CellierBlender)
