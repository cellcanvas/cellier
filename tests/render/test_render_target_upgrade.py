"""Tests for adding cellier's extra render targets after the first draw.

The targets are chosen at canvas construction from the render config, which
is right for a viewer configured up front and wrong for one where a user
ticks the box later.  ``ensure_extra_targets`` closes that gap by installing
a replacement blender mid-session; these tests pin the three things that
makes true and the one invariant it rests on.

Why this is safe, in the order the tests check it:

* The "must run before the first draw" rule belongs to
  ``enable_pick_texture_binding``, which raises usage bits on textures that
  already exist.  A replacement blender has no textures yet, so its own are
  created at the right size and usage on the next draw.
* pygfx keys its pipeline containers on ``(wobject, renderstate, material)``
  with the renderstate derived from ``Blender.hash``, so a new blender
  *branches* the pipeline cache rather than corrupting it.  The cost is one
  recompile frame.
* Both cellier effect passes re-resolve their optional target views every
  frame and fold presence into their pipeline hash, so they pick up a target
  that appears without being told.  **A pass that cached a view across
  frames would keep drawing into the abandoned blender**;
  ``test_a_swap_is_invisible_to_the_colour_output`` is what would catch that.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import pytest
from pygfx.renderers.wgpu.engine.shared import get_shared

from cellier.render._cellier_blender import (
    NORMAL_TARGET,
    OUTLINE_ID_TARGET,
    CellierBlender,
    ensure_extra_targets,
    get_extra_target_view,
)
from cellier.render._pick_buffer import enable_pick_texture_binding, get_pick_view
from cellier.render.shaders._image_volume import ImageVolumeIsoMaterial

SIZE = 64


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _sphere_field(n: int = 40, radius: float = 14.0) -> np.ndarray:
    """A smooth field crossing 0.5 on a sphere, so the iso shader has a surface."""
    centre = (n - 1) / 2
    z, y, x = np.mgrid[:n, :n, :n]
    distance = np.sqrt((x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2)
    return np.clip((radius - distance) / 3.0 + 0.5, 0.0, 1.0).astype(np.float32)


def _iso_scene() -> tuple[gfx.Scene, gfx.Camera]:
    """An iso volume, which is the case that actually writes a normal."""
    scene = gfx.Scene()
    n = 40
    volume = gfx.Volume(
        gfx.Geometry(grid=gfx.Texture(_sphere_field(n), dim=3)),
        ImageVolumeIsoMaterial(clim=(0, 1), threshold=0.5),
    )
    centre = (n - 1) / 2
    volume.local.position = (-centre, -centre, -centre)
    scene.add(volume)

    camera = gfx.PerspectiveCamera(45, 1, depth_range=(1.0, 500.0))
    camera.local.position = (0, 0, 90)
    camera.look_at((0, 0, 0))
    return scene, camera


class _Harness:
    """An offscreen renderer that can be drawn repeatedly."""

    def __init__(self, scene, camera, *, grant_pick: bool) -> None:
        from rendercanvas.offscreen import RenderCanvas

        self.canvas = RenderCanvas(size=(SIZE, SIZE), pixel_ratio=1)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.renderer.pixel_scale = 1
        self.renderer.ppaa = "none"
        self.scene = scene
        self.camera = camera
        # Grant on the *stock* blender, before any swap, so a later
        # assertion can tell whether the grant survived being replaced.
        self.pick_granted = (
            enable_pick_texture_binding(self.renderer) if grant_pick else False
        )

    def draw(self) -> np.ndarray:
        errors: list[BaseException] = []

        def _draw() -> None:
            try:
                self.renderer.render(self.scene, self.camera)
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)
                raise

        self.canvas.request_draw(_draw)
        image = self.canvas.draw()
        if errors:
            cause = errors[0]
            raise RuntimeError(
                f"offscreen draw failed -- {type(cause).__name__}: {cause}"
            ) from cause
        return np.asarray(image)

    def read(self, name: str, dtype, bytes_per_pixel: int, channels: int = 1):
        """Read one extra target back, or ``None`` if it does not exist."""
        texture = self.renderer._blender.get_texture(name)
        if texture is None:
            return None
        raw = get_shared().device.queue.read_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            {
                "offset": 0,
                "bytes_per_row": bytes_per_pixel * SIZE,
                "rows_per_image": SIZE,
            },
            (SIZE, SIZE, 1),
        )
        array = np.frombuffer(raw, dtype)
        if channels > 1:
            return array.reshape(SIZE, SIZE, channels)
        return array.reshape(SIZE, SIZE)

    def normals(self) -> np.ndarray | None:
        written = self.read(NORMAL_TARGET, np.float16, 8, 4)
        return None if written is None else written.astype(np.float32)


@pytest.fixture
def harness(offscreen_renderer):
    """An iso-volume scene on an offscreen renderer with the pick grant made."""
    scene, camera = _iso_scene()
    return _Harness(scene, camera, grant_pick=True)


# ---------------------------------------------------------------------------
# The swap
# ---------------------------------------------------------------------------


def test_the_normal_target_can_be_added_after_the_first_draw(harness):
    """The case a GUI toggle creates: occlusion switched on after drawing.

    Without this the pass would run against normals reconstructed from
    depth, which is 34 degrees out on a raymarched isosurface -- worse, and
    silently so.
    """
    harness.draw()
    assert harness.normals() is None, "the stock blender has no normal target"

    assert ensure_extra_targets(harness.renderer, [NORMAL_TARGET]) is True
    harness.draw()

    normals = harness.normals()
    assert normals is not None
    magnitude = np.linalg.norm(normals[..., :3], axis=-1)
    written = magnitude > 0.5
    assert written.sum() > 100, "the iso shader wrote no normals after the swap"
    assert np.all(np.abs(magnitude[written] - 1.0) < 0.05), "not unit length"


def test_the_target_set_can_grow_after_the_first_draw(harness):
    """Adding a second target keeps the first.

    A user who enables occlusion and *then* outlines must not lose the
    normal target on the way, which is what a naive replacement would do.
    """
    assert ensure_extra_targets(harness.renderer, [NORMAL_TARGET]) is True
    harness.draw()

    assert ensure_extra_targets(harness.renderer, [OUTLINE_ID_TARGET]) is True
    harness.draw()

    blender = harness.renderer._blender
    assert isinstance(blender, CellierBlender)
    assert {t.name for t in blender.extra_targets} == {NORMAL_TARGET, OUTLINE_ID_TARGET}
    assert get_extra_target_view(harness.renderer, OUTLINE_ID_TARGET) is not None
    magnitude = np.linalg.norm(harness.normals()[..., :3], axis=-1)
    assert (magnitude > 0.5).sum() > 100, "the normal target stopped being written"


def test_the_pick_grant_survives_the_swap(harness):
    """Outlines still work after a swap, because usage bits are carried over.

    ``enable_pick_texture_binding`` runs once, on the stock blender, before
    the first draw.  If the replacement dropped that grant the outline pass
    would degrade to a passthrough -- the sort of failure that shows up as
    "it just does nothing".
    """
    assert harness.pick_granted is True
    harness.draw()
    assert get_pick_view(harness.renderer) is not None

    assert ensure_extra_targets(harness.renderer, [OUTLINE_ID_TARGET]) is True
    harness.draw()
    assert get_pick_view(harness.renderer) is not None


def test_a_swap_is_invisible_to_the_colour_output(offscreen_renderer):
    """Adding targets mid-session changes no pixel the user can see.

    The strong form of the safety claim, and the test that would catch a
    pass caching a blender texture view across frames: such a pass would
    keep drawing into the blender that was replaced.
    """
    scene, camera = _iso_scene()
    harness = _Harness(scene, camera, grant_pick=True)
    harness.draw()
    before = harness.draw()

    assert ensure_extra_targets(harness.renderer, [NORMAL_TARGET, OUTLINE_ID_TARGET])
    harness.draw()
    after = harness.draw()

    differing = int(np.count_nonzero(np.any(before != after, axis=-1)))
    assert differing == 0, f"{differing} px changed across the swap"


def test_ensure_is_a_no_op_when_the_targets_are_already_there(harness):
    """The common case costs nothing: no replacement, so no recompile."""
    assert ensure_extra_targets(harness.renderer, [NORMAL_TARGET]) is True
    installed = harness.renderer._blender
    assert ensure_extra_targets(harness.renderer, [NORMAL_TARGET]) is True
    assert harness.renderer._blender is installed


def test_ensure_rejects_an_unknown_target(harness):
    """A typo in cellier's own code, so it raises rather than degrading."""
    with pytest.raises(ValueError, match="unknown extra render target"):
        ensure_extra_targets(harness.renderer, ["not_a_target"])


def test_ensure_degrades_on_unexpected_objects():
    """A pygfx rename disables the feature rather than breaking the canvas."""

    class _NoBlender:
        pass

    assert ensure_extra_targets(_NoBlender(), [NORMAL_TARGET]) is False


# ---------------------------------------------------------------------------
# CanvasView / RenderManager wiring
# ---------------------------------------------------------------------------


def test_enabling_ssao_installs_the_normal_target_on_a_live_canvas(
    controller, offscreen_renderer
):
    """The whole point: a canvas built without the target gains it on demand."""
    scene = controller.add_scene(displayed_axes=("z", "y", "x"), name="scene")
    controller.add_canvas(scene_id=scene.id)
    canvas_view = controller.get_canvas_view(next(iter(scene.canvases)))
    assert canvas_view._normal_target_available is False

    controller.ambient_occlusion_enabled = True

    assert canvas_view._normal_target_available is True
    # The texture itself is created on the next draw, so check the blender
    # carries the target rather than asking for a view that cannot exist yet.
    assert NORMAL_TARGET in canvas_view._renderer._blender.texture_info


def test_enabling_outlines_installs_the_label_key_target_on_a_live_canvas(
    controller, offscreen_renderer
):
    """Same for outlines, whose target carries the per-label key."""
    scene = controller.add_scene(displayed_axes=("z", "y", "x"), name="scene")
    controller.add_canvas(scene_id=scene.id)
    canvas_view = controller.get_canvas_view(next(iter(scene.canvases)))
    assert canvas_view._outline_id_available is False

    controller.outline_enabled = True

    assert canvas_view._outline_id_available is True
    assert OUTLINE_ID_TARGET in canvas_view._renderer._blender.texture_info


def test_a_canvas_built_with_the_feature_on_pays_no_late_swap(offscreen_renderer):
    """Configuring up front keeps the fast path: nothing to install later."""
    from cellier.controller import CellierController
    from cellier.render._config import AmbientOcclusionConfig, RenderManagerConfig

    controller = CellierController(
        render_config=RenderManagerConfig(
            ambient_occlusion=AmbientOcclusionConfig(enabled=True)
        )
    )
    controller.camera_reslice_enabled = False
    try:
        scene = controller.add_scene(displayed_axes=("z", "y", "x"), name="scene")
        controller.add_canvas(scene_id=scene.id)
        canvas_view = controller.get_canvas_view(next(iter(scene.canvases)))
        assert canvas_view._normal_target_available is True

        blender = canvas_view._renderer._blender
        controller.ambient_occlusion_enabled = False
        controller.ambient_occlusion_enabled = True
        assert canvas_view._renderer._blender is blender
    finally:
        controller.close()


def test_disabling_a_feature_keeps_its_target(controller, offscreen_renderer):
    """Targets are only ever added.

    Dropping one would buy back its memory at the price of a second
    recompile, for a feature the user may switch straight back on.
    """
    scene = controller.add_scene(displayed_axes=("z", "y", "x"), name="scene")
    controller.add_canvas(scene_id=scene.id)
    canvas_view = controller.get_canvas_view(next(iter(scene.canvases)))

    controller.ambient_occlusion_enabled = True
    controller.ambient_occlusion_enabled = False

    assert canvas_view._normal_target_available is True
