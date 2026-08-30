"""Tests for the screen-space ambient occlusion pass.

Five groups:

* the pygfx coupling canary -- ``test_pygfx_ssao_coupling_still_valid`` is
  what fails when a pygfx bump invalidates the depth binding or moves the
  normal-reconstruction helpers this pass copies, so it should break CI
  rather than silently degrading the occlusion;
* the occlusion operator itself, driven over real geometry whose correct
  answer is known by construction -- a flat plane must produce *none*, a
  box resting on a plane must produce it in the contact crease and not on
  the silhouette;
* the auto radius, which is the only thing standing between this feature
  and being meaningless in cellier's coordinate systems;
* the hybrid normal: volumes read a written normal, meshes read exactly
  zero and fall back to reconstruction;
* the per-visual opt-out, including the regression the merged LUT map
  exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import pytest
import wgpu
from pygfx.renderers.wgpu.engine.effectpasses import NormalPass
from pygfx.renderers.wgpu.engine.shared import get_shared

from cellier.render._cellier_blender import (
    NORMAL_TARGET,
    OUTLINE_ID_TARGET,
    install_cellier_blender,
)
from cellier.render._config import RenderManagerConfig, SSAOConfig
from cellier.render._ssao import MAX_KERNEL_SAMPLES, SSAOPass, make_kernel
from cellier.render._visual_lut import (
    AO_EXCLUDED_BIT,
    KIND_WHOLE_OBJECT,
    PLACEMENT_INWARD,
    VisualLut,
    decode_entry,
    encode_entry,
)
from cellier.render.shaders._image_volume import (
    ImageVolumeIsoMaterial,
    ImageVolumeMipMaterial,
)

PICK_ID_MAX = 2**20 - 1
SIZE = 96


# ---------------------------------------------------------------------------
# Scene helpers
#
# Every scene here has a known-correct occlusion answer by construction, so
# the assertions are about geometry rather than about a golden image.
# ---------------------------------------------------------------------------


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Shrink a boolean mask by *radius* pixels.

    Used to keep silhouette pixels out of interior assertions.  The 5-tap
    normal reconstruction reads two pixels either side, so a pixel within
    two of an edge has taps landing on the background; the box blur then
    spreads that a further ``blur_radius``.  That fringe is expected and
    is not what an interior test is about.
    """
    out = mask.copy()
    for _ in range(radius):
        shrunk = out.copy()
        # Eight-connected, not four: the box blur is a *square* kernel, so
        # a pixel two away diagonally is still inside it.  A diamond-shaped
        # erosion would leave those corners in and they are exactly where
        # the leftover bleed shows up.
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.zeros_like(out)
                ys = slice(max(dy, 0), out.shape[0] + min(dy, 0))
                xs = slice(max(dx, 0), out.shape[1] + min(dx, 0))
                ys_src = slice(max(-dy, 0), out.shape[0] + min(-dy, 0))
                xs_src = slice(max(-dx, 0), out.shape[1] + min(-dx, 0))
                shifted[ys, xs] = out[ys_src, xs_src]
                shrunk &= shifted
        out = shrunk
    return out


def _lit_scene() -> gfx.Scene:
    scene = gfx.Scene()
    scene.add(gfx.AmbientLight(0.5))
    light = gfx.DirectionalLight(2.0)
    light.local.position = (0, 1, 1)
    scene.add(light)
    return scene


def _plane(scale: float = 1.0) -> gfx.Mesh:
    plane = gfx.Mesh(
        gfx.plane_geometry(120 * scale, 120 * scale),
        gfx.MeshPhongMaterial(color="#888888"),
    )
    # Lay the plane flat: rotate -90 degrees about x.
    plane.local.rotation = (-0.70710678, 0.0, 0.0, 0.70710678)
    return plane


def _camera(scale: float = 1.0) -> gfx.PerspectiveCamera:
    camera = gfx.PerspectiveCamera(50, 1, depth_range=(1.0 * scale, 1000.0 * scale))
    camera.local.position = (60 * scale, 40 * scale, 60 * scale)
    camera.look_at((0.0, 8 * scale, 0.0))
    return camera


def _sphere_field(n: int = 48, radius: float = 16.0) -> np.ndarray:
    """A smooth field crossing 0.5 on a sphere, for the volume cases."""
    centre = (n - 1) / 2
    z, y, x = np.mgrid[:n, :n, :n]
    distance = np.sqrt((x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2)
    return np.clip((radius - distance) / 3.0 + 0.5, 0.0, 1.0).astype(np.float32)


def _volume(material, position=(0.0, 0.0, 0.0), n: int = 48) -> gfx.Volume:
    volume = gfx.Volume(
        gfx.Geometry(grid=gfx.Texture(_sphere_field(n), dim=3)), material
    )
    centre = (n - 1) / 2
    volume.local.position = tuple(p - centre for p in position)
    return volume


def _render(
    scene: gfx.Scene,
    camera: gfx.Camera,
    *,
    extra_targets: list[str] | None = None,
    configure=None,
    size: int = SIZE,
):
    """Draw *scene* offscreen with an SSAO pass; return (pass, renderer, frame)."""
    from rendercanvas.offscreen import RenderCanvas

    canvas = RenderCanvas(size=(size, size), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    if extra_targets:
        assert install_cellier_blender(renderer, extra_targets) is True

    ssao = SSAOPass(renderer, lambda: camera)
    ssao.enabled = True
    ssao.n_samples = 32
    if configure is not None:
        configure(ssao)
    renderer.effect_passes = (ssao, *renderer.effect_passes)

    errors: list[BaseException] = []

    def _draw() -> None:
        try:
            renderer.render(scene, camera)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)
            raise

    canvas.request_draw(_draw)
    image = canvas.draw()
    if errors:
        cause = errors[0]
        raise RuntimeError(
            f"offscreen draw failed -- {type(cause).__name__}: {cause}"
        ) from cause
    return ssao, renderer, np.asarray(image)


def _read_ao(ssao: SSAOPass, *, blurred: bool = True) -> np.ndarray:
    """Read the occlusion field back as floats in [0, 1]."""
    texture = ssao._ao_blur_texture if blurred else ssao._ao_texture
    width, height = ssao._current_size
    raw = get_shared().device.queue.read_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": width, "rows_per_image": height},
        (width, height, 1),
    )
    return np.frombuffer(raw, np.uint8).reshape(height, width) / 255.0


def _read_depth(renderer: gfx.WgpuRenderer, size: int = SIZE) -> np.ndarray:
    raw = get_shared().device.queue.read_texture(
        {
            "texture": renderer._blender.get_texture("depth"),
            "mip_level": 0,
            "origin": (0, 0, 0),
        },
        {"offset": 0, "bytes_per_row": 4 * size, "rows_per_image": size},
        (size, size, 1),
    )
    return np.frombuffer(raw, np.float32).reshape(size, size).copy()


def _read_normal_target(renderer: gfx.WgpuRenderer, size: int = SIZE) -> np.ndarray:
    raw = get_shared().device.queue.read_texture(
        {
            "texture": renderer._blender.get_texture(NORMAL_TARGET),
            "mip_level": 0,
            "origin": (0, 0, 0),
        },
        {"offset": 0, "bytes_per_row": 8 * size, "rows_per_image": size},
        (size, size, 1),
    )
    return np.frombuffer(raw, np.float16).reshape(size, size, 4).astype(np.float32)


# ---------------------------------------------------------------------------
# pygfx coupling canary
# ---------------------------------------------------------------------------


def test_pygfx_ssao_coupling_still_valid(offscreen_renderer):
    """Every pygfx internal this pass leans on is still the expected shape.

    Three separate couplings, all of which fail *silently* if pygfx moves:
    the depth target must still ship with ``TEXTURE_BINDING`` (otherwise
    ``flush()`` hands the pass ``None`` and it becomes a passthrough);
    ``USES_DEPTH`` must still deliver a bindable depth view; and
    ``NormalPass`` must still carry the two helper functions
    ``ssao.wgsl`` copies, with the same signatures.

    If this fails after a pygfx upgrade, check
    ``cellier.render.shaders.wgsl.ssao.wgsl`` against the new
    ``NormalPass.wgsl`` and re-run ``scripts/v2/ssao_spike.py``.
    """
    from pygfx.renderers.wgpu.engine.blender import default_targets
    from rendercanvas.offscreen import RenderCanvas

    depth_format, depth_usage = default_targets["depth"]
    assert depth_format == wgpu.TextureFormat.depth32float
    assert depth_usage & wgpu.TextureUsage.TEXTURE_BINDING

    assert SSAOPass.USES_DEPTH is True

    # The two functions ssao.wgsl copies, with the argument lists it calls.
    wgsl = NormalPass.wgsl
    assert (
        "fn to_view_pos(uv: vec2<f32>, depth: f32, "
        "projection_transform_inv: mat4x4<f32>) -> vec3<f32>" in wgsl
    )
    assert (
        "fn reconstruct_view_normal(uv: vec2<f32>, width: i32, height: i32, "
        "projection_transform_inv: mat4x4<f32>) -> vec3<f32>" in wgsl
    )

    # And the depth view really is bindable on a live renderer.
    canvas = RenderCanvas(size=(32, 32), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    scene = gfx.Scene()
    scene.add(gfx.Mesh(gfx.box_geometry(), gfx.MeshBasicMaterial()))
    camera = gfx.OrthographicCamera()
    camera.show_object(scene)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    canvas.draw()
    view = renderer._blender.get_texture_view(
        "depth", wgpu.TextureUsage.TEXTURE_BINDING, create_if_not_exist=False
    )
    assert view is not None


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


def test_kernel_is_a_hemisphere_with_an_accelerating_ramp():
    """Samples sit in z >= 0 and bunch toward the origin."""
    kernel = make_kernel(32, seed=0)
    assert kernel.shape == (MAX_KERNEL_SAMPLES, 4)
    used = kernel[:32, :3]
    assert np.all(used[:, 2] >= 0.0)
    assert np.all(np.linalg.norm(used, axis=1) <= 1.0 + 1e-6)
    # Unused entries stay zero so the loop can never read a stale sample.
    assert np.all(kernel[32:] == 0.0)
    # The ramp: the first third is closer in than the last third.
    lengths = np.linalg.norm(used, axis=1)
    assert lengths[:10].mean() < lengths[-10:].mean()


def test_kernel_ramp_follows_the_sample_count_not_the_array_size():
    """A 16-sample kernel must still reach the outside of the hemisphere.

    The uniform array is a fixed 64 entries so a sample-count change never
    alters the layout.  Spreading the ``lerp(0.1, 1.0, s * s)`` ramp over
    those 64 instead of over the samples actually used would leave a
    16-sample kernel entirely inside the innermost quarter, which reads as
    almost no occlusion at all.
    """
    small = make_kernel(16, seed=0)[:16, :3]
    full = make_kernel(64, seed=0)[:64, :3]
    assert np.linalg.norm(small, axis=1).max() > 0.5
    assert np.linalg.norm(small, axis=1).max() == pytest.approx(
        np.linalg.norm(full, axis=1).max(), abs=0.5
    )


# ---------------------------------------------------------------------------
# The occlusion operator
# ---------------------------------------------------------------------------


def test_a_flat_plane_produces_no_occlusion(offscreen_renderer):
    """The bias test.  A plane cannot occlude itself.

    This is the check that catches a bias too small for the scene's scale:
    with an absolute bias rather than one proportional to the radius, the
    same plane self-occludes by about 7 percent at a radius of 6.
    """
    scene = _lit_scene()
    scene.add(_plane())
    camera = _camera()
    ssao, renderer, _ = _render(
        scene, camera, configure=lambda p: setattr(p, "radius", 6.0)
    )

    ao = _read_ao(ssao)
    # The plane's own silhouette against the background is in frame, and a
    # 5-tap reconstruction two pixels from an edge has taps on the
    # background.  That fringe is expected; the interior is the test.
    interior = _erode(_read_depth(renderer) < 1.0, 2 + ssao.blur_radius)
    assert interior.sum() > 1000
    # Most of the plane is untouched outright.
    assert np.median(ao[interior]) == 1.0
    assert ao[interior].mean() > 0.99
    # A band near the far edge, where the plane is close to edge-on and one
    # pixel spans a large depth range, comes back a few percent low.  That
    # is the standard grazing-incidence artifact and it is documented; the
    # regression this test guards against is nothing like that small.  An
    # absolute bias rather than one proportional to the radius takes the
    # mean to 0.93 and the minimum to 0.75 on this same scene.
    assert ao[interior].min() > 0.9


def test_a_box_on_a_plane_darkens_the_contact_crease(offscreen_renderer):
    """Occlusion appears on the concave side and not on the convex one.

    The box sits on the plane, so the only concave geometry in the frame is
    the contact line where the two meet.  The convex silhouette of the box
    against the background must stay clean -- darkening there is the classic
    symptom of a missing range check.
    """
    scene = _lit_scene()
    scene.add(_plane())
    box = gfx.Mesh(gfx.box_geometry(20, 20, 20), gfx.MeshPhongMaterial(color="#888888"))
    box.local.position = (0, 10, 0)
    scene.add(box)
    camera = _camera()
    ssao, renderer, _ = _render(
        scene, camera, configure=lambda p: setattr(p, "radius", 6.0)
    )

    ao = _read_ao(ssao)
    depth = _read_depth(renderer)
    surface = depth < 1.0
    occluded = (ao < 0.95) & surface

    assert occluded.sum() > 20, "the contact crease produced no occlusion"
    # Everything occluded is near geometry, not out in the background.
    assert ao[~surface].min() > 0.9


def test_the_background_is_left_alone(offscreen_renderer):
    """An empty frame comes back with occlusion exactly 1 everywhere.

    Without the ``depth >= 1.0`` early-out every silhouette grows a dark
    halo into empty space, and an empty frame goes uniformly grey.
    """
    scene = _lit_scene()
    camera = _camera()
    ssao, _, _ = _render(scene, camera, configure=lambda p: setattr(p, "radius", 6.0))
    assert np.all(_read_ao(ssao) == 1.0)


def test_a_disabled_pass_leaves_the_frame_pixel_identical(offscreen_renderer):
    """The defaults-off guarantee.

    ``flush()`` skips disabled passes entirely, so a canvas that never
    turns occlusion on must produce exactly the frame it produced before
    the feature existed.
    """
    from rendercanvas.offscreen import RenderCanvas

    def _frame(with_pass: bool) -> np.ndarray:
        canvas = RenderCanvas(size=(SIZE, SIZE), pixel_ratio=1)
        renderer = gfx.WgpuRenderer(canvas)
        renderer.pixel_scale = 1
        renderer.ppaa = "none"
        scene = _lit_scene()
        scene.add(_plane())
        box = gfx.Mesh(
            gfx.box_geometry(20, 20, 20), gfx.MeshPhongMaterial(color="#888888")
        )
        box.local.position = (0, 10, 0)
        scene.add(box)
        camera = _camera()
        if with_pass:
            ssao = SSAOPass(renderer, lambda: camera)
            ssao.enabled = False
            renderer.effect_passes = (ssao, *renderer.effect_passes)
        canvas.request_draw(lambda: renderer.render(scene, camera))
        return np.asarray(canvas.draw())

    assert np.array_equal(_frame(False), _frame(True))


def test_the_pass_survives_a_resize(offscreen_renderer):
    """The private textures are reallocated, not reused at the wrong size."""
    scene = _lit_scene()
    scene.add(_plane())
    camera = _camera()

    from rendercanvas.offscreen import RenderCanvas

    canvas = RenderCanvas(size=(64, 64), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    ssao = SSAOPass(renderer, lambda: camera)
    ssao.enabled = True
    ssao.radius = 6.0
    renderer.effect_passes = (ssao, *renderer.effect_passes)

    canvas.request_draw(lambda: renderer.render(scene, camera))
    canvas.draw()
    assert ssao._current_size == (64, 64)

    canvas.set_logical_size(48, 80)
    canvas.draw()
    assert ssao._current_size == (48, 80)
    assert ssao._ao_blur_texture.size[:2] == (48, 80)
    assert _read_ao(ssao).shape == (80, 48)


# ---------------------------------------------------------------------------
# The radius
# ---------------------------------------------------------------------------


def test_the_auto_radius_works_across_three_orders_of_magnitude(offscreen_renderer):
    """The same scene at 0.001x, 1x and 1000x gives the same occlusion.

    This is the whole reason the radius is derived from the scene bounding
    box rather than defaulted to a number.  A fixed radius produces either
    no visible occlusion or a uniformly black frame, and there is no
    default that is right twice.
    """
    fields = []
    for scale in (0.001, 1.0, 1000.0):
        scene = _lit_scene()
        scene.add(_plane(scale))
        box = gfx.Mesh(
            gfx.box_geometry(20 * scale, 20 * scale, 20 * scale),
            gfx.MeshPhongMaterial(color="#888888"),
        )
        box.local.position = (0, 10 * scale, 0)
        scene.add(box)
        camera = _camera(scale)
        box_bounds = np.asarray(scene.get_world_bounding_box())
        diagonal = float(np.linalg.norm(box_bounds[1] - box_bounds[0]))

        ssao, _, _ = _render(
            scene, camera, configure=lambda p, d=diagonal: p.set_scene_extent(d)
        )
        assert ssao.effective_radius == pytest.approx(diagonal * 0.02)
        fields.append(_read_ao(ssao))

    for other in fields[1:]:
        assert np.allclose(fields[0], other, atol=0.02)
    assert fields[0].min() < 0.95, "no occlusion at any scale -- check the scene"


def test_an_explicit_radius_overrides_the_auto_one():
    """``radius`` set explicitly wins, and clearing it restores auto."""
    from rendercanvas.offscreen import RenderCanvas

    canvas = RenderCanvas(size=(16, 16), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    ssao = SSAOPass(renderer, lambda: gfx.PerspectiveCamera())
    ssao.set_scene_extent(100.0)
    assert ssao.effective_radius == pytest.approx(2.0)
    ssao.radius = 7.5
    assert ssao.effective_radius == pytest.approx(7.5)
    ssao.radius = None
    assert ssao.effective_radius == pytest.approx(2.0)


def test_a_degenerate_bounding_box_leaves_the_radius_alone():
    """An empty or half-built scene must not zero the radius.

    ``fit_camera`` can run before the first reslice completes, when the
    scene bounding box is degenerate.  Taking that number would leave the
    pass with a radius of zero and no occlusion at all, which looks exactly
    like the feature being broken.
    """
    from rendercanvas.offscreen import RenderCanvas

    canvas = RenderCanvas(size=(16, 16), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    ssao = SSAOPass(renderer, lambda: gfx.PerspectiveCamera())
    ssao.set_scene_extent(100.0)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        ssao.set_scene_extent(bad)
        assert ssao.effective_radius == pytest.approx(2.0)


def test_the_bias_is_a_fraction_of_the_radius():
    """Bias is dimensionless, so it survives a change of scene scale."""
    from rendercanvas.offscreen import RenderCanvas

    canvas = RenderCanvas(size=(16, 16), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    ssao = SSAOPass(renderer, lambda: gfx.PerspectiveCamera())
    ssao.bias = 0.05
    ssao.radius = 10.0
    assert float(ssao._compute_pass._uniform_data["bias"]) == pytest.approx(0.5)
    ssao.radius = 1000.0
    assert float(ssao._compute_pass._uniform_data["bias"]) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# The hybrid normal
# ---------------------------------------------------------------------------


def test_a_volume_writes_a_normal_and_a_mesh_writes_exactly_zero(offscreen_renderer):
    """The branch the hybrid turns on.

    The occlusion pass chooses per pixel between the written normal and
    depth reconstruction on ``dot(n, n) > 0.25``, which works only because
    WGSL zero-initialises the fragment output: a shader that never writes
    the field emits exactly zero.  If a mesh ever started writing
    something, volumes would keep working and meshes would silently switch
    to a normal nobody computed.
    """
    scene = _lit_scene()
    scene.add(_volume(ImageVolumeIsoMaterial(clim=(0, 1), threshold=0.5), (-30, 0, 0)))
    mesh = gfx.Mesh(
        gfx.sphere_geometry(16, 64, 32), gfx.MeshPhongMaterial(color="#cccccc")
    )
    mesh.local.position = (30, 0, 0)
    scene.add(mesh)
    camera = gfx.PerspectiveCamera(45, 1, depth_range=(1.0, 2000.0))
    camera.local.position = (0, 0, 200)
    camera.look_at((0, 0, 0))

    _, renderer, _ = _render(scene, camera, extra_targets=[NORMAL_TARGET])
    normals = _read_normal_target(renderer)
    surface = _read_depth(renderer) < 1.0
    left = np.zeros_like(surface)
    left[:, : SIZE // 2] = True
    right = ~left

    volume_px = surface & left
    mesh_px = surface & right
    assert volume_px.sum() > 50
    assert mesh_px.sum() > 50

    magnitude = np.linalg.norm(normals[..., :3], axis=-1)
    assert np.all(magnitude[volume_px] > 0.9)
    assert np.all(normals[mesh_px] == 0.0)


def test_the_written_normal_matches_the_analytic_sphere(offscreen_renderer):
    """The normal target carries a real surface normal, not a plausible one.

    Scored against the analytic normal of the sphere the volume encodes,
    using the depth buffer to know which surface point each pixel is.
    Depth reconstruction measures about 34 degrees here; this is the number
    that justifies the target existing at all.
    """
    n, radius = 64, 22.0
    scene = _lit_scene()
    volume = gfx.Volume(
        gfx.Geometry(grid=gfx.Texture(_sphere_field(n, radius), dim=3)),
        ImageVolumeIsoMaterial(clim=(0, 1), threshold=0.5),
    )
    centre = (n - 1) / 2
    volume.local.position = (-centre, -centre, -centre)
    scene.add(volume)
    camera = gfx.PerspectiveCamera(45, 1, depth_range=(1.0, 2000.0))
    camera.local.position = (0, 0, 160)
    camera.look_at((0, 0, 0))

    size = 128
    _, renderer, _ = _render(scene, camera, extra_targets=[NORMAL_TARGET], size=size)
    normals = _read_normal_target(renderer, size)
    depth = _read_depth(renderer, size)

    # CPU port of to_view_pos, for every pixel.
    proj_inv = np.asarray(camera.projection_matrix_inverse, dtype=np.float64)
    rows, cols = np.mgrid[:size, :size]
    uv_x = (cols + 0.5) / size
    uv_y = (rows + 0.5) / size
    ndc = np.stack([uv_x * 2 - 1, 1 - uv_y * 2, depth, np.ones_like(depth)], axis=-1)
    homogeneous = ndc @ proj_inv.T
    view_pos = homogeneous[..., :3] / homogeneous[..., 3:4]

    centre_view = (np.asarray(camera.world.inverse_matrix) @ np.array([0, 0, 0, 1.0]))[
        :3
    ]
    analytic = view_pos - centre_view
    analytic /= np.linalg.norm(analytic, axis=-1, keepdims=True)

    # Trim the silhouette ring, where the surface is edge-on to the ray.
    mask = (depth < 1.0) & (np.abs(analytic[..., 2]) > 0.35)
    assert mask.sum() > 500

    written = normals[..., :3][mask]
    written /= np.linalg.norm(written, axis=-1, keepdims=True)
    cosine = np.clip((written * analytic[mask]).sum(-1), -1.0, 1.0)
    median_error = float(np.median(np.degrees(np.arccos(cosine))))
    assert median_error < 5.0, f"median normal error {median_error:.2f} deg"


@pytest.mark.parametrize(
    "shader_module",
    [
        "multiscale_volume_brick",
        "label_volume",
        "label_volume_brick",
        "image_volume",
    ],
)
def test_every_volume_shader_writes_the_normal_target(shader_module):
    """Each volume shader still has its ``write_normal`` branch.

    Asserted per shader rather than once, because a shader that stops
    writing the field does not fail: it silently falls back to the
    34-degree depth reconstruction and merely looks worse, which is
    precisely the kind of regression nobody notices.
    """
    from pathlib import Path

    import cellier.render.shaders as shaders_pkg

    wgsl_dir = Path(shaders_pkg.__file__).parent / "wgsl"
    source = (wgsl_dir / f"{shader_module}.wgsl").read_text()
    assert "$$ if write_normal" in source
    assert "out.normal = pack_view_normal(" in source
    assert "cellier.view_normal.wgsl" in source


def test_the_colour_frame_is_unchanged_by_the_extra_targets(offscreen_renderer):
    """A scene renders identically with and without the extra targets.

    Stronger than checking the targets read back: it proves the generated
    ``FragmentOutput`` edits did not disturb anything the user can see.
    """
    scene = _lit_scene()
    scene.add(_volume(ImageVolumeIsoMaterial(clim=(0, 1), threshold=0.5), (0, 0, 0)))
    camera = gfx.PerspectiveCamera(45, 1, depth_range=(1.0, 2000.0))
    camera.local.position = (0, 0, 200)
    camera.look_at((0, 0, 0))

    def _frame(targets):
        from rendercanvas.offscreen import RenderCanvas

        canvas = RenderCanvas(size=(SIZE, SIZE), pixel_ratio=1)
        renderer = gfx.WgpuRenderer(canvas)
        renderer.pixel_scale = 1
        renderer.ppaa = "none"
        if targets:
            assert install_cellier_blender(renderer, targets) is True
        canvas.request_draw(lambda: renderer.render(scene, camera))
        return np.asarray(canvas.draw())

    assert np.array_equal(_frame([]), _frame([OUTLINE_ID_TARGET, NORMAL_TARGET]))


# ---------------------------------------------------------------------------
# The shared LUT and the per-visual opt-out
# ---------------------------------------------------------------------------


def test_the_ao_bit_is_independent_of_the_outline_fields():
    """Bit 7 and bits 0-6 do not interfere at any value."""
    for slot in (0, 1, 15):
        for kind in (0, 1, 2):
            outline_only = encode_entry(slot, kind, PLACEMENT_INWARD)
            with_ao = encode_entry(slot, kind, PLACEMENT_INWARD, ao_excluded=True)
            assert with_ao == outline_only | AO_EXCLUDED_BIT
            assert decode_entry(with_ao)[:3] == decode_entry(outline_only)[:3]
            assert decode_entry(with_ao)[3] is True
            assert decode_entry(outline_only)[3] is False


def test_the_ao_bit_round_trips_at_full_range_ids(offscreen_renderer):
    """Entries written for full-range ids read back from the right texel.

    Full-range draws, because pygfx allocates ``global_id`` with
    ``random.randint(1, 1_048_575)`` rather than a counter, so small
    sequential ids would pass a broken index.
    """
    lut = VisualLut()
    rng = np.random.default_rng(7)
    ids = np.unique(rng.integers(1, PICK_ID_MAX, size=64, dtype=np.int64))
    expected = {}
    for index, object_id in enumerate(ids):
        value = encode_entry(
            index % 16, KIND_WHOLE_OBJECT, PLACEMENT_INWARD, ao_excluded=True
        )
        expected[int(object_id)] = value
        lut.set_entry(int(object_id), value)
    for object_id, value in expected.items():
        assert lut.get_entry(object_id) == value
        assert decode_entry(lut.get_entry(object_id))[3] is True


def test_applying_an_outline_state_does_not_clobber_the_ao_bit():
    """The regression the merged authoritative map exists to prevent.

    ``apply`` is a whole-state sync, so an outline-only mapping would clear
    every occlusion exclusion.  The guarantee is at the map level -- one
    caller supplies the whole byte -- and this pins the behaviour that
    makes it necessary, so the two features can never drift back into
    separate maps unnoticed.
    """
    lut = VisualLut()
    outline = encode_entry(2, KIND_WHOLE_OBJECT, PLACEMENT_INWARD)

    # What a merged sync writes: both features in one byte.
    lut.apply({100: outline | AO_EXCLUDED_BIT, 200: AO_EXCLUDED_BIT})
    assert decode_entry(lut.get_entry(100)) == (2, KIND_WHOLE_OBJECT, 0, True)
    assert decode_entry(lut.get_entry(200))[3] is True

    # What a naive outline-only sync would write, and why it is wrong.
    lut.apply({100: outline})
    assert decode_entry(lut.get_entry(100))[3] is False
    assert lut.get_entry(200) == 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_the_config_round_trips():
    """``SSAOConfig`` survives ``model_dump_json`` inside the parent."""
    config = RenderManagerConfig(
        ssao=SSAOConfig(enabled=True, n_samples=24, blur_radius=1, strength=0.8)
    )
    restored = RenderManagerConfig.model_validate_json(config.model_dump_json())
    assert restored == config
    assert restored.ssao.radius is None


def test_the_config_defaults_to_off():
    """Nothing in this feature may change a scene that has not opted in."""
    assert RenderManagerConfig().ssao.enabled is False


def test_apply_config_pushes_every_field(offscreen_renderer):
    """``apply_config`` is the single seam; nothing is left behind."""
    from rendercanvas.offscreen import RenderCanvas

    canvas = RenderCanvas(size=(16, 16), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    ssao = SSAOPass(renderer, lambda: gfx.PerspectiveCamera())
    ssao.apply_config(
        SSAOConfig(
            enabled=True,
            n_samples=24,
            blur_radius=1,
            radius=3.0,
            bias=0.1,
            strength=0.4,
            power=2.0,
        )
    )
    assert ssao.enabled is True
    assert ssao.n_samples == 24
    assert ssao.blur_radius == 1
    assert ssao.effective_radius == pytest.approx(3.0)
    assert ssao.bias == pytest.approx(0.1)
    assert ssao.strength == pytest.approx(0.4)
    assert ssao.power == pytest.approx(2.0)
    assert ssao._compute_pass._template_vars["n_samples"] == 24
    assert ssao._blur_pass._template_vars["blur_radius"] == 1


def test_the_mip_family_is_what_gets_excluded():
    """The default exclusion is keyed on the material's render mode.

    Keyed on ``render_mode`` rather than on the visual class, so the same
    rule covers pygfx materials, cellier's own, and anything added later
    that carries a projection mode.
    """
    from cellier.render.render_manager import MIP_RENDER_MODES

    assert ImageVolumeMipMaterial.render_mode in MIP_RENDER_MODES
    assert ImageVolumeIsoMaterial.render_mode not in MIP_RENDER_MODES
    assert "attenuated_mip" in MIP_RENDER_MODES
    # Label materials carry a categorical mode, which is a real surface.
    assert "iso_categorical" not in MIP_RENDER_MODES
    # Mesh materials have no render mode at all.
    assert getattr(gfx.MeshPhongMaterial(), "render_mode", None) is None


# ---------------------------------------------------------------------------
# Integration: the chain, 2D, and the per-visual opt-out
# ---------------------------------------------------------------------------


@pytest.fixture
def ssao_controller(qtbot, offscreen_renderer):
    """A 3D controller with occlusion enabled and one in-memory volume."""
    from cellier.controller import CellierController
    from cellier.data import ImageMemoryStore
    from cellier.visuals import InMemoryImageAppearance

    data = np.zeros((16, 24, 32), dtype=np.float32)
    data[4:12, 6:18, 8:24] = 1.0

    controller = CellierController(
        render_config=RenderManagerConfig(ssao=SSAOConfig(enabled=True))
    )
    controller.camera_reslice_enabled = False
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_image(
        data=ImageMemoryStore(data=data, name="volume"),
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="mip"
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    return controller, scene, visual


def test_the_pass_sits_ahead_of_outline_and_accumulation(ssao_controller):
    """Chain order is load-bearing in both directions.

    Ahead of the outline because an outline is a UI annotation, not a lit
    surface, and several outline tests assert palette colours by exact
    match.  Ahead of accumulation because the EMA there denoises the
    per-frame kernel rotation for free, which is what lets the sample count
    sit at 16 rather than 64.
    """
    from cellier.render._outline import OutlinePass
    from cellier.render._temporal_accumulation import TemporalAccumulationPass

    controller, _scene, _visual = ssao_controller
    canvas = next(iter(controller._render_manager._canvases.values()))
    passes = list(canvas._renderer.effect_passes)

    index_of = {type(p): i for i, p in enumerate(passes)}
    assert index_of[SSAOPass] < index_of[OutlinePass]
    assert index_of[OutlinePass] < index_of[TemporalAccumulationPass]
    # And DDAA stays last, so it antialiases everything above.
    assert index_of[TemporalAccumulationPass] < len(passes) - 1


def test_a_2d_canvas_never_runs_the_pass(qtbot, offscreen_renderer):
    """2D is a plane at near-constant depth: the occlusion is a constant.

    The request is still remembered, so a 2D -> 3D toggle turns it on
    rather than leaving it off until something touches the config again.
    """
    from cellier.controller import CellierController

    controller = CellierController(
        render_config=RenderManagerConfig(ssao=SSAOConfig(enabled=True))
    )
    controller.camera_reslice_enabled = False
    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_canvas(scene_id=scene.id)
    canvas = next(iter(controller._render_manager._canvases.values()))

    assert canvas._ssao_requested is True
    assert canvas._ssao_pass.enabled is False

    canvas.switch_dim("3d")
    assert canvas._ssao_pass.enabled is True

    canvas.switch_dim("2d")
    assert canvas._ssao_pass.enabled is False


def test_live_setters_reach_every_canvas(ssao_controller):
    """The setters mutate the config and push it, in one step."""
    controller, _scene, _visual = ssao_controller
    canvas = next(iter(controller._render_manager._canvases.values()))

    controller.ssao_strength = 0.4
    assert controller.render_config.ssao.strength == pytest.approx(0.4)
    assert canvas._ssao_pass.strength == pytest.approx(0.4)

    controller.ssao_power = 2.0
    assert canvas._ssao_pass.power == pytest.approx(2.0)

    controller.ssao_radius = 4.0
    assert canvas._ssao_pass.effective_radius == pytest.approx(4.0)

    controller.ssao_enabled = False
    assert canvas._ssao_pass.enabled is False
    controller.ssao_enabled = True
    assert canvas._ssao_pass.enabled is True


def test_a_mip_volume_is_excluded_by_default_and_iso_is_not(ssao_controller):
    """The automatic rule, driven entirely by the visual's render mode.

    A MIP-family mode writes the depth of the brightest sample along the
    ray rather than of a surface, so occlusion derived from it shimmers.
    Switching the same visual to ``iso`` must give it occlusion with no
    explicit call at all.
    """
    from cellier.render._visual_lut import get_shared_visual_lut

    controller, _scene, visual = ssao_controller
    lut = get_shared_visual_lut()
    lut.clear()
    manager = controller._render_manager
    canvas = next(iter(manager._canvases.values()))

    manager._sync_visual_lut()
    entries = lut.entries
    assert entries, "a mip volume should have been excluded"
    assert all(decode_entry(v)[3] for v in entries.values())
    assert canvas._ssao_pass._has_exclusions is True

    visual.appearance.render_mode = "iso"
    manager._sync_visual_lut()
    assert lut.entries == {}
    assert canvas._ssao_pass._has_exclusions is False


def test_an_explicit_choice_survives_a_render_mode_change(ssao_controller):
    """``True`` and ``False`` are decisions, not defaults."""
    from cellier.render._visual_lut import get_shared_visual_lut

    controller, _scene, visual = ssao_controller
    lut = get_shared_visual_lut()
    lut.clear()
    manager = controller._render_manager

    controller.set_visual_ambient_occlusion(visual.id, enabled=True)
    manager._sync_visual_lut()
    assert lut.entries == {}, "an explicit opt-in must beat the mip default"
    assert controller.get_visual_ambient_occlusion(visual.id) is True

    # Still opted in after the render mode changes.
    visual.appearance.render_mode = "iso"
    manager._sync_visual_lut()
    assert lut.entries == {}
    assert controller.get_visual_ambient_occlusion(visual.id) is True

    # An explicit exclusion sticks on a mode that would otherwise receive.
    controller.set_visual_ambient_occlusion(visual.id, enabled=False)
    manager._sync_visual_lut()
    assert lut.entries
    assert all(decode_entry(v)[3] for v in lut.entries.values())

    # And None restores the automatic rule, which now says "include".
    controller.set_visual_ambient_occlusion(visual.id, enabled=None)
    manager._sync_visual_lut()
    assert lut.entries == {}


def test_outlines_and_exclusions_do_not_clobber_each_other(ssao_controller):
    """The regression the merged authoritative map exists to prevent.

    Both features sync the same table with a whole-state ``apply``.  With
    separate maps, whichever synced last would wipe the other's entries --
    and the visible symptom would be "turning outlines on made my MIP
    volume go dark", which no one would connect to the LUT.
    """
    from cellier.render._visual_lut import get_shared_visual_lut

    controller, _scene, visual = ssao_controller
    lut = get_shared_visual_lut()
    lut.clear()
    manager = controller._render_manager

    manager._sync_visual_lut()
    excluded_ids = set(lut.entries)
    assert excluded_ids

    controller.outline_enabled = True
    controller.set_visual_outline(visual.id, slot=3, placement="inward")
    manager._sync_visual_lut()

    entries = lut.entries
    assert set(entries) >= excluded_ids
    for object_id in excluded_ids:
        slot, kind, _placement, ao_excluded = decode_entry(entries[object_id])
        assert ao_excluded is True, "enabling outlines cleared the exclusion"
        assert slot == 3
        assert kind == KIND_WHOLE_OBJECT

    # And removing the outline leaves the exclusion behind.
    controller.set_visual_outline(visual.id, slot=0)
    manager._sync_visual_lut()
    for object_id in excluded_ids:
        assert decode_entry(lut.entries[object_id])[3] is True


def test_nothing_excluded_compiles_the_lookup_away(ssao_controller):
    """A scene with no exclusions binds neither the pick buffer nor the LUT."""
    controller, _scene, visual = ssao_controller
    manager = controller._render_manager
    canvas = next(iter(manager._canvases.values()))

    visual.appearance.render_mode = "iso"
    manager._sync_visual_lut()
    assert canvas._ssao_pass._has_exclusions is False
    assert canvas._ssao_pass._compute_pass._template_vars["has_exclusions"] is False


def test_the_canvas_carries_the_normal_target_when_enabled(ssao_controller):
    """Occlusion on implies the ``normal`` target; off implies neither."""
    from cellier.controller import CellierController
    from cellier.render._cellier_blender import CellierBlender

    controller, _scene, _visual = ssao_controller
    canvas = next(iter(controller._render_manager._canvases.values()))
    blender = canvas._renderer._blender
    assert isinstance(blender, CellierBlender)
    assert NORMAL_TARGET in blender.texture_info
    assert OUTLINE_ID_TARGET not in blender.texture_info
    assert canvas._normal_target_available is True

    plain = CellierController()
    plain.camera_reslice_enabled = False
    scene = plain.add_scene(dim="3d", name="scene")
    plain.add_canvas(scene_id=scene.id)
    plain_canvas = next(iter(plain._render_manager._canvases.values()))
    assert not isinstance(plain_canvas._renderer._blender, CellierBlender)
    assert plain_canvas._normal_target_available is False


def test_the_auto_radius_follows_a_camera_fit(ssao_controller):
    """``fit_camera`` already walks the scene, so it re-derives the radius."""
    controller, scene, _visual = ssao_controller
    canvas = next(iter(controller._render_manager._canvases.values()))

    canvas._ssao_pass.set_scene_extent(1.0)
    assert canvas._ssao_pass.effective_radius == pytest.approx(0.02)

    controller.fit_camera(scene.id)
    box = np.asarray(
        controller._render_manager.get_scene(scene.id).get_world_bounding_box()
    )
    diagonal = float(np.linalg.norm(box[1] - box[0]))
    assert canvas._ssao_pass.effective_radius == pytest.approx(diagonal * 0.02)


# ---------------------------------------------------------------------------
# Every volume shader, rendered for real
#
# The source check above catches a deleted branch; these catch a branch that
# is present and does not compile, or compiles and writes nothing.  Both are
# silent failures: the frame still renders and the occlusion merely falls
# back to the 34-degree depth reconstruction.
# ---------------------------------------------------------------------------


def _render_with_normal_target(controller, scene_id, size=(96, 96)):
    """Draw the controller's live scene through a blender carrying ``normal``.

    The ``render_scene`` fixture builds a stock renderer, so this is its
    counterpart for the cases that need the extra target.  The camera is
    the one the controller's own canvas uses, so the framing matches.
    """
    from rendercanvas.offscreen import RenderCanvas

    controller.fit_camera(scene_id)
    canvas_id = controller.get_canvas_ids(scene_id)[0]
    canvas_view = controller._render_manager._canvases[canvas_id]

    canvas = RenderCanvas(size=size, pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    assert install_cellier_blender(renderer, [NORMAL_TARGET]) is True

    gfx_scene = canvas_view._get_scene_fn(scene_id)
    camera = canvas_view.camera
    errors: list[BaseException] = []

    def _draw() -> None:
        try:
            renderer.render(gfx_scene, camera)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)
            raise

    canvas.request_draw(_draw)
    canvas.draw()
    if errors:
        cause = errors[0]
        raise RuntimeError(
            f"offscreen draw failed -- {type(cause).__name__}: {cause}"
        ) from cause

    normals = _read_normal_target(renderer, size[0])
    depth = _read_depth(renderer, size[0])
    return normals, depth


def _assert_surface_normals_are_written(normals, depth, what, min_coverage=0.9):
    """Assert *what* wrote a usable normal on (nearly) all its surface.

    ``min_coverage`` is below 1 on purpose.  Where the density gradient is
    degenerate -- along a brick edge whose probes read symmetric ghost
    data, for instance -- the shader writes exactly zero rather than an
    invented direction, and the occlusion pass falls back to depth
    reconstruction on that thin line.  That is the right behaviour; the
    regression this guards against takes the coverage to zero, not to 80
    percent.
    """
    surface = depth < 1.0
    assert surface.sum() > 20, f"{what} rendered no surface to check"
    magnitude = np.linalg.norm(normals[..., :3], axis=-1)
    written = magnitude > 0.5
    covered = (written & surface).sum() / surface.sum()
    assert covered > min_coverage, (
        f"{what} wrote a normal on only {covered:.0%} of its surface pixels; "
        "the occlusion pass silently falls back to depth reconstruction there"
    )
    # Anything written must be a real direction: a half-written or
    # unnormalised vector would sit on the wrong side of the pass's
    # dot(n, n) > 0.25 test in a way that depends on the pixel.
    assert np.all(np.abs(magnitude[written] - 1.0) < 0.05), (
        f"{what} wrote normals that are not unit length"
    )


async def test_in_memory_volume_writes_the_normal_target(
    controller, reslice, image_volume
):
    """``image_volume.wgsl`` -- the in-memory image volume, iso mode."""
    from cellier.visuals import InMemoryImageAppearance

    scene = controller.add_scene(dim="3d", name="scene")
    controller.add_image(
        data=image_volume,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(
            color_map="viridis",
            clim=(0.0, 1.0),
            render_mode="iso",
            iso_threshold=0.5,
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    normals, depth = _render_with_normal_target(controller, scene.id)
    _assert_surface_normals_are_written(normals, depth, "the in-memory image volume")


async def test_in_memory_labels_volume_writes_the_normal_target(
    controller, reslice, labels_volume
):
    """``label_volume.wgsl`` -- the in-memory labels volume."""
    from cellier.visuals import InMemoryLabelsAppearance

    scene = controller.add_scene(dim="3d", name="scene")
    controller.add_labels(
        data=labels_volume,
        scene_id=scene.id,
        appearance=InMemoryLabelsAppearance(
            colormap_mode="random", render_mode="iso_categorical"
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    normals, depth = _render_with_normal_target(controller, scene.id)
    _assert_surface_normals_are_written(normals, depth, "the in-memory labels volume")


async def test_multiscale_volume_writes_the_normal_target(
    controller, reslice, multiscale_image_store
):
    """``multiscale_volume_brick.wgsl`` -- the multiscale image volume, iso.

    Also the first test in the suite that renders a multiscale volume in
    ``iso`` mode at all.  Every earlier one uses ``mip``, which is how a
    shader compile error in the iso branch survived unnoticed.
    """
    from cellier.visuals._image import (
        MultiscaleImageAppearance,
        MultiscaleImageRenderConfig,
    )

    scene = controller.add_scene(dim="3d", name="scene")
    controller.add_image_multiscale(
        data=multiscale_image_store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(
            color_map="viridis",
            clim=(0.0, 1.0),
            render_mode="iso",
            iso_threshold=0.5,
            force_level=1,
        ),
        render_config=MultiscaleImageRenderConfig(block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    normals, depth = _render_with_normal_target(controller, scene.id)
    _assert_surface_normals_are_written(
        normals,
        depth,
        "the multiscale image volume",
        # The brick shader's gradient degenerates on the box-edge lines
        # where its probes read symmetric ghost-border data, and it writes
        # zero there rather than the (0, 1, 0) fallback its own shading
        # uses.  About one pixel in six on this small 8-voxel-brick
        # pyramid; far fewer on real data.
        min_coverage=0.75,
    )


async def test_multiscale_labels_volume_writes_the_normal_target(
    controller, reslice, multiscale_labels_store
):
    """``label_volume_brick.wgsl`` -- the multiscale labels volume."""
    from cellier.visuals._labels import (
        MultiscaleLabelRenderConfig,
        MultiscaleLabelsAppearance,
    )

    scene = controller.add_scene(dim="3d", name="scene")
    controller.add_labels_multiscale(
        data=multiscale_labels_store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(
            colormap_mode="random", render_mode="iso_categorical"
        ),
        render_config=MultiscaleLabelRenderConfig(block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    normals, depth = _render_with_normal_target(controller, scene.id)
    _assert_surface_normals_are_written(normals, depth, "the multiscale labels volume")
