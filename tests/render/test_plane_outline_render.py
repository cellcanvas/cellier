"""Plane outlines on screen (plane outline design, sections 5.3, 6 and 10).

Two things are checked against pixels.  That the polygon an outline
follows is where each of the four volume shaders stops drawing the plane:
the outline is computed on the CPU and the plane is drawn by a shader, and
they must agree.  And that the line is drawn on that edge: whole on the
plane it lies in, hidden by what is in front of it.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
import pylinalg as la
import pytest

from cellier.data import ImageMemoryStore, LabelMemoryStore
from cellier.render.shaders._plane_outline import (
    OUTLINE_DEPTH_OFFSET_PX,
    OUTLINE_RENDER_QUEUE,
    PlaneOutlineMaterial,
    PlaneOutlineShader,
)
from cellier.visuals import (
    ClippingPlane,
    InMemoryImageSingleAppearance,
    InMemoryLabelsAppearance,
    PlaneOutline,
)
from tests.render import _plane_outline_pixels as px
from tests.render import _plane_rig as rig_module
from tests.render import _planes as h
from tests.render.conftest import drain_loading

SIZE = (320, 320)
SHAPE = (20, 28, 36)
OBLIQUE = (0.6, -0.5, 1.0)
#: The oblique plane's unit normal in rendered ``(x, y, z)``.
NORMAL = np.array(OBLIQUE[::-1]) / np.linalg.norm(OBLIQUE)
WIDTH = 4.0
RED = PlaneOutline(enabled=True, color=(1.0, 0.0, 0.0, 1.0), width=WIDTH)
TRANSFORMS = {
    "identity": ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
    "anisotropic": ((2.5, 1.0, 1.0), (0.0, 4.0, -7.0)),
}
KINDS = ("memory_image", "memory_labels", "multiscale_image", "multiscale_labels")


def _scene_manager(controller, scene):
    return controller._render_manager._scenes[scene.id]


def _polygon(controller, scene, plane) -> np.ndarray:
    """The polygon the scene draws the outline of *plane* on."""
    drawn = _scene_manager(controller, scene).plane_outlines.drawn
    return drawn[("render", plane.id, 0)].vertices


async def _memory(controller, kind, make_planes, *, transform="identity", clipping=()):
    """An in-memory image or labels visual in plane mode, loaded.

    The labels have no background, so a plane draws on every pixel of it.
    """
    scale, shift = TRANSFORMS[transform]
    scene = controller.add_scene(dim="3d", name="scene")
    world = h.world_of(controller, scene)
    if kind == "memory_image":
        store = ImageMemoryStore(data=h.image_data(SHAPE), name="image")
    else:
        store = LabelMemoryStore(data=np.full(SHAPE, 3, dtype=np.int32), name="labels")
    centre = ((np.array(SHAPE) - 1) / 2 + 0.3) * np.array(scale) + np.array(shift)
    planes = tuple(make_planes(world, centre))
    transform_model = h.scale_translation_transform(
        controller, scene.id, store, scale, shift
    )
    system = store.data_coordinate_systems[0]
    clip = tuple(
        ClippingPlane.from_point_normal(system, p, n, axes=("z", "y", "x"))
        for p, n in clipping
    )
    if kind == "memory_image":
        visual = controller.add_image(
            data=store,
            scene_id=scene.id,
            single=InMemoryImageSingleAppearance(
                render_mode="plane", color_map="gray", clim=(0.0, 1.0)
            ),
            transform=transform_model,
            render_planes=planes,
            clipping_planes=clip,
        )
    else:
        visual = controller.add_labels(
            data=store,
            scene_id=scene.id,
            appearance=InMemoryLabelsAppearance(
                render_mode="plane",
                colormap_mode="direct",
                color_dict=h.label_colours(),
            ),
            transform=transform_model,
            render_planes=planes,
            clipping_planes=clip,
        )
    controller.add_canvas(scene_id=scene.id, canvas_size=SIZE)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    return scene, visual


def _oblique(world, centre):
    return [h.plane_zyx(world, centre, OBLIQUE)]


def _cut(wide: float):
    """An oblique plane with three bounded sides, *wide* times a base size."""

    def make(world, centre):
        return [
            h.plane_zyx(
                world,
                centre,
                OBLIQUE,
                extent_0=(-8.0 * wide, 12.0 * wide),
                extent_1=(None, 10.0 * wide),
            )
        ]

    return make


#: ``case -> (planes, clipping, options)``, per family of visual.
MEMORY_CASES = {
    "oblique": (_oblique, (), {}),
    "extents_and_clipping": (_cut(1.0), [((6.0, 0.0, 20.0), (0.0, 0.2, -1.0))], {}),
    "anisotropic": (
        _cut(1.0),
        [((6.0, 0.0, 20.0), (0.0, 0.2, -1.0))],
        {"transform": "anisotropic"},
    ),
}
MULTISCALE_CASES = {
    "oblique": (_oblique, (), {}),
    "extents_and_clipping": (_cut(2.5), [((0.0, 0.0, 60.0), (0.0, 0.2, -1.0))], {}),
    "coarse_level": (_oblique, [((0.0, 0.0, 60.0), (0.0, 0.2, -1.0))], {"level": 3}),
}


async def _shot_without_outline(controller, tmp_path, kind, case):
    """Draw the plane with no outline; returns what is needed to add one."""
    if kind.startswith("memory"):
        make_planes, clipping, options = MEMORY_CASES[case]
        scene, visual = await _memory(
            controller, kind, make_planes, clipping=clipping, **options
        )
        shot = h.shoot(controller, scene, size=SIZE)
        return scene, visual, shot
    make_planes, clipping, options = MULTISCALE_CASES[case]
    labels = kind == "multiscale_labels"
    spec = rig_module.ANISO
    rig = await rig_module.make_rig(
        controller,
        tmp_path,
        spec,
        make_planes,
        labels=labels,
        fill=(lambda level: np.full(spec.level_shape(level), 3)) if labels else None,
        clipping=clipping,
        appearance={"force_level": options.get("level", 1)},
    )
    shot = await rig.settle()
    return rig.scene, rig.visual, shot


# -- the polygon is where the shader stops ---------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case", ["oblique", "extents_and_clipping", "third"])
async def test_the_outline_s_polygon_is_the_edge_of_what_the_shader_draws(
    controller, tmp_path, kind, case
):
    """Agreement with the shader (5.3): the filled polygon and the pixels a
    plane is drawn on differ only within half a pixel of the edge.

    On voxel edges and not centres, on anisotropic voxels, with extents and
    a clipping plane, and at a coarse level of a pyramid that never
    downsamples z.
    """
    if case == "third":
        case = "anisotropic" if kind.startswith("memory") else "coarse_level"
    scene, visual, shot = await _shot_without_outline(controller, tmp_path, kind, case)
    plane = visual.render_planes[0]
    assert _scene_manager(controller, scene).plane_outlines.drawn == {}

    controller.set_render_planes(
        visual.id, (plane.model_copy(update={"outline": RED}),)
    )
    polygon = _polygon(controller, scene, plane)
    distance = px.edge_distance(polygon, shot.camera, SIZE)
    inside = distance > 0.0
    differ = inside != shot.drawn

    assert inside.sum() > 5000
    assert differ.sum() <= 2, int(differ.sum())
    assert not (np.abs(distance[differ]) > 0.5).any()


@pytest.mark.parametrize("kind", ["memory_image", "memory_labels"])
async def test_the_box_is_the_voxels_edges(controller, kind):
    scene, visual = await _memory(controller, kind, _oblique)
    gfx_visual = _scene_manager(controller, scene).get_visual(visual.id)
    low, high = gfx_visual.outline_box()
    np.testing.assert_array_equal(low, (-0.5, -0.5, -0.5))
    np.testing.assert_array_equal(high, np.array(SHAPE) - 0.5)


@pytest.mark.parametrize("labels", [False, True])
async def test_the_box_of_a_pyramid_is_its_level_0_voxels_edges(
    controller, tmp_path, labels
):
    spec = rig_module.ANISO
    rig = await rig_module.make_rig(controller, tmp_path, spec, _oblique, labels=labels)
    low, high = rig.gfx.outline_box()
    np.testing.assert_array_equal(low, (-0.5, -0.5, -0.5))
    np.testing.assert_array_equal(high, np.array(spec.shape0) - 0.5)


# -- the line ---------------------------------------------------------------------


def _views() -> dict:
    side = np.cross(NORMAL, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)
    return {
        "usual": (-0.5, -0.4, -1.0),
        "facing": tuple(-NORMAL),
        "from_behind": tuple(NORMAL + 0.2 * side),
        "tilted_60": tuple(-0.5 * NORMAL + 0.866 * side),
    }


async def _outlined_image(controller, **kwargs):
    def make(world, centre):
        return [h.plane_zyx(world, centre, OBLIQUE, outline=RED)]

    scene, visual = await _memory(controller, "memory_image", make, **kwargs)
    return scene, visual, _polygon(controller, scene, visual.render_planes[0])


@pytest.mark.parametrize("view", list(_views()))
async def test_the_line_is_whole_on_the_plane_it_lies_in(controller, view):
    """Depth (6.2): an opaque plane writes depth where the line is.  Both
    halves of the line are drawn, from in front, from behind and at a tilt,
    and nothing red is drawn off the edge."""
    scene, _visual, polygon = await _outlined_image(controller)
    size = (480, 480)
    shot = h.shoot(controller, scene, size=size, view_dir=_views()[view])
    shares = px.line_shares(
        shot.frame, px.edge_distance(polygon, shot.camera, size), WIDTH
    )
    assert shares["pixels"] > 800, shares
    assert shares["inner"] > 0.97, shares
    assert shares["outer"] > 0.97, shares
    assert shares["stray"] == 0, shares


async def test_the_line_has_its_width_in_pixels_at_any_distance(controller):
    scene, _visual, polygon = await _outlined_image(controller)
    size = (480, 480)
    for factor in (1.0, 2.5):
        shot = h.shoot(controller, scene, size=size, distance_factor=factor)
        pixels = px.to_pixels(polygon, shot.camera, size)
        perimeter = sum(
            np.linalg.norm(pixels[i] - pixels[(i + 1) % len(pixels)])
            for i in range(len(pixels))
        )
        red = int(px.is_red(shot.frame).sum())
        # The corners add a little to a perimeter times a width.
        assert 0.9 < red / (perimeter * WIDTH) < 1.15, (factor, red, perimeter)


async def test_a_mesh_in_front_hides_the_line(controller):
    """The line is depth tested (B-D7): a quad three units toward the
    camera, over one corner of the polygon, hides the line behind it."""
    scene, _visual, polygon = await _outlined_image(controller)
    size = (480, 480)
    gfx_scene = _scene_manager(controller, scene).scene
    for view_dir in (_views()["usual"], _views()["from_behind"]):
        toward = -np.sign(np.asarray(view_dir) @ NORMAL) * NORMAL
        quad = gfx.Mesh(
            gfx.plane_geometry(10.0, 10.0),
            gfx.MeshBasicMaterial(color=(0.0, 1.0, 0.0, 1.0), side="both"),
        )
        quad.local.position = tuple(polygon[0] + 3.0 * toward)
        quad.local.rotation = la.quat_from_vecs((0.0, 0.0, 1.0), tuple(NORMAL))
        gfx_scene.add(quad)
        try:
            shot = h.shoot(controller, scene, size=size, view_dir=view_dir)
        finally:
            gfx_scene.remove(quad)
        frame = shot.frame
        green = (frame[..., 1] > 200) & (frame[..., 0] < 70) & (frame[..., 2] < 70)
        distance = px.edge_distance(polygon, shot.camera, size)
        on_line = np.abs(distance) < WIDTH / 2.0 - 0.75
        red = px.is_red(frame)
        # The quad covers part of the line: there the quad is seen, not
        # the line.  Everywhere else the line is drawn.
        assert (on_line & green).sum() > 30
        assert (on_line & red).sum() > 300
        assert (on_line & ~red & ~green).sum() == 0


async def test_the_line_is_whole_through_the_canvas_s_own_passes(controller):
    """Through the real pipeline: a perspective camera, an orthographic one
    (a field of view of 0), and ambient occlusion with accumulated frames."""
    scene, _visual, polygon = await _outlined_image(controller)
    size = (480, 480)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    view = controller._render_manager._canvases[canvas_id]

    def shares(frames: int) -> dict:
        frame = np.asarray(controller.screenshot(canvas_id, size=size, frames=frames))
        return px.line_shares(
            frame, px.edge_distance(polygon, view.camera, size), WIDTH
        )

    for step in ("perspective", "orthographic", "ambient occlusion"):
        frames = 1
        if step == "orthographic":
            view.camera.fov = 0.0
            controller.fit_camera(scene.id)
        elif step == "ambient occlusion":
            controller._render_manager.ambient_occlusion_enabled = True
            frames = 8
        found = shares(frames)
        assert found["pixels"] > 800, (step, found)
        assert found["inner"] > 0.97, (step, found)
        assert found["outer"] > 0.97, (step, found)
        assert found["stray"] == 0, (step, found)


async def test_the_line_is_not_picked_and_writes_no_depth(controller):
    scene, _visual, _polygon = await _outlined_image(controller)
    outlines = _scene_manager(controller, scene).plane_outlines
    (line,) = outlines.node.children
    material = line.material
    assert isinstance(material, PlaneOutlineMaterial)
    assert material.pick_write is False
    assert material.depth_test is True
    assert material.depth_write is False
    assert material.render_queue == OUTLINE_RENDER_QUEUE
    assert material.thickness == WIDTH
    assert material.thickness_space == "screen"
    assert material.loop is True
    # After the volume materials, whatever their queue.
    from cellier.render._clipping import drawn_materials

    assert all(
        m.render_queue < OUTLINE_RENDER_QUEUE
        for m in drawn_materials(_scene_manager(controller, scene).scene)
    )


def test_the_shader_patch_names_its_anchor_when_pygfx_moves_it(monkeypatch):
    """The pygfx-bump canary: the depth offset replaces one line of pygfx's
    line shader, and says so if that line is gone."""
    from pygfx.renderers.wgpu.shaders.lineshader import LineShader

    from cellier.render.shaders._alpha_modulated import ShaderAnchorError

    line = gfx.Line(
        gfx.Geometry(positions=np.zeros((3, 3), dtype=np.float32)),
        PlaneOutlineMaterial(color=(1.0, 0.0, 0.0, 1.0), thickness=2.0),
    )
    shader = PlaneOutlineShader(line)
    code = shader.get_code()
    assert repr(OUTLINE_DEPTH_OFFSET_PX) in code
    assert "outline_z / outline_w" in code

    monkeypatch.setattr(LineShader, "get_code", lambda self: "// nothing here")
    with pytest.raises(ShaderAnchorError, match=r"varyings\.position"):
        shader.get_code()
