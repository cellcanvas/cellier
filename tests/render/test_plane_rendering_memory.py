"""In-memory plane rendering: the right voxel at the right pixel.

Plane rendering design v3, section 5 and 14.7 (Phase 6).  Each test draws an
in-memory image or labels visual in ``"plane"`` render mode and compares the
frame with the CPU reference of ``tests/render/_planes.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data import ImageMemoryStore, LabelMemoryStore
from cellier.visuals import (
    ClippingPlane,
    InMemoryImageSingleAppearance,
    InMemoryLabelsAppearance,
)
from tests.render import _planes as h

SHAPE = (20, 28, 36)
#: ``(scale, shift)`` of ``data -> world`` over ``(z, y, x)``.
TRANSFORMS = {
    "identity": ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
    "anisotropic": ((2.5, 1.0, 1.0), (0.0, 4.0, -7.0)),
    "squashed": ((0.5, 1.5, 1.25), (30.0, 0.0, 12.0)),
}


def _centre(scale, shift) -> np.ndarray:
    """A point near the data's centre, in world ``(z, y, x)``.

    A third of a voxel off the centre on every axis: the centre of an
    even-sized axis is a voxel face, where an axis-aligned plane would show
    one voxel or its neighbour by rounding alone.
    """
    return ((np.array(SHAPE) - 1) / 2 + 0.3) * np.array(scale) + np.array(shift)


#: ``name -> normal`` over world ``(z, y, x)``.
POSES = {
    "xy": (1.0, 0.0, 0.0),
    "xz": (0.0, 1.0, 0.0),
    "oblique": (0.6, -0.5, 1.0),
}


async def _image(
    controller, reslice, make_planes, *, transform="identity", clipping=(), **single
):
    scale, shift = TRANSFORMS[transform]
    scene = controller.add_scene(dim="3d", name="scene")
    world = h.world_of(controller, scene)
    store = ImageMemoryStore(data=h.image_data(SHAPE), name="image")
    planes = tuple(make_planes(world, _centre(scale, shift)))
    system_planes = ()
    transform_model = h.scale_translation_transform(
        controller, scene.id, store, scale, shift
    )
    if clipping:
        system = store.data_coordinate_systems[0]
        system_planes = tuple(
            ClippingPlane.from_point_normal(system, p, n, axes=("z", "y", "x"))
            for p, n in clipping
        )
    visual = controller.add_image(
        data=store,
        scene_id=scene.id,
        single=InMemoryImageSingleAppearance(
            **{"render_mode": "plane", "color_map": "gray", "clim": (0.0, 1.0)} | single
        ),
        transform=transform_model,
        render_planes=planes,
        clipping_planes=system_planes,
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    return scene, visual, gfx_visual, planes


@pytest.mark.parametrize("transform", list(TRANSFORMS))
@pytest.mark.parametrize("pose", list(POSES))
async def test_an_image_plane_shows_the_voxel_the_ray_meets(
    controller, reslice, transform, pose
):
    """Right voxel (14.7): every sure pixel shows the voxel under it."""
    scene, _visual, _gfx, planes = await _image(
        controller,
        reslice,
        lambda world, centre: [h.plane_zyx(world, centre, POSES[pose])],
        transform=transform,
    )
    shot = h.shoot(controller, scene)
    scale, shift = TRANSFORMS[transform]
    reference = h.trace(shot, planes, SHAPE, scale_zyx=scale, shift_zyx=shift)
    counts = h.compare_greys(shot, reference, h.voxel_values(SHAPE))
    assert counts["sure_hit"] > 2000, counts
    assert counts["missing"] == 0, counts
    assert counts["extra"] == 0, counts
    assert counts["wrong"] == 0, counts


async def _labels(controller, reslice, make_planes, *, transform="identity", **extra):
    scale, shift = TRANSFORMS[transform]
    scene = controller.add_scene(dim="3d", name="scene")
    world = h.world_of(controller, scene)
    store = LabelMemoryStore(data=h.label_data(SHAPE), name="labels")
    planes = tuple(make_planes(world, _centre(scale, shift)))
    visual = controller.add_labels(
        data=store,
        scene_id=scene.id,
        appearance=InMemoryLabelsAppearance(
            **{
                "render_mode": "plane",
                "colormap_mode": "direct",
                "color_dict": h.label_colours(),
            }
            | extra
        ),
        transform=h.scale_translation_transform(
            controller, scene.id, store, scale, shift
        ),
        render_planes=planes,
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    return scene, visual, gfx_visual, planes


def _assert_matches(counts: dict, least: int = 2000) -> None:
    assert counts["sure_hit"] > least, counts
    assert counts["missing"] == 0, counts
    assert counts["extra"] == 0, counts
    assert counts["wrong"] == 0, counts


def _spun(plane, degrees: float):
    """*plane* with its frame turned about its normal."""
    a0, a1 = np.array(plane.in_plane_axis_0), np.array(plane.in_plane_axis_1)
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return plane.model_copy(
        update={
            "in_plane_axis_0": tuple(c * a0 + s * a1),
            "in_plane_axis_1": tuple(-s * a0 + c * a1),
        }
    )


def _record_plans(monkeypatch) -> list:
    from cellier.render.scene_manager import SceneManager

    plans: list = []
    original = SceneManager.plan_chunked

    def spy(self, request, visual_configs):
        plans.append(tuple(visual_configs))
        return original(self, request, visual_configs)

    monkeypatch.setattr(SceneManager, "plan_chunked", spy)
    return plans


# -- labels ------------------------------------------------------------------


@pytest.mark.parametrize("transform", ["identity", "anisotropic"])
@pytest.mark.parametrize("pose", list(POSES))
async def test_a_label_plane_shows_the_label_and_skips_background(
    controller, reslice, transform, pose
):
    """Labels: 0 wrong; a background voxel draws nothing."""
    scene, _visual, _gfx, planes = await _labels(
        controller,
        reslice,
        lambda world, centre: [h.plane_zyx(world, centre, POSES[pose])],
        transform=transform,
    )
    shot = h.shoot(controller, scene)
    scale, shift = TRANSFORMS[transform]
    labels = h.label_data(SHAPE)
    reference = h.trace(
        shot, planes, SHAPE, scale_zyx=scale, shift_zyx=shift, background=labels == 0
    )
    counts = h.compare_greys(shot, reference, labels, linear_colours=True)
    _assert_matches(counts, least=1500)
    # Background is a ninth of the voxels: holes are in the picture.
    assert counts["sure_empty"] > 0


async def test_a_label_shows_through_another_plane_s_background(controller, reslice):
    """A ray that meets background on the near plane shows the far plane."""
    scene, _visual, _gfx, planes = await _labels(
        controller,
        reslice,
        lambda world, centre: [
            h.plane_zyx(world, centre + np.array([0.0, 0.0, 6.0]), (0.3, 0.0, 1.0)),
            h.plane_zyx(world, centre - np.array([0.0, 0.0, 6.0]), (0.0, 0.2, 1.0)),
            h.plane_zyx(world, centre, (1.0, 0.1, 0.0)),
        ],
    )
    shot = h.shoot(controller, scene, view_dir=(-1.0, -0.3, -0.6))
    labels = h.label_data(SHAPE)
    background = labels == 0
    reference = h.trace(shot, planes, SHAPE, background=background)
    _assert_matches(
        h.compare_greys(shot, reference, labels, linear_colours=True), least=1500
    )

    # Pixels whose nearest plane hit is background but that show a label:
    # the label of a plane behind.
    nearest = h.trace(shot, planes, SHAPE)
    seen_through = reference.sure & reference.hit & nearest.hit
    seen_through &= reference.plane != nearest.plane
    assert seen_through.sum() > 100
    assert shot.drawn[seen_through].all()


# -- extents, clipping, several planes ----------------------------------------


async def test_extents_and_clipping_planes_cut_the_plane(controller, reslice):
    """The rectangle drawn equals the CPU's: 0 wrong, 0 missing."""
    clipping = [((0.0, 0.0, 22.0), (0.0, 0.2, -1.0))]
    scene, _visual, _gfx, planes = await _image(
        controller,
        reslice,
        lambda world, centre: [
            h.plane_zyx(
                world,
                centre,
                (0.6, -0.5, 1.0),
                extent_0=(-6.0, 9.0),
                extent_1=(None, 7.0),
            )
        ],
        transform="anisotropic",
        clipping=clipping,
    )
    shot = h.shoot(controller, scene)
    scale, shift = TRANSFORMS["anisotropic"]
    reference = h.trace(
        shot, planes, SHAPE, scale_zyx=scale, shift_zyx=shift, clipping=clipping
    )
    _assert_matches(h.compare_greys(shot, reference, h.voxel_values(SHAPE)), 800)
    # Both the extent and the clipping plane removed something.
    unbounded = h.trace(
        shot,
        [
            planes[0].model_copy(
                update={"extent_0": (None, None), "extent_1": (None, None)}
            )
        ],
        SHAPE,
        scale_zyx=scale,
        shift_zyx=shift,
    )
    assert (unbounded.hit & ~reference.hit).sum() > 500


async def test_three_planes_draw_the_nearest_at_every_pixel(controller, reslice):
    scene, _visual, _gfx, planes = await _image(
        controller,
        reslice,
        lambda world, centre: [
            h.plane_zyx(world, centre, (1.0, 0.0, 0.0)),
            h.plane_zyx(world, centre, (0.0, 1.0, 0.0)),
            h.plane_zyx(world, centre, (0.0, 0.0, 1.0)),
        ],
    )
    shot = h.shoot(controller, scene)
    reference = h.trace(shot, planes, SHAPE)
    _assert_matches(h.compare_greys(shot, reference, h.voxel_values(SHAPE)), 6000)
    assert set(np.unique(reference.plane[reference.hit])) == {0, 1, 2}


async def test_a_disabled_plane_is_not_drawn(controller, reslice):
    scene, _visual, _gfx, planes = await _image(
        controller,
        reslice,
        lambda world, centre: [
            h.plane_zyx(world, centre, (1.0, 0.0, 0.0), enabled=False),
            h.plane_zyx(world, centre, (0.0, 0.0, 1.0)),
        ],
    )
    shot = h.shoot(controller, scene)
    reference = h.trace(shot, planes, SHAPE)
    _assert_matches(h.compare_greys(shot, reference, h.voxel_values(SHAPE)))
    assert set(np.unique(reference.plane[reference.hit])) == {1}


async def test_moving_and_spinning_a_bounded_plane_is_a_uniform_write(
    controller, reslice, monkeypatch
):
    """C7: an in-plane move and a spin with finite extents; nothing is read."""
    scene, visual, _gfx, planes = await _image(
        controller,
        reslice,
        lambda world, centre: [
            h.plane_zyx(
                world,
                centre,
                (0.6, -0.5, 1.0),
                extent_0=(-7.0, 5.0),
                extent_1=(-4.0, 8.0),
            )
        ],
    )
    plans = _record_plans(monkeypatch)
    values = h.voxel_values(SHAPE)
    start = planes[0]
    compared = 0
    for step in range(1, 5):
        moved = start.model_copy(
            update={
                "origin": tuple(
                    np.array(start.origin)
                    + 1.3 * step * np.array(start.in_plane_axis_0)
                    - 0.7 * step * np.array(start.in_plane_axis_1)
                )
            }
        )
        moved = _spun(moved, 17.0 * step)
        controller.set_render_planes(visual.id, (moved,))
        shot = h.shoot(controller, scene)
        counts = h.compare_greys(shot, h.trace(shot, (moved,), SHAPE), values)
        _assert_matches(counts, least=500)
        compared += counts["sure_hit"]
    assert compared > 4000
    assert plans == []


async def test_a_plane_is_not_drawn_when_its_axes_are_not_displayed(
    controller, reslice
):
    """Design 4.4: a 2D view shows the slice; the mode and planes are ignored."""
    scene, visual, gfx_visual, _planes = await _image(
        controller,
        reslice,
        lambda world, centre: [h.plane_zyx(world, centre, POSES["xy"])],
    )
    assert len(gfx_visual.drawn_render_planes) == 1
    controller.set_displayed_axes(scene.id, (1, 2))
    assert len(gfx_visual.drawn_render_planes) == 0
    controller.set_displayed_axes(scene.id, (0, 1, 2))
    assert len(gfx_visual.drawn_render_planes) == 1
    assert visual.render_planes  # the model is untouched


# -- modes -------------------------------------------------------------------


async def test_switching_an_image_between_plane_and_a_volume_mode(controller, reslice):
    scene, visual, _gfx, planes = await _image(
        controller,
        reslice,
        lambda world, centre: [h.plane_zyx(world, centre, POSES["xy"])],
    )
    values = h.voxel_values(SHAPE)
    shot = h.shoot(controller, scene)
    plane_pixels = int(shot.drawn.sum())
    _assert_matches(h.compare_greys(shot, h.trace(shot, planes, SHAPE), values))

    controller.update_single_appearance_field(visual.id, "render_mode", "mip")
    await reslice(controller, scene.id)
    volume = h.shoot(controller, scene)
    # The whole box draws, not one section of it.
    assert volume.drawn.sum() > 1.5 * plane_pixels

    controller.update_single_appearance_field(visual.id, "render_mode", "plane")
    await reslice(controller, scene.id)
    shot = h.shoot(controller, scene)
    _assert_matches(h.compare_greys(shot, h.trace(shot, planes, SHAPE), values))


async def test_switching_labels_between_plane_and_a_volume_mode(controller, reslice):
    scene, visual, _gfx, planes = await _labels(
        controller,
        reslice,
        lambda world, centre: [h.plane_zyx(world, centre, POSES["xy"])],
    )
    labels = h.label_data(SHAPE)
    background = labels == 0

    def check() -> int:
        shot = h.shoot(controller, scene)
        reference = h.trace(shot, planes, SHAPE, background=background)
        _assert_matches(
            h.compare_greys(shot, reference, labels, linear_colours=True), least=1500
        )
        return int(shot.drawn.sum())

    plane_pixels = check()
    controller.update_appearance_field(visual.id, "render_mode", "flat_categorical")
    await reslice(controller, scene.id)
    assert h.shoot(controller, scene).drawn.sum() > 1.5 * plane_pixels
    controller.update_appearance_field(visual.id, "render_mode", "plane")
    await reslice(controller, scene.id)
    check()


async def test_plane_mode_with_no_plane_draws_nothing_until_one_is_added(
    controller, reslice
):
    scene, visual, _gfx, _planes = await _image(
        controller, reslice, lambda world, centre: []
    )
    assert not h.shoot(controller, scene).drawn.any()
    world = h.world_of(controller, scene)
    plane = h.plane_zyx(world, _centre(*TRANSFORMS["identity"]), POSES["oblique"])
    controller.set_render_planes(visual.id, (plane,))
    await reslice(controller, scene.id)
    shot = h.shoot(controller, scene)
    _assert_matches(
        h.compare_greys(shot, h.trace(shot, (plane,), SHAPE), h.voxel_values(SHAPE))
    )


def test_a_volume_mode_shader_holds_no_plane_code():
    """V8: the branch is compiled in for the plane mode only.

    The byte-for-byte check of every volume mode's generated shader is
    ``test_plane_golden_replay.py``; this guards the mechanism.
    """
    from pathlib import Path

    import cellier.render.shaders as shaders

    wgsl = Path(shaders.__file__).parent / "wgsl"
    for name, guard in (
        ("image_volume.wgsl", "$$ if mode == 'plane'"),
        ("label_volume.wgsl", '$$ if render_mode == "plane"'),
    ):
        source = (wgsl / name).read_text()
        include = source.index("cellier.render_planes.wgsl")
        assert source.rindex(guard, 0, include) > source.rindex("$$ endif", 0, include)
        use = source.index("plane_nearest_hit(")
        assert "$$ else" not in source[source.rindex(guard, 0, use) : use], name


# -- depth and alpha ----------------------------------------------------------


def _crossing(world, centre):
    return [
        h.plane_zyx(world, centre, (1.0, 0.0, 0.0)),
        h.plane_zyx(world, centre, (0.0, 0.0, 1.0)),
    ]


@pytest.mark.parametrize(
    ("mode", "depth_write"),
    [("blend", True), ("blend", False), ("add", True), ("add", False)],
)
async def test_a_translucent_plane_is_blended_once(
    controller, reslice, mode, depth_write
):
    """Depth and alpha (5.4): a pixel is one sample, whatever lies behind it.

    Two planes cross, so many rays meet both.  A pixel that showed both
    blended would differ from a pixel of the same voxel value with nothing
    behind it.
    """
    from cellier.visuals import InMemoryImageAppearance

    scene = controller.add_scene(dim="3d", name="scene")
    world = h.world_of(controller, scene)
    store = ImageMemoryStore(data=h.image_data(SHAPE), name="image")
    planes = tuple(_crossing(world, _centre(*TRANSFORMS["identity"])))
    controller.add_image(
        data=store,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(
            transparency_mode=mode, depth_write=depth_write
        ),
        single=InMemoryImageSingleAppearance(render_mode="plane", opacity=0.5),
        render_planes=planes,
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    shot = h.shoot(controller, scene)
    reference = h.trace(shot, planes, SHAPE)
    both = h.trace(shot, planes[:1], SHAPE).hit & h.trace(shot, planes[1:], SHAPE).hit
    sure = reference.sure & reference.hit
    assert shot.drawn[sure].all()
    assert (sure & both).sum() > 1000 and (sure & ~both).sum() > 1000

    voxel = reference.voxel
    values = h.voxel_values(SHAPE)[voxel[..., 0], voxel[..., 1], voxel[..., 2]]
    for value in range(h.N_VALUES):
        alone = shot.frame[sure & ~both & (values == value)].astype(int)
        behind = shot.frame[sure & both & (values == value)].astype(int)
        assert len(alone) and len(behind)
        colour = np.round(alone.mean(axis=0))
        assert np.abs(alone - colour).max() <= 1, (value, mode)
        assert np.abs(behind - colour).max() <= 1, (value, mode)
    # Half opacity: the picture is not opaque.
    assert shot.frame[sure][:, 3].max() < 200


async def test_a_composite_draws_every_channel_on_the_planes(controller, reslice):
    """A composite is additive and writes no depth, as a composite volume."""
    from cellier.scene import spatial_axes
    from cellier.visuals import InMemoryImageChannelAppearance

    scene = controller.add_scene(
        coordinate_system=[("c", "channel"), *spatial_axes("z", "y", "x")],
        dim="3d",
        name="channels",
    )
    world = h.world_of(controller, scene)
    first = h.voxel_values(SHAPE)
    second = first[:, :, ::-1]
    data = np.stack([first, second]).astype(np.float32)
    data = (data + 1) / (h.N_VALUES + 1)
    planes = tuple(_crossing(world, _centre(*TRANSFORMS["identity"])))
    visual = controller.add_image(
        data=ImageMemoryStore(data=data, name="channels"),
        scene_id=scene.id,
        channel_axis=0,
        composite=True,
        channels={
            0: InMemoryImageChannelAppearance(render_mode="plane", color_map="red"),
            1: InMemoryImageChannelAppearance(render_mode="plane", color_map="blue"),
        },
        render_planes=planes,
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    drawn = [gfx_visual.slots[i] for i in gfx_visual._drawn.values()]
    assert len(drawn) == 2
    assert not any(slot.node_3d.material.depth_write for slot in drawn)
    # One buffer of planes for every channel's material.
    buffers = {id(slot.node_3d.material.render_planes_buffer) for slot in drawn}
    assert buffers == {id(gfx_visual.render_planes_buffer)}

    shot = h.shoot(controller, scene)
    reference = h.trace(shot, planes, SHAPE)
    _assert_matches(h.compare_greys(shot, reference, first, channel=0))
    _assert_matches(h.compare_greys(shot, reference, second, channel=2))


# -- against a mesh -----------------------------------------------------------


def _slab(centre_xyz, size, colour, **material):
    import pygfx as gfx

    mesh = gfx.Mesh(
        gfx.box_geometry(*size), gfx.MeshBasicMaterial(color=colour, **material)
    )
    mesh.local.position = tuple(centre_xyz)
    return mesh


@pytest.mark.parametrize("where", ["in_front", "behind"])
async def test_a_plane_and_an_opaque_mesh_occlude_each_other(
    controller, reslice, where
):
    scene, _visual, _gfx, planes = await _image(
        controller,
        reslice,
        lambda world, centre: [h.plane_zyx(world, centre, POSES["xy"])],
    )
    centre = np.array(planes[0].origin)[::-1]
    # The camera looks down -z, so +z is nearer to it.
    offset = 3.0 if where == "in_front" else -3.0
    gfx_scene = controller._render_manager.get_scene(scene.id)
    gfx_scene.add(
        _slab(centre + np.array([0.0, 0.0, offset]), (10.0, 8.0, 1.0), "#ff0000")
    )
    shot = h.shoot(controller, scene, view_dir=(0.0, 0.0, -1.0))
    reference = h.trace(shot, planes, SHAPE)
    xy = reference.position[..., :2] - centre[:2]
    over_mesh = (np.abs(xy[..., 0]) < 4.5) & (np.abs(xy[..., 1]) < 3.5)
    # Clear of the mesh's outline, which perspective enlarges when in front.
    clear = (np.abs(xy[..., 0]) > 6.5) | (np.abs(xy[..., 1]) > 5.5)
    sure = reference.sure & reference.hit
    assert (sure & over_mesh).sum() > 300 and (sure & clear).sum() > 300
    red = (shot.frame[..., 0] > 250) & (shot.frame[..., 1] < 5)
    if where == "in_front":
        assert red[sure & over_mesh].all()
    else:
        assert not red[sure & over_mesh].any()
    assert not red[sure & clear].any()


@pytest.mark.parametrize("depth_write", [True, False])
async def test_a_translucent_mesh_behind_a_plane_follows_its_depth_write(
    controller, reslice, depth_write
):
    """C11: a mesh drawn after the plane is tested against the plane's depth
    where the visual writes depth, and shows through where it does not."""
    from cellier.visuals import InMemoryImageAppearance

    scene = controller.add_scene(dim="3d", name="scene")
    world = h.world_of(controller, scene)
    store = ImageMemoryStore(data=h.image_data(SHAPE), name="image")
    plane = h.plane_zyx(world, _centre(*TRANSFORMS["identity"]), POSES["xy"])
    controller.add_image(
        data=store,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(depth_write=depth_write),
        single=InMemoryImageSingleAppearance(render_mode="plane"),
        render_planes=(plane,),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    before = h.shoot(controller, scene, view_dir=(0.0, 0.0, -1.0))

    centre = np.array(plane.origin)[::-1]
    mesh = _slab(
        centre - np.array([0.0, 0.0, 3.0]),
        (10.0, 8.0, 1.0),
        "#ff0000",
        opacity=0.5,
        alpha_mode="blend",
    )
    mesh.render_order = 10
    controller._render_manager.get_scene(scene.id).add(mesh)
    after = h.shoot(controller, scene, view_dir=(0.0, 0.0, -1.0))
    reference = h.trace(after, (plane,), SHAPE)
    xy = reference.position[..., :2] - centre[:2]
    over_mesh = (np.abs(xy[..., 0]) < 4.5) & (np.abs(xy[..., 1]) < 3.5)
    over_mesh &= reference.sure & reference.hit
    assert over_mesh.sum() > 300
    changed = (before.frame != after.frame).any(axis=-1)
    if depth_write:
        assert not changed[over_mesh].any()
    else:
        assert changed[over_mesh].all()


# -- pick ---------------------------------------------------------------------


def _sample(mask: np.ndarray, count: int) -> list[tuple[int, int]]:
    rows, cols = np.nonzero(mask)
    chosen = np.linspace(0, len(rows) - 1, count).astype(int)
    return [(int(rows[i]), int(cols[i])) for i in chosen]


@pytest.mark.parametrize("pose", list(POSES))
async def test_a_pick_on_an_image_plane_names_the_voxel_at_the_hit(
    controller, reslice, pose
):
    scene, _visual, gfx_visual, planes = await _image(
        controller,
        reslice,
        lambda world, centre: [h.plane_zyx(world, centre, POSES[pose])],
        transform="anisotropic",
    )
    shot = h.shoot(controller, scene)
    scale, shift = TRANSFORMS["anisotropic"]
    reference = h.trace(shot, planes, SHAPE, scale_zyx=scale, shift_zyx=shift)
    pixels = _sample(reference.sure & reference.hit, 150)
    wrong, worst = [], 0.0
    for row, col in pixels:
        voxel = h.pick_voxel(shot, gfx_visual, row, col)
        if voxel != tuple(reference.voxel[row, col]):
            wrong.append((row, col, voxel, tuple(reference.voxel[row, col])))
        position = h.pick_coordinate(shot, gfx_visual, row, col)
        worst = max(worst, np.abs(position - reference.data_position[row, col]).max())
    assert not wrong, wrong[:5]
    # An eighth of a voxel plus the 14-bit step.
    assert worst < 0.125 + max(SHAPE) / 16383, worst


async def test_a_pick_on_a_label_plane_names_the_label_drawn(controller, reslice):
    scene, _visual, gfx_visual, planes = await _labels(
        controller,
        reslice,
        lambda world, centre: [
            h.plane_zyx(world, centre + np.array([0.0, 0.0, 6.0]), (0.3, 0.0, 1.0)),
            h.plane_zyx(world, centre, (1.0, 0.1, 0.0)),
        ],
    )
    shot = h.shoot(controller, scene, view_dir=(-1.0, -0.3, -0.6))
    labels = h.label_data(SHAPE)
    reference = h.trace(shot, planes, SHAPE, background=labels == 0)
    wrong = []
    for row, col in _sample(reference.sure & reference.hit, 200):
        voxel = h.pick_voxel(shot, gfx_visual, row, col)
        expected = tuple(reference.voxel[row, col])
        if voxel != expected or labels[voxel] == 0:
            wrong.append((row, col, voxel, expected))
    assert not wrong, wrong[:5]


# -- the normal and outline targets -------------------------------------------


def _view_normals(shot, plane) -> np.ndarray:
    """The unit view-space normal of *plane*, faced to the viewer, per pixel."""
    normal = np.cross(plane.in_plane_axis_0[::-1], plane.in_plane_axis_1[::-1])
    view = np.asarray(shot.camera.view_matrix, dtype=np.float64)
    in_view = view[:3, :3] @ normal
    in_view /= np.linalg.norm(in_view)
    return in_view


@pytest.mark.parametrize("kind", ["image", "labels"])
async def test_the_plane_writes_its_normal_facing_the_viewer(controller, reslice, kind):
    """Normal (5.5, P15): the plane's own, the same from both of its sides."""
    from cellier.render._cellier_blender import NORMAL_TARGET

    add = _image if kind == "image" else _labels
    scene, visual, _gfx, planes = await add(
        controller,
        reslice,
        lambda world, centre: [h.plane_zyx(world, centre, POSES["oblique"])],
        transform="anisotropic",
    )
    scale, shift = TRANSFORMS["anisotropic"]
    background = h.label_data(SHAPE) == 0 if kind == "labels" else None

    def normals_and_frame(plane):
        shot = h.shoot(controller, scene, blender_targets=[NORMAL_TARGET])
        written = h.read_target(shot, NORMAL_TARGET, np.float16, 4).astype(np.float64)
        reference = h.trace(
            shot,
            (plane,),
            SHAPE,
            scale_zyx=scale,
            shift_zyx=shift,
            background=background,
        )
        return shot, written, reference

    shot, written, reference = normals_and_frame(planes[0])
    sure = reference.sure & reference.hit
    assert sure.sum() > 1500
    expected = _view_normals(shot, planes[0])
    # Faced to the viewer: against the view-space position of the hit.
    view = np.asarray(shot.camera.view_matrix, dtype=np.float64)
    position = reference.position[sure] @ view[:3, :3].T + view[:3, 3]
    facing = np.where((position @ expected)[:, None] < 0, expected, -expected)
    got = written[sure][:, :3]
    # The target is 16-bit float: unit to three decimal places.
    length = np.linalg.norm(got, axis=-1, keepdims=True)
    assert np.allclose(length, 1.0, atol=0.01)
    got = got / length
    cosine = np.clip(np.sum(got * facing, axis=-1), -1.0, 1.0)
    assert np.degrees(np.arccos(cosine)).max() < 0.35
    # Nothing is written where no plane is drawn.
    assert not written[reference.sure & ~reference.hit][:, :3].any()

    # The other side of the same plane: nothing drawn differs.
    plane = planes[0]
    flipped = plane.model_copy(
        update={
            "in_plane_axis_0": plane.in_plane_axis_1,
            "in_plane_axis_1": plane.in_plane_axis_0,
            "extent_0": plane.extent_1,
            "extent_1": plane.extent_0,
        }
    )
    assert np.allclose(np.array(flipped.normal), -np.array(plane.normal))
    controller.set_render_planes(visual.id, (flipped,))
    flipped_shot, flipped_written, _ = normals_and_frame(flipped)
    assert np.array_equal(flipped_shot.frame, shot.frame)
    assert np.array_equal(flipped_written, written)


async def test_a_label_plane_writes_the_outline_key_of_the_label_drawn(
    controller, reslice
):
    """Outline (5.5): one key per label, none on background or off the plane."""
    from cellier.render._cellier_blender import OUTLINE_ID_TARGET

    scene, _visual, _gfx, planes = await _labels(
        controller,
        reslice,
        lambda world, centre: [h.plane_zyx(world, centre, POSES["oblique"])],
    )
    shot = h.shoot(controller, scene, blender_targets=[OUTLINE_ID_TARGET])
    keys = h.read_target(shot, OUTLINE_ID_TARGET, np.uint32, 1)[..., 0]
    labels = h.label_data(SHAPE)
    reference = h.trace(shot, planes, SHAPE, background=labels == 0)
    voxel = reference.voxel
    label = labels[voxel[..., 0], voxel[..., 1], voxel[..., 2]]
    sure = reference.sure & reference.hit
    assert sure.sum() > 1500
    assert not keys[reference.sure & ~reference.hit].any()
    key_of = {}
    for value in range(1, h.N_VALUES):
        found = np.unique(keys[sure & (label == value)])
        assert len(found) == 1 and found[0] >= 16, (value, found)
        key_of[value] = int(found[0])
    assert len(set(key_of.values())) == h.N_VALUES - 1


# -- nD data (R9) -------------------------------------------------------------


async def test_a_plane_on_three_axes_of_a_4d_scene_shows_the_current_time_point(
    controller, reslice
):
    """A ``zyx`` plane in a ``tzyx`` scene holds at every time point: dims
    supplies ``t``, and the plane shows that time point's voxels."""
    from cellier.scene import spatial_axes

    scene = controller.add_scene(
        coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")],
        dim="3d",
        name="scene",
    )
    world = h.world_of(controller, scene)
    values = np.stack([(h.voxel_values(SHAPE) + 3 * t) % h.N_VALUES for t in range(3)])
    data = ((values + 1) / (h.N_VALUES + 1)).astype(np.float32)
    plane = h.plane_zyx(world, _centre(*TRANSFORMS["identity"]), POSES["oblique"])
    controller.add_image(
        data=ImageMemoryStore(data=data, name="tzyx"),
        scene_id=scene.id,
        single=InMemoryImageSingleAppearance(render_mode="plane"),
        render_planes=(plane,),
    )
    controller.add_canvas(scene_id=scene.id)
    for t in (0, 2):
        controller.update_slice_indices(scene.id, {0: float(t)})
        await reslice(controller, scene.id)
        shot = h.shoot(controller, scene)
        reference = h.trace(shot, (plane,), SHAPE)
        _assert_matches(h.compare_greys(shot, reference, values[t]))
        # Not another time point's picture.
        other = h.compare_greys(shot, reference, values[1])
        assert other["wrong"] > 0.5 * other["sure_hit"]
