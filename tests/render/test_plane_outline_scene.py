"""Which plane outlines a scene draws, and when (plane outline design 3.3,
5.4, 6.1 and 6.4).

Nothing is drawn here: the tests read the lines the scene holds.  What a
line looks like on screen is in ``test_plane_outline_render.py``.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
import pytest

from cellier.data import ImageMemoryStore, LabelMemoryStore
from cellier.scene import spatial_axes
from cellier.visuals import (
    ClippingPlane,
    InMemoryImageChannelAppearance,
    InMemoryImageSingleAppearance,
    InMemoryLabelsAppearance,
    PlaneOutline,
    RenderPlane,
)
from tests.render import _plane_rig as rig_module
from tests.render import _planes as h
from tests.render.conftest import drain_loading

SHAPE = (8, 12, 16)
RED = PlaneOutline(enabled=True, color=(1.0, 0.0, 0.0, 1.0), width=3.0)
BLUE = PlaneOutline(enabled=True, color=(0.0, 0.0, 1.0, 0.5), width=5.0)
#: The plane z = 3 through a volume of ``SHAPE``, as rendered (x, y, z):
#: the voxel-edge box is -0.5 to 15.5, 11.5 and 7.5.
SQUARE = [(-0.5, -0.5, 3.0), (15.5, -0.5, 3.0), (15.5, 11.5, 3.0), (-0.5, 11.5, 3.0)]


def _world(controller, scene):
    return h.world_of(controller, scene)


def _plane(world, z: float = 3.0, outline=RED, **kwargs) -> RenderPlane:
    return RenderPlane.from_point_normal(
        world, (z, 5.0, 7.0), (1, 0, 0), axes=("z", "y", "x"), outline=outline, **kwargs
    )


def _image(controller, scene, planes, *, shape=SHAPE, render_mode="plane", **kwargs):
    store = ImageMemoryStore(data=np.ones(shape, dtype=np.float32), name="image")
    return controller.add_image(
        data=store,
        scene_id=scene.id,
        single=InMemoryImageSingleAppearance(
            render_mode=render_mode, color_map="gray", clim=(0.0, 1.0)
        ),
        render_planes=tuple(planes),
        **kwargs,
    )


def _labels(controller, scene, planes, *, shape=SHAPE, data=None, **kwargs):
    if data is None:
        data = np.ones(shape, dtype=np.int32)
    store = LabelMemoryStore(data=data, name="labels")
    return controller.add_labels(
        data=store,
        scene_id=scene.id,
        appearance=InMemoryLabelsAppearance(render_mode="plane"),
        render_planes=tuple(planes),
        **kwargs,
    )


def _scene(controller):
    scene = controller.add_scene(dim="3d", name="scene")
    return scene, _world(controller, scene)


def _store(controller, visual):
    return controller.get_data_store(UUID(str(visual.data_store_id)))


def _outlines(controller, scene):
    return controller._render_manager._scenes[scene.id].plane_outlines


def _drawn(controller, scene) -> dict:
    return _outlines(controller, scene).drawn


def _same(vertices, expected) -> bool:
    expected = np.asarray(expected, dtype=float)
    if vertices.shape != expected.shape:
        return False
    return any(
        np.abs(vertices - np.roll(candidate, shift, axis=0)).max() < 1e-6
        for candidate in (expected, expected[::-1])
        for shift in range(len(expected))
    )


def _square(z: float):
    return [(x, y, z) for x, y, _ in SQUARE]


# -- when an outline is shown (3.3) ------------------------------------------------


async def test_an_outlined_plane_has_one_line_on_its_polygon(controller):
    scene, world = _scene(controller)
    plane = _plane(world)
    _image(controller, scene, [plane])
    controller.add_canvas(scene_id=scene.id)

    drawn = _drawn(controller, scene)
    assert list(drawn) == [("render", plane.id, 0)]
    polygon = drawn[("render", plane.id, 0)]
    assert _same(polygon.vertices, SQUARE)
    assert polygon.color == RED.color
    assert polygon.width == RED.width
    assert len(_outlines(controller, scene).node.children) == 1


async def test_a_plane_with_no_outline_has_no_line_and_no_node(controller):
    """Nothing is put in the scene for a plane that is not outlined: a
    hidden node would still count toward the scene's bounding box."""
    scene, world = _scene(controller)
    _image(controller, scene, [_plane(world, outline=PlaneOutline())])
    controller.add_canvas(scene_id=scene.id)
    assert _drawn(controller, scene) == {}
    assert _outlines(controller, scene).node.children == ()


async def test_the_outline_follows_its_switch_and_the_plane_s(controller):
    scene, world = _scene(controller)
    plane = _plane(world)
    visual = _image(controller, scene, [plane])
    controller.add_canvas(scene_id=scene.id)

    off = plane.model_copy(
        update={"outline": RED.model_copy(update={"enabled": False})}
    )
    controller.set_render_planes(visual.id, (off,))
    assert _drawn(controller, scene) == {}
    controller.set_render_planes(visual.id, (plane,))
    assert len(_drawn(controller, scene)) == 1

    # A disabled plane is not drawn, and neither is its outline.
    controller.set_render_planes(
        visual.id, (plane.model_copy(update={"enabled": False}),)
    )
    assert _drawn(controller, scene) == {}


async def test_a_hidden_visual_s_planes_have_no_outline(controller):
    scene, world = _scene(controller)
    visual = _image(controller, scene, [_plane(world)])
    controller.add_canvas(scene_id=scene.id)
    visual.appearance.visible = False
    assert _drawn(controller, scene) == {}
    visual.appearance.visible = True
    assert len(_drawn(controller, scene)) == 1


async def test_an_outline_is_shown_only_in_plane_mode(controller):
    scene, world = _scene(controller)
    visual = _image(controller, scene, [_plane(world)], render_mode="mip")
    controller.add_canvas(scene_id=scene.id)
    assert _drawn(controller, scene) == {}
    visual.single.render_mode = "plane"
    assert len(_drawn(controller, scene)) == 1
    visual.single.render_mode = "mip"
    assert _drawn(controller, scene) == {}


async def test_an_outline_is_shown_only_in_a_3d_view(controller):
    scene, world = _scene(controller)
    _image(controller, scene, [_plane(world)])
    controller.add_canvas(scene_id=scene.id)
    controller.set_displayed_axes(scene.id, (1, 2))
    assert _drawn(controller, scene) == {}
    controller.set_displayed_axes(scene.id, (0, 1, 2))
    assert len(_drawn(controller, scene)) == 1


async def test_an_outline_is_shown_only_on_the_plane_s_own_axes(controller):
    """A tzyx scene: a zyx plane is drawn in the zyx view and not in tyx."""
    scene = controller.add_scene(
        coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")],
        dim="3d",
        name="scene",
    )
    world = _world(controller, scene)
    with pytest.warns(UserWarning, match="not drawn"):
        _image(controller, scene, [_plane(world)], shape=(4, *SHAPE))
        controller.add_canvas(scene_id=scene.id)
        assert len(_drawn(controller, scene)) == 1
        controller.set_displayed_axes(scene.id, (0, 2, 3))
    assert _drawn(controller, scene) == {}
    controller.set_displayed_axes(scene.id, (1, 2, 3))
    assert len(_drawn(controller, scene)) == 1


async def test_an_outline_does_not_wait_for_data(controller):
    """B-D6: a labels plane that is all background draws nothing and still
    has its outline."""
    scene, world = _scene(controller)
    _labels(controller, scene, [_plane(world)], data=np.zeros(SHAPE, dtype=np.int32))
    controller.add_canvas(scene_id=scene.id)
    (polygon,) = _drawn(controller, scene).values()
    assert _same(polygon.vertices, SQUARE)


async def test_a_pyramid_s_outline_is_there_before_any_brick(controller, tmp_path):
    """B-D6: nothing has been planned or loaded."""

    def make(world, centre):
        return [h.plane_zyx(world, centre, (1.0, 0.0, 0.0), outline=RED)]

    rig = await rig_module.make_rig(controller, tmp_path, rig_module.ISO, make)
    (polygon,) = _drawn(controller, rig.scene).values()
    assert len(polygon.vertices) == 4
    await rig.settle()
    assert len(_drawn(controller, rig.scene)) == 1


async def test_a_composite_has_one_outline_whatever_its_channels(controller):
    scene = controller.add_scene(
        coordinate_system=[("c", "channel"), *spatial_axes("z", "y", "x")],
        dim="3d",
        name="scene",
    )
    world = _world(controller, scene)
    store = ImageMemoryStore(data=np.ones((3, *SHAPE), dtype=np.float32))
    controller.add_image(
        data=store,
        scene_id=scene.id,
        channel_axis=0,
        composite=True,
        channels={
            k: InMemoryImageChannelAppearance(
                color_map=c, clim=(0.0, 1.0), render_mode="plane"
            )
            for k, c in enumerate(("red", "green", "blue"))
        },
        render_planes=(_plane(world),),
    )
    controller.add_canvas(scene_id=scene.id)
    (polygon,) = _drawn(controller, scene).values()
    assert _same(polygon.vertices, SQUARE)


# -- what moves or restyles a line (6.4) -------------------------------------------


async def test_the_line_follows_its_plane(controller):
    scene, world = _scene(controller)
    plane = _plane(world)
    visual = _image(controller, scene, [plane])
    controller.add_canvas(scene_id=scene.id)
    line = _outlines(controller, scene).line(("render", plane.id, 0))

    controller.set_render_planes(
        visual.id, (plane.model_copy(update={"origin": (5.0, 5.0, 7.0)}),)
    )
    (polygon,) = _drawn(controller, scene).values()
    assert _same(polygon.vertices, _square(5.0))
    # The same line object, moved.
    assert _outlines(controller, scene).line(("render", plane.id, 0)) is line
    np.testing.assert_allclose(line.geometry.positions.data[:, 2], 5.0)

    # Out of the box: no polygon, no line.
    controller.set_render_planes(
        visual.id, (plane.model_copy(update={"origin": (50.0, 5.0, 7.0)}),)
    )
    assert _drawn(controller, scene) == {}


async def test_a_restyle_recolours_the_line_and_keeps_it(controller):
    scene, world = _scene(controller)
    plane = _plane(world)
    visual = _image(controller, scene, [plane])
    controller.add_canvas(scene_id=scene.id)
    line = _outlines(controller, scene).line(("render", plane.id, 0))
    geometry = line.geometry

    controller.set_render_planes(
        visual.id, (plane.model_copy(update={"outline": BLUE}),)
    )
    assert _outlines(controller, scene).line(("render", plane.id, 0)) is line
    assert line.geometry is geometry
    assert tuple(line.material.color) == pytest.approx(BLUE.color)
    assert line.material.thickness == BLUE.width


async def test_the_line_ends_at_a_clipping_plane_and_follows_it(controller):
    scene, world = _scene(controller)
    plane = _plane(world)
    visual = _image(controller, scene, [plane])
    controller.add_canvas(scene_id=scene.id)
    system = _store(controller, visual).data_coordinate_systems[0]

    def clip(x: float) -> ClippingPlane:
        # Kept where x >= x.
        return ClippingPlane.from_point_normal(
            system, (0, 0, x), (0, 0, 1), axes=("z", "y", "x")
        )

    first = clip(4.0)
    controller.set_clipping_planes(visual.id, (first,))
    (polygon,) = _drawn(controller, scene).values()
    assert _same(
        polygon.vertices,
        [(4.0, -0.5, 3.0), (15.5, -0.5, 3.0), (15.5, 11.5, 3.0), (4.0, 11.5, 3.0)],
    )

    controller.set_clipping_planes(
        visual.id, (clip(9.0).model_copy(update={"id": first.id}),)
    )
    (polygon,) = _drawn(controller, scene).values()
    assert polygon.vertices[:, 0].min() == pytest.approx(9.0)

    # A disabled clipping plane cuts nothing.
    controller.set_clipping_planes(
        visual.id, (first.model_copy(update={"enabled": False}),)
    )
    (polygon,) = _drawn(controller, scene).values()
    assert _same(polygon.vertices, SQUARE)


async def test_the_line_follows_the_visual_s_transform(controller):
    """Voxels made twice as wide in x: the box, and so the polygon, grows."""
    scene, world = _scene(controller)
    visual = _image(controller, scene, [_plane(world)])
    controller.add_canvas(scene_id=scene.id)
    store = _store(controller, visual)
    visual.transform = h.scale_translation_transform(
        controller, scene.id, store, (1.0, 1.0, 2.0), (0.0, 0.0, 0.0)
    )
    (polygon,) = _drawn(controller, scene).values()
    assert _same(
        polygon.vertices,
        [(-1.0, -0.5, 3.0), (31.0, -0.5, 3.0), (31.0, 11.5, 3.0), (-1.0, 11.5, 3.0)],
    )


async def test_a_change_draws_the_scene_from_a_clean_history(controller, monkeypatch):
    """A moved or restyled line asks for a frame and discards the frames
    accumulated with the old one."""
    scene, world = _scene(controller)
    plane = _plane(world)
    visual = _image(controller, scene, [plane])
    controller.add_canvas(scene_id=scene.id)
    view = controller._render_manager._canvases[controller.get_canvas_ids(scene.id)[0]]
    calls: list[str] = []
    monkeypatch.setattr(view, "invalidate_accumulation", lambda: calls.append("reset"))
    monkeypatch.setattr(view, "request_draw", lambda: calls.append("draw"))

    controller.set_render_planes(
        visual.id, (plane.model_copy(update={"outline": BLUE}),)
    )
    assert "reset" in calls
    assert "draw" in calls


async def test_removing_the_visual_removes_its_lines(controller):
    scene, world = _scene(controller)
    visual = _image(controller, scene, [_plane(world)])
    other = _image(controller, scene, [_plane(world, z=5.0)])
    controller.add_canvas(scene_id=scene.id)
    assert len(_drawn(controller, scene)) == 2
    controller.remove_visual(visual.id)
    (key,) = _drawn(controller, scene)
    assert key[1] == other.render_planes[0].id
    controller.remove_visual(other.id)
    assert _drawn(controller, scene) == {}
    assert _outlines(controller, scene).node.children == ()


async def test_closing_the_scene_leaves_no_line(controller):
    scene, world = _scene(controller)
    _image(controller, scene, [_plane(world)])
    controller.add_canvas(scene_id=scene.id)
    scene_manager = controller._render_manager._scenes[scene.id]
    scene_manager.close()
    assert scene_manager.plane_outlines.drawn == {}
    assert scene_manager.plane_outlines.node.children == ()


# -- a plane shared by two visuals (5.4) --------------------------------------------


def _pair(controller, *, labels_shape=SHAPE, labels_outline=BLUE):
    """An image and labels sharing one plane id; the image is added first."""
    scene, world = _scene(controller)
    plane = _plane(world)
    image = _image(controller, scene, [plane])
    labels = _labels(
        controller,
        scene,
        [plane.model_copy(update={"outline": labels_outline})],
        shape=labels_shape,
    )
    controller.add_canvas(scene_id=scene.id)
    return scene, plane, image, labels


async def test_a_shared_plane_with_equal_polygons_has_one_outline(controller):
    scene, plane, _image, _labels = _pair(controller)
    drawn = _drawn(controller, scene)
    assert list(drawn) == [("render", plane.id, 0)]
    # The first visual's style.
    assert drawn[("render", plane.id, 0)].color == RED.color
    assert len(_outlines(controller, scene).node.children) == 1


async def test_a_shared_plane_in_two_boxes_has_two_outlines(controller):
    scene, plane, _image, _labels = _pair(controller, labels_shape=(8, 6, 8))
    drawn = _drawn(controller, scene)
    assert list(drawn) == [("render", plane.id, 0), ("render", plane.id, 1)]
    assert _same(drawn[("render", plane.id, 0)].vertices, SQUARE)
    assert _same(
        drawn[("render", plane.id, 1)].vertices,
        [(-0.5, -0.5, 3.0), (7.5, -0.5, 3.0), (7.5, 5.5, 3.0), (-0.5, 5.5, 3.0)],
    )
    # Each in its own visual's style.
    assert drawn[("render", plane.id, 0)].color == RED.color
    assert drawn[("render", plane.id, 1)].color == BLUE.color


async def test_a_shared_plane_clipped_in_one_visual_has_two_outlines(controller):
    scene, plane, _image, labels = _pair(controller)
    system = _store(controller, labels).data_coordinate_systems[0]
    controller.set_clipping_planes(
        labels.id,
        (
            ClippingPlane.from_point_normal(
                system, (0, 0, 4.0), (0, 0, 1), axes=("z", "y", "x")
            ),
        ),
    )
    drawn = _drawn(controller, scene)
    assert len(drawn) == 2
    assert _same(drawn[("render", plane.id, 0)].vertices, SQUARE)
    assert drawn[("render", plane.id, 1)].vertices[:, 0].min() == pytest.approx(4.0)


async def test_a_shared_outline_stays_while_any_visual_shows_it(controller):
    scene, plane, image, labels = _pair(controller)
    image.appearance.visible = False
    drawn = _drawn(controller, scene)
    assert list(drawn) == [("render", plane.id, 0)]
    # Now the second visual's style: the first is not shown.
    assert drawn[("render", plane.id, 0)].color == BLUE.color
    labels.appearance.visible = False
    assert _drawn(controller, scene) == {}
    image.appearance.visible = True
    assert _drawn(controller, scene)[("render", plane.id, 0)].color == RED.color


async def test_two_planes_have_two_outlines(controller):
    scene, world = _scene(controller)
    first, second = _plane(world), _plane(world, z=5.0, outline=BLUE)
    _image(controller, scene, [first, second])
    controller.add_canvas(scene_id=scene.id)
    drawn = _drawn(controller, scene)
    assert set(drawn) == {("render", first.id, 0), ("render", second.id, 0)}
    assert _same(drawn[("render", second.id, 0)].vertices, _square(5.0))


async def test_a_loaded_scene_keeps_its_outline(controller):
    """Loading places the nodes again; the outline is still the plane's."""
    scene, world = _scene(controller)
    _image(controller, scene, [_plane(world)])
    controller.add_canvas(scene_id=scene.id)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    (polygon,) = _drawn(controller, scene).values()
    assert _same(polygon.vertices, SQUARE)
