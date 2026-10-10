"""The outline of a clipping plane: its cut face (plane outline design 3.2).

The plane, cut by the visual's data box and by the visual's other
clipping planes.  Most tests read the lines the scene holds; the last few
draw.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
import pytest

from cellier.data import (
    GraphMemoryStore,
    ImageMemoryStore,
    LinesMemoryStore,
    MeshMemoryStore,
    PointsMemoryStore,
)
from cellier.scene import spatial_axes
from cellier.visuals import (
    ClippingPlane,
    InMemoryImageSingleAppearance,
    MeshFlatAppearance,
    PlaneOutline,
    RenderPlane,
)
from tests._meshes import uv_sphere
from tests.render import _plane_outline_pixels as px
from tests.render import _plane_rig as rig_module
from tests.render import _planes as h
from tests.render.conftest import drain_loading

SHAPE = (8, 12, 16)
RED = PlaneOutline(enabled=True, color=(1.0, 0.0, 0.0, 1.0), width=4.0)
BLUE = PlaneOutline(enabled=True, color=(0.0, 0.0, 1.0, 1.0), width=2.0)
#: The corners of a box from the origin to (8, 12, 16), data ``(z, y, x)``.
CORNERS = np.array(
    [[z, y, x] for z in (0.0, 8.0) for y in (0.0, 12.0) for x in (0.0, 16.0)],
    dtype=np.float32,
)


def _store(controller, visual):
    return controller.get_data_store(UUID(str(visual.data_store_id)))


def _clip(controller, visual, x: float, outline=RED, **kwargs) -> ClippingPlane:
    """The clipping plane keeping ``x >= x``, on the store's last three axes."""
    system = _store(controller, visual).data_coordinate_systems[0]
    return ClippingPlane.from_point_normal(
        system, (0, 0, x), (0, 0, 1), axes=("z", "y", "x"), outline=outline, **kwargs
    )


def _image(controller, scene, *, shape=SHAPE, render_mode="mip", **kwargs):
    store = ImageMemoryStore(data=np.ones(shape, dtype=np.float32), name="image")
    return controller.add_image(
        data=store,
        scene_id=scene.id,
        single=InMemoryImageSingleAppearance(
            render_mode=render_mode, color_map="gray", clim=(0.0, 2.0)
        ),
        **kwargs,
    )


def _drawn(controller, scene) -> dict:
    return controller._render_manager._scenes[scene.id].plane_outlines.drawn


def _same(vertices, expected) -> bool:
    expected = np.asarray(expected, dtype=float)
    if vertices.shape != expected.shape:
        return False
    return any(
        np.abs(vertices - np.roll(candidate, shift, axis=0)).max() < 1e-5
        for candidate in (expected, expected[::-1])
        for shift in range(len(expected))
    )


def _x_face(x: float, low=(-0.5, -0.5), high=(11.5, 7.5)):
    """The face at *x* of a box over y and z, in rendered (x, y, z)."""
    (y0, z0), (y1, z1) = low, high
    return [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]


async def _loaded(controller, scene) -> None:
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)


# -- a volume ----------------------------------------------------------------------


async def test_an_outlined_clipping_plane_has_a_line_round_its_cut_face(controller):
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _image(controller, scene)
    controller.add_canvas(scene_id=scene.id)
    assert _drawn(controller, scene) == {}

    plane = _clip(controller, visual, 4.0)
    controller.set_clipping_planes(visual.id, (plane,))
    drawn = _drawn(controller, scene)
    assert list(drawn) == [("clipping", plane.id, 0)]
    polygon = drawn[("clipping", plane.id, 0)]
    # The voxel-edge box of an 8 x 12 x 16 volume, cut at x = 4.
    assert _same(polygon.vertices, _x_face(4.0))
    assert polygon.color == RED.color
    assert polygon.width == RED.width


async def test_a_clipping_plane_s_outline_follows_its_switches(controller):
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _image(controller, scene)
    controller.add_canvas(scene_id=scene.id)
    plane = _clip(controller, visual, 4.0)

    controller.set_clipping_planes(visual.id, (plane,))
    assert len(_drawn(controller, scene)) == 1
    off = RED.model_copy(update={"enabled": False})
    controller.set_clipping_planes(
        visual.id, (plane.model_copy(update={"outline": off}),)
    )
    assert _drawn(controller, scene) == {}
    # A disabled plane cuts nothing and has no face.
    controller.set_clipping_planes(
        visual.id, (plane.model_copy(update={"enabled": False}),)
    )
    assert _drawn(controller, scene) == {}
    controller.set_clipping_planes(visual.id, (plane,))

    visual.appearance.visible = False
    assert _drawn(controller, scene) == {}
    visual.appearance.visible = True
    assert len(_drawn(controller, scene)) == 1

    # B-P8: 3D views only.
    controller.set_displayed_axes(scene.id, (1, 2))
    assert _drawn(controller, scene) == {}
    controller.set_displayed_axes(scene.id, (0, 1, 2))
    assert len(_drawn(controller, scene)) == 1


async def test_the_cut_face_follows_the_plane_and_its_style(controller):
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _image(controller, scene)
    controller.add_canvas(scene_id=scene.id)
    plane = _clip(controller, visual, 4.0)
    controller.set_clipping_planes(visual.id, (plane,))
    outlines = controller._render_manager._scenes[scene.id].plane_outlines
    line = outlines.line(("clipping", plane.id, 0))

    moved = _clip(controller, visual, 9.0).model_copy(update={"id": plane.id})
    controller.set_clipping_planes(visual.id, (moved,))
    (polygon,) = _drawn(controller, scene).values()
    assert _same(polygon.vertices, _x_face(9.0))
    assert outlines.line(("clipping", plane.id, 0)) is line

    controller.set_clipping_planes(
        visual.id, (moved.model_copy(update={"outline": BLUE}),)
    )
    assert tuple(line.material.color) == pytest.approx(BLUE.color)
    assert line.material.thickness == BLUE.width

    # Past the box: no face.
    controller.set_clipping_planes(
        visual.id,
        (_clip(controller, visual, 40.0).model_copy(update={"id": plane.id}),),
    )
    assert _drawn(controller, scene) == {}


async def test_two_clipping_planes_each_end_at_the_other(controller):
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _image(controller, scene)
    controller.add_canvas(scene_id=scene.id)
    system = _store(controller, visual).data_coordinate_systems[0]
    along_x = _clip(controller, visual, 4.0)
    # Kept where z >= 3.
    along_z = ClippingPlane.from_point_normal(
        system, (3.0, 0, 0), (1, 0, 0), outline=BLUE
    )
    controller.set_clipping_planes(visual.id, (along_x, along_z))
    drawn = _drawn(controller, scene)
    assert _same(
        drawn[("clipping", along_x.id, 0)].vertices,
        _x_face(4.0, low=(-0.5, 3.0), high=(11.5, 7.5)),
    )
    assert _same(
        drawn[("clipping", along_z.id, 0)].vertices,
        [(4.0, -0.5, 3.0), (15.5, -0.5, 3.0), (15.5, 11.5, 3.0), (4.0, 11.5, 3.0)],
    )
    # One of them without an outline still ends the other.
    controller.set_clipping_planes(
        visual.id, (along_x, along_z.model_copy(update={"outline": PlaneOutline()}))
    )
    drawn = _drawn(controller, scene)
    assert list(drawn) == [("clipping", along_x.id, 0)]
    assert drawn[("clipping", along_x.id, 0)].vertices[:, 2].min() == pytest.approx(3.0)


async def test_a_render_plane_and_a_clipping_plane_are_outlined_together(controller):
    """Plane mode: the render plane's outline ends at the cut, and the cut
    face has an outline of its own."""
    scene = controller.add_scene(dim="3d", name="scene")
    world = h.world_of(controller, scene)
    render_plane = RenderPlane.from_point_normal(
        world, (3.0, 5.0, 7.0), (1, 0, 0), axes=("z", "y", "x"), outline=BLUE
    )
    visual = _image(
        controller, scene, render_mode="plane", render_planes=(render_plane,)
    )
    controller.add_canvas(scene_id=scene.id)
    clip = _clip(controller, visual, 4.0)
    controller.set_clipping_planes(visual.id, (clip,))

    drawn = _drawn(controller, scene)
    assert list(drawn) == [("render", render_plane.id, 0), ("clipping", clip.id, 0)]
    assert _same(
        drawn[("render", render_plane.id, 0)].vertices,
        [(4.0, -0.5, 3.0), (15.5, -0.5, 3.0), (15.5, 11.5, 3.0), (4.0, 11.5, 3.0)],
    )
    assert _same(drawn[("clipping", clip.id, 0)].vertices, _x_face(4.0))


async def test_a_clipping_plane_shared_by_two_visuals_has_one_outline(controller):
    """What the ortho linker does: the same tuple, the same ids."""
    scene = controller.add_scene(dim="3d", name="scene")
    first = _image(controller, scene)
    second = _image(controller, scene)
    small = _image(controller, scene, shape=(8, 6, 16))
    controller.add_canvas(scene_id=scene.id)
    plane = _clip(controller, first, 4.0)
    for visual in (first, second):
        # Each store has a coordinate system of its own; the id is shared.
        same = _clip(controller, visual, 4.0).model_copy(update={"id": plane.id})
        controller.set_clipping_planes(visual.id, (same,))
    assert list(_drawn(controller, scene)) == [("clipping", plane.id, 0)]

    # A third visual with another box: the face is another polygon.
    same_plane = _clip(controller, small, 4.0).model_copy(update={"id": plane.id})
    controller.set_clipping_planes(small.id, (same_plane,))
    drawn = _drawn(controller, scene)
    assert list(drawn) == [("clipping", plane.id, 0), ("clipping", plane.id, 1)]
    assert _same(
        drawn[("clipping", plane.id, 1)].vertices, _x_face(4.0, high=(5.5, 7.5))
    )


async def test_a_slider_moves_a_cut_across_a_sliced_axis(controller):
    """Kept where t + z >= 5, in the zyx view of a tzyx image: the face is
    at z = 5 - t, and moves when the time slider does."""
    scene = controller.add_scene(
        coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")],
        dim="3d",
        name="scene",
    )
    visual = _image(controller, scene, shape=(4, *SHAPE))
    controller.add_canvas(scene_id=scene.id)
    system = _store(controller, visual).data_coordinate_systems[0]
    plane = ClippingPlane.from_point_normal(
        system, (5, 0, 0, 0), (1, 1, 0, 0), outline=RED
    )
    controller.set_clipping_planes(visual.id, (plane,))

    for t, z in ((1, 4.0), (3, 2.0), (0, 5.0)):
        controller.update_slice_indices(scene.id, {0: t})
        await drain_loading(controller)
        (polygon,) = _drawn(controller, scene).values()
        assert _same(
            polygon.vertices,
            [(-0.5, -0.5, z), (15.5, -0.5, z), (15.5, 11.5, z), (-0.5, 11.5, z)],
        ), t


async def test_a_pyramid_s_cut_face_is_outlined(controller, tmp_path):
    """A multiscale image in a volume mode, before and after loading."""
    spec = rig_module.ISO
    rig = await rig_module.make_rig(
        controller,
        tmp_path,
        spec,
        lambda world, centre: [],
        single={"render_mode": "mip"},
        appearance={"force_level": 1},
    )
    plane = _clip(controller, rig.visual, 30.0)
    controller.set_clipping_planes(rig.visual.id, (plane,))
    z, y, _x = spec.shape0
    expected = _x_face(30.0, high=(y - 0.5, z - 0.5))
    assert _same(
        _drawn(controller, rig.scene)[("clipping", plane.id, 0)].vertices, expected
    )
    await rig.settle()
    assert _same(
        _drawn(controller, rig.scene)[("clipping", plane.id, 0)].vertices, expected
    )


# -- geometry ----------------------------------------------------------------------


def _geometry_visual(controller, scene, kind):
    if kind == "points":
        return controller.add_points(
            data=PointsMemoryStore(positions=CORNERS), scene_id=scene.id
        )
    if kind == "lines":
        return controller.add_lines(
            data=LinesMemoryStore(positions=CORNERS), scene_id=scene.id
        )
    return controller.add_graph(
        data=GraphMemoryStore(
            positions=CORNERS, edges=np.stack([np.arange(7), np.arange(1, 8)], axis=1)
        ),
        scene_id=scene.id,
    )


@pytest.mark.parametrize("kind", ["points", "lines", "graph"])
async def test_a_geometry_visual_s_cut_face_is_cut_by_the_box_of_its_vertices(
    controller, kind
):
    """No shape, so the box is that of the vertices drawn: from the origin
    to (16, 12, 8) in rendered (x, y, z)."""
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _geometry_visual(controller, scene, kind)
    controller.add_canvas(scene_id=scene.id)
    plane = _clip(controller, visual, 4.0)
    controller.set_clipping_planes(visual.id, (plane,))
    await _loaded(controller, scene)

    (polygon,) = _drawn(controller, scene).values()
    assert _same(polygon.vertices, _x_face(4.0, low=(0.0, 0.0), high=(12.0, 8.0)))


async def test_the_cut_face_follows_the_points_that_arrive(controller):
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _geometry_visual(controller, scene, "points")
    controller.add_canvas(scene_id=scene.id)
    controller.set_clipping_planes(visual.id, (_clip(controller, visual, 4.0),))
    await _loaded(controller, scene)

    # Twice as far along y.
    _store(controller, visual).positions = CORNERS * np.array(
        [1.0, 2.0, 1.0], dtype=np.float32
    )
    await drain_loading(controller)
    (polygon,) = _drawn(controller, scene).values()
    assert _same(polygon.vertices, _x_face(4.0, low=(0.0, 0.0), high=(24.0, 8.0)))


async def test_the_cut_face_follows_a_geometry_visual_s_transform(controller):
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _geometry_visual(controller, scene, "points")
    controller.add_canvas(scene_id=scene.id)
    controller.set_clipping_planes(visual.id, (_clip(controller, visual, 4.0),))
    await _loaded(controller, scene)

    # World x is twice data x, then shifted by 10: the cut at data x = 4
    # is at world x = 18.
    visual.transform = h.scale_translation_transform(
        controller,
        scene.id,
        _store(controller, visual),
        (1.0, 1.0, 2.0),
        (0.0, 0.0, 10.0),
    )
    await drain_loading(controller)
    (polygon,) = _drawn(controller, scene).values()
    assert _same(polygon.vertices, _x_face(18.0, low=(0.0, 0.0), high=(12.0, 8.0)))


async def test_a_mesh_s_cut_face_is_cut_by_the_store_s_extent(controller):
    """A sphere of radius 6 about (8, 8, 8): the box is 2 to 14 on each axis,
    as the mesh's bounding-box wireframe is."""
    positions, indices = uv_sphere(
        radius=6.0, centre=(8.0, 8.0, 8.0), n_lat=8, n_lon=16
    )
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_mesh(
        data=MeshMemoryStore(positions=positions, indices=indices),
        scene_id=scene.id,
        appearance=MeshFlatAppearance(),
    )
    controller.add_canvas(scene_id=scene.id)
    controller.set_clipping_planes(visual.id, (_clip(controller, visual, 8.0),))
    await _loaded(controller, scene)
    (polygon,) = _drawn(controller, scene).values()
    assert _same(polygon.vertices, _x_face(8.0, low=(2.0, 2.0), high=(14.0, 14.0)))


async def test_geometry_clipped_by_the_read_has_no_outline(controller):
    """A plane across a sliced axis cuts points as they are read (they may
    be flattened along that axis): there is no one cut face."""
    scene = controller.add_scene(
        coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")],
        dim="3d",
        name="scene",
    )
    positions = np.hstack([np.zeros((8, 1), dtype=np.float32), CORNERS])
    visual = controller.add_points(
        data=PointsMemoryStore(positions=positions), scene_id=scene.id
    )
    controller.add_canvas(scene_id=scene.id)
    system = _store(controller, visual).data_coordinate_systems[0]
    across_time = ClippingPlane.from_point_normal(
        system, (-1, 0, 0, 4.0), (1, 0, 0, 1), outline=RED
    )
    controller.set_clipping_planes(visual.id, (across_time,))
    await _loaded(controller, scene)
    assert _drawn(controller, scene) == {}

    # A plane on the displayed axes alone is outlined as usual.
    controller.set_clipping_planes(visual.id, (_clip(controller, visual, 4.0),))
    await drain_loading(controller)
    assert len(_drawn(controller, scene)) == 1


# -- on screen ---------------------------------------------------------------------

SIZE = (320, 320)
OBLIQUE = (0.6, -0.5, 1.0)


async def test_the_cut_face_is_where_a_plane_drawn_there_ends(controller):
    """Agreement with the shader: a render plane is drawn a hair inside an
    oblique clipping plane, with a second clipping plane across both.  The
    pixels it is drawn on are the clipping plane's polygon."""
    shape = (20, 28, 36)
    scene = controller.add_scene(dim="3d", name="scene")
    world = h.world_of(controller, scene)
    centre = (np.array(shape) - 1) / 2 + 0.3
    normal = np.array(OBLIQUE) / np.linalg.norm(OBLIQUE)
    store = ImageMemoryStore(data=h.image_data(shape), name="image")
    visual = controller.add_image(
        data=store,
        scene_id=scene.id,
        single=InMemoryImageSingleAppearance(
            render_mode="plane", color_map="gray", clim=(0.0, 1.0)
        ),
        render_planes=(h.plane_zyx(world, centre + 1e-3 * normal, OBLIQUE),),
    )
    controller.add_canvas(scene_id=scene.id, canvas_size=SIZE)
    system = _store(controller, visual).data_coordinate_systems[0]
    # The render plane is on the kept side of this one.
    cut = ClippingPlane.from_point_normal(system, centre, OBLIQUE)
    other = ClippingPlane.from_point_normal(system, (6.0, 0.0, 20.0), (0.0, 0.2, -1.0))
    controller.set_clipping_planes(visual.id, (cut, other))
    await _loaded(controller, scene)
    shot = h.shoot(controller, scene, size=SIZE)

    controller.set_clipping_planes(
        visual.id, (cut.model_copy(update={"outline": RED}), other)
    )
    polygon = _drawn(controller, scene)[("clipping", cut.id, 0)].vertices
    distance = px.edge_distance(polygon, shot.camera, SIZE)
    inside = distance > 0.0
    differ = inside != shot.drawn
    assert inside.sum() > 5000
    # A few more than for a render plane's own outline: the plane drawn is
    # a thousandth of a voxel off the one outlined.
    assert differ.sum() <= 6, int(differ.sum())
    assert not (np.abs(distance[differ]) > 0.5).any()


@pytest.mark.parametrize("render_mode", ["iso", "mip"])
async def test_the_line_is_whole_on_a_cut_face(controller, render_mode):
    """An isosurface writes depth on its cut face, where the line is; a
    maximum projection writes none.  Both halves of the line are drawn."""
    shape = (20, 28, 36)
    size = (480, 480)
    scene = controller.add_scene(dim="3d", name="scene")
    visual = _image(controller, scene, shape=shape, render_mode=render_mode)
    controller.add_canvas(scene_id=scene.id, canvas_size=size)
    system = _store(controller, visual).data_coordinate_systems[0]
    centre = (np.array(shape) - 1) / 2 + 0.3
    # The normal points away from the camera below: the cut face is seen.
    cut = ClippingPlane.from_point_normal(system, centre, OBLIQUE, outline=RED)
    controller.set_clipping_planes(visual.id, (cut,))
    await _loaded(controller, scene)
    view_dir = tuple(np.array(OBLIQUE[::-1]) + np.array([0.3, 0.2, 0.0]))
    shot = h.shoot(controller, scene, size=size, view_dir=view_dir)

    polygon = _drawn(controller, scene)[("clipping", cut.id, 0)].vertices
    shares = px.line_shares(
        shot.frame, px.edge_distance(polygon, shot.camera, size), RED.width
    )
    assert shot.drawn.sum() > 20000
    assert shares["pixels"] > 800, shares
    assert shares["inner"] > 0.97, shares
    assert shares["outer"] > 0.97, shares
    assert shares["stray"] == 0, shares
