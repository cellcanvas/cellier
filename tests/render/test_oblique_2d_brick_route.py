"""Oblique 2D design v3, Phase 2: a multiscale image's tilted 2D slice.

``plans/oblique_2d_raster_design_v3.md`` 5.2-5.6.  A multiscale image whose
transform tilts a 2D view's slice is drawn on the brick route: its 3D node,
in the plane render mode, placed so the slice is rendered depth 0
(the slice node).  Every change here is a jump (D16).

Expectations are analytic (the transform, the camera) or properties of the
data -- never another library's rendering (memory: no cross-library tests).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.convenience.capture import _load_data
from cellier.data import ImageMemoryStore, PointsMemoryStore
from cellier.render import _render_planes
from cellier.render.visuals._chunked import viewport_prism
from cellier.scene.dims import spatial_axes
from cellier.transform import slice_footprint
from cellier.visuals import (
    InMemoryImageSingleAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    PointsMarkerAppearance,
)
from cellier.visuals._image_memory import InMemoryImageAppearance
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._affine import (
    LANDMARK_PX,
    LINEARS,
    P0,
    SHAPE,
    SIZE,
    _multiscale_series,
    bind,
    canvas_view,
    landmark_error,
    load,
    multiscale_store,
    new_viewer,
    render,
    rot,
    to_world,
    world_matrix,
)
from tests.render._pick import PickFrame

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_volume(seed: int = 0) -> np.ndarray:
    """A value per voxel in [16, 255]: a wrong voxel shows as a wrong value."""
    rng = np.random.default_rng(seed)
    return rng.integers(16, 256, size=SHAPE).astype(np.float32)


def _add_tilted(
    viewer: Viewer,
    tmp: Path,
    matrix: np.ndarray,
    data: np.ndarray | None = None,
    *,
    name: str = "tilted",
    clim: tuple[float, float] = (0.0, 255.0),
    render_order: int = 0,
    force_level: int | None = 0,
):
    data = _random_volume() if data is None else data
    store = multiscale_store(tmp, name, data)
    return viewer.add_image_multiscale(
        store,
        appearance=MultiscaleImageAppearance(
            force_level=force_level, interpolation="nearest", render_order=render_order
        ),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
        single=MultiscaleImageSingleAppearance(
            color_map="gray", clim=clim, render_mode="mip"
        ),
        transform=bind(viewer, store, matrix),
    )


def _gfx(viewer: Viewer, visual):
    return viewer.controller._render_manager._scenes[viewer.scene.id].get_visual(
        visual.id
    )


def _active_node(viewer: Viewer, visual):
    manager = viewer.controller._render_manager._scenes[viewer.scene.id]
    return manager.get_active_node(visual.id)


def _pixel_world(camera, size: int = SIZE) -> np.ndarray:
    """Pixel centres' pygfx ``(x, y)`` under an orthographic 2D camera."""
    cols, rows = np.meshgrid(np.arange(size), np.arange(size))
    ndc = np.stack(
        [
            (cols + 0.5) / size * 2.0 - 1.0,
            1.0 - (rows + 0.5) / size * 2.0,
            np.full(cols.shape, 0.5),
            np.ones(cols.shape),
        ],
        axis=-1,
    )
    inverse = np.linalg.inv(np.asarray(camera.camera_matrix, dtype=np.float64))
    world = ndc @ inverse.T
    return world[..., :2] / world[..., 3:4]


def _expected_voxels(camera, matrix: np.ndarray, slice_z: float, shape=SHAPE):
    """Order-0 expectation: each pixel centre's voxel, and a validity mask.

    Pixels whose data point is within 0.02 voxel of a voxel boundary are
    ties (D11), and pixels outside the data have nothing to compare.
    """
    xy = _pixel_world(camera)
    world = np.stack([np.full(xy.shape[:2], slice_z), xy[..., 1], xy[..., 0]], -1)
    linear, translation = matrix[:3, :3], matrix[:3, 3]
    data = (world - translation) @ np.linalg.inv(linear).T
    fraction = data - np.floor(data)
    tie = np.any(np.abs(fraction - 0.5) < 0.02, axis=-1)
    voxel = np.floor(data + 0.5).astype(int)
    inside = np.all((voxel >= 0) & (voxel < np.asarray(shape)), axis=-1)
    return voxel, inside & ~tie


def _wrong_pixels(frame: np.ndarray, data: np.ndarray, voxel, valid) -> int:
    grey = frame[..., 0].astype(np.float64)
    index = tuple(
        np.clip(voxel[..., axis], 0, data.shape[axis] - 1) for axis in range(3)
    )
    expected = data[index]
    return int(np.count_nonzero(np.abs(grey - expected)[valid] > 2.0))


def _run(viewer: Viewer, coroutine_fn) -> None:
    asyncio.run(coroutine_fn())


def _reload(viewer: Viewer) -> None:
    async def _go() -> None:
        await _load_data(viewer)

    asyncio.run(_go())


# ---------------------------------------------------------------------------
# Landmarks and the image rule (D11)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rot_zx", "general"])
def test_a_tilted_landmark_lands_where_the_transform_puts_it(tmp_path, name):
    assert landmark_error("image_multiscale", name, "2d", tmp_path) < LANDMARK_PX


@pytest.mark.parametrize("name", ["rot_zx", "general"])
@pytest.mark.parametrize("fraction", [0.3, 0.5, 0.7])
def test_every_pixel_reads_its_own_voxel(tmp_path, name, fraction):
    """The image rule: 0 wrong pixels against the order-0 pixel-centre voxel."""
    matrix = world_matrix(LINEARS[name])
    data = _random_volume()
    viewer = new_viewer("2d")
    try:
        _add_tilted(viewer, tmp_path, matrix, data)
        centre = to_world(matrix, (np.asarray(SHAPE, float) - 1) / 2)
        corner = to_world(matrix, np.zeros(3))
        slice_z = float(corner[0] + fraction * 2.0 * (centre[0] - corner[0]))
        load(viewer, slice_z=slice_z)
        frame, _, camera = render(viewer)
        voxel, valid = _expected_voxels(camera, matrix, slice_z)
        assert np.count_nonzero(valid) > 10_000
        assert _wrong_pixels(frame, data, voxel, valid) == 0
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("fraction", [0.02, 0.5, 0.98])
@pytest.mark.parametrize("depth_range", [None, (-20.0, 20.0)])
def test_the_2d_camera_never_clips_the_slice_node(tmp_path, fraction, depth_range):
    """K26: the 2D camera's automatic range of an unfitted camera (+-100) or
    a narrow explicit one still draws the whole slice of a volume reaching
    far beyond it in depth: the proxy is flattened onto depth 0 (C1)."""
    matrix = world_matrix(rot(0, 2, 30.0))
    data = np.tile(_random_volume(1), (12, 1, 1))[:256]
    viewer = new_viewer("2d")
    try:
        _add_tilted(viewer, tmp_path, matrix, data)
        corner = to_world(matrix, np.zeros(3))
        far = to_world(matrix, np.array([255.0, 0.0, 0.0]))
        slice_z = float(corner[0] + fraction * (far[0] - corner[0]))
        load(viewer, slice_z=slice_z)
        canvas_id = viewer.controller.get_canvas_ids(viewer.scene.id)[0]
        viewer.controller.set_camera_depth_range(canvas_id, depth_range)
        _reload(viewer)
        if depth_range is None:
            # What an unfitted camera has: pygfx's automatic +-100 * depth,
            # with depth 1 (fitting raises it).  Set just before the draw:
            # a reslice puts the fitted camera state back.
            canvas_view(viewer).camera.depth = 1.0
        frame, _, camera = render(viewer)
        if depth_range is None:
            assert camera.far - camera.near == pytest.approx(200.0)
        voxel, valid = _expected_voxels(camera, matrix, slice_z, data.shape)
        assert np.count_nonzero(valid) > 10_000
        covered = frame[..., 0] > 0
        assert np.count_nonzero(covered & valid) == np.count_nonzero(valid)
        assert _wrong_pixels(frame, data, voxel, valid) == 0
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Placement and route switches (C3, D16)
# ---------------------------------------------------------------------------


def test_a_tilted_view_is_placed_on_the_slice_node(tmp_path):
    viewer = new_viewer("2d")
    try:
        visual = _add_tilted(viewer, tmp_path, world_matrix(LINEARS["rot_zx"]))
        load(viewer, slice_z=float(P0[0]))
        gfx_visual = _gfx(viewer, visual)
        assert gfx_visual.slice_axes == (0, 1, 2)
        assert _active_node(viewer, visual) is gfx_visual.node_3d
        spaces = gfx_visual._spaces
        # The slice node's rendered system: the normal, then the displayed axes.
        world = viewer.scene.dims.world_coordinate_system
        assert [axis.name for axis in spaces.rendered.axes] == [
            world.axes[a].name for a in (0, 1, 2)
        ]
        # The slice sits at rendered depth 0.
        origin = spaces.rendered_to_world.map_coordinates(np.zeros((1, 3)))[0]
        assert origin[0] == pytest.approx(float(P0[0]))
        assert viewer.controller.draw_status(visual.id).drawable
    finally:
        viewer.controller.close()


def test_a_transform_jump_switches_the_route_both_ways(tmp_path):
    viewer = new_viewer("2d")
    try:
        visual = _add_tilted(viewer, tmp_path, world_matrix(np.eye(3)))
        load(viewer, slice_z=float(P0[0]))
        gfx_visual = _gfx(viewer, visual)
        assert gfx_visual.slice_axes is None
        assert _active_node(viewer, visual) is gfx_visual.node_2d
        visual.transform = bind(
            viewer,
            viewer.controller._model.data.stores[
                __import__("uuid").UUID(visual.data_store_id)
            ],
            world_matrix(LINEARS["rot_zx"]),
        )
        _reload(viewer)
        assert gfx_visual.slice_axes == (0, 1, 2)
        assert _active_node(viewer, visual) is gfx_visual.node_3d
        frame, _, _ = render(viewer)
        assert frame[..., :3].max() > 0
        visual.transform = bind(
            viewer,
            viewer.controller._model.data.stores[
                __import__("uuid").UUID(visual.data_store_id)
            ],
            world_matrix(np.eye(3)),
        )
        _reload(viewer)
        assert gfx_visual.slice_axes is None
        assert _active_node(viewer, visual) is gfx_visual.node_2d
    finally:
        viewer.controller.close()


def test_a_displayed_axes_change_into_a_tilted_view(tmp_path):
    viewer = new_viewer("3d")
    try:
        visual = _add_tilted(viewer, tmp_path, world_matrix(LINEARS["general"]))
        load(viewer)
        gfx_visual = _gfx(viewer, visual)
        assert gfx_visual.slice_axes is None
        viewer.set_displayed_dimensions(("y", "x"))
        load(viewer, slice_z=float(to_world(world_matrix(LINEARS["general"]), P0)[0]))
        assert gfx_visual.slice_axes == (0, 1, 2)
        assert _active_node(viewer, visual) is gfx_visual.node_3d
        frame, _, _ = render(viewer)
        assert frame[..., :3].max() > 0
    finally:
        viewer.controller.close()


def test_a_2d_only_scene_gains_the_3d_mode_for_the_slice_node(tmp_path):
    viewer = Viewer(
        spatial_axes("z", "y", "x"),
        displayed_axes=("y", "x"),
        render_modes=("2d",),
        gui="offscreen",
    )
    try:
        visual = _add_tilted(viewer, tmp_path, world_matrix(LINEARS["rot_zx"]))
        load(viewer, slice_z=float(P0[0]))
        gfx_visual = _gfx(viewer, visual)
        assert "3d" in gfx_visual.render_modes
        assert _active_node(viewer, visual) is gfx_visual.node_3d
        frame, _, _ = render(viewer)
        assert frame[..., :3].max() > 0
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Stacking (U9, 5.10): distinct render_order, lower on top
# ---------------------------------------------------------------------------


def _overlap_share(frame, camera, polygon, top_is_brick: bool) -> float:
    """Share of the overlap's pixels that show the top visual's colour.

    The brick image is grey (r == b); the aligned image is viridis yellow
    (b far below r).
    """
    xy = _pixel_world(camera)
    inside_box = (
        (xy[..., 0] > 0.5)
        & (xy[..., 0] < SHAPE[2] - 1.5)
        & (xy[..., 1] > 0.5)
        & (xy[..., 1] < SHAPE[1] - 1.5)
    )
    inside_polygon = np.ones(xy.shape[:2], dtype=bool)
    pts = polygon
    for start, end in zip(pts, np.roll(pts, -1, axis=0), strict=True):
        edge = end - start
        cross = edge[0] * (xy[..., 1] - start[1]) - edge[1] * (xy[..., 0] - start[0])
        inside_polygon &= cross > 1.0 * np.linalg.norm(edge)
    both = inside_box & inside_polygon
    assert np.count_nonzero(both) > 1000
    rgb = frame[..., :3].astype(int)
    brick = np.abs(rgb[..., 0] - rgb[..., 2]) < 8
    shown = brick if top_is_brick else ~brick
    return float(np.count_nonzero(shown & both) / np.count_nonzero(both))


@pytest.mark.parametrize("brick_on_top", [True, False])
def test_the_lower_render_order_is_on_top_after_pan_and_zoom(tmp_path, brick_on_top):
    matrix = world_matrix(LINEARS["rot_zx"])
    viewer = new_viewer("2d")
    try:
        brick = _add_tilted(
            viewer,
            tmp_path,
            matrix,
            np.full(SHAPE, 200.0, np.float32),
            render_order=0 if brick_on_top else 1,
        )
        store = ImageMemoryStore(data=np.ones(SHAPE, np.float32), name="aligned")
        viewer.add_image(
            store,
            appearance=InMemoryImageAppearance(render_order=1 if brick_on_top else 0),
            single=InMemoryImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
            transform=bind(viewer, store, world_matrix(np.eye(3))),
        )
        slice_z = float(to_world(matrix, P0)[0])
        load(viewer, slice_z=slice_z)
        classification = viewer.controller.view_classification(brick.id)
        scene = viewer.scene
        rendered = viewer.controller._scene_rendered(scene.id)
        selection = scene.dims.to_selection(*rendered).with_planes_at_slice(
            classification.sliced_world_axes
        )
        # pygfx (x, y) order; the swap reverses the winding, so reverse it
        # back to counter-clockwise.
        polygon = slice_footprint(SHAPE, classification, selection)[::-1, ::-1]
        camera = canvas_view(viewer).camera
        for move in ("start", "pan", "zoom"):
            if move == "pan":
                x, y, z = camera.local.position
                camera.local.position = (x + 3.3, y - 2.1, z)
            elif move == "zoom":
                camera.zoom = camera.zoom * 1.4
            frame, _, camera = render(viewer)
            assert _overlap_share(frame, camera, polygon, brick_on_top) > 0.99, move
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("points_on_top", [True, False])
def test_points_over_a_brick_image_follow_render_order(tmp_path, points_on_top):
    matrix = world_matrix(LINEARS["rot_zx"])
    viewer = new_viewer("2d")
    try:
        _add_tilted(
            viewer,
            tmp_path,
            matrix,
            np.full(SHAPE, 200.0, np.float32),
            render_order=1 if points_on_top else 0,
        )
        slice_z = float(to_world(matrix, P0)[0])
        point = to_world(matrix, P0)
        store = PointsMemoryStore(positions=point[None].astype(np.float32), name="p")
        viewer.add_points(
            store,
            appearance=PointsMarkerAppearance(
                size=15.0, color=(1, 0, 0, 1), render_order=0 if points_on_top else 1
            ),
            transform=bind(viewer, store, world_matrix(np.eye(3))),
        )
        # A point is selected by a slab (the image still draws one plane).
        load(viewer, slice_z=slice_z, half_thickness=0.6)
        frame, _, camera = render(viewer)
        xy = _pixel_world(camera)
        near = np.hypot(xy[..., 0] - point[2], xy[..., 1] - point[1]) < 0.15
        rgb = frame[..., :3].astype(int)
        red = (rgb[..., 0] > 150) & (rgb[..., 2] < 60)
        share = np.count_nonzero(red & near) / np.count_nonzero(near)
        assert (share > 0.99) if points_on_top else (share < 0.01)
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# The internal plane (D2, K34)
# ---------------------------------------------------------------------------


def test_a_scrub_never_rewrites_the_plane_and_leaves_the_model_alone(
    tmp_path, monkeypatch
):
    matrix = world_matrix(LINEARS["rot_zx"])
    viewer = new_viewer("2d")
    try:
        visual = _add_tilted(viewer, tmp_path, matrix)
        load(viewer, slice_z=float(P0[0]))
        before = visual.model_dump(mode="json")
        changed: list[bool] = []
        original = _render_planes.write_render_planes

        def spy(buffer, reduced):
            result = original(buffer, reduced)
            changed.append(result)
            return result

        monkeypatch.setattr(_render_planes, "write_render_planes", spy)
        events: list = []
        from uuid import uuid4

        from cellier.events import RenderPlanesChangedEvent

        viewer.controller._outgoing_events.subscribe(
            RenderPlanesChangedEvent, events.append, owner_id=uuid4()
        )
        centre = to_world(matrix, (np.asarray(SHAPE, float) - 1) / 2)
        for offset in (-4.0, -2.0, 0.0, 2.0, 4.0):
            load(viewer, slice_z=float(centre[0] + offset))
            frame, _, _ = render(viewer)
            assert frame[..., :3].max() > 0
        assert not any(changed)
        assert events == []
        assert visual.model_dump(mode="json") == before
        assert visual.render_planes == ()
    finally:
        viewer.controller.close()


def test_a_model_render_plane_does_not_reach_the_slice_node(tmp_path):
    from tests.render._planes import plane_zyx

    matrix = world_matrix(LINEARS["rot_zx"])
    data = _random_volume(2)
    viewer = new_viewer("2d")
    try:
        visual = _add_tilted(viewer, tmp_path, matrix, data)
        slice_z = float(to_world(matrix, P0)[0])
        load(viewer, slice_z=slice_z)
        gfx_visual = _gfx(viewer, visual)
        buffer = bytes(gfx_visual.render_planes_buffer.data.tobytes())
        world = viewer.scene.dims.world_coordinate_system
        visual.render_planes = (plane_zyx(world, (5.0, 10.0, 10.0), (0.0, 1.0, 1.0)),)
        visual.render_planes = (plane_zyx(world, (7.0, 12.0, 10.0), (1.0, 1.0, 0.0)),)
        _reload(viewer)
        assert bytes(gfx_visual.render_planes_buffer.data.tobytes()) == buffer
        assert gfx_visual.outline_box() is None
        frame, _, camera = render(viewer)
        voxel, valid = _expected_voxels(camera, matrix, slice_z)
        assert _wrong_pixels(frame, data, voxel, valid) == 0
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Planning (5.3.4, 5.3.5, D3, D4)
# ---------------------------------------------------------------------------


def test_the_viewport_prism_is_unbounded_in_depth(tmp_path):
    viewer = new_viewer("2d")
    try:
        _add_tilted(viewer, tmp_path, world_matrix(LINEARS["rot_zx"]))
        load(viewer, slice_z=float(P0[0]))
        view = canvas_view(viewer)
        request = view.capture_reslicing_request(
            viewer.controller._dims_state_for_scene(viewer.scene.id)
        )
        corners = viewport_prism(request)
        assert corners.shape == (2, 4, 3)
        assert np.all(corners[0, :, 2] >= 1e6)
        assert np.all(corners[1, :, 2] <= -1e6)
        width, height = request.world_extent
        np.testing.assert_allclose(np.ptp(corners[0, :, 0]), width)
        np.testing.assert_allclose(np.ptp(corners[0, :, 1]), height)
    finally:
        viewer.controller.close()


def test_every_backstop_brick_meets_the_plane(tmp_path):
    from cellier.render._plane_planning import plane_cull_masks

    viewer = new_viewer("2d")
    try:
        big = np.tile(_random_volume(6), (3, 8, 8))[:64, :256, :320]
        matrix = world_matrix(LINEARS["general"])
        visual = _add_tilted(viewer, tmp_path, matrix, big, force_level=None)
        load(viewer, slice_z=float(to_world(matrix, P0)[0]))
        gfx_visual = _gfx(viewer, visual)
        slot = gfx_visual.slots[0]
        planes = gfx_visual._plane_planning(slot, 0)
        view = canvas_view(viewer)
        request = view.capture_reslicing_request(
            viewer.controller._dims_state_for_scene(viewer.scene.id)
        )
        loading = visual.render_config.loading
        culled = slot._plan_backstop_3d(
            request.camera_pos, viewport_prism(request), loading, render_planes=planes
        )
        whole = slot._plan_backstop_3d(
            request.camera_pos, viewport_prism(request), loading
        )
        geo = slot._volume_geometry
        level = int(culled[0, 0])
        keep = geo._level_grids[level - 1]["arr"][
            plane_cull_masks(geo, planes, None)[level - 1]
        ]
        as_rows = {tuple(row) for row in keep.tolist()}
        assert {tuple(row) for row in culled.tolist()} <= as_rows
        assert 0 < len(culled) < len(whole)
    finally:
        viewer.controller.close()


def _target_levels(viewer: Viewer, visual) -> set[int]:
    """The levels a FULL plan's target asks for, from the desired keys."""
    from cellier.render.block_cache._image_residency import (
        unpack_keys,
    )
    from cellier.render.scheduling import ChunkClass, PlanMode

    controller = viewer.controller
    view = canvas_view(viewer)
    request = view.capture_reslicing_request(
        controller._dims_state_for_scene(viewer.scene.id),
        selection=controller._selections_for_scene(viewer.scene.id)[view.canvas_id],
    )
    config = controller._render_config_for(viewer.scene.id, visual)
    desired = _gfx(viewer, visual).plan(request, config, PlanMode.FULL)
    levels, _, _ = unpack_keys(desired[0].keys)
    return {int(level) for level in levels[desired[0].cls == ChunkClass.TARGET]}


@pytest.mark.parametrize("pixels_per_voxel", [2.0, 1.0, 0.5, 0.25])
def test_the_level_is_todays_2d_level_at_the_same_zoom(tmp_path, pixels_per_voxel):
    """K29: the plane's orthographic rule picks the aligned 2D level."""
    big = np.tile(_random_volume(3), (4, 8, 8))[:64, :256, :320]
    levels = {}
    for name in ("aligned", "tilted"):
        matrix = world_matrix(np.eye(3) if name == "aligned" else rot(0, 2, 20.0))
        viewer = new_viewer("2d")
        try:
            store = multiscale_store(tmp_path, f"{name}{pixels_per_voxel}", big)
            visual = viewer.add_image_multiscale(
                store,
                render_config=MultiscaleImageRenderConfig(
                    **SMALL_BUDGETS, block_size=16
                ),
                transform=bind(viewer, store, matrix),
            )
            load(viewer, slice_z=10.0)
            camera = canvas_view(viewer).camera
            camera.zoom = pixels_per_voxel * camera.width / SIZE
            levels[name] = _target_levels(viewer, visual)
        finally:
            viewer.controller.close()
    assert len(levels["aligned"]) == 1
    assert levels["tilted"] == levels["aligned"]


def test_a_2d_3d_toggle_keeps_the_atlas(tmp_path):
    viewer = new_viewer("2d")
    try:
        visual = _add_tilted(viewer, tmp_path, world_matrix(LINEARS["rot_zx"]))
        load(viewer, slice_z=float(P0[0]))
        slot = _gfx(viewer, visual).slots[0]
        geometry, cache = slot._volume_geometry, slot._block_cache_3d
        viewer.set_displayed_dimensions(("z", "y", "x"))
        load(viewer)
        viewer.set_displayed_dimensions(("y", "x"))
        load(viewer, slice_z=float(P0[0]))
        assert slot._volume_geometry is geometry
        assert slot._block_cache_3d is cache
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Picks (5.6, C4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rot_zx", "general"])
def test_picks_land_on_the_voxel_under_the_cursor(tmp_path, name):
    matrix = world_matrix(LINEARS[name])
    data = _random_volume(4)
    viewer = new_viewer("2d")
    try:
        visual = _add_tilted(viewer, tmp_path, matrix, data)
        controller = viewer.controller
        manager = controller._render_manager
        rng = np.random.default_rng(7)
        marks = rng.integers(6, np.asarray(SHAPE) - 6, size=(5, 3))
        for mark in marks:
            point = to_world(matrix, mark.astype(float))
            load(viewer, slice_z=float(point[0]))
            frame, renderer, camera = render(viewer)
            del frame
            row, col = (
                np.array(
                    [
                        (1.0 - (camera.camera_matrix @ np.r_[point[::-1], 1.0])[1])
                        / 2.0,
                        ((camera.camera_matrix @ np.r_[point[::-1], 1.0])[0] + 1.0)
                        / 2.0,
                    ]
                )
                * SIZE
            )
            info = PickFrame(renderer).info((float(col), float(row)))
            hit = info["world_object"]
            assert hit is not None
            details = manager._extract_pick_details(viewer.scene.id, hit, info)
            displayed = viewer.scene.dims.selection.displayed_axes
            coordinate = controller._promote_pick_coordinate(
                details.displayed_data_coord,
                displayed_axes=displayed,
                slice_indices={0: float(point[0])},
                world_ndim=3,
                hit_visual_id=visual.id,
                collapsed_data_indices=details.collapsed_data_indices,
            )
            assert np.all(np.abs(np.floor(coordinate) - mark) <= 1)
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Statuses, bounds and fit (5.2.5, 5.5)
# ---------------------------------------------------------------------------


def test_a_slice_through_four_data_axes_is_refused(tmp_path):
    """O19: world t reads data x and the slice is tilted in zx: ``|V| = 4``."""
    from tests.render._affine import new_tzyx_viewer

    viewer = new_tzyx_viewer("2d")
    try:
        data = np.stack([_random_volume(5)] * 4)
        store = _multiscale_series(tmp_path, "tzyx", data)
        matrix = np.eye(5)
        matrix[1:4, 1:4] = rot(0, 2, 30.0)
        matrix[0, 3] = 0.2
        visual = viewer.add_image_multiscale(
            store,
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
            transform=bind(viewer, store, matrix),
        )
        load(viewer)
        status = viewer.controller.draw_status(visual.id)
        assert not status.drawable
        assert "4 data axes" in status.reason
        assert _gfx(viewer, visual).slice_axes is None
    finally:
        viewer.controller.close()


def test_the_bounds_are_the_footprint(tmp_path):
    matrix = world_matrix(LINEARS["general"])
    viewer = new_viewer("2d")
    try:
        visual = _add_tilted(viewer, tmp_path, matrix)
        slice_z = float(to_world(matrix, P0)[0])
        load(viewer, slice_z=slice_z)
        classification = viewer.controller.view_classification(visual.id)
        rendered = viewer.controller._scene_rendered(viewer.scene.id)
        selection = viewer.scene.dims.to_selection(*rendered).with_planes_at_slice(
            classification.sliced_world_axes
        )
        polygon = slice_footprint(SHAPE, classification, selection)
        node = _gfx(viewer, visual).node_3d
        bounds = node.get_world_bounding_box()
        np.testing.assert_allclose(bounds[0, :2], polygon.min(axis=0)[::-1], atol=1e-9)
        np.testing.assert_allclose(bounds[1, :2], polygon.max(axis=0)[::-1], atol=1e-9)
        assert bounds[0, 2] == bounds[1, 2] == 0.0
        # The whole volume's box would be wider than the slice's footprint.
        assert np.ptp(polygon[:, 1]) < SHAPE[2] * 1.5
    finally:
        viewer.controller.close()


def test_the_3d_bounds_forget_the_footprint_after_the_route(tmp_path):
    """Off the route the 3D group is bounded by the volume alone.

    pygfx counts hidden children in a group's box, so a footprint line left
    in the group (at rendered depth 0) would stretch the 3D box to depth 0
    -- far from data placed away from the origin -- and skew the 3D fit.
    """
    offset = np.array([300.0, 200.0, 100.0])
    tilted = world_matrix(LINEARS["rot_zx"])
    tilted[:3, 3] += offset
    aligned = np.eye(4)
    aligned[:3, 3] = offset
    viewer = new_viewer("2d")
    try:
        visual = _add_tilted(viewer, tmp_path, tilted)
        load(viewer, slice_z=float(to_world(tilted, P0)[0]))
        gfx_visual = _gfx(viewer, visual)
        assert gfx_visual.slice_axes is not None
        assert gfx_visual._footprint_line is not None
        visual.transform = bind(viewer, visual_store(viewer, visual), aligned)
        assert gfx_visual.slice_axes is None
        viewer.set_displayed_dimensions(("z", "y", "x"))
        load(viewer)
        bounds = gfx_visual.node_3d.get_world_bounding_box()
        # pygfx (x, y, z) is world (x, y, z): the voxel-edge box of the data.
        low = offset[::-1] - 0.5
        high = offset[::-1] + np.asarray(SHAPE[::-1]) - 0.5
        np.testing.assert_allclose(bounds[0], low, atol=1e-3)
        np.testing.assert_allclose(bounds[1], high, atol=1e-3)
        # Back on the route, the footprint is drawn and bounds the node again.
        viewer.set_displayed_dimensions(("y", "x"))
        visual.transform = bind(viewer, visual_store(viewer, visual), tilted)
        load(viewer, slice_z=float(to_world(tilted, P0)[0]))
        assert gfx_visual.slice_axes is not None
        assert gfx_visual._footprint_line.parent is gfx_visual.node_3d
        assert gfx_visual.node_3d.get_world_bounding_box()[0, 2] == 0.0
    finally:
        viewer.controller.close()


def visual_store(viewer: Viewer, visual):
    """The data store *visual* reads."""
    from uuid import UUID

    return viewer.controller._model.data.stores[UUID(visual.data_store_id)]


def test_a_slice_outside_the_data_has_no_bounds(tmp_path):
    viewer = new_viewer("2d")
    try:
        visual = _add_tilted(viewer, tmp_path, world_matrix(LINEARS["rot_zx"]))
        load(viewer, slice_z=500.0)
        assert _gfx(viewer, visual).node_3d.get_bounding_box() is None
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Slices along a line (U13): nothing drawn, a status, never a crash
# ---------------------------------------------------------------------------


def _flat_store(tmp: Path, family: str):
    """A ``yx`` store of ones, in memory or as a two-level pyramid."""
    from cellier.data import (
        LabelMemoryStore,
        MultiscaleZarrDataStore,
    )
    from tests.render.conftest import _write_multiscale_zarr

    dtype = np.int32 if family.startswith("labels") else np.float32
    data = np.ones((32, 40), dtype=dtype)
    if family == "image":
        return ImageMemoryStore(data=data, name="flat")
    if family == "labels":
        return LabelMemoryStore(data=data, name="flat")
    root = tmp / family
    root.mkdir(parents=True, exist_ok=True)
    coarse = data[::2, ::2]

    def fill(arr):
        arr[...] = data if arr.shape == data.shape else coarse

    _write_multiscale_zarr(
        root,
        levels=[("s0", data.shape), ("s1", coarse.shape)],
        fill=fill,
        dtype=str(data.dtype),
    )
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0), (2.0, 2.0)],
        level_translations=[(0.0, 0.0), (0.5, 0.5)],
        name=family,
    )


def _add_flat(viewer: Viewer, family: str, store, z_reads_x: float):
    matrix = np.zeros((4, 3))
    matrix[0, 1] = z_reads_x
    matrix[1, 0] = matrix[2, 1] = matrix[3, 2] = 1.0
    transform = bind(viewer, store, matrix)
    if family == "image_multiscale":
        return viewer.add_image_multiscale(
            store,
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
            transform=transform,
        )
    if family == "labels_multiscale":
        from cellier.visuals import (
            MultiscaleLabelRenderConfig,
            MultiscaleLabelsAppearance,
        )

        return viewer.add_labels_multiscale(
            store,
            appearance=MultiscaleLabelsAppearance(),
            render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=16),
            transform=transform,
        )
    add = viewer.add_image if family == "image" else viewer.add_labels
    return add(store, transform=transform)


@pytest.mark.parametrize(
    "family", ["image", "labels", "image_multiscale", "labels_multiscale"]
)
@pytest.mark.parametrize(
    ("displayed", "z_reads_x"),
    [(("y", "x"), 0.5), (("z", "x"), 0.0)],
    ids=["sheared", "edge_on"],
)
def test_a_slice_along_a_line_draws_nothing_and_says_why(
    tmp_path, family, displayed, z_reads_x
):
    viewer = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=displayed, gui="offscreen"
    )
    try:
        visual = _add_flat(viewer, family, _flat_store(tmp_path, family), z_reads_x)
        # Adding the canvas used to raise in node_matrix (U13).
        load(viewer)
        frame, _, _ = render(viewer)
        assert frame[..., :3].max() == 0
        status = viewer.controller.draw_status(visual.id)
        assert not status.drawable
        assert "along a line" in status.reason
        # Shown in its own plane, the same store draws.
        store = viewer.controller._model.data.stores[
            __import__("uuid").UUID(visual.data_store_id)
        ]
        matrix = np.zeros((4, 3))
        matrix[1, 0] = matrix[2, 1] = matrix[3, 2] = 1.0

        async def _flatten() -> None:
            # Inside a loop: an in-memory visual reslices on assignment.
            viewer.set_displayed_dimensions(("y", "x"))
            visual.transform = bind(viewer, store, matrix)
            viewer.set_slice_positions({0: 0.0})
            await _load_data(viewer)

        asyncio.run(_flatten())
        assert viewer.controller.draw_status(visual.id).drawable
        frame, _, _ = render(viewer)
        assert frame[..., :3].max() > 0
    finally:
        viewer.controller.close()
