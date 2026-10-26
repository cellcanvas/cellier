"""Oblique 2D design v3, Phase 3: multiscale labels on the brick route.

``plans/oblique_2d_raster_design_v3.md`` 5.8 and Phase 3.  A multiscale
labels visual whose transform tilts a 2D view's slice is drawn by its 3D
node in the plane render mode (the slice node), as multiscale images are
from Phase 2.  The labels rule: every drawn pixel shows the label of the
voxel under its centre, and no label that is not in the data.

Expectations are analytic (the transform, the camera) or properties of the
data -- never another library's rendering (memory: no cross-library tests).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
import pygfx as gfx
import pytest

from cellier.convenience import Viewer
from cellier.convenience.capture import _load_data
from cellier.data import ImageMemoryStore
from cellier.render import RenderManagerConfig
from cellier.render._config import OutlineConfig, OutlineLayerConfig
from cellier.scene.dims import spatial_axes
from cellier.transform import slice_footprint
from cellier.visuals import (
    InMemoryImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
    VisualOutline,
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
    new_tzyx_viewer,
    new_viewer,
    render,
    rot,
    to_world,
    world_matrix,
)
from tests.render._pick import PickFrame
from tests.render.test_oblique_2d_brick_route import (
    _expected_voxels,
    _overlap_share,
    _pixel_world,
)

if TYPE_CHECKING:
    from pathlib import Path

#: Label ids four apart, so a drawn grey names its label unambiguously.
_IDS = np.arange(4, 244, 4)


def _random_labels(seed: int = 0, shape=SHAPE) -> np.ndarray:
    """A label from ``_IDS`` per voxel: a wrong voxel shows as a wrong id."""
    rng = np.random.default_rng(seed)
    return rng.choice(_IDS, size=shape).astype(np.int32)


def _grey_colours() -> dict[int, tuple[float, float, float, float]]:
    """A grey per id whose drawn (sRGB-encoded) value is the id."""

    def linear(value: float) -> float:
        value = value / 255.0
        if value <= 0.04045:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    return {int(i): (linear(i),) * 3 + (1.0,) for i in _IDS}


def _add_tilted_labels(
    viewer: Viewer,
    tmp: Path,
    matrix: np.ndarray,
    data: np.ndarray | None = None,
    *,
    name: str = "tilted",
    render_order: int = 0,
    outline: VisualOutline | None = None,
    outline_mode: str = "per_label",
):
    data = _random_labels() if data is None else data
    store = multiscale_store(tmp, name, data)
    return viewer.add_labels_multiscale(
        store,
        appearance=MultiscaleLabelsAppearance(
            force_level=0,
            colormap_mode="direct",
            color_dict=_grey_colours(),
            render_order=render_order,
        ),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=16),
        transform=bind(viewer, store, matrix),
        outline=outline,
        outline_mode=outline_mode,
    )


def _gfx(viewer: Viewer, visual):
    return viewer.controller._render_manager._scenes[viewer.scene.id].get_visual(
        visual.id
    )


def _active_node(viewer: Viewer, visual):
    manager = viewer.controller._render_manager._scenes[viewer.scene.id]
    return manager.get_active_node(visual.id)


def _reload(viewer: Viewer) -> None:
    asyncio.run(_load_data(viewer))


def _slice_at(matrix: np.ndarray, fraction: float) -> float:
    centre = to_world(matrix, (np.asarray(SHAPE, float) - 1) / 2)
    corner = to_world(matrix, np.zeros(3))
    return float(corner[0] + fraction * 2.0 * (centre[0] - corner[0]))


def _expected_labels(camera, matrix, slice_z, data):
    voxel, valid = _expected_voxels(camera, matrix, slice_z, data.shape)
    index = tuple(
        np.clip(voxel[..., axis], 0, data.shape[axis] - 1) for axis in range(3)
    )
    return data[index], valid


# ---------------------------------------------------------------------------
# Placement, landmarks and the labels rule (D11)
# ---------------------------------------------------------------------------


def test_a_tilted_labels_view_is_placed_on_the_slice_node(tmp_path):
    viewer = new_viewer("2d")
    try:
        visual = _add_tilted_labels(viewer, tmp_path, world_matrix(LINEARS["rot_zx"]))
        load(viewer, slice_z=float(P0[0]))
        gfx_visual = _gfx(viewer, visual)
        assert gfx_visual.slice_axes == (0, 1, 2)
        assert _active_node(viewer, visual) is gfx_visual.slice_node
        assert gfx_visual.material_3d.render_mode == "plane"
        assert gfx_visual.material_3d.slice_2d
        assert viewer.controller.draw_status(visual.id).drawable
        # The model's render mode is untouched (D5).
        assert visual.appearance.render_mode == "iso_categorical"
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("name", ["rot_zx", "general"])
def test_a_tilted_labels_landmark_lands_where_the_transform_puts_it(tmp_path, name):
    assert landmark_error("labels_multiscale", name, "2d", tmp_path) < LANDMARK_PX


@pytest.mark.parametrize("name", ["rot_zx", "general"])
@pytest.mark.parametrize("fraction", [0.3, 0.7])
def test_every_pixel_shows_the_label_of_its_voxel(tmp_path, name, fraction):
    """The labels rule: 0 wrong pixels and no label that is not in the data."""
    matrix = world_matrix(LINEARS[name])
    data = _random_labels(1)
    viewer = new_viewer("2d")
    try:
        _add_tilted_labels(viewer, tmp_path, matrix, data)
        slice_z = _slice_at(matrix, fraction)
        load(viewer, slice_z=slice_z)
        frame, _, camera = render(viewer)
        expected, valid = _expected_labels(camera, matrix, slice_z, data)
        assert np.count_nonzero(valid) > 10_000
        grey = frame[..., 0].astype(int)
        drawn = frame[..., :3].max(axis=-1) > 0
        assert np.count_nonzero(np.abs(grey - expected)[valid] > 1) == 0
        # No foreign id anywhere, ties and edges included.
        nearest = np.abs(grey[drawn][:, None] - _IDS[None, :]).min(axis=1)
        assert np.all(nearest <= 1)
    finally:
        viewer.controller.close()


def test_labels_move_to_and_from_the_route_with_the_transform(tmp_path):
    viewer = new_viewer("2d")
    try:
        visual = _add_tilted_labels(viewer, tmp_path, world_matrix(np.eye(3)))
        load(viewer, slice_z=float(P0[0]))
        gfx_visual = _gfx(viewer, visual)
        assert gfx_visual.slice_axes is None
        assert _active_node(viewer, visual) is gfx_visual.node_2d
        store = viewer.controller._model.data.stores[UUID(visual.data_store_id)]

        async def _assign(matrix) -> None:
            visual.transform = bind(viewer, store, matrix)
            await _load_data(viewer)

        asyncio.run(_assign(world_matrix(LINEARS["rot_zx"])))
        assert gfx_visual.slice_axes == (0, 1, 2)
        assert _active_node(viewer, visual) is gfx_visual.slice_node
        assert render(viewer)[0][..., :3].max() > 0
        asyncio.run(_assign(world_matrix(np.eye(3))))
        assert gfx_visual.slice_axes is None
        assert _active_node(viewer, visual) is gfx_visual.node_2d
        # Back on today's path the 3D material is in the model's mode again.
        assert gfx_visual.material_3d.render_mode == "iso_categorical"
        assert not gfx_visual.material_3d.slice_2d
        assert render(viewer)[0][..., :3].max() > 0
    finally:
        viewer.controller.close()


def test_a_2d_only_scene_gains_the_3d_mode_for_labels(tmp_path):
    viewer = Viewer(
        spatial_axes("z", "y", "x"),
        displayed_axes=("y", "x"),
        render_modes=("2d",),
        gui="offscreen",
    )
    try:
        visual = _add_tilted_labels(viewer, tmp_path, world_matrix(LINEARS["rot_zx"]))
        load(viewer, slice_z=float(P0[0]))
        gfx_visual = _gfx(viewer, visual)
        assert "3d" in gfx_visual.render_modes
        assert _active_node(viewer, visual) is gfx_visual.slice_node
        assert render(viewer)[0][..., :3].max() > 0
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("fraction", [0.02, 0.98])
def test_the_2d_camera_never_clips_the_labels_slice_node(tmp_path, fraction):
    """K26 for labels: a narrow explicit depth range still draws it all."""
    matrix = world_matrix(rot(0, 2, 30.0))
    data = np.tile(_random_labels(2), (12, 1, 1))[:256]
    viewer = new_viewer("2d")
    try:
        _add_tilted_labels(viewer, tmp_path, matrix, data)
        corner = to_world(matrix, np.zeros(3))
        far = to_world(matrix, np.array([255.0, 0.0, 0.0]))
        slice_z = float(corner[0] + fraction * (far[0] - corner[0]))
        load(viewer, slice_z=slice_z)
        canvas_id = viewer.controller.get_canvas_ids(viewer.scene.id)[0]
        viewer.controller.set_camera_depth_range(canvas_id, (-20.0, 20.0))
        _reload(viewer)
        frame, _, camera = render(viewer)
        expected, valid = _expected_labels(camera, matrix, slice_z, data)
        assert np.count_nonzero(valid) > 10_000
        assert np.all(frame[..., :3].max(axis=-1)[valid] > 0)
        grey = frame[..., 0].astype(int)
        assert np.count_nonzero(np.abs(grey - expected)[valid] > 1) == 0
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Outlines with the camera never fitted (5.8, 0c P-f)
# ---------------------------------------------------------------------------

_BOUNDARY = (0.0, 1.0, 0.0, 1.0)
_SLOT1 = (1.0, 0.0, 1.0, 1.0)


def _outline_config() -> RenderManagerConfig:
    return RenderManagerConfig(
        outline=OutlineConfig(
            enabled=True,
            boundaries=OutlineLayerConfig(
                enabled=True,
                inward_thickness=1,
                outward_thickness=0,
                color=_BOUNDARY,
            ),
            selection=OutlineLayerConfig(
                enabled=True, inward_thickness=1, outward_thickness=0
            ),
            inner_thickness=0,
            palette=[_SLOT1],
        )
    )


def _blocky_labels() -> np.ndarray:
    """Labels in 4-voxel blocks: boundaries a pixel test can find."""
    rng = np.random.default_rng(3)
    coarse = rng.choice(_IDS, size=tuple(-(-n // 4) for n in SHAPE))
    blocks = np.kron(coarse, np.ones((4, 4, 4), dtype=np.int64))
    return blocks[: SHAPE[0], : SHAPE[1], : SHAPE[2]].astype(np.int32)


def _draw_outlined(viewer: Viewer, size: int = 320) -> tuple[np.ndarray, object]:
    """Draw the live scene through an outline pass, with the view's camera."""
    from pygfx.renderers.wgpu.engine.effectpasses import PPAAPass
    from rendercanvas.offscreen import RenderCanvas

    from cellier.render._cellier_blender import (
        OUTLINE_ID_TARGET,
        install_cellier_blender,
    )
    from cellier.render._outline import OutlinePass
    from cellier.render._pick_buffer import enable_pick_texture_binding
    from cellier.render._visual_lut import get_shared_visual_lut

    controller = viewer.controller
    controller.update_background_field(viewer.scene.id, "visible", False)
    view = canvas_view(viewer)
    camera = view.camera
    # What an unfitted camera has: pygfx's automatic +-100 depth range.
    camera.depth = 1.0
    gfx_scene = controller._render_manager.get_scene(viewer.scene.id)
    canvas = RenderCanvas(size=(size, size), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    install_cellier_blender(renderer, [OUTLINE_ID_TARGET])
    enable_pick_texture_binding(renderer)
    outline = OutlinePass(renderer, get_shared_visual_lut(), lambda: camera)
    outline.apply_config(controller.render_config.outline)
    outline.set_placements(has_inward=True, has_outward=False)
    outline.set_depth_edges(False)
    renderer.effect_passes = (
        outline,
        *(p for p in renderer.effect_passes if not isinstance(p, PPAAPass)),
    )
    canvas.request_draw(lambda: renderer.render(gfx_scene, camera))
    return np.asarray(canvas.draw()), camera


def _on_a_boundary(mask: np.ndarray, labels: np.ndarray, radius: int = 2) -> bool:
    """Whether every pixel of *mask* has a different label within *radius*."""
    padded = np.pad(labels, radius, mode="edge")
    differs = np.zeros(labels.shape, dtype=bool)
    height, width = labels.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = padded[
                radius + dy : radius + dy + height, radius + dx : radius + dx + width
            ]
            differs |= shifted != labels
    return bool(np.all(differs[mask]))


def _outline_viewer(tmp, mode: str):
    viewer = Viewer(
        spatial_axes("z", "y", "x"),
        displayed_axes=("y", "x"),
        render_config=_outline_config(),
        gui="offscreen",
    )
    matrix = world_matrix(LINEARS["rot_zx"])
    data = _blocky_labels()
    visual = _add_tilted_labels(
        viewer,
        tmp,
        matrix,
        data,
        outline=VisualOutline(slot=1, placement="inward"),
        outline_mode=mode,
    )
    slice_z = _slice_at(matrix, 0.5)
    load(viewer, slice_z=slice_z)
    return viewer, visual, matrix, data, slice_z


@pytest.mark.parametrize("mode", ["all_boundaries", "per_label", "depth_threshold"])
def test_every_outline_pixel_lies_on_a_label_boundary(tmp_path, mode):
    viewer, _visual, matrix, data, slice_z = _outline_viewer(tmp_path, mode)
    try:
        frame, camera = _draw_outlined(viewer)
        colour = _BOUNDARY if mode == "per_label" else _SLOT1
        rgb = np.round(np.asarray(colour[:3]) * 255).astype(int)
        lines = np.all(np.abs(frame[..., :3].astype(int) - rgb) <= 1, axis=-1)
        assert np.count_nonzero(lines) > 200
        size = frame.shape[0]
        xy = _pixel_world(camera, size)
        world = np.stack([np.full(xy.shape[:2], slice_z), xy[..., 1], xy[..., 0]], -1)
        linear, translation = matrix[:3, :3], matrix[:3, 3]
        voxel = np.floor((world - translation) @ np.linalg.inv(linear).T + 0.5).astype(
            int
        )
        inside = np.all((voxel >= 0) & (voxel < np.asarray(SHAPE)), axis=-1)
        index = tuple(np.clip(voxel[..., a], 0, SHAPE[a] - 1) for a in range(3))
        labels = np.where(inside, data[index], 0)
        assert _on_a_boundary(lines, labels)
    finally:
        viewer.controller.close()


def test_depth_threshold_draws_as_all_boundaries_on_the_slice_node(tmp_path):
    """5.8: the slice node is one plane; no depth range is sent for it."""
    viewer, visual, *_ = _outline_viewer(tmp_path, "all_boundaries")
    try:
        expected, _ = _draw_outlined(viewer)
        viewer.controller.update_visual_render_field(
            visual.id, "outline_mode", "depth_threshold"
        )
        assert viewer.controller._render_manager.get_visual_outline(visual.id)[3] == 0
        frame, _ = _draw_outlined(viewer)
        np.testing.assert_array_equal(frame, expected)
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Stacking (U9): labels on the brick node over today's image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("labels_on_top", [True, False])
def test_brick_labels_over_todays_image_follow_render_order(tmp_path, labels_on_top):
    matrix = world_matrix(LINEARS["rot_zx"])
    viewer = new_viewer("2d")
    try:
        labels = _add_tilted_labels(
            viewer,
            tmp_path,
            matrix,
            np.full(SHAPE, 200, np.int32),
            render_order=0 if labels_on_top else 1,
        )
        store = ImageMemoryStore(data=np.ones(SHAPE, np.float32), name="aligned")
        viewer.add_image(
            store,
            appearance=InMemoryImageAppearance(render_order=1 if labels_on_top else 0),
            single=InMemoryImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
            transform=bind(viewer, store, world_matrix(np.eye(3))),
        )
        slice_z = float(to_world(matrix, P0)[0])
        load(viewer, slice_z=slice_z)
        classification = viewer.controller.view_classification(labels.id)
        rendered = viewer.controller._scene_rendered(viewer.scene.id)
        selection = viewer.scene.dims.to_selection(*rendered).with_planes_at_slice(
            classification.sliced_world_axes
        )
        polygon = slice_footprint(SHAPE, classification, selection)[::-1, ::-1]
        camera = canvas_view(viewer).camera
        for move in ("start", "pan", "zoom"):
            if move == "pan":
                x, y, z = camera.local.position
                camera.local.position = (x + 3.3, y - 2.1, z)
            elif move == "zoom":
                camera.zoom = camera.zoom * 1.4
            frame, _, camera = render(viewer)
            # The grey labels read as "brick" (r == b) in the helper.
            assert _overlap_share(frame, camera, polygon, labels_on_top) > 0.99, move
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Picks (5.6) and statuses (5.2.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rot_zx", "general"])
def test_labels_picks_land_on_the_voxel_under_the_cursor(tmp_path, name):
    matrix = world_matrix(LINEARS[name])
    data = _random_labels(4)
    viewer = new_viewer("2d")
    try:
        visual = _add_tilted_labels(viewer, tmp_path, matrix, data)
        controller = viewer.controller
        manager = controller._render_manager
        rng = np.random.default_rng(8)
        for mark in rng.integers(6, np.asarray(SHAPE) - 6, size=(5, 3)):
            point = to_world(matrix, mark.astype(float))
            load(viewer, slice_z=float(point[0]))
            _, renderer, camera = render(viewer)
            ndc = camera.camera_matrix @ np.r_[point[::-1], 1.0]
            row, col = (1.0 - ndc[1]) / 2.0 * SIZE, (ndc[0] + 1.0) / 2.0 * SIZE
            info = PickFrame(renderer).info((float(col), float(row)))
            assert info["world_object"] is not None
            details = manager._extract_pick_details(
                viewer.scene.id, info["world_object"], info
            )
            coordinate = controller._promote_pick_coordinate(
                details.displayed_data_coord,
                displayed_axes=viewer.scene.dims.selection.displayed_axes,
                slice_indices={0: float(point[0])},
                world_ndim=3,
                hit_visual_id=visual.id,
                collapsed_data_indices=details.collapsed_data_indices,
            )
            assert np.all(np.abs(np.floor(coordinate) - mark) <= 1)
    finally:
        viewer.controller.close()


def test_a_labels_slice_through_four_data_axes_is_refused(tmp_path):
    viewer = new_tzyx_viewer("2d")
    try:
        data = np.stack([_random_labels(5)] * 4)
        store = _multiscale_series(tmp_path, "tzyx", data)
        matrix = np.eye(5)
        matrix[1:4, 1:4] = rot(0, 2, 30.0)
        matrix[0, 3] = 0.2
        visual = viewer.add_labels_multiscale(
            store,
            appearance=MultiscaleLabelsAppearance(),
            render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=16),
            transform=bind(viewer, store, matrix),
        )
        load(viewer)
        status = viewer.controller.draw_status(visual.id)
        assert not status.drawable
        assert "4 data axes" in status.reason
        assert _gfx(viewer, visual).slice_axes is None
    finally:
        viewer.controller.close()


def test_the_labels_bounds_are_the_footprint(tmp_path):
    matrix = world_matrix(LINEARS["general"])
    viewer = new_viewer("2d")
    try:
        visual = _add_tilted_labels(viewer, tmp_path, matrix)
        load(viewer, slice_z=float(to_world(matrix, P0)[0]))
        classification = viewer.controller.view_classification(visual.id)
        rendered = viewer.controller._scene_rendered(viewer.scene.id)
        selection = viewer.scene.dims.to_selection(*rendered).with_planes_at_slice(
            classification.sliced_world_axes
        )
        polygon = slice_footprint(SHAPE, classification, selection)
        bounds = _gfx(viewer, visual).slice_node.get_world_bounding_box()
        np.testing.assert_allclose(bounds[0, :2], polygon.min(axis=0)[::-1], atol=1e-9)
        np.testing.assert_allclose(bounds[1, :2], polygon.max(axis=0)[::-1], atol=1e-9)
        load(viewer, slice_z=500.0)
        assert _gfx(viewer, visual).slice_node.get_bounding_box() is None
    finally:
        viewer.controller.close()
