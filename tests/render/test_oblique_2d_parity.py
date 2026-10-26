"""Oblique 2D design v3, Phase 5: feature parity in tilted views (O17).

``plans/oblique_2d_raster_design_v3.md`` 5.7 (clipping), 5.9 (progressive
display), multichannel composite, and the orthoviewer under ``rot_zx`` and
``general`` with every panel drawing.

Expectations are analytic (the transform, the camera) or properties of the
data -- never another library's rendering (memory: no cross-library tests).
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import numpy as np
import pygfx as gfx
import pytest
from rendercanvas.offscreen import RenderCanvas

from cellier.convenience import OrthoViewer, Viewer
from cellier.convenience.capture import _load_data
from cellier.data import ImageMemoryStore
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.render.scheduling import ChunkClass
from cellier.scene import spatial_axes
from cellier.transform import AffineTransform
from cellier.visuals import (
    ClippingPlane,
    InMemoryImageChannelAppearance,
    InMemoryImageRenderConfig,
    InMemoryImageSingleAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageChannelAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
    ProgressiveLoadingConfig,
)
from cellier.visuals._image_memory import InMemoryImageAppearance
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._affine import (
    LINEARS,
    SHAPE,
    SIZE,
    bind,
    load,
    multiscale_store,
    new_viewer,
    render,
    rot,
    to_world,
    world_matrix,
)
from tests.render._pick import PickFrame
from tests.render.conftest import _write_multiscale_zarr
from tests.render.test_oblique_2d_brick_route import _pixel_world, _random_volume

_CZYX = [("c", "channel"), *spatial_axes("z", "y", "x")]


def _add_image(viewer, family, tmp_path, matrix, data, **appearance):
    if family == "image_multiscale":
        store = multiscale_store(tmp_path, "s", data)
        return viewer.add_image_multiscale(
            store,
            appearance=MultiscaleImageAppearance(
                force_level=0, interpolation="nearest", **appearance
            ),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
            single=MultiscaleImageSingleAppearance(
                color_map="gray", clim=(0.0, 255.0), render_mode="mip"
            ),
            transform=bind(viewer, store, matrix),
        )
    store = ImageMemoryStore(data=data, name="m")
    return viewer.add_image(
        store,
        appearance=InMemoryImageAppearance(interpolation="nearest", **appearance),
        single=InMemoryImageSingleAppearance(
            color_map="gray", clim=(0.0, 255.0), render_mode="mip"
        ),
        render_config=InMemoryImageRenderConfig(
            gpu_budget_bytes=64 * 2**20, block_size=16
        ),
        transform=bind(viewer, store, matrix),
    )


def _data_points(camera, matrix, slice_z, size=SIZE):
    """Each pixel centre's level-0 data point on the slice."""
    xy = _pixel_world(camera, size)
    world = np.stack([np.full(xy.shape[:2], slice_z), xy[..., 1], xy[..., 0]], -1)
    return (world - matrix[:3, 3]) @ np.linalg.inv(matrix[:3, :3]).T


# ---------------------------------------------------------------------------
# Clip lines (5.7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["image_multiscale", "image"])
def test_a_clipping_plane_cuts_the_tilted_slice_where_it_meets_it(tmp_path, family):
    """The kept side draws, the other side does not, with a landmark either
    side of the plane (data x = 20, keeping x >= 20)."""
    matrix = world_matrix(LINEARS["rot_zx"])
    data = np.full(SHAPE, 100.0, np.float32)
    viewer = new_viewer("2d")
    try:
        visual = _add_image(viewer, family, tmp_path, matrix, data)
        store = viewer.controller._model.data.stores[UUID(visual.data_store_id)]
        system = store.data_coordinate_systems[0]
        visual.clipping_planes = (
            ClippingPlane.from_point_normal(system, (0, 0, 20.0), (0, 0, 1.0)),
        )
        centre = (np.asarray(SHAPE, float) - 1) / 2
        slice_z = float(to_world(matrix, centre)[0])
        load(viewer, slice_z=slice_z)
        frame, _, camera = render(viewer)
        p = _data_points(camera, matrix, slice_z)
        inside = np.all((p > 0.5) & (p < np.asarray(SHAPE) - 1.5), axis=-1)
        drawn = frame[..., :3].max(axis=-1) > 0
        kept = inside & (p[..., 2] > 20.5)
        cut = inside & (p[..., 2] < 19.5)
        assert np.count_nonzero(kept) > 5000 and np.count_nonzero(cut) > 5000
        assert np.all(drawn[kept])
        assert not np.any(drawn[cut])
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Composite (O13 for plain slices)
# ---------------------------------------------------------------------------


def _czyx_store(tmp_path, data):
    root = tmp_path / "czyx"
    root.mkdir(parents=True, exist_ok=True)
    coarse = data[:, ::2, ::2, ::2]

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
        level_scales=[(1.0, 1.0, 1.0, 1.0), (1.0, 2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0, 0.0), (0.0, 0.5, 0.5, 0.5)],
        name="czyx",
    )


@pytest.mark.parametrize("family", ["image_multiscale", "image"])
def test_a_tilted_composite_is_each_channel_drawn_alone_blended(tmp_path, family):
    data = np.stack([_random_volume(20), _random_volume(21)]).astype(np.float32)
    viewer = Viewer(_CZYX, displayed_axes=("y", "x"), gui="offscreen")
    from cellier.scene._background import BackgroundAppearance

    viewer.background = BackgroundAppearance(mode="uniform", color=(0, 0, 0, 1))
    try:
        matrix = np.eye(5)
        matrix[1:4, 1:4] = rot(0, 2, 25.0)
        centre = (np.asarray(SHAPE, float) - 1) / 2
        matrix[1:4, 4] = centre - rot(0, 2, 25.0) @ centre
        if family == "image_multiscale":
            store = _czyx_store(tmp_path, data)
            channel_cls = MultiscaleImageChannelAppearance
            visual = viewer.add_image_multiscale(
                store,
                appearance=MultiscaleImageAppearance(
                    force_level=0, interpolation="nearest"
                ),
                render_config=MultiscaleImageRenderConfig(
                    **SMALL_BUDGETS, block_size=16
                ),
                channel_axis=0,
                composite=True,
                channels={
                    0: channel_cls(color_map="red", clim=(0.0, 255.0)),
                    1: channel_cls(color_map="green", clim=(0.0, 255.0)),
                },
                transform=bind(viewer, store, matrix),
            )
        else:
            store = ImageMemoryStore(data=data, name="czyx")
            channel_cls = InMemoryImageChannelAppearance
            visual = viewer.add_image(
                store,
                appearance=InMemoryImageAppearance(interpolation="nearest"),
                render_config=InMemoryImageRenderConfig(
                    gpu_budget_bytes=64 * 2**20, block_size=16
                ),
                channel_axis=0,
                composite=True,
                channels={
                    0: channel_cls(color_map="red", clim=(0.0, 255.0)),
                    1: channel_cls(color_map="green", clim=(0.0, 255.0)),
                },
                transform=bind(viewer, store, matrix),
            )
        slice_z = float(centre[0])
        gfx_visual = viewer.controller._render_manager._scenes[
            viewer.scene.id
        ].get_visual(visual.id)

        async def _load(channels_visible) -> None:
            for index, visible in channels_visible.items():
                visual.channels[index].visible = visible
            viewer.set_slice_positions({1: slice_z})
            await _load_data(viewer)

        from cellier.convenience.capture import _ensure_canvases

        _ensure_canvases(viewer, (SIZE, SIZE))
        frames = {}
        for name, vis in (
            ("both", {0: True, 1: True}),
            ("red", {0: True, 1: False}),
            ("green", {0: False, 1: True}),
        ):
            asyncio.run(_load(vis))
            assert gfx_visual.slice_axes is not None
            frames[name] = render(viewer)[0][..., :3].astype(int)
        both, red, green = frames["both"], frames["red"], frames["green"]
        assert red[..., 0].max() > 100 and green[..., 1].max() > 100
        assert np.abs(both - np.clip(red + green, 0, 255)).max() <= 2
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Progressive display (5.9): intermediate stages on the slice node
# ---------------------------------------------------------------------------


def test_intermediate_levels_are_planned_and_read_in_stage_order(tmp_path, monkeypatch):
    data = np.tile(_random_volume(22), (3, 4, 4))[:64, :128, :160]
    root = tmp_path / "p"
    root.mkdir()
    levels = [data, data[::2, ::2, ::2], data[::4, ::4, ::4]]
    by_shape = {level.shape: level for level in levels}

    def fill(arr):
        arr[...] = by_shape[arr.shape]

    _write_multiscale_zarr(
        root,
        levels=[(f"s{i}", level.shape) for i, level in enumerate(levels)],
        fill=fill,
        dtype="float32",
    )
    store = MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=["s0", "s1", "s2"],
        level_scales=[(1.0,) * 3, (2.0,) * 3, (4.0,) * 3],
        level_translations=[(0.0,) * 3, (0.5,) * 3, (1.5,) * 3],
        name="p",
    )
    reads: list[int] = []
    original = MultiscaleZarrDataStore.get_data

    async def spy(self, request):
        reads.append(request.scale_index)
        return await original(self, request)

    monkeypatch.setattr(MultiscaleZarrDataStore, "get_data", spy)
    viewer = new_viewer("2d")
    try:
        matrix = world_matrix(rot(0, 2, 20.0))
        visual = viewer.add_image_multiscale(
            store,
            appearance=MultiscaleImageAppearance(force_level=1),
            render_config=MultiscaleImageRenderConfig(
                **SMALL_BUDGETS,
                block_size=16,
                loading=ProgressiveLoadingConfig(intermediate_offsets=(1,)),
            ),
            transform=bind(viewer, store, matrix),
        )
        load(viewer, slice_z=30.0)
        gfx_visual = viewer.controller._render_manager._scenes[
            viewer.scene.id
        ].get_visual(visual.id)
        assert gfx_visual.slice_axes is not None
        from cellier.render.block_cache._image_residency import unpack_keys
        from cellier.render.scheduling._core import DEAD, Tier

        residency = gfx_visual.slots[0].residency_3d()
        registry = viewer.controller._render_manager.scheduler.core.registry(
            residency.cache_id
        )
        wanted = (registry.tier == Tier.VISIBLE) & (registry.state != DEAD)
        level = unpack_keys(registry.key[wanted])[0]
        classes = {
            int(cls): sorted(set(level[registry.cls[wanted] == cls].tolist()))
            for cls in np.unique(registry.cls[wanted])
        }
        assert classes == {
            int(ChunkClass.TARGET): [1],
            1: [2],
            int(ChunkClass.BACKSTOP): [3],
        }
        # Read coarsest first: backstop, then the stage, then the target.
        firsts = [reads.index(index) for index in (2, 1, 0)]
        assert firsts == sorted(firsts)
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# The orthoviewer under a tilt: every panel draws, statuses and picks
# ---------------------------------------------------------------------------


def _draw_panel(controller, scene_id, size=320):
    if not controller.get_canvas_ids(scene_id):
        controller.add_canvas(scene_id=scene_id, canvas_size=(size, size))
    canvas_id = controller.get_canvas_ids(scene_id)[0]
    view = controller._render_manager._canvases[canvas_id]
    controller.update_background_field(scene_id, "visible", False)
    scene = view._get_scene_fn(scene_id)
    canvas = RenderCanvas(size=(size, size), pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    canvas.request_draw(lambda: renderer.render(scene, view.camera))
    return np.asarray(canvas.draw()), renderer, view.camera


@pytest.mark.parametrize("name", ["rot_zx", "general"])
def test_the_orthoviewer_draws_every_panel_of_a_tilted_volume(tmp_path, name):
    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui="offscreen")
    try:
        data = _random_volume(23)
        store = multiscale_store(tmp_path, "s", data)
        labels_store = multiscale_store(
            tmp_path, "l", (data > 128).astype(np.int32) * 3
        )
        controller = ortho.controller
        scene = next(iter(ortho.scenes.values()))
        controller._ensure_data_coordinate_systems(scene.id, store)
        controller._ensure_data_coordinate_systems(scene.id, labels_store)
        world = scene.dims.world_coordinate_system
        matrix = world_matrix(LINEARS[name])
        images = ortho.add_image_multiscale(
            store,
            appearance=MultiscaleImageAppearance(force_level=0),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
            single=MultiscaleImageSingleAppearance(
                color_map="gray", clim=(0.0, 255.0), render_mode="mip"
            ),
            transform=AffineTransform.from_matrix(
                matrix, store.data_coordinate_systems[0], world
            ),
        )
        labels = ortho.add_labels_multiscale(
            labels_store,
            appearance=MultiscaleLabelsAppearance(force_level=0, render_order=-1),
            render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=16),
            transform=AffineTransform.from_matrix(
                matrix, labels_store.data_coordinate_systems[0], world
            ),
        )
        centre = to_world(matrix, (np.asarray(SHAPE, float) - 1) / 2)

        from cellier.convenience.capture import _ensure_canvases

        _ensure_canvases(ortho, (320, 320))

        async def _load() -> None:
            ortho.set_slice_positions(
                {0: float(centre[0]), 1: float(centre[1]), 2: float(centre[2])}
            )
            for scene_ in ortho.scenes.values():
                controller.fit_camera(scene_.id)
            controller.reslice_all()
            from tests.render.conftest import drain_loading

            await drain_loading(controller)

        asyncio.run(_load())
        for panel in ("xy", "xz", "yz", "vol"):
            assert controller.draw_status(images[panel].id).drawable, panel
            assert controller.draw_status(labels[panel].id).drawable, panel
        for panel in ("xy", "xz", "yz"):
            scene_id = controller._visual_to_scene[images[panel].id]
            frame, renderer, _camera = _draw_panel(controller, scene_id)
            assert frame[..., :3].max() > 0, panel
            # A pick at the frame centre hits one of the panel's visuals.
            info = PickFrame(renderer).info((160.0, 160.0))
            hit = info["world_object"]
            assert hit is not None, panel
            hit_id = controller._render_manager._scenes[
                scene_id
            ].get_visual_id_for_node(hit)
            assert hit_id in (images[panel].id, labels[panel].id), panel
    finally:
        ortho.controller.close()


def test_a_channel_axis_may_sit_beside_a_spatial_rotation():
    """A rotation of the spatial axes leaves the channel axis on its own world
    axis; coupling the channel axis to a spatial one is still refused."""
    viewer = Viewer(_CZYX, displayed_axes=("y", "x"), gui="offscreen")
    try:
        store = ImageMemoryStore(data=np.zeros((2, 8, 8, 8), np.float32), name="c")
        matrix = np.eye(5)
        matrix[1:4, 1:4] = rot(0, 2, 25.0)
        viewer.add_image(store, channel_axis=0, transform=bind(viewer, store, matrix))
        coupled = np.eye(5)
        coupled[3, 0] = 0.5  # world y reads the channel axis
        with pytest.raises(ValueError, match="channel_axis=0 is rotated"):
            viewer.add_image(
                store, channel_axis=0, transform=bind(viewer, store, coupled)
            )
    finally:
        viewer.controller.close()
