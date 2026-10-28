"""Oblique 2D design v3, Phase 6: slab projection in ``ALONG_AXIS`` views.

``plans/oblique_2d_raster_design_v3.md`` 4.2, 5.2.5, 5.3.8, D10 and D13.  An
image whose appearance projects (``slab_projection``, ``projection_axes``)
reduces the slab the scene's thickness selects on the brick route, in any 2D
view; labels never project.

Expectations are the reductions over the slab's voxel centres, computed from
the data here -- never another library's rendering (memory: no
cross-library tests).
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.convenience.capture import _load_data
from cellier.data import ImageMemoryStore, LabelMemoryStore
from cellier.events import DimsInteractionEvent
from cellier.render import RenderManagerConfig
from cellier.render._config import SchedulerConfig
from cellier.render.scheduling import PlanMode
from cellier.render.visuals._image import GFXMultiscaleImageVisual
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    InMemoryImageRenderConfig,
    InMemoryImageSingleAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
)
from cellier.visuals._image_memory import InMemoryImageAppearance
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._affine import (
    SHAPE,
    SIZE,
    bind,
    canvas_view,
    load,
    multiscale_store,
    new_viewer,
    render,
    world_matrix,
)
from tests.render._pick import PickFrame
from tests.render.test_oblique_2d_brick_route import _pixel_world, _random_volume

FAMILIES = ["image_multiscale", "image"]


def _add(
    viewer,
    family,
    tmp_path,
    data,
    *,
    op="mip",
    axes=("z",),
    matrix=None,
    name="p",
    interpolation="nearest",
    **appearance,
):
    matrix = world_matrix(np.eye(3)) if matrix is None else matrix
    fields = dict(
        interpolation=interpolation,
        slab_projection=op,
        projection_axes=tuple(axes),
        **appearance,
    )
    if family == "image_multiscale":
        store = multiscale_store(tmp_path, name, data)
        return viewer.add_image_multiscale(
            store,
            appearance=MultiscaleImageAppearance(force_level=0, **fields),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
            single=MultiscaleImageSingleAppearance(
                color_map="gray", clim=(0.0, 255.0), render_mode="mip"
            ),
            transform=bind(viewer, store, matrix),
        )
    store = ImageMemoryStore(data=data, name=name)
    return viewer.add_image(
        store,
        appearance=InMemoryImageAppearance(force_level=1, **fields),
        single=InMemoryImageSingleAppearance(
            color_map="gray", clim=(0.0, 255.0), render_mode="mip"
        ),
        render_config=InMemoryImageRenderConfig(
            gpu_budget_bytes=64 * 2**20, block_size=16
        ),
        transform=bind(viewer, store, matrix),
    )


def _gfx(viewer, visual):
    return viewer.controller._render_manager._scenes[viewer.scene.id].get_visual(
        visual.id
    )


def _pixels(camera, shape):
    """Each pixel centre's ``(row, col)`` voxel in an identity ``yx`` view."""
    xy = _pixel_world(camera)
    col = np.floor(xy[..., 0] + 0.5).astype(int)
    row = np.floor(xy[..., 1] + 0.5).astype(int)
    inside = (col >= 0) & (col < shape[2]) & (row >= 0) & (row < shape[1])
    tie = np.any(np.abs(np.abs(xy - np.round(xy)) - 0.5) < 0.02, axis=-1)
    return np.clip(row, 0, shape[1] - 1), np.clip(col, 0, shape[2] - 1), inside & ~tie


def _reduce(data, op, lo, hi):
    """The reduction over the voxel centres in ``[lo, hi)`` along data z."""
    lo_i, hi_i = int(np.ceil(lo)), int(np.ceil(hi))
    block = data[max(0, lo_i) : max(0, min(data.shape[0], hi_i))]
    return {"mip": block.max(0), "minip": block.min(0), "mean": block.mean(0)}[op]


def _wrong(frame, expected, row, col, valid, tol=1.0):
    got = frame[..., 0].astype(float)
    return int(np.count_nonzero(np.abs(got - expected[row, col])[valid] > tol))


# ---------------------------------------------------------------------------
# Reductions (4.2): exact over the slab's voxel centres
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("op", ["mip", "minip", "mean"])
@pytest.mark.parametrize("thickness", [1, 8, 64])
def test_a_projection_reduces_the_slab_voxel_centres(tmp_path, family, op, thickness):
    """Thickness ``T`` voxels: half-thickness ``T / 2`` about the slice."""
    data = np.tile(_random_volume(31), (4, 1, 1))[:80].astype(np.float32)
    shape = data.shape
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, family, tmp_path, data, op=op)
        slice_z, half = 40.0, thickness / 2.0
        load(viewer, slice_z=slice_z, half_thickness=half)
        assert _gfx(viewer, visual).slice_axes == (0, 1, 2)
        frame, _, camera = render(viewer)
        row, col, valid = _pixels(camera, shape)
        expected = _reduce(data, op, slice_z - half, slice_z + half)
        assert np.count_nonzero(valid) > 10_000
        assert _wrong(frame, expected, row, col, valid) == 0
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_linear_sampling_on_an_along_axis_ray_is_exact_too(tmp_path, family):
    """Samples sit on the voxel centres in depth, so linear sampling reduces
    the in-plane bilinear values of the slab's planes, exactly."""
    data = _random_volume(32)
    viewer = new_viewer("2d")
    try:
        _add(viewer, family, tmp_path, data, op="mip", interpolation="linear")
        load(viewer, slice_z=10.0, half_thickness=3.0)
        frame, _, camera = render(viewer)
        xy = _pixel_world(camera)
        x0 = np.floor(xy[..., 0]).astype(int)
        y0 = np.floor(xy[..., 1]).astype(int)
        fx, fy = xy[..., 0] - x0, xy[..., 1] - y0
        inside = (x0 >= 0) & (x0 < SHAPE[2] - 1) & (y0 >= 0) & (y0 < SHAPE[1] - 1)
        x0c = np.clip(x0, 0, SHAPE[2] - 2)
        y0c = np.clip(y0, 0, SHAPE[1] - 2)
        planes = data[7:13]
        bilinear = (
            planes[:, y0c, x0c] * (1 - fx) * (1 - fy)
            + planes[:, y0c, x0c + 1] * fx * (1 - fy)
            + planes[:, y0c + 1, x0c] * (1 - fx) * fy
            + planes[:, y0c + 1, x0c + 1] * fx * fy
        )
        expected = bilinear.max(axis=0)
        got = frame[..., 0].astype(float)
        assert np.count_nonzero(inside) > 10_000
        # The sampler's 8-bit fraction weights: within a few levels.
        assert np.percentile(np.abs(got - expected)[inside], 99) <= 3.0
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
@pytest.mark.parametrize("depth", [8.0e4, 3.0e5])
def test_the_window_faces_do_not_depend_on_the_near_plane(
    tmp_path, family, interpolation, depth
):
    """The window ``[7, 13)`` holds the same planes however far the camera's
    near plane is.  Rays measured from a distant near plane carried f32 error
    as large as the face tolerance: the face samples came and went with the
    GPU's rounding (lavapipe dropped plane 7 at the default range; Metal did
    at these, linear at 8e4 and nearest at 3e5)."""
    data = _random_volume(32)
    viewer = new_viewer("2d")
    try:
        _add(viewer, family, tmp_path, data, op="mip", interpolation=interpolation)
        load(viewer, slice_z=10.0, half_thickness=3.0)
        canvas_view(viewer).camera.depth_range = (-depth, depth)
        frame, _, camera = render(viewer)
        if interpolation == "nearest":
            row, col, valid = _pixels(camera, SHAPE)
            assert _wrong(frame, data[7:13].max(0), row, col, valid) == 0
            return
        xy = _pixel_world(camera)
        x0 = np.floor(xy[..., 0]).astype(int)
        y0 = np.floor(xy[..., 1]).astype(int)
        fx, fy = xy[..., 0] - x0, xy[..., 1] - y0
        inside = (x0 >= 0) & (x0 < SHAPE[2] - 1) & (y0 >= 0) & (y0 < SHAPE[1] - 1)
        x0c = np.clip(x0, 0, SHAPE[2] - 2)
        y0c = np.clip(y0, 0, SHAPE[1] - 2)
        planes = data[7:13]
        expected = (
            planes[:, y0c, x0c] * (1 - fx) * (1 - fy)
            + planes[:, y0c, x0c + 1] * fx * (1 - fy)
            + planes[:, y0c + 1, x0c] * (1 - fx) * fy
            + planes[:, y0c + 1, x0c + 1] * fx * fy
        ).max(axis=0)
        got = frame[..., 0].astype(float)
        assert np.percentile(np.abs(got - expected)[inside], 99) <= 3.0
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# D10: the sign of the slab (boundary), and the winning sample's pick (O12)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("bright", [7, 13])
def test_the_window_is_half_open_in_the_slab_direction(tmp_path, family, bright):
    """``s = 10``, ``h = 3``: the window is ``[7, 13)``.  A bright plane at 7
    wins the MIP; one at 13 does not.  Fails if the walk is mirrored."""
    data = np.full(SHAPE, 16.0, np.float32)
    data[bright] = 250.0
    viewer = new_viewer("2d")
    try:
        _add(viewer, family, tmp_path, data, op="mip")
        load(viewer, slice_z=10.0, half_thickness=3.0)
        frame, _, camera = render(viewer)
        _row, _col, valid = _pixels(camera, SHAPE)
        values = frame[..., 0][valid]
        if bright == 7:
            assert np.all(values >= 249)
        else:
            assert np.all(values <= 17)
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_a_mip_pick_returns_the_winning_sample(tmp_path, family):
    """O12: the pick of a landmark at depth +k is its own data coordinate."""
    data = np.full(SHAPE, 16.0, np.float32)
    mark = np.array([13, 17, 22])
    data[tuple(mark)] = 250.0
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, family, tmp_path, data, op="mip")
        load(viewer, slice_z=10.0, half_thickness=4.0)
        frame, renderer, camera = render(viewer)
        ndc = camera.camera_matrix @ np.r_[mark[::-1].astype(float), 1.0]
        row, col = (1.0 - ndc[1]) / 2.0 * SIZE, (ndc[0] + 1.0) / 2.0 * SIZE
        assert frame[int(row), int(col), 0] >= 249
        info = PickFrame(renderer).info((float(col), float(row)))
        manager = viewer.controller._render_manager
        details = manager._extract_pick_details(
            viewer.scene.id, info["world_object"], info
        )
        coordinate = viewer.controller._promote_pick_coordinate(
            details.displayed_data_coord,
            displayed_axes=viewer.scene.dims.selection.displayed_axes,
            slice_indices={0: 10.0},
            world_ndim=3,
            hit_visual_id=visual.id,
            collapsed_data_indices=details.collapsed_data_indices,
        )
        # Index space: voxel i spans [i, i + 1).
        np.testing.assert_array_equal(np.floor(np.asarray(coordinate)), mark)
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Edge cases (O9, zero thickness) and the slab's reach (K35)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_samples_outside_the_data_are_skipped(tmp_path, family):
    """A slab reaching past the data's first plane reduces what exists."""
    data = _random_volume(33)
    viewer = new_viewer("2d")
    try:
        _add(viewer, family, tmp_path, data, op="minip")
        load(viewer, slice_z=1.0, half_thickness=3.0)
        frame, _, camera = render(viewer)
        row, col, valid = _pixels(camera, SHAPE)
        assert _wrong(frame, _reduce(data, "minip", -2.0, 4.0), row, col, valid) == 0
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_a_slab_thinner_than_a_coarse_voxel_reads_one_coarse_sample(tmp_path, family):
    """4.2, level of detail: a window that holds no centre of the drawn
    level's voxels reads the voxel its centre plane is in.  Phase 8 found the
    backstop drawing nothing under a thin slab while moving."""
    data = _random_volume(36)
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, family, tmp_path, data, op="mip")
        visual.appearance.force_level = 2
        gfx = _gfx(viewer, visual)
        if family == "image_multiscale":
            coarse = data[::2, ::2, ::2]
            scale, trans = np.full(3, 2.0), np.full(3, 0.5)
        else:
            asyncio.run(gfx.brick_source.wait_built())
            coarse = gfx.brick_source._coarse
            scale = np.asarray(gfx.brick_source.level_scales[1])
            trans = np.asarray(gfx.brick_source.level_translations[1])
        # A quarter of a coarse voxel off a centre; +-0.1 of one holds none.
        slice_z = trans[0] + scale[0] * 2.25
        half = 0.1 * scale[0]
        load(viewer, slice_z=slice_z, half_thickness=half)
        assert gfx.slice_axes == (0, 1, 2)
        frame, _, camera = render(viewer)
        xy = _pixel_world(camera)
        world = np.stack([np.full(xy.shape[:2], slice_z), xy[..., 1], xy[..., 0]], -1)
        u = (world - trans) / scale
        index = np.floor(u + 0.5).astype(int)
        shape = np.asarray(coarse.shape)
        valid = np.all((index >= 0) & (index < shape), axis=-1)
        valid &= ~np.any(np.abs(np.abs(u - np.round(u)) - 0.5) < 0.02, axis=-1)
        index = np.clip(index, 0, shape - 1)
        expected = coarse[index[..., 0], index[..., 1], index[..., 2]]
        got = frame[..., 0].astype(float)
        assert np.count_nonzero(valid) > 10_000
        assert np.count_nonzero(np.abs(got - expected)[valid] > 1.0) == 0
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_zero_thickness_draws_the_plane_on_todays_path(tmp_path, family):
    data = _random_volume(34)
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, family, tmp_path, data, op="mip")
        load(viewer, slice_z=10.0)
        assert _gfx(viewer, visual).slice_axes is None
        status = viewer.controller.draw_status(visual.id)
        assert status.drawable and status.reason is None
        frame, _, camera = render(viewer)
        row, col, valid = _pixels(camera, SHAPE)
        assert _wrong(frame, data[10], row, col, valid) == 0
    finally:
        viewer.controller.close()


def test_a_wide_slab_reaches_beyond_the_unfitted_cameras_range(tmp_path):
    """K35: rays come from the near and far points, so a slab of +-120 under
    a +-100 camera still counts samples at depth +-118."""
    data = np.full((256, 32, 40), 16.0, np.float32)
    data[12, 10, 10] = 250.0  # depth -118
    data[248, 20, 30] = 240.0  # depth +118
    viewer = new_viewer("2d")
    try:
        _add(viewer, "image_multiscale", tmp_path, data, op="mip")
        load(viewer, slice_z=130.0, half_thickness=120.0)
        canvas_view(viewer).camera.depth = 1.0
        frame, _, camera = render(viewer)
        assert camera.far - camera.near == pytest.approx(200.0)
        for (z, y, x), value in (((12, 10, 10), 250), ((248, 20, 30), 240)):
            ndc = camera.camera_matrix @ np.array([x, y, 0.0, 1.0])
            row = int((1.0 - ndc[1]) / 2.0 * SIZE)
            col = int((ndc[0] + 1.0) / 2.0 * SIZE)
            assert abs(int(frame[row, col, 0]) - value) <= 1, (z, y, x)
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Routing (5.2.2, 5.2.3) and O14
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_turning_projection_on_and_off_switches_the_route(tmp_path, family):
    data = _random_volume(35)
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, family, tmp_path, data, op="none")
        load(viewer, slice_z=10.0, half_thickness=3.0)
        gfx_visual = _gfx(viewer, visual)
        assert gfx_visual.slice_axes is None
        before = render(viewer)[0]

        async def _set(op) -> None:
            visual.appearance.slab_projection = op
            await _load_data(viewer)

        asyncio.run(_set("mip"))
        assert gfx_visual.slice_axes == (0, 1, 2)
        assert not np.array_equal(render(viewer)[0], before)
        asyncio.run(_set("none"))
        assert gfx_visual.slice_axes is None
        np.testing.assert_array_equal(render(viewer)[0], before)
    finally:
        viewer.controller.close()


def test_a_thickness_drag_through_zero_stays_on_the_route_until_release(tmp_path):
    viewer = Viewer(
        spatial_axes("z", "y", "x"),
        displayed_axes=("y", "x"),
        render_config=RenderManagerConfig(scheduler=SchedulerConfig(dims_settle_s=60)),
        gui="offscreen",
    )
    try:
        visual = _add(viewer, "image_multiscale", tmp_path, _random_volume(36))
        load(viewer, slice_z=10.0, half_thickness=2.0)
        gfx_visual = _gfx(viewer, visual)
        controller = viewer.controller
        seen = []

        async def _drag() -> None:
            with controller.dims_interaction(viewer.scene.id):
                for half in (1.0, 0.0, 1.0, 0.0):
                    controller.update_thickness(viewer.scene.id, {0: half})
                    seen.append(gfx_visual.slice_axes is not None)
            await _load_data(viewer)
            seen.append(gfx_visual.slice_axes is not None)

        asyncio.run(_drag())
        assert seen == [True, True, True, True, False]
        # At zero thickness on the route, the plane is drawn (no slab).
        assert gfx_visual.slab is None
    finally:
        viewer.controller.close()


def test_labels_beside_a_projecting_image_do_not_project_and_stay_paintable(tmp_path):
    """O14: projection is per visual."""
    viewer = new_viewer("2d")
    try:
        image = _add(viewer, "image_multiscale", tmp_path, _random_volume(37))
        store = LabelMemoryStore(data=np.zeros(SHAPE, np.int32), name="labels")
        labels = viewer.add_labels(store, transform=bind(viewer, store, np.eye(4)))
        load(viewer, slice_z=10.0, half_thickness=3.0)
        controller = viewer.controller
        assert _gfx(viewer, image).slice_axes is not None
        assert _gfx(viewer, labels).slice_axes is None
        assert controller.paint_refusal(labels.id) is None
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Statuses (5.2.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("axes", "reason"),
    [
        (("z", "y"), "projects over one axis only; drawing the slice"),
        (("x",), "cannot project along a displayed axis"),
        (("q",), "no world axis named q"),
    ],
)
def test_a_projection_that_cannot_run_draws_the_plane_with_a_status(
    tmp_path, axes, reason
):
    data = _random_volume(38)
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, "image_multiscale", tmp_path, data, axes=axes)
        load(viewer, slice_z=10.0, half_thickness=3.0)
        status = viewer.controller.draw_status(visual.id)
        assert status.drawable and status.reason == reason
        assert _gfx(viewer, visual).slice_axes is None
        assert viewer.controller._draw_status_lines(viewer.scene.id) == [
            f"{visual.name}: {reason}"
        ]
        frame, _, camera = render(viewer)
        row, col, valid = _pixels(camera, SHAPE)
        assert _wrong(frame, data[10], row, col, valid) == 0
    finally:
        viewer.controller.close()


def test_a_slab_over_a_tilted_2d_store_draws_nothing_and_says_why(tmp_path):
    """G0-4: deferred; the status is the line slice's (O18)."""
    viewer = new_viewer("2d")
    try:
        store = ImageMemoryStore(data=np.ones((32, 40), np.float32), name="flat")
        matrix = np.zeros((4, 3))
        matrix[0, 1] = 0.5
        matrix[1, 0] = matrix[2, 1] = matrix[3, 2] = 1.0
        visual = viewer.add_image(
            store,
            appearance=InMemoryImageAppearance(
                slab_projection="mip", projection_axes=("z",)
            ),
            transform=bind(viewer, store, matrix),
        )
        load(viewer, slice_z=5.0, half_thickness=3.0)
        status = viewer.controller.draw_status(visual.id)
        assert not status.drawable
        assert "along a line" in status.reason
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# The thickness flag (D13)
# ---------------------------------------------------------------------------


def test_an_interactive_thickness_change_is_a_tick_of_a_drag(tmp_path, monkeypatch):
    calls = []
    original = GFXMultiscaleImageVisual.plan

    def spy(self, request, config, mode=PlanMode.FULL):
        calls.append(mode)
        return original(self, request, config, mode)

    monkeypatch.setattr(GFXMultiscaleImageVisual, "plan", spy)
    viewer = Viewer(
        spatial_axes("z", "y", "x"),
        displayed_axes=("y", "x"),
        render_config=RenderManagerConfig(
            scheduler=SchedulerConfig(dims_settle_s=0.05)
        ),
        gui="offscreen",
    )
    try:
        _add(
            viewer,
            "image_multiscale",
            tmp_path,
            _random_volume(39),
            coarsest_while_moving_2d=True,
        )
        load(viewer, slice_z=10.0, half_thickness=2.0)
        controller = viewer.controller
        events = []
        controller._outgoing_events.subscribe(
            DimsInteractionEvent, events.append, owner_id=uuid4()
        )
        calls.clear()

        async def _step() -> None:
            controller.update_thickness(viewer.scene.id, {0: 3.0}, interactive=True)
            assert controller.dims_interaction_state(viewer.scene.id) == "active"
            await asyncio.sleep(0.3)
            await _load_data(viewer)

        asyncio.run(_step())
        assert [e.phase for e in events] == ["start", "end"]
        assert events[-1].reason == "settle"
        assert calls[0] is PlanMode.BACKSTOP_ONLY
        assert calls[-1] is PlanMode.FULL
        # A jump starts no drag.
        events.clear()
        controller.update_thickness(viewer.scene.id, {0: 4.0})
        assert [e.phase for e in events if e.phase == "start"] == []
    finally:
        viewer.controller.close()


def test_a_qt_thickness_step_is_interactive_and_a_typed_value_a_jump(qtbot):
    from cellier.gui.qt._scene import _ThicknessSpinBox

    box = _ThicknessSpinBox()
    qtbot.addWidget(box)
    seen = []
    box.valueChanged.connect(lambda _v: seen.append(box.stepping))
    box.stepBy(1)
    box.setValue(7.0)
    assert seen == [True, False]


def test_an_anywidget_thickness_step_is_interactive_and_a_commit_a_jump():
    from cellier.events import DimsUpdateEvent
    from cellier.gui._axis_values import ContinuousAxisValues
    from cellier.gui.anywidget._dims_panel import AnywidgetDimsPanel

    panel = AnywidgetDimsPanel(
        scene_id=uuid4(),
        axis_values={0: ContinuousAxisValues(min=0, max=10, step_size=1)},
        axis_labels={0: "z"},
        slice_indices={0: 1.0},
        displayed_axes=(1, 2),
        thickness_axes=[0],
        thickness={0: 0.0},
    )
    emitted: list[DimsUpdateEvent] = []
    panel.changed.connect(emitted.append)
    panel._on_custom_message(panel, {"type": "thickness", "axis": 0, "value": 2.0}, [])
    panel.thickness = {"0": 5.0}
    assert [(e.thickness, e.interactive) for e in emitted] == [
        ({0: 2.0}, True),
        ({0: 5.0}, False),
    ]
    assert panel.thickness == {"0": 5.0}


# ---------------------------------------------------------------------------
# The image control's projection rows
# ---------------------------------------------------------------------------


def test_the_image_control_offers_the_projection_rows_in_2d(qtbot):
    from cellier.events import AppearanceUpdateEvent
    from cellier.gui._image_controls import image_control_values
    from cellier.gui.qt.visuals._image_controls import QtImageControls
    from cellier.visuals import ImageVisual

    visual = ImageVisual(name="i", data_store_id="s")
    values = image_control_values(
        visual,
        fields=["clim", "slab_projection", "projection_axes"],
        axis_names=["z", "y", "x"],
    )
    assert values["projection"]["axes"] == ["(none)", "z", "y", "x"]
    control = QtImageControls(visual.id, values, n_displayed_dimensions=2)
    qtbot.addWidget(control.widget)
    emitted = []
    control.changed.connect(emitted.append)
    combo = control._controls[("shared", None, "projection_axes")]
    combo.setCurrentText("z")
    control._controls[("shared", None, "slab_projection")].setCurrentText("mean")
    updates = [
        (e.field, e.value) for e in emitted if isinstance(e, AppearanceUpdateEvent)
    ]
    assert updates == [("projection_axes", ("z",)), ("slab_projection", "mean")]
    control.widget.show()
    assert combo.isVisible()
    control.n_displayed_dimensions = 3
    assert not combo.isVisible()
    # Without the fields, the values are what they were before projection.
    plain = image_control_values(visual, fields=["clim"])
    assert "projection" not in plain and "slab_projection" not in plain["shared"]


def test_the_controller_routes_an_axis_update_from_the_control(tmp_path):
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, "image", tmp_path, _random_volume(40), axes=())
        load(viewer, slice_z=10.0, half_thickness=3.0)
        assert _gfx(viewer, visual).slice_axes is None
        from cellier.events import AppearanceUpdateEvent

        async def _update() -> None:
            viewer.controller._on_appearance_update(
                AppearanceUpdateEvent(uuid4(), visual.id, "projection_axes", ("z",))
            )
            await _load_data(viewer)

        asyncio.run(_update())
        assert visual.appearance.projection_axes == ("z",)
        assert _gfx(viewer, visual).slice_axes == (0, 1, 2)
        store = viewer.controller._model.data.stores[UUID(visual.data_store_id)]
        assert store is not None
    finally:
        viewer.controller.close()


def test_each_channel_projects_on_its_own_then_composites(tmp_path):
    """O13: per-channel projection, then the additive composite."""
    from cellier.convenience.capture import _ensure_canvases
    from cellier.scene._background import BackgroundAppearance
    from cellier.visuals import InMemoryImageChannelAppearance

    data = np.stack([_random_volume(41), _random_volume(42)]).astype(np.float32)
    viewer = Viewer(
        [("c", "channel"), *spatial_axes("z", "y", "x")],
        displayed_axes=("y", "x"),
        gui="offscreen",
    )
    viewer.background = BackgroundAppearance(mode="uniform", color=(0, 0, 0, 1))
    try:
        store = ImageMemoryStore(data=data, name="czyx")
        viewer.add_image(
            store,
            appearance=InMemoryImageAppearance(
                interpolation="nearest", slab_projection="mip", projection_axes=("z",)
            ),
            render_config=InMemoryImageRenderConfig(
                gpu_budget_bytes=64 * 2**20, block_size=16
            ),
            channel_axis=0,
            composite=True,
            channels={
                0: InMemoryImageChannelAppearance(color_map="red", clim=(0.0, 255.0)),
                1: InMemoryImageChannelAppearance(color_map="green", clim=(0.0, 255.0)),
            },
            transform=bind(viewer, store, np.eye(5)),
        )
        _ensure_canvases(viewer, (SIZE, SIZE))

        async def _load() -> None:
            viewer.set_slice_positions({1: 10.0})
            viewer.controller.update_thickness(viewer.scene.id, {1: 3.0})
            await _load_data(viewer)

        asyncio.run(_load())
        frame, _, camera = render(viewer)
        row, col, valid = _pixels(camera, SHAPE)
        for index in (0, 1):
            expected = data[index, 7:13].max(axis=0)
            got = frame[..., index].astype(float)
            # A colormap from cmap: the channel's own colour ramp, near-linear.
            err = np.abs(got - expected[row, col])[valid]
            assert np.percentile(err, 99) <= 3.0, index
    finally:
        viewer.controller.close()
