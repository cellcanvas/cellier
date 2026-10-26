"""Full affine design v2, Phase 4: 3D and ``IN_PLANE`` for every family.

Every family draws under any spatial affine in 3D and under an in-plane
rotation or shear in 2D; a view that cannot draw a visual draws nothing of
it and says why (D3, D8).  Oracles are the D10 kinds: analytic landmarks
(``tests/render/_affine.py``), in-cellier equivalences, picks and
structural checks.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cellier.convenience import OrthoViewer
from cellier.convenience.capture import _wait_for_slicer
from cellier.data import ImageMemoryStore, LabelMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.transform import AffineTransform
from cellier.visuals import (
    InMemoryImageSingleAppearance,
    InMemoryLabelsAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    PointsMarkerAppearance,
    RenderPlane,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._affine import (
    FAMILIES,
    GEOMETRY,
    LANDMARK_PX,
    LINEARS,
    P0,
    RASTER,
    SHAPE,
    add_landmark,
    bind,
    block_volume,
    canvas_view,
    centroid,
    cube_corners,
    drawn,
    hull_centroid,
    landmark_error,
    load,
    new_viewer,
    nudge_2d_camera,
    project,
    render,
    to_world,
    world_matrix,
)
from tests.render.conftest import _write_multiscale_zarr

# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------

THREE_D = ["rot_yx", "rot_zx", "shear_yx", "shear_x_by_z", "general", "rot90_yx"]
THREE_D += ["flip_x"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("name", THREE_D)
def test_3d_landmark_every_family(tmp_path, name, family):
    """The landmark lands where ``L p + t`` projects (D10, 0.5 px at 960 px)."""
    assert landmark_error(family, name, "3d", tmp_path) < LANDMARK_PX


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("name", ["rot_yx", "shear_yx"])
def test_2d_in_plane_landmark_every_family(tmp_path, name, family):
    assert landmark_error(family, name, "2d", tmp_path) < LANDMARK_PX


# ---------------------------------------------------------------------------
# A rotated slide, zoomed onto a corner: every visible tile is resident
# ---------------------------------------------------------------------------

SLIDE = (2, 512, 512)


def _slide_store(root: Path):
    """Level 0 is 1.0 everywhere and level 1 is 0.25: a coarse fallback shows."""
    root.mkdir(parents=True, exist_ok=True)

    def fill(arr):
        arr[...] = 1.0 if arr.shape == SLIDE else 0.25

    _write_multiscale_zarr(
        root,
        levels=[("s0", SLIDE), ("s1", (1, 256, 256))],
        fill=fill,
        dtype="float32",
    )
    from cellier.data import MultiscaleZarrDataStore

    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        name="slide",
    )


def test_a_rotated_slide_zoomed_onto_a_corner_is_sharp(tmp_path):
    """K10: the LOD and the view's tiles under an in-plane rotation.

    Every pixel the rotated slide covers must show level 0 (white): a
    missing tile would fall back to level 1 (dark grey), and a too-coarse
    LOD choice would draw level 1 throughout.
    """
    linear = np.diag([1.0, 1.0, 1.0])
    theta = np.deg2rad(30.0)
    linear[1:, 1:] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    centre = (np.asarray(SLIDE, dtype=float) - 1.0) / 2.0
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = centre - linear @ centre
    viewer = new_viewer("2d")
    try:
        store = _slide_store(tmp_path / "slide")
        viewer.add_image_multiscale(
            store,
            appearance=MultiscaleImageAppearance(),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=32),
            single=MultiscaleImageSingleAppearance(color_map="gray", clim=(0.0, 1.0)),
            transform=bind(viewer, store, matrix),
        )
        load(viewer, slice_z=0.0)
        camera = canvas_view(viewer).camera
        corner = to_world(matrix, np.array([0.0, -0.5, -0.5]))
        camera.width = camera.height = 120.0
        camera.local.position = (float(corner[2]), float(corner[1]), 0.0)
        nudge_2d_camera(camera)

        async def _reload():
            viewer.controller.reslice_all()
            await _wait_for_slicer(viewer.controller)

        asyncio.run(_reload())
        frame, _, camera = render(viewer)
    finally:
        viewer.controller.close()
    # Which pixels the slide covers, analytically: pull each pixel centre
    # back through the camera and the transform.
    size = frame.shape[0]
    rows, cols = np.mgrid[:size, :size]
    ndc = np.stack(
        [(cols + 0.5) / size * 2 - 1, 1 - (rows + 0.5) / size * 2], -1
    ).reshape(-1, 2)
    inverse = np.linalg.inv(np.asarray(camera.camera_matrix, dtype=np.float64))
    h = np.concatenate([ndc, np.zeros((len(ndc), 1)), np.ones((len(ndc), 1))], 1)
    world_xyz = (h @ inverse.T)[:, :3]
    world_zyx = world_xyz[:, ::-1].copy()
    world_zyx[:, 0] = 0.0
    data = (world_zyx - matrix[:3, 3]) @ np.linalg.inv(matrix[:3, :3]).T
    margin = 0.6  # a little over half a pixel, in voxels (120 / 960 per px)
    inside = np.all(
        (data[:, 1:] > -0.5 + margin) & (data[:, 1:] < 511.5 - margin), axis=1
    ).reshape(size, size)
    outside = np.all(
        (data[:, 1:] < -0.5 - margin) | (data[:, 1:] > 511.5 + margin), axis=1
    ).reshape(size, size) | np.any(
        (data[:, 1:] < -0.5 - margin) | (data[:, 1:] > 511.5 + margin), axis=1
    ).reshape(size, size)
    assert inside.sum() > 1000 and outside.sum() > 1000
    luminance = frame[..., :3].max(-1)
    assert luminance[inside].min() == 255
    assert luminance[outside].max() == 0


# ---------------------------------------------------------------------------
# Equivalence: rot90_yx draws exactly what transposed, flipped data draws
# ---------------------------------------------------------------------------


def _f_volume() -> np.ndarray:
    """An asymmetric "F" of three boxes: a wrong turn or flip shows."""
    data = np.zeros(SHAPE, dtype=np.float32)
    data[4:20, 4:28, 4:12] = 1.0
    data[4:20, 4:10, 12:34] = 0.75
    data[8:16, 14:20, 12:26] = 0.5
    return data


@pytest.mark.parametrize("dim", ["2d", "3d"])
@pytest.mark.parametrize("kind", ["image", "labels"])
def test_rot90_equals_transposed_and_flipped_data(dim, kind):
    """``world_y = 35 - x``, ``world_x = y + 4`` against the same data moved.

    Pixel identical in 2D; in 3D within D10's equivalence threshold.
    """
    data = _f_volume()
    if kind == "labels":
        data = np.round(data * 4).astype(np.int32)
    moved = np.ascontiguousarray(data[:, :, ::-1].transpose(0, 2, 1))
    shift = np.eye(4)
    shift[:3, 3] = (0.0, -4.0, 4.0)
    frames = []
    state = None
    for volume, matrix in ((data, world_matrix(LINEARS["rot90_yx"])), (moved, shift)):
        viewer = new_viewer(dim)
        try:
            if kind == "image":
                store = ImageMemoryStore(data=volume, name="f")
                viewer.add_image(
                    store,
                    single=InMemoryImageSingleAppearance(
                        color_map="gray", clim=(0.0, 1.0), render_mode="mip"
                    ),
                    transform=bind(viewer, store, matrix),
                )
            else:
                store = LabelMemoryStore(data=volume, name="f")
                viewer.add_labels(
                    store,
                    appearance=InMemoryLabelsAppearance(render_mode="iso_categorical"),
                    transform=bind(viewer, store, matrix),
                )
            load(viewer, slice_z=12.0 if dim == "2d" else None)
            view = canvas_view(viewer)
            if state is None:
                state = view.camera.get_state()
            else:
                view.camera.set_state(state)
            frames.append(render(viewer, 480)[0][..., :3].astype(int))
        finally:
            viewer.controller.close()
    assert drawn(frames[0]).sum() > 1000
    if dim == "2d":
        assert np.array_equal(frames[0], frames[1])
        return
    # A raymarch meets the box faces with different rounding when the data
    # is stored the other way round: a few silhouette pixels (15 of 32k
    # measured) differ.  D10's equivalence threshold, plus a cap on how many.
    diff = np.abs(frames[0] - frames[1]).max(-1)
    either = drawn(frames[0]) | drawn(frames[1])
    assert np.percentile(diff[either], 99) <= 8
    assert np.count_nonzero(diff) <= 0.001 * either.sum()


# ---------------------------------------------------------------------------
# Picks under the general affine: within one voxel of the data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", RASTER)
def test_picks_under_the_general_affine_decode_to_the_data(tmp_path, family):
    matrix = world_matrix(LINEARS["general"])
    viewer = new_viewer("3d")
    try:
        visual = add_landmark(viewer, family, matrix, tmp_path)
        load(viewer)
        frame, renderer, _ = render(viewer, 480)
        gfx_visual = viewer.controller._render_manager._scenes[
            viewer.scene.id
        ].get_visual(visual.id)
        rows, cols = np.nonzero(drawn(frame))
        rng = np.random.default_rng(0)
        picks = rng.choice(rows.size, size=min(30, rows.size), replace=False)
        hits = 0
        for index in picks:
            info = renderer.get_pick_info((cols[index] + 0.5, rows[index] + 0.5))
            coord = gfx_visual.pick_data_coordinate(info.get("world_object"), info)
            assert coord is not None
            # Through the controller, as a pick event decodes it.
            full = viewer.controller._promote_pick_coordinate(
                coord,
                displayed_axes=(0, 1, 2),
                slice_indices={},
                world_ndim=3,
                hit_visual_id=visual.id,
            )
            voxel = np.floor(np.asarray(full))
            hits += bool(np.all(np.abs(voxel - P0) <= 2.0))
    finally:
        viewer.controller.close()
    assert hits == len(picks)


# ---------------------------------------------------------------------------
# Draw status
# ---------------------------------------------------------------------------


def _status_events(viewer) -> list:
    events: list = []
    viewer.controller.on_draw_status_changed(
        viewer.scene.id, events.append, owner_id=viewer.scene.id
    )
    return events


def test_rot_zx_image_status_follows_the_view(tmp_path, caplog):
    """3D draws, a ``(y, x)`` view cuts it at an angle and draws it on the
    brick route (oblique 2D design v3, Phase 4: a pre-approved flip of
    "draws nothing"), 3D draws again."""
    matrix = world_matrix(LINEARS["rot_zx"])
    viewer = new_viewer("3d")
    try:
        visual = add_landmark(viewer, "image", matrix, tmp_path)
        events = _status_events(viewer)
        controller = viewer.controller
        scene_manager = controller._render_manager._scenes[viewer.scene.id]
        load(viewer)
        first = controller.draw_status(visual.id)
        assert first.drawable and first.alignment == "in_plane"

        with caplog.at_level(logging.WARNING, logger="cellier.draw_status"):
            viewer.set_displayed_dimensions(("y", "x"))
        second = controller.draw_status(visual.id)
        assert second.drawable and second.alignment == "oblique"
        assert second.reason is None
        assert not scene_manager.is_visual_suppressed(visual.id)
        assert controller._render_config_for(viewer.scene.id, visual).slicing_enabled
        assert not [
            r for r in caplog.records if "is not drawn in this view" in r.getMessage()
        ]
        # The alignment changed, so the status is announced; it draws.
        assert [event.status.drawable for event in events] == [True]
        assert controller._draw_status_lines(viewer.scene.id) == []

        viewer.set_displayed_dimensions(("z", "y", "x"))
        load(viewer)
        assert controller.draw_status(visual.id).drawable
        assert not scene_manager.is_visual_suppressed(visual.id)
        assert all(event.status.drawable for event in events)
        frame, _, _ = render(viewer)
        assert centroid(frame) is not None  # resliced and drawn on return
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_a_slice_shift_view_draws(tmp_path, family):
    """``SLICE_SHIFT`` is drawn by every family (Phase 5; Phase 4 drew none)."""
    viewer = new_viewer("2d")
    try:
        visual = add_landmark(
            viewer, family, world_matrix(LINEARS["shear_x_by_z"]), tmp_path
        )
        status = viewer.controller.draw_status(visual.id)
        assert status.alignment == "slice_shift" and status.drawable
    finally:
        viewer.controller.close()


def test_the_orthoviewer_under_rot_zx_reports_its_panels():
    """Design 6.2: every panel draws the image -- ``xy`` and ``yz`` on the
    brick route (oblique 2D design v3, Phase 4: a pre-approved flip of
    "``xy``/``yz`` don't")."""
    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui="offscreen")
    try:
        store = ImageMemoryStore(data=block_volume(np.float32), name="F image")
        ortho.controller.add_data_store(store)
        visuals = ortho.add_image(store, name="F image")
        scene = next(iter(ortho.scenes.values()))
        ortho.controller._ensure_data_coordinate_systems(scene.id, store)
        transform = AffineTransform.from_matrix(
            world_matrix(LINEARS["rot_zx"]),
            store.data_coordinate_systems[0],
            scene.dims.world_coordinate_system,
        )
        for visual in visuals.values():
            visual.transform = transform
        drawn_by_panel = {
            panel: ortho.controller.draw_status(visual.id).drawable
            for panel, visual in visuals.items()
        }
        assert drawn_by_panel == {"xy": True, "xz": True, "yz": True, "vol": True}
        assert ortho.draw_status_overlay is not None
    finally:
        ortho.controller.close()


def test_the_status_overlay_draws_in_a_capture(tmp_path):
    """D8 B: the canvas text shows in a screenshot (captures carry overlays)."""
    viewer = new_viewer("2d")
    try:
        # A view that still draws nothing: a flat yx store cut along a line
        # (oblique 2D design v3, U13).  A tilted volume draws since Phase 4.
        store = ImageMemoryStore(data=np.ones((32, 40), np.float32), name="flat")
        matrix = np.zeros((4, 3))
        matrix[0, 1] = 0.5
        matrix[1, 0] = matrix[2, 1] = matrix[3, 2] = 1.0
        viewer.add_image(store, transform=bind(viewer, store, matrix))
        load(viewer, slice_z=P0[0])
        frame = viewer.screenshot(size=(320, 320))
        assert drawn(frame).sum() > 50  # the text; the image draws nothing
        viewer.draw_status_overlay.visible = False
        assert drawn(viewer.screenshot(size=(320, 320))).sum() == 0
    finally:
        viewer.controller.close()


def test_a_viewer_can_be_built_without_the_status_overlay(tmp_path):
    from cellier.convenience import Viewer

    viewer = Viewer(
        spatial_axes("z", "y", "x"),
        displayed_axes=("y", "x"),
        gui="offscreen",
        draw_status_overlay=False,
    )
    try:
        assert viewer.draw_status_overlay is None
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Plane render mode under shear and the general affine
# ---------------------------------------------------------------------------


def _plane_section_centroid(camera, matrix) -> np.ndarray:
    """Where the block's cross-section by the world plane ``z = W0_z`` projects.

    The cube's 12 edges are intersected with the plane in world space, and
    the area centroid of the hull of their projections is the expectation.
    """
    corners = np.array([to_world(matrix, c) for c in cube_corners()])
    level = to_world(matrix, P0)[0]
    bits = [
        (a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1
    ]
    points = []
    for a, b in bits:
        za, zb = corners[a, 0], corners[b, 0]
        if (za - level) * (zb - level) < 0:
            t = (level - za) / (zb - za)
            points.append(corners[a] + t * (corners[b] - corners[a]))
    return hull_centroid(np.array([project(camera, p) for p in points]))


@pytest.mark.parametrize("name", ["shear_yx", "general"])
def test_plane_mode_draws_the_landmark_section(tmp_path, name):
    matrix = world_matrix(LINEARS[name])
    viewer = new_viewer("3d")
    try:
        world = viewer.scene.dims.world_coordinate_system
        plane = RenderPlane.from_point_normal(
            world,
            tuple(float(v) for v in to_world(matrix, P0)),
            (1.0, 0.0, 0.0),
            axes=("z", "y", "x"),
        )
        store = ImageMemoryStore(data=block_volume(np.float32), name="plane")
        viewer.add_image(
            store,
            single=InMemoryImageSingleAppearance(
                color_map="gray", clim=(0.0, 1.0), render_mode="plane"
            ),
            render_planes=(plane,),
            transform=bind(viewer, store, matrix),
        )
        load(viewer)
        frame, _, camera = render(viewer)
        got = centroid(frame)
        assert got is not None
        expected = _plane_section_centroid(camera, matrix)
        assert np.linalg.norm(got - expected) < 1.0
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Painting (D5)
# ---------------------------------------------------------------------------


def _paint_setup(name):
    viewer = new_viewer("2d")
    store = LabelMemoryStore(data=np.zeros(SHAPE, dtype=np.int32), name="paint")
    matrix = world_matrix(LINEARS[name])
    visual = viewer.add_labels(
        store,
        appearance=InMemoryLabelsAppearance(),
        transform=bind(viewer, store, matrix),
    )
    viewer.controller.add_canvas(scene_id=viewer.scene.id)
    viewer.set_slice_positions({0: float(P0[0])})
    return viewer, store, visual, matrix


async def test_painting_under_an_in_plane_rotation_lands_on_the_voxel():
    viewer, store, visual, matrix = _paint_setup("rot_yx")
    try:
        assert viewer.controller.paint_refusal(visual.id) is None
        canvas_id = viewer.controller.get_canvas_ids(viewer.scene.id)[0]
        paint = viewer.controller.add_paint_controller(
            visual_id=visual.id,
            canvas_id=canvas_id,
            brush_value=7,
            brush_radius_voxels=0.0,
        )
        # A pointer event carries the displayed axes in pygfx (x, y) order
        # (``CellierController`` pointer bridge); the controller swaps back.
        world = to_world(matrix, P0)
        world[1], world[2] = world[2], world[1]
        paint._apply_brush(world)
        painted = np.argwhere(store.data == 7)
        assert painted.tolist() == [P0.astype(int).tolist()]
    finally:
        viewer.controller.close()


async def test_painting_under_shear_is_refused_naming_the_alignment():
    viewer, _store, visual, _matrix = _paint_setup("shear_yx")
    try:
        refusal = viewer.controller.paint_refusal(visual.id)
        assert refusal is not None and "IN_PLANE" in refusal
        canvas_id = viewer.controller.get_canvas_ids(viewer.scene.id)[0]
        with pytest.raises(ValueError, match="IN_PLANE"):
            viewer.controller.add_paint_controller(
                visual_id=visual.id, canvas_id=canvas_id, brush_value=1
            )
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Assignment refusals (D6, design 4.2)
# ---------------------------------------------------------------------------

_CZYX = [("c", "channel"), *spatial_axes("z", "y", "x")]
_TZYX = [("t", "time"), *spatial_axes("z", "y", "x")]


def _four_axis_matrix(row: int, column: int, value: float) -> np.ndarray:
    matrix = np.eye(5)
    matrix[row, column] = value
    return matrix


def test_a_channel_coupled_to_space_is_refused_at_assignment():
    from cellier.controller import CellierController

    controller = CellierController(gui="offscreen")
    try:
        scene = controller.add_scene(coordinate_system=_CZYX, displayed_axes=("y", "x"))
        store = ImageMemoryStore(data=np.zeros((2, 4, 6, 8), dtype=np.float32))
        controller._ensure_data_coordinate_systems(scene.id, store)
        world = scene.dims.world_coordinate_system
        coupled = AffineTransform.from_matrix(
            _four_axis_matrix(3, 0, 0.5), store.data_coordinate_systems[0], world
        )
        with pytest.raises(ValueError, match="channel axis"):
            controller.add_image(store, scene.id, channel_axis=0, transform=coupled)
        assert scene.visuals == []
        visual = controller.add_image(store, scene.id, channel_axis=0)
        accepted = visual.transform
        with pytest.raises(Exception, match="channel axis"):
            visual.transform = coupled
        assert visual.transform is accepted
    finally:
        controller.close()


def test_time_coupled_to_space_is_accepted():
    from cellier.controller import CellierController

    controller = CellierController(gui="offscreen")
    try:
        scene = controller.add_scene(coordinate_system=_TZYX, displayed_axes=("y", "x"))
        store = PointsMemoryStore(positions=np.zeros((3, 4), dtype=np.float32))
        controller._ensure_data_coordinate_systems(scene.id, store)
        drift = AffineTransform.from_matrix(
            _four_axis_matrix(3, 0, 0.5),
            store.data_coordinate_systems[0],
            scene.dims.world_coordinate_system,
        )
        visual = controller.add_points(
            data=store,
            scene_id=scene.id,
            appearance=PointsMarkerAppearance(),
            transform=drift,
        )
        assert visual.transform is drift
    finally:
        controller.close()


def test_a_singular_transform_is_refused_at_assignment():
    viewer = new_viewer("3d")
    try:
        store = ImageMemoryStore(data=np.zeros(SHAPE, dtype=np.float32))
        singular = np.eye(4)
        singular[1, :3] = (0.0, 1.0, 1.0)  # world y and world x are both y + x
        singular[2, :3] = (0.0, 1.0, 1.0)
        with pytest.raises(ValueError, match="singular"):
            viewer.add_image(store, transform=bind(viewer, store, singular))
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# I2: nothing decides drawability through axis_correspondence
# ---------------------------------------------------------------------------


def test_axis_correspondence_is_only_called_where_the_design_allows():
    """Design v2 I2 (the Phase 4 grep).

    Outside the transform package it is called only by helpers that run
    after the classifier: the axis-aligned branches of the image helpers and
    the controller's retained axes, the channel-axis check's first try, and
    the ``_spaces`` delegation itself.
    """
    root = Path(__file__).resolve().parents[2] / "src" / "cellier"
    result = subprocess.run(
        ["grep", "-rn", "--include=*.py", "axis_correspondence(", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    sites = []
    for line in result.stdout.splitlines():
        path, _number, text = line.split(":", 2)
        relative = Path(path).relative_to(root).as_posix()
        code = text.split("#", 1)[0]
        if relative.startswith("transform/") or not re.search(
            r"(?<!exclusive_)axis_correspondence\(", code
        ):
            continue
        if code.strip().startswith(("``", '"""')) or "``" in code:
            continue
        sites.append(relative)
    assert sorted(set(sites)) == [
        "controller.py",
        "render/_spaces.py",
        "render/visuals/_image.py",
    ]


# ---------------------------------------------------------------------------
# D7 end to end: a rotation built with cos/sin is drawn as the axis-aligned
# view it is
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_a_trig_built_quarter_turn_draws_in_a_2d_view(tmp_path, family):
    """``cos(pi / 2)`` is ``6e-17``: snapped (D7), every consumer agrees.

    The classifier calls the ``(y, x)`` view of a 90 degree turn about ``y``
    ``AXIS_ALIGNED``; the render and slicing paths read the same snapped
    matrix (``effective_transform``), so the visual draws where the exact
    rotation puts it instead of raising on the noise.
    """
    from tests.render._affine import rot

    linear = rot(0, 2, 90.0)
    assert 0.0 < np.abs(linear[np.abs(linear) < 1e-3]).max() < 1e-15
    matrix = world_matrix(linear)
    viewer = new_viewer("2d")
    try:
        visual = add_landmark(viewer, family, matrix, tmp_path)
        assert viewer.controller.draw_status(visual.id).alignment == "axis_aligned"
        load(
            viewer,
            slice_z=to_world(matrix, P0)[0],
            half_thickness=0.6 if family in GEOMETRY else None,
        )
        nudge_2d_camera(canvas_view(viewer).camera)
        frame, _, camera = render(viewer)
        got = centroid(frame)
        assert got is not None
        if family in ("points", "lines", "graph"):
            expected = project(camera, to_world(matrix, P0))
        else:
            # Axis aligned after the turn: the block's footprint is the hull
            # of all eight projected corners.
            corners = [to_world(matrix, c) for c in cube_corners().astype(float)]
            expected = hull_centroid(np.array([project(camera, c) for c in corners]))
        assert np.linalg.norm(got - expected) < LANDMARK_PX
    finally:
        viewer.controller.close()
