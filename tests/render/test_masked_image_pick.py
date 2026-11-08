"""MaskedImage picks.

A pick reports, at level 0 and the nearest voxel, the mask and the value at
the hit and the value projected along the inward normal as the surface is
coloured.  The pure part, :class:`cellier._masked_image.MaskedPick`, is
checked on small cases with known answers; the whole path (the pick buffer,
the visual's decode, the controller's two passes) on the example's band
pair, where the projected band is known from the threshold, the thickness
and the placement, and every other value must equal the arrays.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from cellier._masked_image import MaskedPick
from cellier.convenience.capture import _wait_until_loaded
from cellier.events import ImagePickEvent, MaskedImagePickEvent, ViewRay
from cellier.events._events import _CanvasRawPointerEvent
from cellier.transform import Plane
from cellier.visuals import ClippingPlane, MaskedImageSingleAppearance
from tests.render._affine import canvas_view, render
from tests.render._masked import (
    band_colormap,
    band_pair,
    load,
    masked_config,
    new_viewer,
    store,
    value_to_mask_matrix,
)
from tests.render._pick import PickFrame

SIDE = 64
SIZE = 256


# -- the pure part -----------------------------------------------------------------


def _pick(hit=(10.0, 10.0, 10.0), **fields) -> MaskedPick:
    """A pick on a 32 cubed pair on one grid, world = data, unless overridden."""
    defaults = {
        "hit": np.asarray(hit, dtype=np.float64),
        "mask_shape": (32, 32, 32),
        "value_shape": (32, 32, 32),
        "mask_space": (0, 1, 2),
        "mask_to_world": np.eye(3),
        "mask_to_value": np.eye(4),
        "three_d": True,
        "iso_threshold": 0.5,
        "surface_thickness": 6.0,
        "surface_placement": "inward",
        "surface_projection": "max",
        "steps_per_voxel": 1.0,
    }
    return MaskedPick(**{**defaults, **fields})


def _ramp_values(pick: MaskedPick, axis: int, slope: float = 1.0) -> dict:
    """Mask values of *pick*'s reads for ``mask = slope * index[axis]``."""
    return {key: slope * index[axis] for key, index in pick.mask_reads().items()}


def test_a_2d_pick_reads_the_mask_at_the_hit_only():
    pick = _pick(three_d=False)
    assert pick.mask_reads() == {"hit": (10, 10, 10)}
    assert set(pick.value_reads({"hit": 1.0})) == {"hit"}


def test_a_3d_pick_with_no_thickness_reads_no_stencil():
    assert set(_pick(surface_thickness=0.0).mask_reads()) == {"hit"}


def test_a_3d_pick_reads_the_mask_stencil():
    reads = _pick(hit=(0.2, 10.0, 31.0)).mask_reads()
    assert reads["hit"] == (0, 10, 31)
    # Clamped at the data's edge, one sided there.
    assert reads[(0, -1)] == (0, 10, 31) and reads[(0, 1)] == (1, 10, 31)
    assert reads[(2, -1)] == (0, 10, 30) and reads[(2, 1)] == (0, 10, 31)
    assert len(reads) == 7


def test_a_hit_outside_the_mask_reads_nothing():
    assert _pick(hit=(-0.6, 10.0, 10.0)).mask_reads() == {}


def test_samples_follow_the_inward_normal_at_the_shaders_spacing():
    """``mask = x``: inward is +x; six steps of one voxel, at their midpoints."""
    pick = _pick()
    reads = pick.value_reads(_ramp_values(pick, 2))
    samples = [reads[("sample", i)] for i in range(6)]
    # x = 10.5, 11.5, ... 15.5, nearest voxel half up.
    assert samples == [(10, 10, x) for x in range(11, 17)]
    assert reads["hit"] == (10, 10, 10)


def test_the_sample_count_follows_the_value_voxels_and_the_cap():
    pick = _pick(steps_per_voxel=2.0)
    assert len(pick.value_reads(_ramp_values(pick, 2))) == 1 + 12
    # A value twice as fine: twice the voxels along the same world length.
    fine = np.diag([2.0, 2.0, 2.0, 1.0])
    fine[:3, 3] = 0.5
    pick = _pick(mask_to_value=fine, value_shape=(64, 64, 64))
    assert len(pick.value_reads(_ramp_values(pick, 2))) == 1 + 12
    pick = _pick(surface_thickness=100.0, hit=(10.0, 10.0, 1.0))
    reads = pick.value_reads(_ramp_values(pick, 2))
    # 64 steps of 100/64 from x = 1: those past x = 31.5 are not read.
    assert sum(1 for key in reads if key != "hit") == 20


def test_the_normal_is_one_world_unit_long_on_anisotropic_voxels():
    """World x is half a data voxel: one world unit inward is two voxels."""
    pick = _pick(mask_to_world=np.diag([2.0, 1.0, 0.5]), surface_thickness=1.0)
    reads = pick.value_reads(_ramp_values(pick, 2))
    # One world unit is two value voxels: two steps, at x = 10.5 and 11.5.
    assert [reads[("sample", i)] for i in range(2)] == [(10, 10, 11), (10, 10, 12)]


@pytest.mark.parametrize(
    ("placement", "op", "expected"),
    [
        ("inward", "max", 15.5),
        ("inward", "min", 10.5),
        ("inward", "mean", 13.0),
        ("outward", "max", 9.5),
        ("centred", "min", 7.5),
    ],
)
def test_the_projection_reduces_the_samples(placement, op, expected):
    """``value = x + 0.5`` at voxel x: the samples are the voxel centres read."""
    pick = _pick(surface_placement=placement, surface_projection=op)
    mask_values = _ramp_values(pick, 2)
    value_values = {
        key: index[2] - 0.5 for key, index in pick.value_reads(mask_values).items()
    }
    info = pick.info(mask_values, value_values)
    assert info.projected_value == expected
    assert info.surface_value == 9.5
    assert info.mask_value == 10.0
    assert info.data_coordinate == (10.5, 10.5, 10.5)
    assert info.value_coordinate == (10.5, 10.5, 10.5)


def test_a_flat_mask_projects_nothing():
    pick = _pick()
    flat = dict.fromkeys(pick.mask_reads(), 1.0)
    reads = pick.value_reads(flat)
    assert set(reads) == {"hit"}
    assert pick.info(flat, {"hit": 3.0}).projected_value is None


def test_a_cut_face_projects_along_the_planes_normal():
    """Inside the mask on a plane keeping ``x <= 10``: inward is -x."""
    plane = Plane(coordinate_system=uuid4(), normal=(0.0, 0.0, -1.0), offset=-10.0)
    pick = _pick(planes=(plane,))
    # The mask is inside (above 0.5) and rises along +x: the stencil says +x.
    mask_values = _ramp_values(pick, 2)
    reads = pick.value_reads(mask_values)
    assert [reads[("sample", i)][2] for i in range(6)] == [10, 9, 8, 7, 6, 5]
    # Outside the mask there, the hit is the surface, not the cut face.
    outside = {key: value - 10.0 for key, value in mask_values.items()}
    reads = pick.value_reads(outside)
    assert reads[("sample", 0)][2] == 11


def test_a_value_on_its_own_grid():
    """Mask voxel c is value voxel 2c + 1/2; a hit outside the value has none."""
    to_value = np.diag([2.0, 2.0, 2.0, 1.0])
    to_value[:3, 3] = 0.5
    pick = _pick(mask_to_value=to_value, value_shape=(20, 64, 64), three_d=False)
    info = pick.info({"hit": 1.0}, {})
    assert info.value_coordinate == (21.0, 21.0, 21.0)
    assert info.surface_value is None and info.projected_value is None
    assert pick.value_reads({"hit": 1.0}) == {}


# -- the whole path ----------------------------------------------------------------


def _viewer(tmp_path, kind, dim, value_scale=1, **single):
    """The band pair, loaded; ``(viewer, model, mask array, value array)``."""
    mask_data, value_data = band_pair(SIDE, value_scale)
    viewer = new_viewer(dim)
    value_budget = 48 if kind == "memory" and value_scale == 2 else 16
    model = viewer.controller.add_masked_image(
        store(kind, tmp_path, "mask", mask_data),
        store(kind, tmp_path, "value", value_data),
        viewer.scene.id,
        value_to_mask=value_to_mask_matrix(value_scale),
        single=MaskedImageSingleAppearance(
            color_map=band_colormap(), clim=(0.5, 3.5), **single
        ),
        render_config=masked_config(value_gpu_budget_bytes=value_budget * 1024**2),
    )
    load(viewer, slice_z=31.0 if dim == "2d" else None, size=(SIZE, SIZE))
    return viewer, model, mask_data, value_data


def _raw(viewer, details, camera, action="press"):
    canvas_id = viewer.controller.get_canvas_ids(viewer.scene.id)[0]
    return _CanvasRawPointerEvent(
        canvas_id=canvas_id,
        scene_id=viewer.scene.id,
        action=action,
        camera_type=camera,
        position_2d=np.array([1.0, 2.0]) if camera == "2d" else None,
        ray=(
            ViewRay(origin=np.zeros(3), direction=np.array([0.0, 0.0, 1.0]))
            if camera == "3d"
            else None
        ),
        hit_visual_id=details[0],
        button=1,
        modifiers=(),
        buttons=(1,) if action != "move" else (),
        gesture_id=uuid4() if action != "move" else None,
        pick_details=details[1],
    )


def _hits(viewer, step=6) -> list:
    """``(visual id, pick details)`` of drawn pixels on a grid, from one frame."""
    frame, renderer, _camera = render(viewer, SIZE)
    pick = PickFrame(renderer)
    manager = viewer.controller._render_manager
    scene = manager._scenes[viewer.scene.id]
    out = []
    for y in range(step // 2, SIZE, step):
        for x in range(step // 2, SIZE, step):
            if frame[y, x, :3].max() <= 8:
                continue
            info = pick.info((x + 0.5, y + 0.5))
            hit = info.get("world_object")
            if hit is None:
                continue
            details = manager._extract_pick_details(viewer.scene.id, hit, info)
            if details is None:
                continue
            visual_id = scene.get_visual_id_for_node(hit)
            out.append((visual_id, details))
    return out


def _picks(viewer, camera, hits, action="press") -> tuple[list, list]:
    """Send each hit through the controller; ``(masked events, image events)``."""
    controller = viewer.controller
    canvas_id = controller.get_canvas_ids(viewer.scene.id)[0]
    masked, image = [], []
    controller.on_pick(canvas_id, MaskedImagePickEvent, masked.append, owner_id=uuid4())
    controller.on_pick(canvas_id, ImagePickEvent, image.append, owner_id=uuid4())

    async def _run() -> None:
        for details in hits:
            controller._on_raw_pointer_event(_raw(viewer, details, camera, action))
        await _wait_until_loaded(controller)

    asyncio.run(_run())
    return masked, image


def _index(coordinate) -> tuple[int, ...]:
    return tuple(int(np.floor(v)) for v in coordinate)


#: ``(op, band)``: inward from iso 4 over 8, max is band 2 and min band 1.
OP_BANDS = [("max", 2.0), ("min", 1.0)]


@pytest.fixture(scope="module")
def picks_3d(tmp_path_factory):
    """``picks_3d(kind)``: ``op -> (hits, events, image events)``, and the arrays.

    The ops differ only in the single appearance, so one viewer per store
    kind serves both: the projection is set on the live model, then the
    frame drawn and its hits picked again.
    """
    cache: dict[str, tuple] = {}

    def get(kind: str) -> tuple[dict, np.ndarray, np.ndarray]:
        if kind not in cache:
            root = tmp_path_factory.mktemp(f"pick_{kind}")
            viewer, model, mask_data, value_data = _viewer(
                root, kind, "3d", iso_threshold=4.0, surface_thickness=8.0
            )
            out = {}
            try:
                for op, _band in OP_BANDS:
                    model.single.surface_projection = op
                    asyncio.run(_wait_until_loaded(viewer.controller))
                    hits = _hits(viewer)
                    events, image_events = _picks(viewer, "3d", hits)
                    # Copies: the handlers stay subscribed for the next op.
                    out[op] = (hits, list(events), list(image_events))
            finally:
                viewer.controller.close()
            cache[kind] = (out, mask_data, value_data)
        return cache[kind]

    return get


@pytest.mark.parametrize("kind", ["memory", "lazy"])
@pytest.mark.parametrize(("op", "band"), OP_BANDS)
def test_3d_picks_report_both_images(picks_3d, kind, op, band):
    """Inward from iso 4 over 8: max is band 2 and min band 1 (as drawn)."""
    by_op, mask_data, value_data = picks_3d(kind)
    hits, events, image_events = by_op[op]
    assert len(hits) > 40
    assert image_events == []
    assert len(events) == len(hits)
    for event in events:
        info = event.pick_info
        assert event.camera_type == "3d"
        index = _index(info.data_coordinate)
        assert info.mask_value == float(mask_data[index])
        assert info.value_coordinate == info.data_coordinate
        assert info.surface_value == float(value_data[index])
        # The hit is on the surface, within a voxel's worth of the iso.
        assert abs(info.mask_value - 4.0) <= 1.0, info
        assert info.projected_value == band, info


@pytest.mark.parametrize("kind", ["memory", "lazy"])
def test_2d_picks_report_the_value_under_the_pixel(tmp_path, kind):
    """In the band, a pick's projected value is its own value."""
    viewer, _model, mask_data, value_data = _viewer(
        tmp_path, kind, "2d", iso_threshold=4.0, surface_thickness=8.0
    )
    try:
        hits = _hits(viewer, step=6)
        assert len(hits) > 40
        events, image_events = _picks(viewer, "2d", hits)
        assert image_events == []
        assert len(events) == len(hits)
        bands = set()
        for event in events:
            info = event.pick_info
            index = _index(info.data_coordinate)
            assert index[0] == 31  # the slice drawn
            assert info.mask_value == float(mask_data[index])
            assert info.surface_value == float(value_data[index])
            assert info.projected_value == info.surface_value
            # An inward band from iso 4 over 8 lies between 4 and 12.
            assert 3.0 <= info.mask_value <= 13.0, info
            bands.add(info.surface_value)
        assert bands == {1.0, 2.0}
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_picks_read_a_value_on_its_own_grid(tmp_path, dim):
    """A value twice as fine: its coordinate is the mask's mapped, its value read."""
    viewer, _model, _mask_data, value_data = _viewer(
        tmp_path, "memory", dim, value_scale=2, iso_threshold=4.0
    )
    try:
        hits = _hits(viewer, step=4)
        assert len(hits) > 20
        events, _ = _picks(viewer, dim, hits)
        assert len(events) == len(hits)
        for event in events:
            info = event.pick_info
            # Value voxel 2c + 1/2 (centred) is mask voxel c.
            expected = 2.0 * (np.asarray(info.data_coordinate) - 0.5) + 1.0
            assert np.allclose(info.value_coordinate, expected), info
            assert info.surface_value == float(
                value_data[_index(info.value_coordinate)]
            )
    finally:
        viewer.controller.close()


def test_a_newer_move_cancels_a_lazy_pick_in_flight(tmp_path):
    """Of three hover moves only the last is answered, through both passes."""
    viewer, _model, _mask_data, _value_data = _viewer(
        tmp_path, "lazy", "3d", iso_threshold=4.0, surface_thickness=8.0
    )
    try:
        hits = _hits(viewer)[:3]
        events, _ = _picks(viewer, "3d", hits, action="move")
        assert len(events) == 1
    finally:
        viewer.controller.close()


def test_a_cut_face_pick_projects_into_the_kept_side(tmp_path):
    """The cross section of Phase 6: a z ramp read along +z from z = 32.

    The samples are at z = 32.5 to 37.5, read at the nearest voxel: max 37
    or 38 and mean 34.5 or 35.5 by the hit's side of 32.  The wrong sign
    would read 32 and 29.5 or below.  Face picks within a voxel of the
    contour are left out: there the nearest voxel can read the mask below
    the threshold where the drawn (linear) read is above it, and the pick
    then takes the surface's normal, a pick's known approximation.
    """
    mask_data, _ = band_pair(SIDE)
    ramp = np.broadcast_to(
        np.arange(SIDE, dtype=np.float32)[:, None, None], (SIDE,) * 3
    ).copy()
    viewer = new_viewer("3d")
    mask = store("memory", tmp_path, "mask", mask_data)
    model = viewer.controller.add_masked_image(
        mask,
        store("memory", tmp_path, "value", ramp),
        viewer.scene.id,
        single=MaskedImageSingleAppearance(
            color_map="viridis",
            clim=(30.0, 39.0),
            iso_threshold=4.0,
            surface_thickness=6.0,
        ),
        render_config=masked_config(),
    )
    centre = (SIDE - 1) / 2.0
    model.clipping_planes = [
        ClippingPlane.from_point_normal(
            mask.data_coordinate_systems[0], (32.0, centre, centre), (1.0, 0.0, 0.0)
        )
    ]
    try:
        load(viewer, size=(SIZE, SIZE))
        view = canvas_view(viewer)
        view.camera.show_object(
            view._get_scene_fn(viewer.scene.id), view_dir=(0.0, 0.0, 1.0)
        )
        events, _ = _picks(viewer, "3d", _hits(viewer, step=4))
        on_face = [
            event.pick_info
            for event in events
            if abs(event.pick_info.data_coordinate[0] - 0.5 - 32.0) <= 0.1
            and event.pick_info.mask_value >= 5.0
        ]
        assert len(on_face) > 40
        for info in on_face:
            assert info.projected_value in (37.0, 38.0), info
    finally:
        viewer.controller.close()
