"""The MaskedImage's 2D band.

The band is compared with the exact in-plane truth stored in
``masked_band_truth.json`` (``tests/render/_masked_band.py`` has the scenes;
``scripts/masked_image/freeze_band_truth.py`` wrote the truth).  The bar: no
extra pixel, edges within a pixel, and the fan's known shortfall (a few
percent of a band's far edge at 16 rays).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pytest
from scipy import ndimage as ndi

from cellier.convenience.capture import _wait_until_loaded
from cellier.data import ImageMemoryStore
from cellier.render.block_cache._cell_ranges import CellRanges
from cellier.render.block_cache._image_residency import pack_keys
from tests.render import _masked_band as band
from tests.render._masked import gfx_visual, masked_config, new_viewer, pyramid_store

TRUTH = band.load_truth()
MASKS = band.masks()
CASES = [
    (mask, view, name)
    for mask in band.MASK_KINDS
    for view in band.VIEWS
    for name in band.BANDS
]


def _truth(mask: str, view: str) -> tuple[np.ndarray, float]:
    case = TRUTH["cases"][f"{mask}/{view}"]
    return band.decode_distance(case["d"]), float(case["px"])


def _frame(viewer) -> np.ndarray:
    frames = band.Frames(viewer)
    frames.frame()
    return frames.frame()


def _ranges(viewer, model) -> CellRanges:
    return gfx_visual(viewer, model.id)._slots[0]._residency_3d.cell_ranges


def _without_skip(viewer, model) -> np.ndarray:
    """A frame with every cell's range level unmatched: nothing skips."""
    ranges = _ranges(viewer, model)
    ranges.data[..., 2] = -1.0
    ranges.texture.update_full()
    return _frame(viewer)


def _set_band(viewer, model, name: str) -> None:
    """Give the live model band *name*'s thickness and placement."""
    interval = band.BANDS[name]
    model.single.surface_thickness = interval[1] - interval[0]
    model.single.surface_placement = band.placement(interval)
    asyncio.run(_wait_until_loaded(viewer.controller))


# -- one viewer a scene -------------------------------------------------------------
#
# The truth and skip cases differ only in the band, an appearance, so each
# (mask, view) scene is built once and every band drawn from it.  On CI's
# software GPU the build and its first frames are most of a case's time.

#: The bands the skip is checked on.
SKIP_BANDS = ("inward_t8", "outward_t8")


@dataclass
class _Scene:
    """Frames of one (mask, view) scene, and its read uniforms."""

    band: dict[str, np.ndarray] = field(default_factory=dict)
    skipped: dict[str, np.ndarray] = field(default_factory=dict)
    plain: dict[str, np.ndarray] = field(default_factory=dict)
    uniforms: dict[str, object] = field(default_factory=dict)


def _render_scene(root, mask: str, view: str) -> _Scene:
    data, iso = MASKS[mask]
    names = list(band.BANDS)
    viewer, model = band.build(root, data, iso, view, band.BANDS[names[0]])
    out = _Scene()
    try:
        visual = gfx_visual(viewer, model.id)
        uniforms = visual._slots[0].material_3d.uniform_buffer.data
        out.uniforms = {
            "mask_snap": list(uniforms["mask_snap"][:3]),
            "value_snap": list(uniforms["value_snap"][:3]),
            "mask_edge_fade": float(uniforms["mask_edge_fade"]),
        }
        for name in names:
            _set_band(viewer, model, name)
            out.band[name] = _frame(viewer)
            if name in SKIP_BANDS:
                visual.tick()
                out.skipped[name] = _frame(viewer)
        # Last: the unmatched table stays unmatched for the rest of the viewer.
        for name in SKIP_BANDS:
            _set_band(viewer, model, name)
            out.plain[name] = _without_skip(viewer, model)
    finally:
        viewer.controller.close()
    return out


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """``scene(mask, view)``: that scene's frames, drawn on first use."""
    cache: dict[tuple[str, str], _Scene] = {}

    def get(mask: str, view: str) -> _Scene:
        if (mask, view) not in cache:
            root = tmp_path_factory.mktemp(f"band_{mask}_{view}")
            cache[mask, view] = _render_scene(root, mask, view)
        return cache[mask, view]

    return get


# -- the band against the truth ---------------------------------------------------


@pytest.mark.parametrize(("mask", "view", "name"), CASES)
def test_the_band_matches_the_in_plane_truth(scene, mask, view, name):
    """Every mask kind, aligned and tilted, two levels, four placements."""
    frame = scene(mask, view).band[name]
    d, px = _truth(mask, view)
    drawn = band.drawn(frame)
    score = band.score(drawn, d, band.BANDS[name], px)
    assert score["drawn"] > 0, score
    assert score["missing"] <= max(8, 0.025 * score["expected"]), score
    # Edges within a pixel, never past the band.
    for edge in ("reach_past_outer_px", "reach_past_inner_px"):
        assert -1.0 <= score[edge] <= 0.25, score
    # No pixel the truth does not draw, at any coverage, as P0-18 counted (a
    # crossing past the reach does not count: the tilted cases).
    covered = frame[..., :3].max(axis=-1) > 0
    assert band.score(covered, d, band.BANDS[name], px)["extra"] == 0
    if name == "hairline":
        # Thickness 0 is one closed ring.  Any coverage counts: where the
        # contour lies on a pixel boundary two rows share it at half.
        labels, _n = ndi.label(~covered)
        rows, cols = np.nonzero(covered)
        middle = (int(np.median(rows)), int(np.median(cols)))
        assert labels[middle] != labels[0, 0]


def test_a_flat_mask_at_its_value_draws_nothing(tmp_path):
    """The example's zero outside at threshold 0: no contour anywhere."""
    data, _iso = MASKS["edt_clamped"]
    viewer, _model = band.build(tmp_path, data, 0.0, "L0_aligned", (0.0, 3.4))
    try:
        assert not band.drawn(_frame(viewer)).any()
    finally:
        viewer.controller.close()


# -- the per-cell skip -------------------------------------------------------------


@pytest.mark.parametrize("mask", band.MASK_KINDS)
@pytest.mark.parametrize("view", band.VIEWS)
@pytest.mark.parametrize("name", SKIP_BANDS)
def test_the_skip_changes_no_pixel(scene, mask, view, name):
    frames = scene(mask, view)
    np.testing.assert_array_equal(frames.skipped[name], frames.plain[name])


def test_a_uniformly_inside_slice_casts_no_ray(tmp_path):
    """A counting readout (debug_mode 'band_skip'), not a timing.

    The mask is inside everywhere, so every cell is wholly on one side and
    clear of any other: every pixel whose reach stays on the data's own
    reads exits before its first ray.  With the table unmatched, none does.
    """
    data = np.ones((band.SIDE,) * 3, np.float32)
    viewer, model = band.build(tmp_path, data, 0.5, "L0_aligned", (0.0, 8.0))
    try:
        material = gfx_visual(viewer, model.id)._slots[0].material_3d
        material.debug_mode = "band_skip"
        gfx_visual(viewer, model.id).tick()
        skipped = band.drawn(_frame(viewer)) | (_frame(viewer)[..., :3].max(-1) > 8)
        world, _px = band.pixel_world(band.Frames(viewer).view.camera)
        # The data's own reads, a reach (8 world units) in from every edge.
        interior = (
            (world[..., 1] > 8.5)
            & (world[..., 1] < band.SIDE - 9.5)
            & (world[..., 2] > 8.5)
            & (world[..., 2] < band.SIDE - 9.5)
        )
        assert interior.sum() > 5000
        assert skipped[interior].mean() > 0.99
        none = _without_skip(viewer, model)[..., :3].max(-1) > 8
        assert not none.any()
    finally:
        viewer.controller.close()


# -- unloaded cells ---------------------------------------------------------------


#: The bands the unloaded-cell case draws.
UNLOADED_BANDS = ("inward_t8", "outward_t8", "hairline")


@pytest.fixture(scope="module")
def unloaded(tmp_path_factory) -> dict:
    """Half the binary mask's cells blanked: ``name -> (frame, plain, world)``.

    One viewer for every band; the frames with the skip first, then the
    unmatched table (which stays unmatched) for each.
    """
    data, iso = MASKS["binary"]
    root = tmp_path_factory.mktemp("band_unloaded")
    viewer, model = band.build(
        root, data, iso, "L0_aligned", band.BANDS[UNLOADED_BANDS[0]]
    )
    out = {}
    try:
        visual = gfx_visual(viewer, model.id)
        lut = visual._slots[0]._lut_manager_3d
        # Cells of x index 2 and 3: x from 31.5 on.
        lut.lut_data[:, :, 2:, :] = 0
        lut.lut_tex.update_range((0, 0, 0), lut.lut_tex.size)
        _ranges(viewer, model).repaint(lut.lut_data)
        world, _px = band.pixel_world(band.Frames(viewer).view.camera)
        frames = {}
        for name in UNLOADED_BANDS:
            _set_band(viewer, model, name)
            frames[name] = _frame(viewer)
        for name in UNLOADED_BANDS:
            _set_band(viewer, model, name)
            out[name] = (frames[name], _without_skip(viewer, model), world)
    finally:
        viewer.controller.close()
    return out


@pytest.mark.parametrize("name", UNLOADED_BANDS)
def test_an_unloaded_cell_invents_no_contour(unloaded, name):
    """Half the mask's cells blanked: nothing drawn in them, nothing extra.

    A ray that read the blank half as zero would draw a contour along the
    loaded region's edge, inside the ellipse; the rule goes on past it, so
    every drawn pixel is one the fully loaded truth draws, and none lies in
    the blank half.  The skip, which never crosses an unloaded cell, gives
    the same frame.
    """
    frame, plain, world = unloaded[name]
    drawn = band.drawn(frame)
    assert drawn.any()
    assert not drawn[world[..., 2] > 31.5].any()
    d, px = _truth("binary", "L0_aligned")
    assert band.score(drawn, d, band.BANDS[name], px)["extra"] == 0
    np.testing.assert_array_equal(frame, plain)


# -- the reading per side ---------------------------------------------------------


def _memory_or_lazy(tmp_path, kind: str, view: str) -> np.ndarray:
    """The probability mask's inward band, its two images in memory or lazy."""
    from cellier.transform import AffineTransform
    from cellier.visuals import MaskedImageAppearance, MaskedImageSingleAppearance
    from tests.render._masked import load

    data, iso = MASKS["probability"]
    value_data = np.linspace(0.0, 3.0, data.size, dtype=np.float32).reshape(data.shape)
    if kind == "memory":
        mask = ImageMemoryStore(data=data, name="mask")
        value = ImageMemoryStore(data=value_data, name="value")
    else:
        mask = pyramid_store(tmp_path, "mask", data, 1)
        value = pyramid_store(tmp_path, "value", value_data, 1)
    viewer = new_viewer("2d")
    try:
        controller = viewer.controller
        controller._ensure_data_coordinate_systems(viewer.scene.id, mask)
        transform = AffineTransform.from_matrix(
            band.data_matrix(band.VIEWS[view][1]),
            mask.data_coordinate_systems[0],
            viewer.scene.dims.world_coordinate_system,
        )
        controller.add_masked_image(
            mask,
            value,
            viewer.scene.id,
            transform=transform,
            appearance=MaskedImageAppearance(frustum_cull=False),
            single=MaskedImageSingleAppearance(
                color_map="viridis",
                clim=(0.0, 3.0),
                iso_threshold=iso,
                surface_thickness=6.0,
            ),
            render_config=masked_config(),
        )
        load(viewer, slice_z=band.SLICE + 0.3, size=(band.SIZE, band.SIZE))
        camera = band.canvas_view(viewer).camera
        camera.width = camera.height = band.SIZE * band.ZOOM
        return _frame(viewer)
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("view", ["L0_aligned", "L0_tilted30"])
def test_an_in_memory_side_reads_as_a_one_level_store(tmp_path, view):
    """The same arrays in memory and as one-level zarr stores."""
    memory = _memory_or_lazy(tmp_path / "m", "memory", view)
    lazy = _memory_or_lazy(tmp_path / "l", "lazy", view)
    assert band.drawn(memory).sum() > 1000
    # Within one code: a lazy store's bricks land in atlas slots in the order
    # their reads finish, and a read at another atlas coordinate can round
    # the other way (two lazy runs of the tilted view differ by a pixel about
    # half the time).  A wrong reading rule moves whole edges.
    np.testing.assert_allclose(memory.astype(int), lazy.astype(int), rtol=0, atol=1)


def test_the_data_edge_follows_the_tile_rule(tmp_path):
    """The mask's outer half voxel, read as a 2D tile reads it.

    A mask inside everywhere fades over the outer half voxel before voxel 0
    and not past the last one at level 1, so at threshold 0.75 the contour
    is at x = -0.25 on the low side.  On the high side the mask is one up to
    where it reads zero off the data, so its contour is in the last half
    voxel the slice draws.
    """
    data = np.ones((band.SIDE,) * 3, np.float32)
    viewer, _model = band.build(tmp_path, data, 0.75, "L0_aligned", (0.0, 0.0))
    try:
        frame = _frame(viewer)
        world, _px = band.pixel_world(band.Frames(viewer).view.camera)
    finally:
        viewer.controller.close()
    lin = frame[..., 0].astype(np.float64) / 255.0
    lin = np.where(lin <= 0.04045, lin / 12.92, ((lin + 0.055) / 1.055) ** 2.4)
    rows = (world[..., 1] > 10.0) & (world[..., 1] < band.SIDE - 10.0)
    low = rows & (world[..., 2] < 2.0)
    centre = float((lin[low] * world[..., 2][low]).sum() / lin[low].sum())
    assert centre == pytest.approx(-0.25, abs=0.1)
    high = rows & (world[..., 2] > band.SIDE - 4.0)
    assert band.drawn(frame)[high].any()
    assert not band.drawn(frame)[high & (world[..., 2] < band.SIDE - 1.5)].any()


def test_the_extent_clamp_leaves_2d_frames_alone(tmp_path):
    """The clamp is 3D only: every 2D frame is identical without it."""
    data, iso = MASKS["signed_distance"]
    viewer, model = band.build(tmp_path, data, iso, "L1_aligned", (0.0, 8.0))
    try:
        material = gfx_visual(viewer, model.id)._slots[0].material_3d
        frames = []
        for clamp in (True, False):
            material.extent_clamp = clamp
            frames.append(_frame(viewer))
    finally:
        viewer.controller.close()
    np.testing.assert_array_equal(*frames)


@pytest.mark.parametrize(
    ("view", "expected"), [("L0_aligned", [0, 0, 1]), ("L0_tilted30", [0, 0, 0])]
)
def test_each_side_reads_its_own_sliced_axis(scene, view, expected):
    """Aligned: both snap z (shader axis 2); tilted against the mask: neither."""
    uniforms = scene("binary", view).uniforms
    assert uniforms["mask_snap"] == expected
    assert uniforms["value_snap"] == expected
    assert uniforms["mask_edge_fade"] == float(any(expected))


# -- the range table --------------------------------------------------------------


def _table(level_shapes=((32, 32, 32),), lut_shape=(2, 2, 2)) -> CellRanges:
    return CellRanges(
        level_shapes, block_size=16, overlap=3, n_slots=26, grid_side=3,
        lut_shape=lut_shape,
    )  # fmt: skip


def _brick(value: float, edge_zero: bool = False) -> np.ndarray:
    data = np.full((22, 22, 22), value, np.float32)
    if edge_zero:
        # Past the array's far edge on x: the store's zero padding.
        data[..., 19:] = 0.0
    return data


def test_a_brick_range_leaves_out_the_padding_past_the_array():
    table = _table()
    key = int(pack_keys(np.array([1]), np.array([0]), np.array([[0, 0, 1]]))[0])
    table.record(4, key, _brick(5.0, edge_zero=True))
    assert (table.brick_min[4], table.brick_max[4]) == (5.0, 5.0)


def _paint(table: CellRanges, slots: np.ndarray) -> np.ndarray:
    """A LUT whose cells name scheduler slots *slots* at level 1 (0: none)."""
    lut = np.zeros((*slots.shape, 4), np.uint8)
    flat = slots + 1
    lut[..., 0] = flat % 3
    lut[..., 1] = (flat // 3) % 3
    lut[..., 2] = flat // 9
    lut[..., 3] = np.where(slots >= 0, 1, 0)
    return lut


def test_cells_take_their_brick_range_and_level():
    table = _table()
    for slot, value in ((0, 1.0), (1, 3.0)):
        key = int(pack_keys(np.array([1]), np.array([0]), np.array([[0, 0, 0]]))[0])
        table.record(slot, key, _brick(value))
    slots = np.array([[[0, 1], [0, 0]], [[0, 0], [-1, 0]]])
    table.set_threshold(2.0)
    table.repaint(_paint(table, slots))
    assert table.data[0, 0, 1, :3].tolist() == [3.0, 3.0, 1.0]
    assert table.data[0, 0, 0, :3].tolist() == [1.0, 1.0, 1.0]
    # The unloaded cell: no range, level 0.
    assert table.data[1, 1, 0, :3].tolist() == [0.0, 0.0, 0.0]
    # Cells below 2 next to the one above it (or the unloaded one): 0 voxels
    # of clearance; the far corner is one cell further (16 voxels).
    assert table.data[0, 0, 0, 3] == 0.0
    assert table.data[1, 1, 1, 3] == 0.0
    assert table.data[1, 0, 0, 3] == 0.0


def test_a_threshold_change_zeroes_the_clearance_until_refreshed():
    table = _table(lut_shape=(1, 3, 3))
    key = int(pack_keys(np.array([1]), np.array([0]), np.array([[0, 0, 0]]))[0])
    table.record(0, key, _brick(1.0))
    table.record(1, key, _brick(3.0))
    slots = np.zeros((1, 3, 3), np.int64)
    slots[0, 0, 0] = 1
    table.set_threshold(2.0)
    table.repaint(_paint(table, slots))
    assert table.data[0, 2, 2, 3] == 16.0
    table.set_threshold(0.5)
    assert not table.data[..., 3].any()
    table.refresh()
    # Every cell is now above 0.5: wholly clear.
    assert table.data[0, 2, 2, 3] > 1e8


def test_the_clearance_of_an_aligned_plane_is_measured_in_its_layer():
    """The layers above and below a 2D slice are not loaded."""
    table = _table(lut_shape=(3, 3, 3))
    key = int(pack_keys(np.array([1]), np.array([0]), np.array([[0, 0, 0]]))[0])
    table.record(0, key, _brick(1.0))
    slots = np.full((3, 3, 3), -1, np.int64)
    slots[1] = 0
    table.set_threshold(2.0)
    table.repaint(_paint(table, slots))
    # In 3D the unloaded layers are next to every cell.
    assert table.data[1, 1, 1, 3] == 0.0
    table.set_layer_axis(0)
    table.refresh()
    assert table.data[1, 1, 1, 3] > 1e8
