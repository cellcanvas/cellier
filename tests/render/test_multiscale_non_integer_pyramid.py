"""Rendering a multiscale pyramid whose level ratio is not an integer.

y is 98 -> 49 -> 24 voxels (ratios 2.0 and 4.083) with an 8-voxel block, so
the base LUT grid has 13 rows while level 3 has 3 bricks x 4 base cells.
Before the shared cell -> brick rule the coarse render disagreed with the
finest one: the MIP shader sampled base cells 4 and 8 against the previous
brick, the 2D tile shader wrapped slivers of those cells into the wrong end of
the tile, and nothing wrote the last row at level 3.  See
``docs/Explanations/multiscale_brick_lookup.md``.

Each test renders the same data forced to the finest and to the coarsest
level and compares the two.  Comparisons are made away from the silhouette:
coarse levels draw their outermost half texel slightly differently from the
finest level on every pyramid, power-of-two ones included, which is a separate
edge convention.  The LUT-level guarantees (every cell written, by the brick
the rule names) are in ``lut_indirection/test_non_integer_pyramid_lut.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorstore as ts

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.visuals import MultiscaleImageSingleAppearance
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)

BLOCK_SIZE = 8
SHAPES = [(16, 98, 16), (8, 49, 8), (4, 24, 4)]
COARSEST = len(SHAPES)
SCALES = [tuple(s0 / sk for s0, sk in zip(SHAPES[0], shape)) for shape in SHAPES]


def _write_pyramid(root, fill, dtype: str) -> None:
    for index, shape in enumerate(SHAPES):
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(root / f"s{index}")},
            "metadata": {
                "shape": list(shape),
                "data_type": dtype,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": list(shape)},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        arr = np.zeros(shape, dtype=dtype)
        fill(arr, SCALES[index])
        ts.open(spec).result()[...].write(arr).result()


def _store(root, name: str) -> MultiscaleZarrDataStore:
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=[f"s{index}" for index in range(len(SHAPES))],
        level_scales=SCALES,
        level_translations=[tuple((s - 1.0) / 2.0 for s in scale) for scale in SCALES],
        name=name,
    )


def _level0_y(arr: np.ndarray, scale) -> np.ndarray:
    """The level-0 y coordinate each level-k row samples."""
    return np.arange(arr.shape[1], dtype=np.float64) * scale[1] + 0.5


@pytest.fixture
def ramp_root(tmp_path):
    """Intensity rises along y and is constant along z and x."""

    def _fill(arr, scale):
        arr[:] = (_level0_y(arr, scale) / SHAPES[0][1]).astype(arr.dtype)[None, :, None]

    _write_pyramid(tmp_path, _fill, "float32")
    return tmp_path


@pytest.fixture
def bands_root(tmp_path):
    """Labels 1..7 in 16-voxel bands along y, constant along z and x."""

    def _fill(arr, scale):
        bands = np.floor(_level0_y(arr, scale) / 16.0) + 1
        arr[:] = bands.astype(arr.dtype)[None, :, None]

    _write_pyramid(tmp_path, _fill, "int32")
    return tmp_path


# ---------------------------------------------------------------------------
# Frame comparison helpers
# ---------------------------------------------------------------------------


def _interior(fine: np.ndarray, coarse: np.ndarray, erosion: int = 4) -> np.ndarray:
    """Pixels both frames drew, at least *erosion* pixels from either silhouette."""
    mask = (fine[..., 3] > 0) & (coarse[..., 3] > 0)
    for _ in range(erosion):
        mask = (
            mask
            & np.roll(mask, 1, 0)
            & np.roll(mask, -1, 0)
            & np.roll(mask, 1, 1)
            & np.roll(mask, -1, 1)
        )
    return mask


def _coverage_mismatch(fine: np.ndarray, coarse: np.ndarray) -> float:
    fine_mask = fine[..., 3] > 0
    coarse_mask = coarse[..., 3] > 0
    return np.count_nonzero(fine_mask != coarse_mask) / fine_mask.sum()


def _color_error_99th(fine: np.ndarray, coarse: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs(fine[..., :3].astype(int) - coarse[..., :3].astype(int))
    return float(np.percentile(diff[mask].max(axis=1), 99))


def _band_sequence(frame: np.ndarray, mask: np.ndarray) -> list[tuple[int, ...]]:
    """The colours met along the long axis, one entry per run of 3+ pixels."""
    rows, cols = np.nonzero(mask)
    if np.ptp(cols) > np.ptp(rows):
        row = (rows.min() + rows.max()) // 2
        line, keep = frame[row, :, :3], mask[row, :]
    else:
        col = (cols.min() + cols.max()) // 2
        line, keep = frame[:, col, :3], mask[:, col]
    runs: list[list] = []
    for color in map(tuple, line[keep]):
        if runs and runs[-1][0] == color:
            runs[-1][1] += 1
        else:
            runs.append([color, 1])
    return [color for color, length in runs if length >= 3]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _resident_levels(tile_manager) -> set[int]:
    return {key.level for key in tile_manager.tilemap}


async def _render_image(controller, render_scene, reslice, root, dim, level):
    scene = controller.add_scene(dim=dim, name=f"image-{dim}-{level}")
    visual = controller.add_image_multiscale(
        data=_store(root, f"ramp-{dim}-{level}"),
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(force_level=level),
        render_config=MultiscaleImageRenderConfig(block_size=BLOCK_SIZE),
        single=MultiscaleImageSingleAppearance(
            color_map="gray", clim=(0.0, 1.0), render_mode="mip"
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    cache = gfx._block_cache_3d if dim == "3d" else gfx._block_cache_2d
    assert _resident_levels(cache.tile_manager) == {level}
    return render_scene(controller, scene.id)


async def _render_labels(controller, render_scene, reslice, root, dim, level):
    scene = controller.add_scene(dim=dim, name=f"labels-{dim}-{level}")
    visual = controller.add_labels_multiscale(
        data=_store(root, f"bands-{dim}-{level}"),
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(force_level=level),
        render_config=MultiscaleLabelRenderConfig(block_size=BLOCK_SIZE),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    cache = gfx._block_cache_3d if dim == "3d" else gfx._block_cache_2d
    assert _resident_levels(cache.tile_manager) == {level}
    return render_scene(controller, scene.id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", ["2d", "3d"])
async def test_coarse_image_matches_the_finest(
    controller, render_scene, reslice, ramp_root, dim
):
    """No band of wrong intensity at the coarsest level.

    A wrong-brick read lands about a third of the ramp away from where it
    should (roughly 85 of 255 levels) across a whole base cell; resampling at
    the coarsest level stays within about 10.
    """
    fine = await _render_image(controller, render_scene, reslice, ramp_root, dim, 1)
    coarse = await _render_image(
        controller, render_scene, reslice, ramp_root, dim, COARSEST
    )

    inner = _interior(fine, coarse)
    assert inner.sum() > 500
    assert _color_error_99th(fine, coarse, inner) <= 30
    if dim == "2d":
        # 2D draws the same footprint at every level.  Before the rule the last
        # row of base cells was never written at level 3.
        assert _coverage_mismatch(fine, coarse) < 0.01


@pytest.mark.parametrize("dim", ["2d", "3d"])
async def test_coarse_labels_keep_every_band_in_order(
    controller, render_scene, reslice, bands_root, dim
):
    """The label shaders draw the coarsest level, every band in order."""
    fine = await _render_labels(controller, render_scene, reslice, bands_root, dim, 1)
    coarse = await _render_labels(
        controller, render_scene, reslice, bands_root, dim, COARSEST
    )

    fine_pixels = np.count_nonzero(fine[..., 3])
    assert fine_pixels > 0
    assert np.count_nonzero(coarse[..., 3]) >= 0.5 * fine_pixels
    if dim == "2d":
        inner = _interior(fine, coarse)
        bands = _band_sequence(fine, inner)
        assert len(bands) >= 5
        assert _band_sequence(coarse, inner) == bands
