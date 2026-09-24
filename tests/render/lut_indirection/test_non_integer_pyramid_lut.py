"""LUT writers and scale buffers on a pyramid whose level ratio is not an integer.

The y axis is 49 -> 24 -> 12 voxels (ratios 2.04 and 4.08) with a block size
of 4, so the base grid has 13 rows but level 3 only has 3 bricks x 4 cells.
Before the shared cell -> brick rule the last row was never written at level
3, and the shaders picked bricks with the float ratio while the LUT used the
rounded one.  See ``docs/Explanations/multiscale_brick_lookup.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render.block_cache import (
    BlockKey3D,
    TileManager3D,
    compute_block_cache_parameters_3d,
)
from cellier.render.lut_indirection import BlockLayout3D, LutIndirectionManager3D
from cellier.render.lut_indirection._cell_brick_rule import (
    UNBOUNDED_BRICK_COUNT,
    brick_for_cell,
    level_brick_counts,
    level_cell_spans,
)
from cellier.render.lut_indirection._layout_2d import BlockLayout2D
from cellier.render.lut_indirection._lut_buffers_2d import build_block_scales_buffer_2d
from cellier.render.lut_indirection._lut_indirection_manager_2d import (
    LutIndirectionManager2D,
)
from cellier.render.shaders._multiscale_volume_brick import build_brick_scales_buffer

BLOCK_SIZE = 4

SHAPES_3D = [(4, 49, 12), (4, 24, 6), (4, 12, 3)]
SCALES_3D = [
    tuple(s0 / sk for s0, sk in zip(SHAPES_3D[0], shape)) for shape in SHAPES_3D
]
SHAPES_2D = [shape[1:] for shape in SHAPES_3D]
SCALES_2D = [scale[1:] for scale in SCALES_3D]


def _grid(shape):
    return tuple(-(-n // BLOCK_SIZE) for n in shape)


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------


def _manager_3d(level_shapes=SHAPES_3D):
    layout = BlockLayout3D(volume_shape=SHAPES_3D[0], block_size=BLOCK_SIZE)
    manager = LutIndirectionManager3D(
        layout,
        n_levels=len(SHAPES_3D),
        level_scale_vecs_data=SCALES_3D,
        level_shapes=level_shapes,
    )
    params = compute_block_cache_parameters_3d(
        block_size=BLOCK_SIZE, gpu_budget_bytes=512 * 6**3 * 4
    )
    return manager, TileManager3D(params)


def _paint_3d(manager, tile_manager, keys) -> dict:
    """Paint *keys* (one phase), each in its own atlas slot; grid pos -> key."""
    grid_pos = np.array(
        [tile_manager._slot_grid_pos(i + 1) for i in range(len(keys))]
    ).reshape(-1, 3)
    manager.paint(
        np.array([k.level for k in keys]),
        np.array([(k.g0, k.g1, k.g2) for k in keys]).reshape(-1, 3),
        grid_pos,
        np.zeros(len(keys), np.float32),
        [np.arange(len(keys))],
    )
    return {tuple(int(v) for v in pos): key for pos, key in zip(grid_pos, keys)}


@pytest.mark.parametrize("level", [1, 2, 3])
def test_3d_every_cell_is_written_by_the_brick_the_rule_names(level):
    manager, tile_manager = _manager_3d()
    counts = level_brick_counts(
        level_cell_spans(3, 3, SCALES_3D), _grid(SHAPES_3D[0]), BLOCK_SIZE, SHAPES_3D
    )[level - 1]
    spans = level_cell_spans(3, 3, SCALES_3D)[level - 1]
    keys = [
        BlockKey3D(level=level, g0=a, g1=b, g2=c)
        for a in range(counts[0])
        for b in range(counts[1])
        for c in range(counts[2])
    ]
    slot_to_key = _paint_3d(manager, tile_manager, keys)

    lut = manager.lut_data
    assert np.all(lut[..., 3] == level)  # no cell left unwritten
    for d, h, w in np.ndindex(lut.shape[:3]):
        sx, sy, sz, _level = (int(v) for v in lut[d, h, w])
        key = slot_to_key[(sz, sy, sx)]
        assert key.g0 == brick_for_cell(d, spans[0], counts[0])
        assert key.g1 == brick_for_cell(h, spans[1], counts[1])
        assert key.g2 == brick_for_cell(w, spans[2], counts[2])


def test_3d_without_level_shapes_the_tail_row_is_not_owned():
    """Why the managers take ``level_shapes``: without it, no brick count."""
    manager, tile_manager = _manager_3d(level_shapes=None)
    keys = [BlockKey3D(level=3, g0=0, g1=b, g2=0) for b in range(3)]
    _paint_3d(manager, tile_manager, keys)

    assert np.all(manager.lut_data[:, 12, :, 3] == 0)
    assert np.all(manager.lut_data[:, :12, :, 3] == 3)


def test_brick_scales_buffer_carries_the_rule_in_shader_order():
    buffer = build_brick_scales_buffer(
        SCALES_3D, level_shapes=SHAPES_3D, block_size=BLOCK_SIZE
    )
    data = buffer.data
    # data order (z, y, x) -> shader order (x, y, z)
    assert tuple(data["span_2"][:3]) == (2.0, 2.0, 1.0)
    assert tuple(data["span_3"][:3]) == (4.0, 4.0, 1.0)
    assert tuple(data["bricks_3"][:3]) == (1.0, 3.0, 1.0)
    assert data["scale_3"][1] == pytest.approx(49 / 12)
    # Unused levels are harmless: span 1, no clamp.
    assert tuple(data["span_7"][:3]) == (1.0, 1.0, 1.0)
    assert tuple(data["bricks_7"][:3]) == (float(UNBOUNDED_BRICK_COUNT),) * 3


def test_brick_scales_buffer_without_shapes_leaves_counts_unbounded():
    data = build_brick_scales_buffer(SCALES_3D).data
    assert tuple(data["span_3"][:3]) == (4.0, 4.0, 1.0)
    assert tuple(data["bricks_3"][:3]) == (float(UNBOUNDED_BRICK_COUNT),) * 3


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [1, 2, 3])
def test_2d_every_cell_is_written_by_the_tile_the_rule_names(level):
    layout = BlockLayout2D.from_shape(shape=SHAPES_2D[0], block_size=BLOCK_SIZE)
    manager = LutIndirectionManager2D(
        layout,
        n_levels=3,
        scale_vecs_data=SCALES_2D,
        level_shapes=SHAPES_2D,
    )
    spans = level_cell_spans(3, 2, SCALES_2D)[level - 1]
    counts = level_brick_counts(
        level_cell_spans(3, 2, SCALES_2D), _grid(SHAPES_2D[0]), BLOCK_SIZE, SHAPES_2D
    )[level - 1]
    grids = np.array(
        [(a, b) for a in range(counts[0]) for b in range(counts[1])], dtype=np.int64
    )
    # Each tile gets its own slot position, so a cell names its tile.
    slots = np.column_stack([np.arange(len(grids)) // 16, np.arange(len(grids)) % 16])
    manager.paint(np.full(len(grids), level), grids, slots, [np.arange(len(grids))])

    lut = manager.lut_data
    assert np.all(lut[..., 2] == level)
    slot_to_grid = {(int(sy), int(sx)): tuple(g) for (sy, sx), g in zip(slots, grids)}
    for h, w in np.ndindex(lut.shape[:2]):
        sx, sy = int(lut[h, w, 0]), int(lut[h, w, 1])
        g0, g1 = slot_to_grid[(sy, sx)]
        assert g0 == brick_for_cell(h, spans[0], counts[0])
        assert g1 == brick_for_cell(w, spans[1], counts[1])


def test_block_scales_buffer_2d_carries_the_rule_in_shader_order():
    data = build_block_scales_buffer_2d(
        level_scale_vecs_data=SCALES_2D, level_shapes=SHAPES_2D, block_size=BLOCK_SIZE
    ).data
    # data order (y, x) -> shader order (x, y)
    assert tuple(data["span_2"][:2]) == (2.0, 2.0)
    assert tuple(data["span_3"][:2]) == (4.0, 4.0)
    assert tuple(data["bricks_3"][:2]) == (1.0, 3.0)
    assert data["scale_3"][1] == pytest.approx(12 / 49)


def test_block_scales_buffer_2d_power_of_two_fallback():
    data = build_block_scales_buffer_2d(n_levels=3).data
    assert tuple(data["span_3"][:2]) == (4.0, 4.0)
    assert tuple(data["bricks_3"][:2]) == (float(UNBOUNDED_BRICK_COUNT),) * 2


# -- Level offsets (plan v2, Phase 4) ------------------------------------------


def test_brick_scales_buffer_carries_level_offsets_in_shader_order():
    translations = [np.array([0.0, 0.0, 0.0]), np.array([0.5, 1.0, 1.5])]
    data = build_brick_scales_buffer(
        [np.ones(3), np.array([2.0, 3.0, 4.0])],
        level_translation_vecs_data=translations,
    ).data
    # LUT level k (1-based) is data level k - 1; shader order reverses (z, y, x).
    np.testing.assert_array_equal(data["offset_1"][:3], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(data["offset_2"][:3], [1.5, 1.0, 0.5])
    # Without translations every offset is zero.
    plain = build_brick_scales_buffer([np.ones(3), np.array([2.0, 3.0, 4.0])]).data
    assert not any(plain[f"offset_{k}"].any() for k in range(10))


def test_block_scales_buffer_2d_carries_level_offsets_uninverted():
    data = build_block_scales_buffer_2d(
        level_scale_vecs_data=[np.ones(2), np.array([2.0, 4.0])],
        level_translation_vecs_data=[np.zeros(2), np.array([0.5, 1.5])],
    ).data
    # scale_k holds 1 / s; offset_k holds t itself, shader order (x, y).
    np.testing.assert_allclose(data["scale_2"][:2], [0.25, 0.5])
    np.testing.assert_array_equal(data["offset_2"][:2], [1.5, 0.5])
