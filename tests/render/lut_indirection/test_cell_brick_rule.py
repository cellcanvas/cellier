"""Tests for the cell -> brick rule shared by the LUT writers and the shaders.

The motivating pyramid is the light-sheet organoid dataset: z is never
downsampled and y/x are halved with odd sizes, so the level ratios are not
integers (389 -> 194 -> 96 -> 47 -> 23 in y).  See
``docs/Explanations/multiscale_brick_lookup.md``.
"""

from __future__ import annotations

import math

import pytest

from cellier.render.lut_indirection._cell_brick_rule import (
    brick_for_cell,
    brick_rule_issues,
    cell_range,
    level_brick_counts,
    level_cell_spans,
    max_out_of_brick,
)

BLOCK_SIZE = 32

#: (z, y, x) level shapes of the light-sheet pyramid.
LIGHTSHEET_SHAPES = [
    (76, 389, 610),
    (76, 194, 305),
    (76, 96, 152),
    (76, 47, 76),
    (76, 23, 38),
]
#: Extent-preserving scales, exactly as the dataset's OME metadata records them.
LIGHTSHEET_SCALES = [
    tuple(s0 / sk for s0, sk in zip(LIGHTSHEET_SHAPES[0], shape))
    for shape in LIGHTSHEET_SHAPES
]


def _rule_tables(shapes, scales, ndim):
    spans = level_cell_spans(len(shapes), ndim, scales)
    grid = tuple(math.ceil(n / BLOCK_SIZE) for n in shapes[0])
    counts = level_brick_counts(spans, grid, BLOCK_SIZE, shapes)
    return spans, counts, grid


def test_spans_round_half_up_and_never_drop_below_one():
    scales = [(1.0, 0.4), (2.005, 2.5), (4.052, 3.49), (16.9, 16.05)]
    spans = level_cell_spans(4, 2, scales)
    assert spans == [(1, 1), (2, 3), (4, 3), (17, 16)]


def test_spans_fall_back_to_powers_of_two():
    assert level_cell_spans(3, 3) == [(1, 1, 1), (2, 2, 2), (4, 4, 4)]
    # A level past the supplied vectors also falls back.
    assert level_cell_spans(2, 2, [(1.0, 1.0)]) == [(1, 1), (2, 2)]


def test_lightsheet_spans_and_counts():
    spans, counts, grid = _rule_tables(LIGHTSHEET_SHAPES, LIGHTSHEET_SCALES, 3)
    assert grid == (3, 13, 20)
    assert spans == [(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 8, 8), (1, 17, 16)]
    assert counts == [(3, 13, 20), (3, 7, 10), (3, 3, 5), (3, 2, 3), (3, 1, 2)]


def test_counts_without_shapes_never_clamp_inside_the_grid():
    spans = [(1, 1), (2, 2), (4, 4)]
    counts = level_brick_counts(spans, (13, 20), BLOCK_SIZE)
    assert counts == [(13, 20), (7, 10), (4, 5)]
    for span, count in zip(spans, counts):
        for axis, grid_dim in enumerate((13, 20)):
            for cell in range(grid_dim):
                assert cell // span[axis] <= count[axis] - 1


@pytest.mark.parametrize("level", range(5))
def test_cell_ranges_partition_the_grid_and_agree_with_brick_for_cell(level):
    """Every base cell is written by exactly the brick the shader will pick."""
    spans, counts, grid = _rule_tables(LIGHTSHEET_SHAPES, LIGHTSHEET_SCALES, 3)
    for axis in range(3):
        span, count = spans[level][axis], counts[level][axis]
        owners: dict[int, int] = {}
        for brick in range(count):
            start, stop = cell_range(brick, span, count, grid[axis])
            for cell in range(start, stop):
                assert cell not in owners
                owners[cell] = brick
                assert brick_for_cell(cell, span, count) == brick
        assert sorted(owners) == list(range(grid[axis]))


def test_the_last_brick_owns_the_tail():
    """Level 3 y: 3 bricks x 4 cells = 12 < 13, so cell 12 goes to brick 2.

    Before the rule this row was never written at level 3.
    """
    assert cell_range(2, 4, 3, 13) == (8, 13)
    assert brick_for_cell(12, 4, 3) == 2


@pytest.mark.parametrize(
    ("level", "axis", "expected"),
    [
        (2, 1, 0.49),
        (3, 1, 0.82),
        (4, 1, 1.07),
        (5, 1, 0.00),
        (2, 2, 0.00),
        (3, 2, 0.42),
        (4, 2, 0.21),
        (5, 2, 0.10),
        (4, 0, 0.00),
    ],
)
def test_lightsheet_out_of_brick_distances(level, axis, expected):
    spans, counts, _grid = _rule_tables(LIGHTSHEET_SHAPES, LIGHTSHEET_SCALES, 3)
    out_of_brick, unreachable = max_out_of_brick(
        LIGHTSHEET_SCALES[level - 1][axis],
        spans[level - 1][axis],
        counts[level - 1][axis],
        LIGHTSHEET_SHAPES[0][axis],
        BLOCK_SIZE,
    )
    assert out_of_brick == pytest.approx(expected, abs=0.01)
    assert unreachable == ()


def test_a_power_of_two_pyramid_lands_exactly_inside_its_bricks():
    shapes = [(256, 256), (128, 128), (64, 64), (32, 32)]
    scales = [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (8.0, 8.0)]
    spans, counts, _grid = _rule_tables(shapes, scales, 2)
    for level in range(4):
        for axis in range(2):
            out_of_brick, unreachable = max_out_of_brick(
                scales[level][axis],
                spans[level][axis],
                counts[level][axis],
                shapes[0][axis],
                BLOCK_SIZE,
            )
            assert out_of_brick == pytest.approx(0.0, abs=1e-9)
            assert unreachable == ()


def test_bricks_that_own_no_cell_are_reported():
    """Rounding the span *up* over many bricks can push the last ones off the grid.

    Scale 1.6 rounds to span 2; 7 level-k bricks x 2 cells = 14 > 10 base cells.
    """
    out_of_brick, unreachable = max_out_of_brick(1.6, 2, 7, 320, BLOCK_SIZE)
    assert unreachable == (5, 6)
    assert out_of_brick > 0.0


def test_lightsheet_3d_padding_fits_images_and_labels():
    """Image bricks carry 3 voxels of padding and label bricks 2."""
    assert brick_rule_issues(LIGHTSHEET_SCALES, LIGHTSHEET_SHAPES, BLOCK_SIZE, 3) == []
    assert brick_rule_issues(LIGHTSHEET_SCALES, LIGHTSHEET_SHAPES, BLOCK_SIZE, 2) == []


def test_lightsheet_2d_tiles_with_one_voxel_of_padding_are_reported():
    shapes_2d = [shape[1:] for shape in LIGHTSHEET_SHAPES]
    scales_2d = [scale[1:] for scale in LIGHTSHEET_SCALES]
    issues = brick_rule_issues(scales_2d, shapes_2d, BLOCK_SIZE, 1)
    assert [(issue.level, issue.axis) for issue in issues] == [(3, 0), (4, 0)]
    assert issues[1].out_of_brick == pytest.approx(1.07, abs=0.01)
    assert "level 4 axis 0" in issues[1].describe()


def test_no_issues_without_scales_or_shapes():
    assert brick_rule_issues(None, LIGHTSHEET_SHAPES, BLOCK_SIZE, 1) == []
    assert brick_rule_issues(LIGHTSHEET_SCALES, None, BLOCK_SIZE, 1) == []


# -- Translation term (plan v2, Phase 4) ---------------------------------------


@pytest.mark.parametrize(
    ("kind", "translation", "expected"),
    [
        # Block averaging sits exactly on the bricks: t = (s - 1) / 2.
        ("block_average", lambda s: (s - 1) / 2, lambda s: 0.0),
        # Plain striding is shifted by -(s - 1) / 2 level-0 voxels.
        ("plain", lambda s: 0.0, lambda s: (s - 1) / (2 * s)),
        # Offset striding (t = s // 2) is shifted by +0.5 level-0 voxel.
        ("offset", lambda s: float(s // 2), lambda s: 0.5 / s),
    ],
)
def test_translation_term_on_a_power_of_two_pyramid(kind, translation, expected):
    """Out-of-brick distance = the shift from block averaging, in level-k voxels."""
    shapes = [(256,), (128,), (64,), (32,)]
    scales = [(1.0,), (2.0,), (4.0,), (8.0,)]
    spans, counts, _grid = _rule_tables(shapes, scales, 1)
    for level in range(1, 4):
        s = scales[level][0]
        out_of_brick, _ = max_out_of_brick(
            s, spans[level][0], counts[level][0], 256, BLOCK_SIZE, translation(s)
        )
        assert out_of_brick == pytest.approx(expected(s), abs=1e-9), (kind, s)


def test_no_translation_means_block_averaging():
    spans, counts, _grid = _rule_tables(LIGHTSHEET_SHAPES, LIGHTSHEET_SCALES, 3)
    for level in range(5):
        for axis in range(3):
            s = LIGHTSHEET_SCALES[level][axis]
            args = (s, spans[level][axis], counts[level][axis])
            rest = (LIGHTSHEET_SHAPES[0][axis], BLOCK_SIZE)
            assert max_out_of_brick(*args, *rest) == max_out_of_brick(
                *args, *rest, (s - 1) / 2
            )


def test_lightsheet_with_its_published_translations():
    """Scale-only metadata (t = 0): the worst case drops from 1.07 to 0.63."""
    zeros = [(0.0, 0.0, 0.0)] * len(LIGHTSHEET_SHAPES)
    spans, counts, _grid = _rule_tables(LIGHTSHEET_SHAPES, LIGHTSHEET_SCALES, 3)
    worst = max(
        max_out_of_brick(
            LIGHTSHEET_SCALES[level][axis],
            spans[level][axis],
            counts[level][axis],
            LIGHTSHEET_SHAPES[0][axis],
            BLOCK_SIZE,
            zeros[level][axis],
        )[0]
        for level in range(5)
        for axis in range(3)
    )
    assert worst == pytest.approx(0.63, abs=0.01)


def test_issues_report_the_translation_and_honour_the_margin():
    shapes_2d = [shape[1:] for shape in LIGHTSHEET_SHAPES]
    scales_2d = [scale[1:] for scale in LIGHTSHEET_SCALES]
    zeros = [(0.0, 0.0)] * len(shapes_2d)
    # Border 1, nearest sampling (2D labels): 0.63 fits in 1.0.
    assert (
        brick_rule_issues(
            scales_2d,
            shapes_2d,
            BLOCK_SIZE,
            1,
            sampling_margin=0.0,
            translation_vecs_data=zeros,
        )
        == []
    )
    # Border 1, linear (2D image): allowance 0.5.
    issues = brick_rule_issues(
        scales_2d, shapes_2d, BLOCK_SIZE, 1, translation_vecs_data=zeros
    )
    assert issues
    assert all(issue.translation == 0.0 for issue in issues)
    assert "translation 0" in issues[0].describe()
