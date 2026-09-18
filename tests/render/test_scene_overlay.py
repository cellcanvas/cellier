"""Tests for the render-layer scene overlays (``GFXSceneBoundingBox``).

The node is tested directly with world bounds handed in, as the controller
would -- no controller needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render.visuals._scene_overlay import (
    GFXSceneBoundingBox,
    box_edge_positions,
    project_bounds,
)
from cellier.visuals import SceneBoundingBox, SceneBoundingBoxAppearance

_BOUNDS = (np.array([0.0, -1.0, -2.0, -3.0]), np.array([5.0, 1.0, 2.0, 3.0]))


def _edges(positions: np.ndarray) -> set[frozenset]:
    pairs = positions.reshape(-1, 2, 3)
    return {frozenset(map(tuple, pair)) for pair in pairs}


def test_a_3d_box_has_the_twelve_unit_cube_edges():
    positions = box_edge_positions(np.zeros(3), np.ones(3))

    assert positions.shape == (24, 3)
    assert positions.dtype == np.float32
    edges = _edges(positions)
    assert len(edges) == 12
    for edge in edges:
        a, b = (np.array(corner) for corner in edge)
        assert np.count_nonzero(a != b) == 1


def test_a_2d_frame_has_four_edges_at_z_zero():
    positions = box_edge_positions(np.zeros(2), np.array([2.0, 3.0]))

    assert positions.shape == (8, 3)
    assert len(_edges(positions)) == 4
    np.testing.assert_array_equal(positions[:, 2], 0.0)


def test_positions_are_reversed_to_pygfx_order():
    """Cellier displayed order (z, y, x) becomes pygfx (x, y, z)."""
    positions = box_edge_positions(np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0]))

    np.testing.assert_allclose(positions.max(axis=0), (3.0, 2.0, 1.0))


def test_project_bounds_selects_the_displayed_axes_in_order():
    low, high = project_bounds(_BOUNDS, (3, 1))

    np.testing.assert_allclose(low, (-3.0, -1.0))
    np.testing.assert_allclose(high, (3.0, 1.0))


def test_project_bounds_gives_up_on_an_axis_nothing_reaches():
    low = _BOUNDS[0].copy()
    low[2] = np.nan

    assert project_bounds((low, _BOUNDS[1]), (1, 2, 3)) is None
    assert project_bounds((low, _BOUNDS[1]), (0, 1)) is not None
    assert project_bounds(None, (1, 2)) is None


@pytest.fixture
def box() -> GFXSceneBoundingBox:
    return GFXSceneBoundingBox(SceneBoundingBox(name="box"))


def test_the_box_stays_hidden_until_it_has_bounds(box):
    assert not box.node.visible

    box.update_scene_extent(_BOUNDS, (1, 2, 3))
    assert box.node.visible

    box.update_scene_extent(None, (1, 2, 3))
    assert not box.node.visible


def test_visibility_needs_both_the_flag_and_bounds(box):
    box.apply("visible", False)
    box.update_scene_extent(_BOUNDS, (1, 2, 3))
    assert not box.node.visible

    box.apply("visible", True)
    assert box.node.visible


def test_depth_test_is_on_in_3d_and_off_in_2d(box):
    box.update_scene_extent(_BOUNDS, (1, 2, 3))
    assert box.node.material.depth_test

    box.update_scene_extent(_BOUNDS, (2, 3))
    assert not box.node.material.depth_test


def test_the_material_never_writes_depth_or_pick(box):
    assert not box.node.material.depth_write
    assert not box.node.material.pick_write


def test_apply_updates_the_material(box):
    box.apply("appearance.color", (1.0, 0.0, 0.0, 1.0))
    box.apply("appearance.thickness", 6.0)
    box.apply("appearance.render_order", 4)

    np.testing.assert_allclose(tuple(box.node.material.color), (1.0, 0.0, 0.0, 1.0))
    assert box.node.material.thickness == pytest.approx(6.0)
    assert box.node.render_order == 4


def test_apply_takes_a_whole_appearance(box):
    box.apply(
        "appearance",
        SceneBoundingBoxAppearance(
            color=(0.0, 1.0, 0.0, 1.0), thickness=3.0, render_order=2
        ),
    )

    np.testing.assert_allclose(tuple(box.node.material.color), (0.0, 1.0, 0.0, 1.0))
    assert box.node.material.thickness == pytest.approx(3.0)
    assert box.node.render_order == 2
