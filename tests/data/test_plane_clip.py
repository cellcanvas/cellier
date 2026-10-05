"""The CPU clipping kernels (clipping planes design 5.2)."""

from __future__ import annotations

import numpy as np

from cellier.data._plane_clip import clip_segments, kept_points
from cellier.data.mesh._section import SectionParts, clip_parts

PLANE = ((0.0, 0.0, 1.0), 5.0)  # keeps x >= 5 over (z, y, x)


def test_kept_points_keeps_the_plane_itself():
    points = np.array([[0, 0, 4.9], [0, 0, 5.0], [9, 9, 7.0]])
    np.testing.assert_array_equal(kept_points(points, [PLANE]), [False, True, True])


def test_kept_points_is_the_intersection():
    points = np.array([[0, 0, 6.0], [0, 8, 6.0]])
    planes = [PLANE, ((0.0, -1.0, 0.0), -5.0)]  # and y <= 5
    np.testing.assert_array_equal(kept_points(points, planes), [True, False])


def test_a_segment_is_kept_dropped_or_cut():
    start = np.array([[0, 0, 6.0], [0, 0, 1.0], [0, 0, 3.0], [0, 2, 9.0]])
    end = np.array([[0, 0, 8.0], [0, 0, 2.0], [0, 4, 7.0], [0, 2, 1.0]])
    new_start, new_end, kept = clip_segments(start, end, [PLANE])
    np.testing.assert_array_equal(kept, [0, 2, 3])
    np.testing.assert_allclose(new_start, [[0, 0, 6], [0, 2, 5], [0, 2, 9]])
    np.testing.assert_allclose(new_end, [[0, 0, 8], [0, 4, 7], [0, 2, 5]])


def test_attribute_columns_are_interpolated_to_the_cut():
    # Position (z, y, x) then one attribute that runs 0 -> 1 along the segment.
    start = np.array([[0, 0, 0.0, 0.0]])
    end = np.array([[0, 0, 10.0, 1.0]])
    new_start, new_end, _ = clip_segments(start, end, [PLANE], n_position_columns=3)
    np.testing.assert_allclose(new_start, [[0, 0, 5.0, 0.5]])
    np.testing.assert_allclose(new_end, end)


def test_every_point_of_a_clipped_segment_is_on_the_kept_side():
    rng = np.random.default_rng(0)
    start, end = rng.uniform(0, 10, (500, 3)), rng.uniform(0, 10, (500, 3))
    planes = [((0.3, 0.5, 1.0), 9.0), ((0.0, -1.0, 0.2), -6.0)]
    new_start, new_end, kept = clip_segments(start, end, planes)
    for normal, offset in planes:
        assert (new_start @ np.array(normal) >= offset - 1e-9).all()
        assert (new_end @ np.array(normal) >= offset - 1e-9).all()
    # A clipped segment lies on its source segment.
    direction = end[kept] - start[kept]
    for points in (new_start, new_end):
        offset = points - start[kept]
        np.testing.assert_allclose(np.cross(offset, direction), 0, atol=1e-9)
    # Nothing that had a kept part was dropped: sample the dropped segments.
    dropped = np.setdiff1d(np.arange(500), kept)
    t = np.linspace(0, 1, 21)[:, None, None]
    samples = start[dropped] + t * (end[dropped] - start[dropped])
    inside = np.ones(samples.shape[:2], dtype=bool)
    for normal, offset in planes:
        inside &= samples @ np.array(normal) > offset + 1e-9
    assert not inside.any()


def _unit_square_parts(colored: bool) -> SectionParts:
    # One square in the plane z = 0, corners at (y, x) in {0, 10}^2: two
    # triangles, and its outline.
    positions = np.array([[0, 0, 0], [0, 0, 10], [0, 10, 10], [0, 10, 0]], dtype=float)
    colors = None
    outline_colors = None
    if colored:
        colors = np.array(
            [[0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1]], float
        )
    outline = positions[[0, 1, 1, 2, 2, 3, 3, 0]]
    if colored:
        outline_colors = colors[[0, 1, 1, 2, 2, 3, 3, 0]]
    return SectionParts(
        fill_positions=positions,
        fill_indices=np.array([[0, 1, 2], [0, 2, 3]]),
        fill_colors=colors,
        fill_face_ids=np.array([7, 8]),
        outline_positions=outline,
        outline_colors=outline_colors,
        outline_face_ids=np.array([7, 7, 8, 8]),
        n_closed_loops=1,
    )


def _area(parts: SectionParts) -> float:
    corners = parts.fill_positions[parts.fill_indices]
    return float(
        0.5
        * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        ).sum()
    )


def test_clip_parts_cuts_the_fill_and_the_outline():
    clipped = clip_parts(_unit_square_parts(colored=False), [PLANE])
    assert _area(clipped) == 50.0  # half of the 10 x 10 square
    assert clipped.fill_positions[:, 2].min() == 5.0
    assert set(clipped.fill_face_ids.tolist()) == {7, 8}
    # The edge at x = 0 is gone; the two crossing edges are cut at x = 5.
    assert len(clipped.outline_face_ids) == 3
    assert clipped.outline_positions[:, 2].min() == 5.0
    assert clipped.n_closed_loops == 1


def test_clip_parts_interpolates_colours():
    clipped = clip_parts(_unit_square_parts(colored=True), [PLANE])
    # Red runs 0 -> 1 with x, so it is x / 10 everywhere after the cut.
    np.testing.assert_allclose(
        clipped.fill_colors[:, 0], clipped.fill_positions[:, 2] / 10.0
    )
    np.testing.assert_allclose(
        clipped.outline_colors[:, 0], clipped.outline_positions[:, 2] / 10.0
    )


def test_clip_parts_without_planes_is_the_input():
    parts = _unit_square_parts(colored=False)
    assert clip_parts(parts, []) is parts


def test_clip_parts_can_remove_everything():
    clipped = clip_parts(_unit_square_parts(colored=True), [((0.0, 0.0, 1.0), 50.0)])
    assert clipped.is_empty
    assert clipped.fill_indices.shape == (0, 3)
