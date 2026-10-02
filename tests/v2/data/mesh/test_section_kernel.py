"""The mesh section kernel (``plans/mesh_refactor_v3.md`` 5.5, X1-X5).

Two kinds of check:

- **stored expected cuts** (``expected/section_cases.json``): the input
  mesh, a plane, and the cut VTK gives for it, written once by
  ``scripts/mesh_refactor_v3/make_section_expected.py``.  No test imports
  another library, and the file is not regenerated to make a test pass;
- **analytic** checks on meshes built here: loop closure, areas, the
  conventions for planes through vertices, edges and faces, slab clipping,
  colours.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from cellier.data.mesh._section import (
    closure_report,
    cut_plane,
    fill_loops,
    section_cut,
    section_slab,
    stitch,
)

EXPECTED = json.loads(
    (Path(__file__).parent / "expected" / "section_cases.json").read_text()
)
CUTS = [
    pytest.param(case, cut, id=f"{case['name']}-{index}")
    for case in EXPECTED["cases"]
    for index, cut in enumerate(case["cuts"])
]


def _tol(positions: np.ndarray) -> float:
    return 1e-7 * float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))


def _both_ways(segments: np.ndarray) -> np.ndarray:
    """``(2 * S, 6)``: every segment, then every segment reversed.

    A segment has no direction, and its two endpoints can agree on a
    coordinate to within rounding, so no ordering of them is stable.
    """
    flat = segments.reshape(-1, 6)
    return np.concatenate([flat, flat[:, [3, 4, 5, 0, 1, 2]]])


def _match(segments: np.ndarray, stored: np.ndarray) -> tuple[float, int]:
    """Nearest stored segment for each: ``(worst distance, distinct partners)``."""
    distance, partner = cKDTree(_both_ways(stored)).query(segments.reshape(-1, 6))
    return float(distance.max()), len(np.unique(partner % len(stored)))


def _fill_area(parts) -> float:
    triangles = parts.fill_positions[parts.fill_indices]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    return 0.5 * float(np.linalg.norm(cross, axis=1).sum())


def _outline_segments(parts) -> np.ndarray:
    return parts.outline_positions.reshape(-1, 2, 3)


# -- stored expected cuts -----------------------------------------------------


def test_the_expected_file_says_where_it_came_from():
    assert EXPECTED["reference"] == "pyvista.PolyData.slice"
    assert EXPECTED["written_by"].endswith("make_section_expected.py")
    assert len(EXPECTED["cases"]) >= 6


@pytest.mark.parametrize(("case", "expected"), CUTS)
def test_the_cut_matches_the_stored_cut(case, expected):
    positions = np.array(case["positions"], dtype=np.float32)
    indices = np.array(case["indices"], dtype=np.int32)
    normal = np.array(expected["normal"])
    scale = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    parts = section_cut(positions, indices, normal, expected["offset"], _tol(positions))

    segments = _outline_segments(parts)
    stored = np.array(expected["segments"]).reshape(-1, 2, 3)
    assert len(segments) == expected["n_segments"]
    length = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1).sum()
    assert length == pytest.approx(expected["length"], rel=1e-5, abs=1e-6 * scale)
    if len(stored):
        # One to one: every stored segment has its own nearest partner.
        worst, partners = _match(segments, stored)
        assert worst < 1e-5 * scale
        assert partners == len(stored)

    assert parts.n_closed_loops == expected["n_closed_loops"]
    assert _fill_area(parts) == pytest.approx(
        expected["area"], rel=1e-5, abs=1e-6 * scale**2
    )
    # Every cap triangle is marked as no face's.
    assert (parts.fill_face_ids == -1).all()


@pytest.mark.parametrize(("case", "expected"), CUTS)
def test_a_candidate_superset_gives_the_same_cut(case, expected):
    """S4's overlap query hands the kernel a subset; the cut must not change."""
    positions = np.array(case["positions"], dtype=np.float32)
    indices = np.array(case["indices"], dtype=np.int32)
    normal = np.array(expected["normal"])
    tol = _tol(positions)
    full = section_cut(positions, indices, normal, expected["offset"], tol)
    distance = positions.astype(np.float64) @ normal - expected["offset"]
    per_face = distance[indices]
    touching = np.flatnonzero(
        (per_face.min(axis=1) <= tol) & (per_face.max(axis=1) >= -tol)
    )
    subset = section_cut(
        positions, indices, normal, expected["offset"], tol, candidates=touching
    )
    full_segments, subset_segments = _outline_segments(full), _outline_segments(subset)
    assert len(full_segments) == len(subset_segments)
    if len(full_segments):
        # Bit for bit the same points: the same vertices classify the same.
        worst, partners = _match(subset_segments, full_segments)
        assert worst == 0.0
        assert partners == len(full_segments)
    np.testing.assert_array_equal(
        np.sort(full.outline_face_ids), np.sort(subset.outline_face_ids)
    )
    assert _fill_area(subset) == pytest.approx(_fill_area(full), rel=1e-12)


# -- meshes built here --------------------------------------------------------


def _cube(low=0.0, high=1.0, flip=False):
    """A closed cube of 12 triangles, corners on integer-like coordinates."""
    corners = np.array(
        [[x, y, z] for x in (low, high) for y in (low, high) for z in (low, high)],
        dtype=np.float32,
    )
    quads = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    faces = []
    for a, b, c, d in quads:
        faces += [[a, b, c], [a, c, d]]
    faces = np.array(faces, dtype=np.int32)
    return corners, faces[:, ::-1] if flip else faces


def _octahedron(radius=1.0):
    positions = radius * np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float32,
    )
    indices = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int32,
    )
    return positions, indices


X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def test_a_cube_cut_is_one_closed_unit_square():
    positions, indices = _cube()
    parts = section_cut(positions, indices, X, 0.4, _tol(positions))
    assert parts.n_closed_loops == 1
    assert parts.n_open_segments == 0
    assert _fill_area(parts) == pytest.approx(1.0)
    np.testing.assert_allclose(parts.fill_positions[:, 0], 0.4)
    assert len(parts.outline_face_ids) == 8  # four side faces, two triangles each


def test_winding_does_not_matter():
    """Loops are stitched by edge keys, not by face orientation."""
    positions, indices = _cube()
    rng = np.random.default_rng(0)
    flipped = indices.copy()
    which = rng.random(len(indices)) < 0.5
    flipped[which] = flipped[which][:, ::-1]
    parts = section_cut(positions, flipped, X, 0.4, _tol(positions))
    assert parts.n_closed_loops == 1
    assert _fill_area(parts) == pytest.approx(1.0)


def test_a_plane_through_vertices_closes_through_them():
    """The octahedron's equator: four vertices lie on the plane."""
    positions, indices = _octahedron()
    parts = section_cut(positions, indices, Z, 0.0, _tol(positions))
    assert parts.n_closed_loops == 1
    assert parts.n_open_segments == 0
    assert _fill_area(parts) == pytest.approx(2.0)  # a square of diagonal 2
    # No zero-length segment reaches the outline.
    segments = _outline_segments(parts)
    assert (np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) > 0).all()


def test_a_plane_touching_one_vertex_draws_nothing():
    positions, indices = _octahedron()
    parts = section_cut(positions, indices, Z, 1.0, _tol(positions))
    assert len(parts.outline_face_ids) == 0
    assert len(parts.fill_indices) == 0
    assert parts.is_empty


def test_a_plane_missing_the_mesh_is_empty():
    positions, indices = _cube()
    parts = section_cut(positions, indices, X, 5.0, _tol(positions))
    assert parts.is_empty
    assert parts.fill_positions.shape == (0, 3)
    assert parts.outline_positions.shape == (0, 3)


def test_a_face_in_the_plane_is_drawn_as_itself():
    """The cube's bottom face lies in the plane: fill it, outline its border."""
    positions, indices = _cube()
    parts = section_cut(positions, indices, X, 0.0, _tol(positions))
    # Two in-plane triangles, with their own face ids.
    assert sorted(parts.fill_face_ids.tolist()) == [0, 1]
    assert _fill_area(parts) == pytest.approx(1.0)
    # The outline is the square's four boundary edges, not its diagonal.
    segments = _outline_segments(parts)
    assert len(segments) == 4
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    np.testing.assert_allclose(lengths, 1.0)
    assert set(parts.outline_face_ids.tolist()) <= {0, 1}


def test_a_flat_stack_draws_only_the_sheet_on_the_plane():
    """Flat sheets at x = 0, 1, 2: the plane picks one."""
    sheets, faces = [], []
    for level in range(3):
        base = len(sheets)
        sheets += [[level, 0, 0], [level, 1, 0], [level, 1, 1], [level, 0, 1]]
        faces += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    positions = np.array(sheets, dtype=np.float32)
    indices = np.array(faces, dtype=np.int32)
    parts = section_cut(positions, indices, X, 1.0, _tol(positions))
    assert sorted(parts.fill_face_ids.tolist()) == [2, 3]
    assert len(parts.outline_face_ids) == 4


def test_a_degenerate_in_plane_face_is_dropped():
    positions = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 1, 1]], dtype=np.int32)
    parts = section_cut(positions, indices, X, 0.0, _tol(positions))
    assert parts.fill_face_ids.tolist() == [0]


def _joined(parts_in):
    positions, indices, offset = [], [], 0
    for cube_positions, cube_indices in parts_in:
        positions.append(cube_positions)
        indices.append(cube_indices + offset)
        offset += len(cube_positions)
    return np.concatenate(positions), np.concatenate(indices)


def test_a_hole_is_cut_out_and_an_island_filled_again():
    """A solid, a cavity in it, a solid in the cavity: 16 - 4 + 1."""
    positions, indices = _joined(
        [_cube(-2.0, 2.0), _cube(-1.0, 1.0, flip=True), _cube(-0.5, 0.5)]
    )
    parts = section_cut(positions, indices, X, 0.1, _tol(positions))
    assert parts.n_closed_loops == 3
    assert _fill_area(parts) == pytest.approx(16.0 - 4.0 + 1.0)


@pytest.mark.parametrize("normal", [X, -X, Y, Z], ids=["x", "-x", "y", "z"])
def test_an_object_inside_another_is_filled(normal):
    """Two solids, one inside the other: the union, not a hole."""
    positions, indices = _joined([_cube(-2.0, 2.0), _cube(-1.0, 1.0)])
    parts = section_cut(positions, indices, normal, 0.1, _tol(positions))
    assert parts.n_closed_loops == 2
    assert _fill_area(parts) == pytest.approx(16.0)
    # The outline still shows both objects.
    assert len(parts.outline_face_ids) == 16


@pytest.mark.parametrize("normal", [X, -X], ids=["x", "-x"])
def test_a_cavity_is_a_hole_whichever_way_the_plane_faces(normal):
    positions, indices = _joined([_cube(-2.0, 2.0), _cube(-1.0, 1.0, flip=True)])
    parts = section_cut(positions, indices, normal, 0.1, _tol(positions))
    assert _fill_area(parts) == pytest.approx(16.0 - 4.0)


def test_a_mesh_wound_inside_out_is_filled_the_same():
    """Every face reversed: sides swap, the picture does not."""
    for cubes, area in (
        ([_cube(-2.0, 2.0, flip=True)], 16.0),
        ([_cube(-2.0, 2.0, flip=True), _cube(-1.0, 1.0, flip=True)], 16.0),
        ([_cube(-2.0, 2.0, flip=True), _cube(-1.0, 1.0)], 12.0),
    ):
        positions, indices = _joined(cubes)
        parts = section_cut(positions, indices, X, 0.1, _tol(positions))
        assert _fill_area(parts) == pytest.approx(area)


def test_nested_loops_with_mixed_winding_fall_back_to_even_odd():
    """The inner cube's faces disagree, so its side is unknown: a hole."""
    inner_positions, inner_indices = _cube(-1.0, 1.0)
    mixed = inner_indices.copy()
    mixed[::2] = mixed[::2][:, ::-1]
    positions, indices = _joined([_cube(-2.0, 2.0), (inner_positions, mixed)])
    parts = section_cut(positions, indices, X, 0.1, _tol(positions))
    assert parts.n_closed_loops == 2
    assert _fill_area(parts) == pytest.approx(16.0 - 4.0)


def test_a_slab_caps_nested_objects_as_their_union():
    positions, indices = _joined([_cube(-2.0, 2.0), _cube(-1.0, 1.0)])
    parts = section_slab(positions, indices, X, -0.25, 0.25, _tol(positions))
    caps = parts.fill_positions[parts.fill_indices[parts.fill_face_ids == -1]]
    cross = np.cross(caps[:, 1] - caps[:, 0], caps[:, 2] - caps[:, 0])
    assert 0.5 * np.linalg.norm(cross, axis=1).sum() == pytest.approx(2 * 16.0)


def test_an_open_mesh_is_outline_only():
    """A cube with no lid on the cutting path: an open chain."""
    positions, indices = _cube()
    # Drop the two triangles of one side face the plane crosses.
    opened = np.delete(indices, [4, 5], axis=0)
    parts = section_cut(positions, opened, X, 0.4, _tol(positions))
    assert parts.n_closed_loops == 0
    assert parts.n_open_segments == 6
    assert len(parts.fill_indices) == 0
    assert len(parts.outline_face_ids) == 6


def test_a_non_manifold_fan_is_outline_only():
    """Three sheets sharing one edge: the cut has a node of degree three."""
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0], [0.5, 0, 1]],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int32)
    parts = section_cut(positions, indices, X, 0.25, _tol(positions))
    assert parts.n_closed_loops == 0
    assert parts.n_open_segments == 3
    assert len(parts.fill_indices) == 0
    assert len(parts.outline_face_ids) == 3


def test_parts_can_be_turned_off():
    positions, indices = _cube()
    tol = _tol(positions)
    no_fill = section_cut(positions, indices, X, 0.4, tol, fill=False)
    assert len(no_fill.fill_indices) == 0
    assert len(no_fill.outline_face_ids) == 8
    no_outline = section_cut(positions, indices, X, 0.4, tol, outline=False)
    assert len(no_outline.outline_face_ids) == 0
    assert _fill_area(no_outline) == pytest.approx(1.0)


def test_stitch_and_fill_guard_the_empty_case():
    empty = np.zeros(0, dtype=np.int64)
    assert stitch(empty, empty).loops == []
    used, triangles = fill_loops([])
    assert used.shape == (0,)
    assert triangles.shape == (0, 3)


def test_the_source_face_of_every_segment_is_reported():
    positions, indices = _cube()
    parts = section_cut(positions, indices, X, 0.4, _tol(positions))
    cut = cut_plane(positions, indices, X, 0.4, _tol(positions))
    # Every reported face really straddles the plane.
    per_face = positions[indices[parts.outline_face_ids]][..., 0]
    assert ((per_face.min(axis=1) < 0.4) & (per_face.max(axis=1) > 0.4)).all()
    assert sorted(cut.face.tolist()) == sorted(parts.outline_face_ids.tolist())


# -- colours ------------------------------------------------------------------


def test_vertex_colours_are_interpolated_along_the_cut_edges():
    """Colour = position, so every output colour must equal its position."""
    positions, indices = _cube()
    colours = np.concatenate([positions, np.ones((len(positions), 1), np.float32)], 1)
    parts = section_cut(
        positions, indices, X, 0.4, _tol(positions), vertex_colors=colours
    )
    np.testing.assert_allclose(
        parts.outline_colors[:, :3], parts.outline_positions, atol=1e-6
    )
    np.testing.assert_allclose(
        parts.fill_colors[:, :3], parts.fill_positions, atol=1e-6
    )
    np.testing.assert_allclose(parts.fill_colors[:, 3], 1.0)


def test_face_colours_are_copied():
    positions, indices = _cube()
    colours = np.random.default_rng(1).random((len(indices), 4)).astype(np.float32)
    parts = section_cut(
        positions, indices, X, 0.4, _tol(positions), face_colors=colours
    )
    np.testing.assert_array_equal(
        parts.outline_colors, np.repeat(colours[parts.outline_face_ids], 2, axis=0)
    )
    assert parts.fill_colors.shape == (len(parts.fill_positions), 4)


def test_no_colours_in_means_no_colours_out():
    positions, indices = _cube()
    parts = section_cut(positions, indices, X, 0.4, _tol(positions))
    assert parts.fill_colors is None
    assert parts.outline_colors is None


# -- slab mode (X3) -----------------------------------------------------------


def test_a_slab_through_a_cube_has_the_analytic_area():
    """Four side strips of 1 x 0.3, and two unit caps."""
    positions, indices = _cube()
    parts = section_slab(positions, indices, X, 0.2, 0.5, _tol(positions))
    surface = parts.fill_face_ids >= 0
    triangles = parts.fill_positions[parts.fill_indices]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    assert areas[surface].sum() == pytest.approx(4 * 0.3)
    assert areas[~surface].sum() == pytest.approx(2.0)
    assert parts.n_closed_loops == 2
    # Nothing outside the slab.
    assert parts.fill_positions[:, 0].min() >= 0.2 - 1e-6
    assert parts.fill_positions[:, 0].max() <= 0.5 + 1e-6
    # The outline is the two cuts.
    assert sorted(set(np.round(parts.outline_positions[:, 0], 6))) == [0.2, 0.5]


def test_a_slab_holding_the_whole_mesh_keeps_every_face():
    positions, indices = _cube()
    parts = section_slab(positions, indices, X, -1.0, 2.0, _tol(positions))
    assert sorted(parts.fill_face_ids.tolist()) == list(range(len(indices)))
    assert _fill_area(parts) == pytest.approx(6.0)
    assert len(parts.outline_face_ids) == 0


def test_a_slab_of_no_thickness_is_the_cut():
    positions, indices = _cube()
    tol = _tol(positions)
    slab = section_slab(positions, indices, X, 0.4, 0.4, tol)
    cut = section_cut(positions, indices, X, 0.4, tol)
    np.testing.assert_array_equal(slab.outline_positions, cut.outline_positions)
    np.testing.assert_array_equal(slab.fill_indices, cut.fill_indices)


def test_slab_vertex_colours_follow_the_clipped_positions():
    positions, indices = _cube()
    colours = np.concatenate([positions, np.ones((len(positions), 1), np.float32)], 1)
    parts = section_slab(
        positions, indices, X, 0.2, 0.5, _tol(positions), vertex_colors=colours
    )
    np.testing.assert_allclose(
        parts.fill_colors[:, :3], parts.fill_positions, atol=1e-6
    )


def test_a_slab_on_an_oblique_plane_stays_inside_it():
    positions = np.array(EXPECTED["cases"][0]["positions"], dtype=np.float32)
    indices = np.array(EXPECTED["cases"][0]["indices"], dtype=np.int32)
    normal = np.array([0.3, 0.2, 0.93])
    normal /= np.linalg.norm(normal)
    parts = section_slab(positions, indices, normal, -0.5, 0.8, _tol(positions))
    along = parts.fill_positions @ normal
    assert along.min() >= -0.5 - 1e-5
    assert along.max() <= 0.8 + 1e-5
    assert parts.n_closed_loops == 2


# -- closure report (X5) ------------------------------------------------------


def test_a_closed_mesh_reports_closed():
    report = closure_report(_cube()[1], 8)
    assert report.closed
    assert (report.boundary_edges, report.nonmanifold_edges) == (0, 0)


def test_an_open_mesh_reports_its_boundary():
    positions, indices = _cube()
    report = closure_report(np.delete(indices, [4, 5], axis=0), len(positions))
    assert not report.closed
    assert report.boundary_edges == 4
    assert report.nonmanifold_edges == 0


def test_a_fan_reports_its_non_manifold_edge():
    indices = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int32)
    report = closure_report(indices, 5)
    assert report.nonmanifold_edges == 1
    assert report.boundary_edges == 6


def test_a_degenerate_edge_is_not_counted():
    report = closure_report(np.array([[0, 1, 1]], dtype=np.int32), 2)
    # The edge 0-1 is used twice by the one face; 1-1 is not an edge.
    assert (report.boundary_edges, report.nonmanifold_edges) == (0, 0)
    assert closure_report(np.zeros((0, 3), np.int32), 0).closed
