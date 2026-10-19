"""Tests for cellier.transform._geometry_ops (design section 4)."""

import numpy as np
import pytest

from cellier.transform._geometry_ops import (
    DegenerateNormalError,
    affine_bounding_box,
    axis_aligned_bounds,
    drop_broadcast_constraints,
    imap_directions,
    imap_half_spaces,
    imap_normals,
    imap_plane,
    imap_points,
    is_axis_aligned,
    map_directions,
    map_half_spaces,
    map_normals,
    map_plane,
    map_points,
    polytope_bounds,
    pseudo_inverse_affine,
)
from tests.transform._use_cases import UC3, USE_CASES

CASE_IDS = list(USE_CASES)


def case(name):
    return USE_CASES[name]


def inverse_of(name):
    linear, translation, _ = case(name)
    inverse = pseudo_inverse_affine(linear, translation)
    assert inverse is not None
    return inverse


def brute_force_box(linear, translation, lower, upper):
    """Ground truth AABB by enumerating the 2**n corners."""
    n = len(lower)
    corners = np.array(
        [
            [upper[i] if (k >> i) & 1 else lower[i] for i in range(n)]
            for k in range(2**n)
        ]
    )
    image = corners @ linear.T + translation
    return image.min(0), image.max(0)


# --- D24: the three-case inverse rule --------------------------------


@pytest.mark.parametrize("name", CASE_IDS)
def test_every_use_case_has_a_left_inverse_d24(name):
    linear, translation, _ = case(name)
    inverse_linear, inverse_translation = inverse_of(name)
    rng = np.random.default_rng(0)
    points = rng.normal(size=(6, linear.shape[1]))
    mapped = map_points(points, linear, translation)
    assert np.allclose(imap_points(mapped, inverse_linear, translation), points)
    # the derived translation is the same statement, written the other way
    assert np.allclose(mapped @ inverse_linear.T + inverse_translation, points)


def test_dimension_reducing_affine_has_no_inverse_d24_case_3():
    linear = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert pseudo_inverse_affine(linear, np.zeros(2)) is None


def test_singular_square_matrix_has_no_inverse_d24_case_1():
    linear = np.array([[1.0, 2.0], [2.0, 4.0]])
    assert pseudo_inverse_affine(linear, np.zeros(2)) is None


def test_rank_deficient_embedding_has_no_inverse_d24_case_3():
    linear = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    assert pseudo_inverse_affine(linear, np.zeros(3)) is None


def test_uc3_embedding_has_full_column_rank():
    linear, _, _ = UC3
    assert np.linalg.matrix_rank(linear) == linear.shape[1] == 4


# --- points (D11, D12) -----------------------------------------------


def test_identity_affine_returns_a_distinct_array_d11():
    """transformnd's identity Affine.apply returns its input; we must not."""
    points = np.array([[1.0, 2.0, 3.0]])
    result = map_points(points, np.eye(3), np.zeros(3))
    assert result is not points
    result[0, 0] = 99.0
    assert points[0, 0] == 1.0


def test_input_array_is_not_mutated():
    points = np.array([[1.0, 2.0, 3.0]])
    original = points.copy()
    map_points(points, np.diag([2.0, 3.0, 4.0]), np.ones(3))
    assert np.array_equal(points, original)


def test_rank_of_the_output_matches_the_input_d12():
    linear, translation, _ = UC3
    assert map_points(np.zeros(4), linear, translation).shape == (5,)
    assert map_points(np.zeros((7, 4)), linear, translation).shape == (7, 5)


def test_homogeneous_input_is_rejected_d12():
    linear, translation, _ = UC3
    with pytest.raises(ValueError, match="Homogeneous"):
        map_points(np.zeros((7, 5)), linear, translation)


def test_three_dimensional_input_is_rejected_d12():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        map_points(np.zeros((2, 2, 3)), np.eye(3), np.zeros(3))


def test_wrong_component_count_on_1d_input_is_rejected_d12():
    with pytest.raises(ValueError, match="components"):
        map_points(np.zeros(4), np.eye(3), np.zeros(3))


# --- directions and normals (D27, D28) -------------------------------


@pytest.mark.parametrize("name", CASE_IDS)
def test_directions_round_trip(name):
    linear, _, _ = case(name)
    inverse_linear, _ = inverse_of(name)
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(6, linear.shape[1]))
    forward = map_directions(vectors, linear)
    assert np.allclose(imap_directions(forward, inverse_linear), vectors)


@pytest.mark.parametrize("name", CASE_IDS)
def test_normals_use_the_covariant_rule_d27(name):
    """The defining property, not the formula: <n', A v> == <n, v>.

    This fails if ``A`` is used where ``(A+)^T`` is required, which a test
    that mirrors the formula would not catch.
    """
    linear, _, _ = case(name)
    inverse_linear, _ = inverse_of(name)
    rng = np.random.default_rng(2)
    normal = rng.normal(size=linear.shape[1])
    vector = rng.normal(size=linear.shape[1])
    assert np.allclose(
        map_normals(normal, inverse_linear) @ map_directions(vector, linear),
        normal @ vector,
    )


@pytest.mark.parametrize("name", CASE_IDS)
def test_normals_round_trip(name):
    linear, _, _ = case(name)
    inverse_linear, _ = inverse_of(name)
    rng = np.random.default_rng(3)
    normal = rng.normal(size=linear.shape[1])
    forward = map_normals(normal, inverse_linear)
    assert np.allclose(imap_normals(forward, linear), normal)


def test_directions_and_normals_differ_under_a_scale_d27():
    linear = np.diag([0.5, 0.1, 0.1])
    inverse_linear, _ = pseudo_inverse_affine(linear, np.zeros(3))
    vector = np.array([1.0, 1.0, 1.0])
    assert not np.allclose(
        map_directions(vector, linear), map_normals(vector, inverse_linear)
    )


def test_nothing_is_normalised_d28():
    linear = np.diag([2.0, 2.0, 2.0])
    inverse_linear, _ = pseudo_inverse_affine(linear, np.zeros(3))
    normal = np.array([3.0, 0.0, 0.0])
    assert np.linalg.norm(map_normals(normal, inverse_linear)) == pytest.approx(1.5)
    assert np.linalg.norm(map_directions(normal, linear)) == pytest.approx(6.0)
    assert np.linalg.norm(imap_normals(normal, linear)) == pytest.approx(6.0)
    assert np.linalg.norm(imap_directions(normal, inverse_linear)) == pytest.approx(1.5)


def test_map_direction_and_imap_normal_work_without_an_inverse_d27():
    """Both need only A and A^T, so they must not be gated on invertibility."""
    linear = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert pseudo_inverse_affine(linear, np.zeros(2)) is None
    assert map_directions(np.array([1.0, 2.0, 3.0]), linear).shape == (2,)
    assert imap_normals(np.array([1.0, 2.0]), linear).shape == (3,)


# --- degeneracy (D32) ------------------------------------------------


def test_normal_parallel_to_the_broadcast_axis_raises_d32():
    linear, translation, _ = UC3
    parallel_to_c = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.raises(DegenerateNormalError):
        imap_normals(parallel_to_c, linear)
    with pytest.raises(DegenerateNormalError):
        imap_plane(parallel_to_c, 5.0, linear, translation)


def test_a_plane_tilted_in_c_and_x_does_not_raise_and_loses_the_tilt_d32():
    """Pinned as intended behaviour: the reverse map is not injective."""
    linear, translation, _ = UC3
    tilted = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    pulled, _ = imap_plane(tilted, 0.0, linear, translation)
    assert np.allclose(pulled, [0.0, 0.0, 0.0, 0.1])


def test_degenerate_error_names_the_offending_row():
    linear, _, _ = UC3
    normals = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    with pytest.raises(DegenerateNormalError, match=r"\[1\]"):
        imap_normals(normals, linear)


# --- planes ----------------------------------------------------------


@pytest.mark.parametrize("name", CASE_IDS)
def test_points_on_a_plane_stay_on_the_mapped_plane(name):
    linear, translation, _ = case(name)
    inverse_linear, _ = inverse_of(name)
    rng = np.random.default_rng(4)
    ndim = linear.shape[1]
    normal = rng.normal(size=ndim)
    offset = 1.7

    sample = rng.normal(size=(8, ndim))
    on_plane = (
        sample + ((offset - sample @ normal) / (normal @ normal))[:, None] * normal
    )
    assert np.allclose(on_plane @ normal, offset)

    mapped_normal, mapped_offset = map_plane(
        normal, offset, inverse_linear, translation
    )
    mapped_points = map_points(on_plane, linear, translation)
    assert np.allclose(mapped_points @ mapped_normal, mapped_offset)


@pytest.mark.parametrize("name", CASE_IDS)
def test_plane_round_trips_up_to_scale(name):
    linear, translation, _ = case(name)
    inverse_linear, _ = inverse_of(name)
    rng = np.random.default_rng(5)
    normal = rng.normal(size=linear.shape[1])
    offset = 0.9
    forward = map_plane(normal, offset, inverse_linear, translation)
    back_normal, back_offset = imap_plane(*forward, linear, translation)
    scale = np.linalg.norm(back_normal) / np.linalg.norm(normal)
    assert np.allclose(back_normal, normal * scale)
    assert back_offset == pytest.approx(offset * scale)


# --- bounding boxes (D30, D31) ---------------------------------------


@pytest.mark.parametrize("name", ["UC1", "UC2", "UC4"])
def test_bounding_box_is_exact_for_diagonal_and_permutation_cases_d30(name):
    linear, translation, _ = case(name)
    ndim = linear.shape[1]
    lower = np.full(ndim, -1.0)
    upper = np.array([2.0, 3.0, 4.0, 5.0])[:ndim]
    got = affine_bounding_box(lower, upper, linear, translation)
    expected = brute_force_box(linear, translation, lower, upper)
    assert np.allclose(got[0], expected[0])
    assert np.allclose(got[1], expected[1])


def test_rotated_affine_gives_a_strict_superset_and_inflates_d30():
    angle = np.pi / 6
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    lower, upper = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    got_lower, got_upper = affine_bounding_box(lower, upper, rotation, np.zeros(2))
    expected = brute_force_box(rotation, np.zeros(2), lower, upper)
    # the corner hull is inside the enclosing box, and it is a strict superset
    assert np.all(got_lower <= expected[0] + 1e-12)
    assert np.all(got_upper >= expected[1] - 1e-12)
    assert got_upper[0] == pytest.approx(np.cos(angle) + np.sin(angle))

    inverse_linear, inverse_translation = pseudo_inverse_affine(rotation, np.zeros(2))
    back = affine_bounding_box(
        got_lower, got_upper, inverse_linear, inverse_translation
    )
    assert np.all(back[1] > upper)  # inflated, not equal


def test_reflection_keeps_min_below_max_e4():
    """The case transformnd's BoundingBoxAdapter gets wrong."""
    reflection = np.diag([-1.0, 1.0])
    lower, upper = np.array([1.0, 1.0]), np.array([3.0, 4.0])
    got_lower, got_upper = affine_bounding_box(lower, upper, reflection, np.zeros(2))
    assert np.all(got_lower <= got_upper)
    assert np.allclose(got_lower, [-3.0, 1.0])
    assert np.allclose(got_upper, [-1.0, 4.0])


def test_uc3_broadcast_axis_is_unbounded_and_the_rest_exact_d31():
    linear, translation, broadcast = UC3
    lower = np.full(4, -1.0)
    upper = np.array([2.0, 3.0, 4.0, 5.0])
    got_lower, got_upper = affine_bounding_box(
        lower, upper, linear, translation, broadcast
    )
    expected = brute_force_box(linear, translation, lower, upper)

    assert got_lower[1] == -np.inf
    assert got_upper[1] == np.inf
    kept = [0, 2, 3, 4]
    assert np.allclose(got_lower[kept], expected[0][kept])
    assert np.allclose(got_upper[kept], expected[1][kept])


def test_without_broadcast_axes_the_c_extent_would_be_degenerate_d31():
    """The wrong answer D25 exists to prevent, pinned so it stays visible."""
    linear, translation, _ = UC3
    lower, upper = np.full(4, -1.0), np.array([2.0, 3.0, 4.0, 5.0])
    got_lower, got_upper = affine_bounding_box(lower, upper, linear, translation)
    assert got_lower[1] == got_upper[1] == 0.0


def test_infinite_extent_produces_no_nan_on_any_axis_e3():
    """The direct regression test for the masked contraction (D31)."""
    linear, translation, _ = UC3
    inverse_linear, inverse_translation = pseudo_inverse_affine(linear, translation)
    lower = np.array([-1.0, -np.inf, -1.0, -1.0, -1.0])
    upper = np.array([1.0, np.inf, 1.0, 1.0, 1.0])
    got_lower, got_upper = affine_bounding_box(
        lower, upper, inverse_linear, inverse_translation
    )
    assert not np.any(np.isnan(got_lower))
    assert not np.any(np.isnan(got_upper))
    assert np.all(np.isfinite(got_lower))
    assert np.all(np.isfinite(got_upper))


@pytest.mark.parametrize("extent", [1.0, np.inf])
def test_imap_bounding_box_drops_the_broadcast_axis_either_way(extent):
    linear, translation, _ = UC3
    inverse_linear, inverse_translation = pseudo_inverse_affine(linear, translation)
    lower = np.array([-1.0, -extent, -1.0, -1.0, -1.0])
    upper = np.array([1.0, extent, 1.0, 1.0, 1.0])
    got = affine_bounding_box(lower, upper, inverse_linear, inverse_translation)
    reference_lower = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
    reference_upper = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    reference = affine_bounding_box(
        reference_lower, reference_upper, inverse_linear, inverse_translation
    )
    assert np.allclose(got[0], reference[0])
    assert np.allclose(got[1], reference[1])


# --- half-spaces (D39) -----------------------------------------------


def test_half_spaces_agree_with_the_direct_predicate_d39():
    linear, translation, _ = UC3
    rng = np.random.default_rng(3)
    normals = rng.normal(size=(4, 5))
    offsets = rng.normal(size=4) * 3
    data_normals, data_offsets = imap_half_spaces(normals, offsets, linear, translation)
    points = rng.normal(size=(3000, 4)) * 4
    world = map_points(points, linear, translation)
    assert np.array_equal(
        (points @ data_normals.T <= data_offsets).all(1),
        (world @ normals.T <= offsets).all(1),
    )


@pytest.mark.parametrize(("offset", "satisfiable"), [(5.0, True), (-5.0, False)])
def test_a_broadcast_axis_constraint_is_vacuous_or_infeasible_not_an_error(
    offset, satisfiable
):
    """D39 with no broadcast axes declared: contrast with D32, which raises.

    This is the raw ``A^T`` behaviour, still reachable by passing no
    ``broadcast_axes``.  Once the axis is declared broadcast, D8 drops
    the constraint entirely rather than letting it become a zero normal;
    see :func:`test_declaring_the_axis_broadcast_drops_the_constraint_d8`.
    """
    linear, translation, _ = UC3
    normals = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
    pulled, pulled_offsets = imap_half_spaces(
        normals, np.array([offset]), linear, translation
    )
    assert np.allclose(pulled, 0.0)
    assert bool(pulled_offsets[0] >= 0) is satisfiable


@pytest.mark.parametrize("offset", [5.0, -5.0])
def test_declaring_the_axis_broadcast_drops_the_constraint_d8(offset):
    """Both signs go away: a free variable satisfies either bound."""
    linear, translation, broadcast = UC3
    normals = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
    pulled, pulled_offsets = imap_half_spaces(
        normals, np.array([offset]), linear, translation, broadcast
    )
    assert pulled.shape == (0, 4)
    assert pulled_offsets.shape == (0,)


def test_drop_broadcast_constraints_keeps_everything_it_does_not_touch():
    normals = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],  # T only        -> kept
            [0.0, 1.0, 0.0, 0.0, 0.0],  # C only        -> dropped
            [0.0, 0.0, 1.0, 0.0, 0.0],  # Z only        -> kept
            [1.0, 0.5, 0.0, 0.0, 0.0],  # T and C       -> dropped
        ]
    )
    offsets = np.array([1.0, 2.0, 3.0, 4.0])
    kept, kept_offsets = drop_broadcast_constraints(normals, offsets, (1,))
    assert np.allclose(kept, normals[[0, 2]])
    assert np.allclose(kept_offsets, offsets[[0, 2]])


def test_drop_broadcast_constraints_is_a_no_op_without_broadcast_axes():
    normals = np.array([[1.0, 0.0], [0.0, 1.0]])
    offsets = np.array([1.0, 2.0])
    kept, kept_offsets = drop_broadcast_constraints(normals, offsets, ())
    assert kept is normals
    assert kept_offsets is offsets


def test_drop_broadcast_constraints_handles_an_unbounded_region():
    """A region with no constraints at all still has a rank."""
    normals = np.zeros((0, 5))
    offsets = np.zeros(0)
    kept, kept_offsets = drop_broadcast_constraints(normals, offsets, (1,))
    assert kept.shape == (0, 5)
    assert kept_offsets.shape == (0,)


def test_half_spaces_round_trip_forward_and_back():
    linear, translation, _ = UC3
    inverse_linear, _ = pseudo_inverse_affine(linear, translation)
    rng = np.random.default_rng(6)
    normals = rng.normal(size=(3, 4))
    offsets = rng.normal(size=3)
    world = map_half_spaces(normals, offsets, inverse_linear, translation)
    back = imap_half_spaces(*world, linear, translation)
    assert np.allclose(back[0], normals)
    assert np.allclose(back[1], offsets)


# --- polytope bounds (D40) -------------------------------------------


AXIS_ALIGNED_NORMALS = np.array(
    [
        [1, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, -1, 0, 0, 0],
    ],
    float,
)
AXIS_ALIGNED_OFFSETS = np.array([7.5, -6.5, 2.5, -1.5])


def oblique_constraints():
    diagonal = np.array([0, 0, 1, 1, 0]) / np.sqrt(2)
    normals = np.vstack(
        [
            AXIS_ALIGNED_NORMALS,
            diagonal,
            -diagonal,
            [0, 0, 1, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, -1, 0],
        ]
    )
    offsets = np.concatenate(
        [AXIS_ALIGNED_OFFSETS, [5.5, -4.5], [10.0, 0.0, 10.0, 0.0]]
    )
    return normals, offsets


def test_is_axis_aligned_d40():
    assert is_axis_aligned(AXIS_ALIGNED_NORMALS)
    assert not is_axis_aligned(oblique_constraints()[0])
    assert is_axis_aligned(np.array([[3.0, 0.0], [0.0, -2.0]]))
    assert is_axis_aligned(np.zeros((1, 3)))
    assert is_axis_aligned(np.zeros((0, 3)))


def test_fast_path_and_lp_agree_on_an_axis_aligned_region_d40():
    fast = axis_aligned_bounds(AXIS_ALIGNED_NORMALS, AXIS_ALIGNED_OFFSETS, 5)
    slow = polytope_bounds(AXIS_ALIGNED_NORMALS, AXIS_ALIGNED_OFFSETS, 5)
    assert np.array_equal(fast[0], slow[0])
    assert np.array_equal(fast[1], slow[1])
    assert np.array_equal(fast[0], [6.5, 1.5, -np.inf, -np.inf, -np.inf])
    assert np.array_equal(fast[1], [7.5, 2.5, np.inf, np.inf, np.inf])


def test_fast_path_divides_out_a_non_unit_normal():
    normals = np.array([[2.0, 0.0], [-4.0, 0.0]])
    offsets = np.array([6.0, -4.0])
    lower, upper = axis_aligned_bounds(normals, offsets, 2)
    assert lower[0] == pytest.approx(1.0)
    assert upper[0] == pytest.approx(3.0)


def test_fast_path_rejects_an_oblique_normal():
    normals, offsets = oblique_constraints()
    with pytest.raises(ValueError, match="single"):
        axis_aligned_bounds(normals, offsets, 5)


def test_lp_bounds_an_oblique_region_d40():
    normals, offsets = oblique_constraints()
    lower, upper = polytope_bounds(normals, offsets, 5)
    assert np.allclose(lower, [6.5, 1.5, 0.0, 0.0, -np.inf])
    assert np.allclose(upper[:4], [7.5, 2.5, 7.7781746, 7.7781746])
    assert upper[4] == np.inf


def test_lp_reads_a_slab_tilted_through_every_axis_as_unbounded():
    """A world ``z`` slab pulled back through a general affine (design v2 K13).

    The constraints are what a ``(y, x)`` view of the probes' ``general``
    transform selects.  HiGHS's presolve called three of the six bound
    problems infeasible, so the region read as empty and the slice failed.
    """
    normal = np.array([1.8901682009848177, -0.30997551921944466, -0.10359160429827452])
    normals = np.stack([normal, -normal])
    offsets = np.array([17.212277479607657, -15.212277479607657])
    bounds = polytope_bounds(normals, offsets, 3)
    assert bounds is not None
    lower, upper = bounds
    assert np.all(lower == -np.inf)
    assert np.all(upper == np.inf)


def test_no_constraints_is_all_of_space():
    lower, upper = polytope_bounds(np.zeros((0, 3)), np.zeros(0), 3)
    assert np.all(lower == -np.inf)
    assert np.all(upper == np.inf)


def test_infeasible_regions_are_signalled_not_raised():
    normals = np.array([[1.0, 0.0], [-1.0, 0.0]])
    offsets = np.array([1.0, -5.0])
    assert polytope_bounds(normals, offsets, 2) is None
    assert axis_aligned_bounds(normals, offsets, 2) is None


def test_fast_path_resolves_a_zero_normal_constraint():
    zero_row = np.array([[0.0, 0.0, 0.0]])
    assert axis_aligned_bounds(zero_row, np.array([5.0]), 3) is not None
    assert axis_aligned_bounds(zero_row, np.array([-5.0]), 3) is None
