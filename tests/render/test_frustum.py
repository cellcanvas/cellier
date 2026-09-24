"""Tests for cellier.render._frustum.

Pure-NumPy frustum geometry and conservative AABB visibility.  These
functions encode the DHW-to-XYZ axis swap (``g0/g1/g2`` map to ``z/y/x``)
that has caused real bugs, so the tests below pin down both the axis
order and the isotropic/anisotropic parity.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render._frustum import (
    _compute_plane_parameters,
    bricks_in_frustum,
    bricks_in_frustum_arr,
    compute_brick_aabb_corners,
    frustum_planes_from_corners,
)
from cellier.render.block_cache import BlockKey3D

TIMING_KEYS = {"build_corners_ms", "einsum_ms", "mask_ms"}


def _box_frustum(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> np.ndarray:
    """Build the (6, 4) planes of an axis-aligned box frustum.

    The returned frustum's interior is exactly the box
    ``[xmin, xmax] x [ymin, ymax] x [zmin, zmax]`` with inward normals.
    The near plane is placed at ``zmax`` and the far plane at ``zmin`` to
    match the winding assumed by ``frustum_planes_from_corners``.
    """
    # corner order within each plane: (left-bottom, right-bottom,
    # right-top, left-top)
    near = np.array(
        [
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ]
    )
    far = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
        ]
    )
    return frustum_planes_from_corners(np.stack([near, far]))


def _inside(planes: np.ndarray, point: np.ndarray) -> bool:
    """Whether ``point`` is inside every half-space (inclusive)."""
    return bool(np.all(planes[:, :3] @ point + planes[:, 3] >= -1e-9))


# ---------------------------------------------------------------------------
# _compute_plane_parameters
# ---------------------------------------------------------------------------


def test_compute_plane_parameters_degenerate_returns_zero_plane():
    # three collinear points -> zero-area triangle -> zero plane
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 1.0, 1.0])
    p2 = np.array([2.0, 2.0, 2.0])
    plane = _compute_plane_parameters(p0, p1, p2)
    np.testing.assert_array_equal(plane, np.zeros(4))


def test_compute_plane_parameters_coincident_returns_zero_plane():
    p = np.array([3.0, -1.0, 2.0])
    plane = _compute_plane_parameters(p, p, p)
    np.testing.assert_array_equal(plane, np.zeros(4))


def test_compute_plane_parameters_known_triangle_unit_normal():
    # triangle in the z=0 plane, CCW -> +z normal, passing through origin
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    plane = _compute_plane_parameters(p0, p1, p2)
    np.testing.assert_allclose(plane, [0.0, 0.0, 1.0, 0.0], atol=1e-12)
    # unit normal
    assert np.isclose(np.linalg.norm(plane[:3]), 1.0)


def test_compute_plane_parameters_offset_plane_d_term():
    # same triangle shifted to z=2 -> normal (0,0,1), d = -2
    p0 = np.array([0.0, 0.0, 2.0])
    p1 = np.array([1.0, 0.0, 2.0])
    p2 = np.array([0.0, 1.0, 2.0])
    plane = _compute_plane_parameters(p0, p1, p2)
    np.testing.assert_allclose(plane, [0.0, 0.0, 1.0, -2.0], atol=1e-12)


# ---------------------------------------------------------------------------
# frustum_planes_from_corners
# ---------------------------------------------------------------------------


def test_frustum_planes_shape_and_inward_normals():
    planes = _box_frustum(-1, 1, -1, 1, 1, 5)
    assert planes.shape == (6, 4)
    # all normals unit length
    np.testing.assert_allclose(np.linalg.norm(planes[:, :3], axis=1), 1.0)

    # centre of the frustum passes all 6 planes
    centre = np.array([0.0, 0.0, 3.0])
    assert _inside(planes, centre)
    # every plane is strictly satisfied at the centre
    assert np.all(planes[:, :3] @ centre + planes[:, 3] > 0)


def test_frustum_planes_reject_exterior_points():
    planes = _box_frustum(-1, 1, -1, 1, 1, 5)
    for outside in [
        np.array([10.0, 0.0, 3.0]),  # +x
        np.array([-10.0, 0.0, 3.0]),  # -x
        np.array([0.0, 10.0, 3.0]),  # +y
        np.array([0.0, 0.0, 100.0]),  # past near
        np.array([0.0, 0.0, -100.0]),  # past far
    ]:
        assert not _inside(planes, outside)


# ---------------------------------------------------------------------------
# compute_brick_aabb_corners
# ---------------------------------------------------------------------------


def test_aabb_corners_isotropic_axis_swap_level1():
    # g0/g1/g2 -> z/y/x
    key = BlockKey3D(level=1, g0=1, g1=2, g2=3)
    corners = compute_brick_aabb_corners(key, block_size=8)
    assert corners.shape == (8, 3)
    # Centre convention (plan v2, D1): x spans voxels g2*8 .. (g2+1)*8 - 1,
    # i.e. [g2*8 - 0.5, (g2+1)*8 - 0.5]; y from g1, z from g0.
    assert corners[:, 0].min() == 23.5 and corners[:, 0].max() == 31.5  # x <- g2
    assert corners[:, 1].min() == 15.5 and corners[:, 1].max() == 23.5  # y <- g1
    assert corners[:, 2].min() == 7.5 and corners[:, 2].max() == 15.5  # z <- g0


def test_aabb_corners_isotropic_level2_doubles_block_world():
    key = BlockKey3D(level=2, g0=1, g1=2, g2=3)
    corners = compute_brick_aabb_corners(key, block_size=8)
    # block_world = 8 * 2**(2-1) = 16; the implied block-averaged level
    # (t = 0.5) starts at level-0 voxel 16 * g, low edge 16 * g - 0.5.
    assert corners[:, 0].min() == 47.5 and corners[:, 0].max() == 63.5  # x
    assert corners[:, 1].min() == 31.5 and corners[:, 1].max() == 47.5  # y
    assert corners[:, 2].min() == 15.5 and corners[:, 2].max() == 31.5  # z


def test_aabb_corners_anisotropic_honours_scale_and_translation():
    key = BlockKey3D(level=1, g0=1, g1=1, g2=1)
    scale = np.array([[2.0, 3.0, 4.0]])  # (sx=W, sy=H, sz=D) for level 1
    translation = np.array([[10.0, 20.0, 30.0]])
    corners = compute_brick_aabb_corners(
        key,
        block_size=8,
        level_scale_arr_shader=scale,
        level_translation_arr_shader=translation,
    )
    # low = s * (g * 8 - 0.5) + t.  x width = 8*2 = 16, min = 2*7.5 + 10 = 25
    assert corners[:, 0].min() == 25 and corners[:, 0].max() == 41
    # y width = 8*3 = 24, min = 3*7.5 + 20 = 42.5
    assert corners[:, 1].min() == 42.5 and corners[:, 1].max() == 66.5
    # z width = 8*4 = 32, min = 4*7.5 + 30 = 60
    assert corners[:, 2].min() == 60 and corners[:, 2].max() == 92


def test_aabb_corners_level1_isotropic_matches_unit_anisotropic():
    # At level 1 the isotropic block_world = block_size * 2**0 = block_size,
    # which equals the anisotropic block_size * scale when scale == 1.  The two
    # branches must therefore produce identical corners here.
    key = BlockKey3D(level=1, g0=2, g1=1, g2=3)
    iso = compute_brick_aabb_corners(key, block_size=8)
    aniso = compute_brick_aabb_corners(
        key,
        block_size=8,
        level_scale_arr_shader=np.ones((1, 3)),
        level_translation_arr_shader=np.zeros((1, 3)),
    )
    np.testing.assert_array_equal(iso, aniso)


# ---------------------------------------------------------------------------
# bricks_in_frustum_arr
# ---------------------------------------------------------------------------


def test_bricks_in_frustum_arr_empty_input():
    arr = np.empty((0, 4), dtype=np.int32)
    planes = _box_frustum(0, 20, 0, 20, 0, 20)
    visible, timings = bricks_in_frustum_arr(arr, block_size=8, frustum_planes=planes)
    assert visible.shape == (0, 4)
    assert set(timings) == TIMING_KEYS
    assert all(v == 0.0 for v in timings.values())


def test_bricks_in_frustum_arr_inside_outside_straddle():
    planes = _box_frustum(0, 20, 0, 20, 0, 20)
    # columns: [level, gz, gy, gx]
    arr = np.array(
        [
            [1, 0, 0, 0],  # brick spans [0,8]^3 -> inside
            [1, 4, 4, 4],  # brick spans [32,40]^3 -> outside
            [1, 2, 2, 2],  # brick spans [16,24]^3 -> straddles max face
        ],
        dtype=np.int32,
    )
    visible, timings = bricks_in_frustum_arr(arr, block_size=8, frustum_planes=planes)
    assert set(timings) == TIMING_KEYS
    survivors = {tuple(r) for r in visible}
    assert (1, 0, 0, 0) in survivors  # inside kept
    assert (1, 2, 2, 2) in survivors  # straddler kept (conservative)
    assert (1, 4, 4, 4) not in survivors  # fully outside culled


def test_bricks_in_frustum_arr_isotropic_anisotropic_parity():
    planes = _box_frustum(0, 40, 0, 40, 0, 40)
    arr = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 2, 3],
            [2, 1, 1, 1],
            [1, 6, 6, 6],
        ],
        dtype=np.int32,
    )
    iso_visible, _ = bricks_in_frustum_arr(arr, block_size=8, frustum_planes=planes)

    n_levels = 2
    scale = np.ones((n_levels, 3))
    translation = np.zeros((n_levels, 3))
    # NOTE: the anisotropic branch uses block_world = block_size * scale, i.e.
    # NO power-of-2 level scaling.  To reproduce the isotropic 2**(level-1)
    # widths we bake the level scale into the per-level scale vectors.
    scale[1] = 2.0  # level 2 -> 2x
    aniso_visible, _ = bricks_in_frustum_arr(
        arr,
        block_size=8,
        frustum_planes=planes,
        level_scale_arr_shader=scale,
        level_translation_arr_shader=translation,
    )
    np.testing.assert_array_equal(iso_visible, aniso_visible)


# ---------------------------------------------------------------------------
# bricks_in_frustum (dict/set path)
# ---------------------------------------------------------------------------


def test_bricks_in_frustum_empty():
    planes = _box_frustum(0, 20, 0, 20, 0, 20)
    visible, timings = bricks_in_frustum(set(), block_size=8, frustum_planes=planes)
    assert visible == {}
    assert set(timings) == TIMING_KEYS
    assert all(v == 0.0 for v in timings.values())


def test_bricks_in_frustum_set_input():
    planes = _box_frustum(0, 20, 0, 20, 0, 20)
    inside = BlockKey3D(level=1, g0=0, g1=0, g2=0)
    outside = BlockKey3D(level=1, g0=4, g1=4, g2=4)
    visible, _ = bricks_in_frustum(
        {inside, outside}, block_size=8, frustum_planes=planes
    )
    assert inside in visible
    assert outside not in visible


def test_bricks_in_frustum_dict_preserves_values():
    planes = _box_frustum(0, 20, 0, 20, 0, 20)
    inside = BlockKey3D(level=1, g0=0, g1=0, g2=0)
    outside = BlockKey3D(level=1, g0=4, g1=4, g2=4)
    visible, _ = bricks_in_frustum(
        {inside: 7, outside: 9}, block_size=8, frustum_planes=planes
    )
    assert visible == {inside: 7}


def test_bricks_in_frustum_agrees_with_arr_pipeline():
    planes = _box_frustum(0, 30, 0, 30, 0, 30)
    keys = [
        BlockKey3D(level=1, g0=0, g1=0, g2=0),
        BlockKey3D(level=1, g0=2, g1=1, g2=3),
        BlockKey3D(level=1, g0=5, g1=5, g2=5),
        BlockKey3D(level=2, g0=1, g1=1, g2=1),
    ]
    dict_visible, _ = bricks_in_frustum(set(keys), block_size=8, frustum_planes=planes)

    arr = np.array([[k.level, k.g0, k.g1, k.g2] for k in keys], dtype=np.int32)
    arr_visible, _ = bricks_in_frustum_arr(arr, block_size=8, frustum_planes=planes)
    arr_keys = {
        BlockKey3D(level=int(r[0]), g0=int(r[1]), g1=int(r[2]), g2=int(r[3]))
        for r in arr_visible
    }
    assert set(dict_visible.keys()) == arr_keys


@pytest.mark.parametrize("container", ["set", "dict"])
def test_bricks_in_frustum_container_types(container):
    planes = _box_frustum(0, 20, 0, 20, 0, 20)
    key = BlockKey3D(level=1, g0=0, g1=0, g2=0)
    brick_keys = {key} if container == "set" else {key: 3}
    visible, _ = bricks_in_frustum(brick_keys, block_size=8, frustum_planes=planes)
    assert key in visible
