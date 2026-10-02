"""Mesh slicing: the fast paths give what the full pass gives.

``plans/mesh_refactor_v3.md`` 5.4 (S1-S4).  The reference is the rule
itself, written the slow way: a face survives when ``region.contains`` holds
for all three of its vertices, the survivors are reindexed, and the normals
are computed from the survivors.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import numpy as np
import pytest

from cellier.data.mesh import _mesh_slicing
from cellier.data.mesh._mesh_requests import MeshSliceRequest
from cellier.data.mesh._mesh_slicing import (
    LevelCache,
    MeshLevelArrays,
    compute_vertex_normals,
    face_bounds_index,
    slice_mesh,
)
from cellier.transform import ConvexRegion, HalfSpace
from tests._v2 import data_region, systems


def _random_mesh(rng, n_vertices=300, n_faces=500, ndim=3, colors=None):
    positions = rng.uniform(-10.0, 10.0, (n_vertices, ndim)).astype(np.float32)
    # Short faces: three vertices near each other in index and so in space
    # only by chance; the long ones below are the deliberate outliers.
    indices = rng.integers(0, n_vertices, (n_faces, 3)).astype(np.int32)
    values, layout = None, None
    if colors == "vertex":
        values, layout = rng.random((n_vertices, 4)).astype(np.float32), "vertex"
    elif colors == "face":
        values, layout = rng.random((n_faces, 4)).astype(np.float32), "face"
    return MeshLevelArrays(positions, indices, values, layout)


def _local_mesh(rng, n_faces=400, ndim=3):
    """Small triangles scattered in a box, plus one face spanning the box."""
    centres = rng.uniform(-10.0, 10.0, (n_faces, 1, ndim))
    corners = centres + rng.uniform(-0.3, 0.3, (n_faces, 3, ndim))
    positions = corners.reshape(-1, ndim).astype(np.float32)
    indices = np.arange(3 * n_faces, dtype=np.int32).reshape(n_faces, 3)
    # Share some vertices, so a dropped face can change a kept vertex.
    indices[1::7, 0] = indices[0::7, 0][: len(indices[1::7])]
    outlier = np.array([[-10] * ndim, [10] * ndim, [10] + [-10] * (ndim - 1)])
    positions = np.concatenate([positions, outlier.astype(np.float32)])
    indices = np.concatenate(
        [indices, [[len(positions) - 3, len(positions) - 2, len(positions) - 1]]]
    ).astype(np.int32)
    return MeshLevelArrays(positions, indices)


def _request(region, ndim, retained):
    sid = uuid4()
    return MeshSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=0,
        displayed_axes=tuple(retained),
        retained_axes=tuple(retained),
        region=region,
        output_axes=tuple(reversed(retained)),
    )


def _reference(arrays, request):
    """The whole-face rule, the slow way."""
    positions, indices = arrays.positions, arrays.indices
    mask = request.region.contains(positions)[indices].all(axis=1)
    faces = np.flatnonzero(mask)
    if not len(faces):
        return None
    surviving = indices[faces]
    kept = np.unique(surviving)
    remap = np.full(len(positions), -1, dtype=np.int32)
    remap[kept] = np.arange(len(kept), dtype=np.int32)
    out = np.zeros((len(kept), 3), dtype=np.float32)
    for column, axis in enumerate(request.output_axes):
        out[:, column] = positions[kept, axis]
    new_indices = remap[surviving]
    normals = (
        compute_vertex_normals(out, new_indices)
        if len(request.output_axes) == 3
        else None
    )
    return faces, kept, out, new_indices, normals


def _assert_equal(arrays, request, data):
    reference = _reference(arrays, request)
    if reference is None:
        assert data.is_empty
        return
    faces, kept, positions, indices, normals = reference
    assert not data.is_empty
    drawn = (
        np.arange(len(arrays.indices))
        if data.original_face_indices is None
        else data.original_face_indices
    )
    np.testing.assert_array_equal(drawn, faces)
    np.testing.assert_array_equal(data.positions, positions)
    np.testing.assert_array_equal(data.indices, indices)
    if normals is None:
        assert data.normals is None
    else:
        np.testing.assert_allclose(data.normals, normals, atol=2e-5)
    np.testing.assert_allclose(
        data.bounds, [positions.min(axis=0), positions.max(axis=0)]
    )
    if arrays.colors is not None:
        rows = faces if arrays.colors_layout == "face" else kept
        np.testing.assert_array_equal(data.colors, arrays.colors[rows])
    for name in ("positions", "indices", "normals", "colors"):
        array = getattr(data, name)
        if array is not None:
            assert array.flags.c_contiguous, name


def _slabs(rng, ndim, axes, n):
    for _ in range(n):
        yield {
            axis: (float(rng.uniform(-9.0, 9.0)), float(rng.uniform(0.0, 6.0)))
            for axis in axes
        }


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("colors", [None, "vertex", "face"])
def test_indexed_slicing_equals_contains_slicing_3d(seed, colors):
    """A 4D mesh in a 3D view, sliced along the fourth axis."""
    rng = np.random.default_rng(seed)
    arrays = _random_mesh(rng, ndim=4, colors=colors)
    cache = LevelCache()
    for slabs in _slabs(rng, 4, (0,), 12):
        request = _request(data_region(4, slabs), 4, (1, 2, 3))
        _assert_equal(arrays, request, slice_mesh(arrays, request, cache))
        _assert_equal(arrays, request, slice_mesh(arrays, request, None))


@pytest.mark.parametrize("seed", range(5))
def test_indexed_slicing_equals_contains_slicing_2d(seed):
    rng = np.random.default_rng(seed)
    arrays = _local_mesh(rng)
    cache = LevelCache()
    for slabs in _slabs(rng, 3, (0,), 12):
        request = _request(data_region(3, slabs), 3, (1, 2))
        _assert_equal(arrays, request, slice_mesh(arrays, request, cache))


@pytest.mark.parametrize("seed", range(4))
def test_several_constrained_axes_intersect(seed):
    rng = np.random.default_rng(seed)
    arrays = _local_mesh(rng, ndim=4)
    cache = LevelCache()
    for slabs in _slabs(rng, 4, (0, 1), 10):
        request = _request(data_region(4, slabs), 4, (2, 3))
        _assert_equal(arrays, request, slice_mesh(arrays, request, cache))
    assert cache.peek(("index", 0)) is not None
    assert cache.peek(("index", 1)) is not None


def test_a_straddling_face_recomputes_the_normals():
    """Faces crossing the slab edge are dropped; kept vertices they shared
    get normals from the surviving faces only."""
    rng = np.random.default_rng(3)
    arrays = _local_mesh(rng, n_faces=600)
    cache = LevelCache()
    checked = 0
    for slabs in _slabs(rng, 3, (0,), 40):
        # Slice along axis 0 but keep it: a 3D result with a clipped mesh.
        request = _request(data_region(3, slabs), 3, (0, 1, 2))
        data = slice_mesh(arrays, request, cache)
        _assert_equal(arrays, request, data)
        checked += not data.is_empty
    assert checked > 5


def test_an_oblique_region_takes_the_full_pass():
    rng = np.random.default_rng(4)
    arrays = _local_mesh(rng)
    data_system, _ = systems(3)
    region = ConvexRegion(
        coordinate_system=data_system.id,
        ndim=3,
        half_spaces=(
            HalfSpace(normal=np.array([1.0, 1.0, 0.0]), offset=3.0),
            HalfSpace(normal=np.array([-1.0, -1.0, 0.0]), offset=6.0),
        ),
    )
    cache = LevelCache()
    request = _request(region, 3, (0, 1, 2))
    _assert_equal(arrays, request, slice_mesh(arrays, request, cache))
    assert cache.peek(("index", 0)) is None


def test_scaled_normals_match_contains_exactly():
    """A region pulled back through a scale has non-unit normals."""
    rng = np.random.default_rng(5)
    arrays = _random_mesh(rng)
    data_system, _ = systems(3)
    cache = LevelCache()
    for _ in range(30):
        scale = float(rng.uniform(0.1, 7.0))
        centre, half = rng.uniform(-8.0, 8.0), rng.uniform(0.0, 5.0)
        region = ConvexRegion(
            coordinate_system=data_system.id,
            ndim=3,
            half_spaces=(
                HalfSpace(
                    normal=np.array([scale, 0.0, 0.0]), offset=scale * (centre + half)
                ),
                HalfSpace(
                    normal=np.array([-scale, 0.0, 0.0]),
                    offset=-scale * (centre - half),
                ),
            ),
        )
        request = _request(region, 3, (1, 2))
        _assert_equal(arrays, request, slice_mesh(arrays, request, cache))


def test_a_plane_through_exact_coordinates_keeps_the_faces_on_it():
    """Zero thickness: the comparison is exact, and inclusive on both sides."""
    positions = np.array(
        [[2, 0, 0], [2, 1, 0], [2, 0, 1], [3, 0, 0], [3, 1, 0], [3, 0, 1]], np.float32
    )
    arrays = MeshLevelArrays(positions, np.array([[0, 1, 2], [3, 4, 5]], np.int32))
    request = _request(data_region(3, {0: (3.0, 0.0)}), 3, (1, 2))
    data = slice_mesh(arrays, request, LevelCache())
    assert data.original_face_indices.tolist() == [1]


# -- S3: the identity path ----------------------------------------------------


def test_every_face_passing_returns_the_level_itself():
    rng = np.random.default_rng(6)
    arrays = _random_mesh(rng, colors="vertex")
    cache = LevelCache()
    request = _request(data_region(3), 3, (0, 1, 2))
    first = slice_mesh(arrays, request, cache)
    second = slice_mesh(arrays, request, cache)

    assert first.original_face_indices is None
    assert first.indices is arrays.indices
    assert first.colors is arrays.colors
    # Cached: the second read hands out the same arrays, and computes nothing.
    assert second.positions is first.positions
    assert second.normals is first.normals
    assert cache.builds == {
        ("projected", (2, 1, 0)): 1,
        ("normals", (2, 1, 0)): 1,
    }
    np.testing.assert_array_equal(first.positions, arrays.positions[:, ::-1])
    np.testing.assert_allclose(
        first.normals, compute_vertex_normals(first.positions, arrays.indices)
    )


def test_a_slab_holding_every_face_is_the_identity_too():
    rng = np.random.default_rng(7)
    arrays = _random_mesh(rng, ndim=4)
    arrays.positions[:, 0] = 2.0  # one timepoint
    cache = LevelCache()
    request = _request(data_region(4, {0: (2.0, 0.0)}), 4, (1, 2, 3))
    data = slice_mesh(arrays, request, cache)
    assert data.original_face_indices is None
    assert data.indices is arrays.indices


def test_an_empty_mesh_is_empty():
    arrays = MeshLevelArrays(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
    request = _request(data_region(3), 3, (0, 1, 2))
    assert slice_mesh(arrays, request, LevelCache()).is_empty


# -- S2: cached normals -------------------------------------------------------


def _series(n_t=5, faces_per_t=40, seed=8):
    """A ``(t, z, y, x)`` series: no face straddles a timepoint."""
    rng = np.random.default_rng(seed)
    positions, indices = [], []
    for t in range(n_t):
        frame = _local_mesh(rng, n_faces=faces_per_t)
        column = np.full((len(frame.positions), 1), t, np.float32)
        indices.append(frame.indices + sum(len(p) for p in positions))
        positions.append(np.concatenate([column, frame.positions], axis=1))
    return MeshLevelArrays(
        np.concatenate(positions), np.concatenate(indices).astype(np.int32)
    )


def test_a_series_gathers_cached_normals(monkeypatch):
    arrays = _series()
    cache = LevelCache()
    calls: list[int] = []
    inner = _mesh_slicing.compute_vertex_normals

    def counting(positions, indices):
        calls.append(len(indices))
        return inner(positions, indices)

    monkeypatch.setattr(_mesh_slicing, "compute_vertex_normals", counting)
    for t in (0, 3, 1, 3):
        request = _request(data_region(4, {0: (float(t), 0.0)}), 4, (1, 2, 3))
        data = slice_mesh(arrays, request, cache)
        _assert_equal(arrays, request, data)
    # Once, over the whole level; every timepoint gathers from it.
    assert calls == [len(arrays.indices)]
    # The projected level was used for the normals and not kept.
    assert cache.peek(("projected", (3, 2, 1))) is None


def test_normals_are_dropped_with_the_cache():
    arrays = _series()
    cache = LevelCache()
    request = _request(data_region(4, {0: (1.0, 0.0)}), 4, (1, 2, 3))
    slice_mesh(arrays, request, cache)
    assert cache.peek(("normals", (3, 2, 1))) is not None
    # A store change replaces the cache (see the store tests); a new cache
    # holds nothing.
    assert LevelCache().peek(("normals", (3, 2, 1))) is None


def test_bincount_normals_match_the_accumulating_ones():
    rng = np.random.default_rng(9)
    arrays = _random_mesh(rng)
    positions, indices = arrays.positions, arrays.indices
    e1 = positions[indices[:, 1]] - positions[indices[:, 0]]
    e2 = positions[indices[:, 2]] - positions[indices[:, 0]]
    face_normals = np.cross(e1, e2)
    expected = np.zeros_like(positions)
    for corner in range(3):
        np.add.at(expected, indices[:, corner], face_normals)
    norms = np.linalg.norm(expected, axis=1, keepdims=True)
    expected = np.where(norms > 0, expected / np.where(norms > 0, norms, 1), [0, 0, 1])
    np.testing.assert_allclose(
        compute_vertex_normals(positions, indices), expected, atol=1e-5
    )


def test_a_vertex_no_face_uses_gets_the_default_normal():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], np.float32)
    normals = compute_vertex_normals(positions, np.array([[0, 1, 2]], np.int32))
    np.testing.assert_array_equal(normals[3], [0.0, 0.0, 1.0])


# -- S4: the inline build -----------------------------------------------------


def test_concurrent_first_reads_build_one_entry_per_axis():
    arrays = _series(n_t=6, faces_per_t=300)
    cache = LevelCache()
    requests = [
        _request(data_region(4, {0: (float(t), 0.0)}), 4, (1, 2, 3)) for t in range(6)
    ]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda r: slice_mesh(arrays, r, cache), requests))
    for request, data in zip(requests, results, strict=True):
        _assert_equal(arrays, request, data)
    assert cache.builds[("index", 0)] == 1
    assert cache.builds[("normals", (3, 2, 1))] == 1


def test_builds_for_different_axes_do_not_wait_on_each_other():
    """The lock is per entry: axis 1 builds while axis 0's build is held."""
    cache = LevelCache()
    holding = threading.Event()
    release = threading.Event()

    def slow():
        holding.set()
        assert release.wait(5.0)
        return "axis 0"

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.get, ("index", 0), slow)
        assert holding.wait(5.0)
        # Axis 0 is mid-build; axis 1 is built and read without waiting.
        assert cache.get(("index", 1), lambda: "axis 1") == "axis 1"
        assert cache.peek(("index", 0)) is None
        release.set()
        assert first.result(5.0) == "axis 0"
    assert cache.get(("index", 0), lambda: "again") == "axis 0"
    assert cache.builds == {("index", 0): 1, ("index", 1): 1}


def test_the_face_index_is_built_from_the_faces():
    arrays = _local_mesh(np.random.default_rng(10))
    index = face_bounds_index(arrays, LevelCache(), 0)
    column = arrays.positions[:, 0][arrays.indices]
    overlapping = np.flatnonzero(
        (column.min(axis=1) <= 1.0) & (column.max(axis=1) >= 0.0)
    )
    np.testing.assert_array_equal(np.sort(index.overlapping(0.0, 1.0)), overlapping)
    # The box-spanning face is a long item.
    assert len(arrays.indices) - 1 in index.long_ids
