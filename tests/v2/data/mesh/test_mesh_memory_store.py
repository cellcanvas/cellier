"""Tests for MeshMemoryStore construction and get_data."""

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.mesh._mesh_requests import MeshSliceRequest
from tests._v2 import data_region


def _simple_store() -> MeshMemoryStore:
    """Tetrahedron: 4 vertices, 4 faces."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return MeshMemoryStore(positions=positions, indices=indices, name="tet")


def _req(displayed=(1, 2), sliced=None, thickness=0.5, ndim=3):
    """One request, with the filter as the region the store now takes.

    ``sliced`` maps axis to a **data**-space position; each becomes a slab of
    half-width *thickness*, which is exactly what ``slice_indices`` plus
    ``thickness`` meant before Phase 8 deleted them (R8.3).
    """
    if sliced is None:
        sliced = {0: 0}
    sid = uuid4()
    return MeshSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=0,
        displayed_axes=displayed,
        retained_axes=tuple(sorted(displayed)),
        region=data_region(
            ndim, {axis: (position, thickness) for axis, position in sliced.items()}
        ),
        output_axes=tuple(sorted(displayed, reverse=True)),
    )


# ── Construction ──────────────────────────────────────────────────────────────


def test_normals_in_3d_get_data_result():
    """Normals are computed from projected geometry for 3-D display."""
    store = _simple_store()
    sid = uuid4()
    req = MeshSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=0,
        displayed_axes=(0, 1, 2),
        retained_axes=(0, 1, 2),
        region=data_region(3),
        output_axes=(2, 1, 0),
    )
    result = asyncio.run(store.get_data(req))
    assert result.normals is not None
    assert result.normals.shape == (result.positions.shape[0], 3)
    assert result.normals.dtype == np.float32


def test_no_normals_for_2d_display():
    """A 2-D result carries no normals: it is drawn unlit."""
    store = _simple_store()
    result = asyncio.run(store.get_data(_req(displayed=(1, 2), sliced={0: 0})))
    assert not result.is_empty
    assert result.normals is None


def test_int64_indices_coerced_to_int32():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int64)
    store = MeshMemoryStore(positions=positions, indices=indices)
    assert store.indices.dtype == np.int32


def _triangle():
    """3 vertices, 1 face."""
    return (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.int32),
    )


def _tetrahedron():
    """4 vertices and 4 faces -- the count the old inference could not read."""
    return (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32),
    )


def test_colors_mode_reports_the_declared_layout():
    positions, indices = _triangle()
    store = MeshMemoryStore(
        positions=positions,
        indices=indices,
        colors=np.ones((3, 4), dtype=np.float32),
        colors_layout="vertex",
    )
    assert store.colors_mode == "vertex"


def test_colors_mode_face():
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 1]],
        dtype=np.float32,
    )
    indices = np.array(
        [[0, 1, 2], [1, 3, 2], [0, 1, 4], [1, 3, 4], [0, 2, 4]], dtype=np.int32
    )
    store = MeshMemoryStore(
        positions=positions,
        indices=indices,
        colors=np.ones((5, 4), dtype=np.float32),
        colors_layout="face",
    )
    assert store.colors_mode == "face"


def test_colors_mode_none_without_colors():
    positions, indices = _triangle()
    store = MeshMemoryStore(positions=positions, indices=indices)
    assert store.colors_mode == "none"


def test_equal_vertex_and_face_counts_honour_the_declaration():
    """A tetrahedron has 4 vertices and 4 faces.

    The old inference compared ``colors.shape[0]`` against ``n_faces`` and
    so reported per-vertex colours here as per-face -- and then gathered
    the wrong rows when slicing.  With the layout declared, both readings
    are available and neither is guessed.
    """
    positions, indices = _tetrahedron()
    colors = np.ones((4, 4), dtype=np.float32)

    as_vertex = MeshMemoryStore(
        positions=positions, indices=indices, colors=colors, colors_layout="vertex"
    )
    as_face = MeshMemoryStore(
        positions=positions, indices=indices, colors=colors, colors_layout="face"
    )
    assert as_vertex.colors_mode == "vertex"
    assert as_face.colors_mode == "face"


def test_colors_without_layout_raises():
    positions, indices = _triangle()
    with pytest.raises(ValueError, match="explicit colors_layout"):
        MeshMemoryStore(
            positions=positions,
            indices=indices,
            colors=np.ones((3, 4), dtype=np.float32),
        )


def test_layout_without_colors_raises():
    positions, indices = _triangle()
    with pytest.raises(ValueError, match="without colors"):
        MeshMemoryStore(positions=positions, indices=indices, colors_layout="vertex")


def test_layout_disagreeing_with_the_array_length_raises():
    positions, indices = _triangle()  # 3 vertices, 1 face
    with pytest.raises(ValueError, match="expects 1 rows"):
        MeshMemoryStore(
            positions=positions,
            indices=indices,
            colors=np.ones((7, 4), dtype=np.float32),
            colors_layout="face",
        )


def test_assigning_colors_without_a_layout_raises():
    """validate_assignment keeps the invariant past construction.

    Without it the check held only at __init__, and a later
    ``store.colors = ...`` left colors_layout None -- which colors_mode
    returned, and get_data's ``== "face"`` test silently read as vertex.
    """
    from pydantic import ValidationError

    positions, indices = _triangle()
    store = MeshMemoryStore(positions=positions, indices=indices)
    with pytest.raises(ValidationError, match="explicit colors_layout"):
        store.colors = np.ones((3, 4), dtype=np.float32)


def test_reassigning_colors_with_a_layout_set_is_fine():
    """The demo path: declare the layout up front, swap colours later."""
    positions, indices = _triangle()
    store = MeshMemoryStore(
        positions=positions,
        indices=indices,
        colors=np.zeros((3, 4), dtype=np.float32),
        colors_layout="vertex",
    )
    store.colors = np.ones((3, 4), dtype=np.float32)
    assert store.colors_mode == "vertex"
    assert np.allclose(store.colors, 1.0)


# ── get_data — 3D (all axes displayed) ───────────────────────────────────────


def test_get_data_3d_returns_all_faces():
    store = _simple_store()
    sid = uuid4()
    req = MeshSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=0,
        displayed_axes=(0, 1, 2),
        retained_axes=(0, 1, 2),
        region=data_region(3),
        output_axes=(2, 1, 0),
    )
    result = asyncio.run(store.get_data(req))
    assert result.is_empty is False
    assert result.indices.shape[0] == store.n_faces
    # Upload-ready: the columns are the output axes, (x, y, z).
    np.testing.assert_array_equal(result.positions, store.positions[:, ::-1])
    assert result.positions.flags.c_contiguous
    assert result.positions.dtype == np.float32
    np.testing.assert_allclose(result.bounds, [[0, 0, 0], [1, 1, 1]])
    assert result.level == 0


# ── get_data — 2D (slab filter) ──────────────────────────────────────────────


def test_get_data_2d_empty_slab():
    store = _simple_store()
    # Slice at z=100, far outside the tetrahedron.
    result = asyncio.run(store.get_data(_req(sliced={0: 100})))
    assert result.is_empty is True
    assert result.indices.shape == (1, 3)  # placeholder


def test_get_data_2d_positions_projected():
    store = _simple_store()
    # Tetrahedron vertices: 0=(0,0,0), 1=(1,0,0), 2=(0,1,0), 3=(0,0,1).
    # Axis 0 (the sliced axis) values: 0→0, 1→1, 2→0, 3→0.
    # Slice at axis0=0, thickness=0.5: vertices 0, 2, 3 are in the slab;
    # vertex 1 is not.
    # All-vertices rule: only face [0,2,3] has every vertex in the slab.
    result = asyncio.run(store.get_data(_req(sliced={0: 0}, thickness=0.5)))
    assert not result.is_empty
    # Upload-ready: (x, y) of the two retained axes, and a zero third column.
    assert result.positions.shape[1] == 3
    np.testing.assert_array_equal(result.positions[:, 2], 0.0)
    np.testing.assert_array_equal(
        result.positions[:, :2], store.positions[[0, 2, 3]][:, [2, 1]]
    )
    # Exactly one face survives — the one whose vertices are all on the slice.
    assert result.indices.shape == (1, 3)
    # Exactly three vertices survive.
    assert result.positions.shape[0] == 3


def test_get_data_2d_all_vertices_must_be_in_slab():
    """Faces with any off-slab vertex are excluded (all-vertex rule)."""
    store = _simple_store()
    # Tetrahedron at z=0 slice: vertices 0, 2, 3 in slab; vertex 1 at z=1.
    # Faces touching vertex 1 ([0,1,2], [0,1,3], [1,2,3]) must be excluded.
    result = asyncio.run(store.get_data(_req(sliced={0: 0}, thickness=0.5)))
    assert not result.is_empty
    assert result.indices.shape[0] == 1  # only face [0,2,3] survives


def test_get_data_2d_off_slab_face_excluded():
    """A face whose vertices are entirely off-slab produces an empty result."""
    store = _simple_store()
    # Slice at z=1, thickness=0.5: only vertex 1 (z=1) is in the slab.
    # No face has ALL vertices at z≈1, so result must be empty.
    result = asyncio.run(store.get_data(_req(sliced={0: 1}, thickness=0.5)))
    assert result.is_empty


def test_get_data_2d_indices_reindexed():
    """All index values must be valid into the compacted positions array."""
    store = _simple_store()
    result = asyncio.run(store.get_data(_req(sliced={0: 0}, thickness=0.5)))
    if not result.is_empty:
        n_verts = result.positions.shape[0]
        assert result.indices.max() < n_verts
        assert result.indices.min() >= 0


def test_get_data_2d_vertex_colors_gathered():
    # Square base: 4 vertices, 2 faces (2 triangles).
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    store = MeshMemoryStore(
        positions=positions,
        indices=indices,
        colors=np.eye(4, dtype=np.float32),
        colors_layout="vertex",
    )
    result = asyncio.run(store.get_data(_req(sliced={0: 0}, thickness=0.5)))
    if not result.is_empty:
        assert result.colors is not None
        assert result.colors.shape[0] == result.positions.shape[0]
        assert result.color_mode == "vertex"


# ── original_face_indices (pick index mapping) ────────────────────────────────


def _stacked_faces_store() -> MeshMemoryStore:
    """Three independent triangles, each planar at z = 0, 10, 20."""
    verts = []
    faces = []
    for z in (0, 10, 20):
        b = len(verts)
        verts += [[z, 0, 0], [z, 1, 0], [z, 0, 1]]
        faces.append([b, b + 1, b + 2])
    return MeshMemoryStore(
        positions=np.array(verts, dtype=np.float32),
        indices=np.array(faces, dtype=np.int32),
    )


def test_original_face_indices_identity_in_3d():
    """A full 3-D view keeps every face: no map, and the store's own faces."""
    store = _stacked_faces_store()
    result = asyncio.run(store.get_data(_req(displayed=(0, 1, 2), sliced={})))
    assert result.original_face_indices is None
    assert result.indices is store.indices


def test_original_face_indices_track_surviving_subset_in_2d():
    """A 2-D slab keeping one face reports that face's original index.

    pygfx reports the rendered face index (here 0, the only survivor);
    ``original_face_indices`` maps it back to original face 1.
    """
    store = _stacked_faces_store()
    result = asyncio.run(
        store.get_data(_req(displayed=(1, 2), sliced={0: 10}, thickness=0.5))
    )
    assert not result.is_empty
    assert result.indices.shape[0] == 1
    assert list(result.original_face_indices) == [1]


# ── The read runs off the event loop ──────────────────────────────────────────


def test_get_data_runs_in_an_executor_thread():
    """The slicing work is not done on the event loop's thread."""
    import threading

    from cellier.data.mesh import _mesh_slicing

    store = _simple_store()
    seen: list[int] = []
    inner = _mesh_slicing.slice_mesh

    def recording(arrays, request, cache):
        seen.append(threading.get_ident())
        return inner(arrays, request, cache)

    async def run():
        loop_thread = threading.get_ident()
        _mesh_slicing.slice_mesh = recording
        try:
            await store.get_data(_req(sliced={0: 0}))
        finally:
            _mesh_slicing.slice_mesh = inner
        return loop_thread

    loop_thread = asyncio.run(run())
    assert seen and seen[0] != loop_thread


def test_a_store_change_drops_the_slicing_cache():
    store = _stacked_faces_store()
    asyncio.run(store.get_data(_req(displayed=(1, 2), sliced={0: 10})))
    before = store.level_cache()
    assert before.peek(("index", 0)) is not None
    store.positions = store.positions + 1.0
    assert store.level_cache() is not before
    assert store.level_cache().peek(("index", 0)) is None


def test_a_copied_store_gets_its_own_cache():
    store = _stacked_faces_store()
    asyncio.run(store.get_data(_req(displayed=(1, 2), sliced={0: 10})))
    copy = store.model_copy(deep=True)
    assert copy.level_cache() is not store.level_cache()
    assert copy.level_cache().peek(("index", 0)) is None


def test_only_level_zero_exists():
    with pytest.raises(ValueError, match="one level"):
        _simple_store().level_arrays(1)
