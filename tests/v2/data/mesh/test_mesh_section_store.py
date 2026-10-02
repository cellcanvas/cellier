"""A mesh store's section read (``plans/mesh_refactor_v3.md`` 5.5, 5.6).

``MeshMemoryStore.get_data`` with a request that carries a ``section``: the
filter on the axes the cut does not replace, then the cut, then upload-ready
arrays.  The kernel itself is tested in ``test_section_kernel.py``.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.mesh._mesh_requests import (
    MeshData,
    MeshSectionData,
    MeshSectionRequest,
    MeshSliceRequest,
)
from cellier.data.mesh._mesh_slicing import (
    LevelCache,
    MeshLevelArrays,
    level_closure,
    slice_mesh,
)
from tests._meshes import uv_sphere
from tests._v2 import data_region

R = 5.0


def _request(ndim, section, retained, filter_slabs=None):
    sid = uuid4()
    return MeshSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=0,
        displayed_axes=tuple(retained),
        retained_axes=tuple(retained),
        region=data_region(ndim, filter_slabs),
        output_axes=tuple(reversed(retained)),
        section=section,
    )


def _z_cut(z, ndim=3, axis=0, **kwargs):
    normal = [0.0] * ndim
    normal[axis] = 1.0
    return MeshSectionRequest(tuple(normal), (float(z),), **kwargs)


def _area(data: MeshSectionData) -> float:
    triangles = data.fill_positions[data.fill_indices]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    return 0.5 * float(np.linalg.norm(cross, axis=1).sum())


def _sphere_store(**kwargs) -> MeshMemoryStore:
    positions, indices = uv_sphere(R, (10.0, 20.0, 30.0), n_lat=24, n_lon=48)
    return MeshMemoryStore(positions=positions, indices=indices, **kwargs)


def _series_arrays(n_t=4):
    """One sphere per timepoint, radius growing with t: ``(t, z, y, x)``."""
    positions, indices, offset = [], [], 0
    for t in range(n_t):
        p, i = uv_sphere(2.0 + t, (10.0, 10.0, 10.0), n_lat=16, n_lon=32)
        positions.append(np.concatenate([np.full((len(p), 1), t, np.float32), p], 1))
        indices.append(i + offset)
        offset += len(p)
    return np.concatenate(positions), np.concatenate(indices).astype(np.int32)


def test_a_zyx_mesh_is_cut_at_z():
    store = _sphere_store()
    data = asyncio.run(store.get_data(_request(3, _z_cut(11.0), (1, 2))))
    assert isinstance(data, MeshSectionData)
    assert not data.is_empty
    # A circle of radius sqrt(R^2 - 1), centred on (x, y) = (30, 20).
    expected = np.pi * (R**2 - 1.0)
    assert _area(data) == pytest.approx(expected, rel=0.02)
    assert data.n_closed_loops == 1
    centre = data.outline_positions[:, :2].mean(axis=0)
    np.testing.assert_allclose(centre, [30.0, 20.0], atol=0.05)
    radius = np.linalg.norm(data.outline_positions[:, :2] - [30.0, 20.0], axis=1)
    assert radius.max() <= np.sqrt(R**2 - 1.0) + 1e-4
    # Upload-ready: (x, y, 0) float32, contiguous, int32 indices.
    for name in ("fill_positions", "outline_positions"):
        array = getattr(data, name)
        assert array.dtype == np.float32 and array.flags.c_contiguous
        np.testing.assert_array_equal(array[:, 2], 0.0)
    assert data.fill_indices.dtype == np.int32
    np.testing.assert_allclose(
        data.outline_bounds,
        [data.outline_positions.min(axis=0), data.outline_positions.max(axis=0)],
    )
    assert data.level == 0


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_each_spatial_axis_can_be_the_cut(axis):
    """The ortho viewer's three 2D panels: the same circle on every axis."""
    store = _sphere_store()
    centre = (10.0, 20.0, 30.0)
    retained = tuple(a for a in range(3) if a != axis)
    data = asyncio.run(
        store.get_data(_request(3, _z_cut(centre[axis] + 1.0, axis=axis), retained))
    )
    assert data.n_closed_loops == 1
    assert _area(data) == pytest.approx(np.pi * (R**2 - 1.0), rel=0.03)


def test_a_time_series_is_filtered_by_t_then_cut_at_z():
    positions, indices = _series_arrays()
    store = MeshMemoryStore(positions=positions, indices=indices)
    for t in range(4):
        request = _request(
            4, _z_cut(10.0, ndim=4, axis=1), (2, 3), {0: (float(t), 0.0)}
        )
        data = asyncio.run(store.get_data(request))
        # The equator of that timepoint's sphere only.
        assert data.n_closed_loops == 1
        radius = np.linalg.norm(data.outline_positions[:, :2] - [10.0, 10.0], axis=1)
        np.testing.assert_allclose(radius.max(), 2.0 + t, rtol=1e-3)
        faces_per_t = len(indices) // 4
        assert set(data.outline_face_ids // faces_per_t) == {t}


def test_a_time_between_frames_cuts_nothing():
    positions, indices = _series_arrays()
    store = MeshMemoryStore(positions=positions, indices=indices)
    request = _request(4, _z_cut(10.0, ndim=4, axis=1), (2, 3), {0: (1.5, 0.0)})
    data = asyncio.run(store.get_data(request))
    assert data.is_empty
    assert data.fill_bounds is None and data.outline_bounds is None


def test_the_candidate_subset_equals_the_full_pass():
    """S4's overlap query against no cache at all: the same cut, exactly."""
    positions, indices = _series_arrays()
    arrays = MeshLevelArrays(positions, indices)
    cache = LevelCache()
    for t, z in ((0, 9.3), (2, 10.0), (3, 13.9), (1, 30.0)):
        request = _request(4, _z_cut(z, ndim=4, axis=1), (2, 3), {0: (float(t), 0.0)})
        indexed = slice_mesh(arrays, request, cache)
        full = slice_mesh(arrays, request, None)
        np.testing.assert_array_equal(indexed.outline_positions, full.outline_positions)
        np.testing.assert_array_equal(indexed.outline_face_ids, full.outline_face_ids)
        np.testing.assert_array_equal(indexed.fill_positions, full.fill_positions)
        np.testing.assert_array_equal(indexed.fill_indices, full.fill_indices)
    assert cache.peek(("index", 1)) is not None


def test_a_mesh_lying_at_one_z_is_drawn_as_itself():
    """Flat polygons: every face is in the plane, or none is."""
    positions = np.array([[7, 0, 0], [7, 4, 0], [7, 4, 3], [7, 0, 3]], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    store = MeshMemoryStore(positions=positions, indices=indices)
    on = asyncio.run(store.get_data(_request(3, _z_cut(7.0), (1, 2))))
    assert sorted(on.fill_face_ids.tolist()) == [0, 1]
    assert _area(on) == pytest.approx(12.0)
    assert len(on.outline_face_ids) == 4  # the rectangle's border
    off = asyncio.run(store.get_data(_request(3, _z_cut(7.5), (1, 2))))
    assert off.is_empty


def test_cut_against_slab():
    store = _sphere_store()
    cut = asyncio.run(store.get_data(_request(3, _z_cut(10.0), (1, 2))))
    slab_section = MeshSectionRequest((1.0, 0.0, 0.0), (9.0, 11.0), mode="slab")
    slab = asyncio.run(store.get_data(_request(3, slab_section, (1, 2))))
    # The slab draws the surface between its faces as well as two caps.
    assert (slab.fill_face_ids >= 0).any()
    assert (slab.fill_face_ids == -1).any()
    assert slab.n_closed_loops == 2
    assert len(slab.outline_face_ids) > len(cut.outline_face_ids)
    assert (cut.fill_face_ids == -1).all()
    # A slab with no thickness is the cut.
    thin = MeshSectionRequest((1.0, 0.0, 0.0), (10.0, 10.0), mode="slab")
    same = asyncio.run(store.get_data(_request(3, thin, (1, 2))))
    np.testing.assert_array_equal(same.outline_positions, cut.outline_positions)


def test_a_scaled_or_reversed_normal_is_the_same_plane():
    """What an anisotropic or flipped transform pulls a plane back to."""
    store = _sphere_store()
    reference = asyncio.run(store.get_data(_request(3, _z_cut(11.0), (1, 2))))
    for normal, offset in (((2.5, 0.0, 0.0), 27.5), ((-0.5, 0.0, 0.0), -5.5)):
        section = MeshSectionRequest(normal, (offset,))
        data = asyncio.run(store.get_data(_request(3, section, (1, 2))))
        np.testing.assert_array_equal(
            data.outline_positions, reference.outline_positions
        )
    slab = MeshSectionRequest((-1.0, 0.0, 0.0), (-11.0, -9.0), mode="slab")
    data = asyncio.run(store.get_data(_request(3, slab, (1, 2))))
    assert data.n_closed_loops == 2


def test_parts_follow_the_request():
    store = _sphere_store()
    no_fill = asyncio.run(store.get_data(_request(3, _z_cut(11.0, fill=False), (1, 2))))
    assert len(no_fill.fill_indices) == 0 and len(no_fill.outline_face_ids) > 0
    no_outline = asyncio.run(
        store.get_data(_request(3, _z_cut(11.0, outline=False), (1, 2)))
    )
    assert len(no_outline.outline_face_ids) == 0 and len(no_outline.fill_indices) > 0


@pytest.mark.parametrize("layout", ["vertex", "face"])
def test_colours_come_out_per_vertex(layout):
    positions, indices = uv_sphere(R, (10.0, 20.0, 30.0))
    n = len(positions) if layout == "vertex" else len(indices)
    colors = np.random.default_rng(0).random((n, 4)).astype(np.float32)
    store = MeshMemoryStore(
        positions=positions, indices=indices, colors=colors, colors_layout=layout
    )
    data = asyncio.run(store.get_data(_request(3, _z_cut(11.0), (1, 2))))
    assert data.color_mode == layout
    assert data.fill_colors.shape == (len(data.fill_positions), 4)
    assert data.outline_colors.shape == (len(data.outline_positions), 4)
    assert data.fill_colors.dtype == np.float32


def test_no_section_is_the_whole_face_read():
    store = _sphere_store()
    data = asyncio.run(store.get_data(_request(3, None, (1, 2))))
    assert isinstance(data, MeshData)


def test_a_section_needs_one_axis_across_the_view():
    store = _sphere_store()
    in_view = MeshSectionRequest((0.0, 1.0, 0.0), (20.0,))
    with pytest.raises(ValueError, match="exactly one data axis"):
        asyncio.run(store.get_data(_request(3, in_view, (1, 2))))
    with pytest.raises(ValueError, match="two output axes"):
        asyncio.run(store.get_data(_request(3, _z_cut(10.0), (0, 1, 2))))


def test_the_closure_report_is_computed_by_the_first_section_read():
    """X5: a mesh that is only ever drawn in 3D never pays for it."""
    store = _sphere_store()
    assert store.level_cache().peek(("closure",)) is None
    asyncio.run(store.get_data(_request(3, None, (0, 1, 2))))
    assert store.level_cache().peek(("closure",)) is None
    asyncio.run(store.get_data(_request(3, _z_cut(11.0), (1, 2))))
    report = store.level_cache().peek(("closure",))
    assert report.closed
    assert level_closure(store.level_arrays(), store.level_cache()) is report


def _info_rows(store) -> dict[str, str]:
    return dict(store.dataset_info().sections[0].rows)


def test_dataset_info_reports_closure_after_a_section_read():
    store = _sphere_store()
    assert _info_rows(store)["Closed"] == "not computed"
    asyncio.run(store.get_data(_request(3, _z_cut(11.0), (1, 2))))
    assert _info_rows(store)["Closed"] == "yes"

    positions, indices = uv_sphere(R, (10.0, 20.0, 30.0))
    opened = MeshMemoryStore(positions=positions, indices=indices[:-3])
    asyncio.run(opened.get_data(_request(3, _z_cut(11.0), (1, 2))))
    assert _info_rows(opened)["Closed"].startswith("no (")
    assert "boundary" in _info_rows(opened)["Closed"]
