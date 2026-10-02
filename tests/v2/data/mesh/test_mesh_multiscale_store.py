"""``MultiscaleMeshStore``: a mesh at several levels (plan M3).

Each level is read by the same code as a ``MeshMemoryStore``; what is new is
the validation across levels, which level a request reads, and that the
extent is the finest level's.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from cellier.data.mesh import MeshLevel, MeshMemoryStore, MultiscaleMeshStore
from cellier.data.mesh._mesh_requests import (
    MeshSectionData,
    MeshSectionRequest,
    MeshSliceRequest,
)
from cellier.visuals import GeometryLodConfig
from tests._meshes import uv_sphere
from tests._v2 import data_region

CENTRE = (16.0, 16.0, 16.0)


def _level(n_lat: int, radius: float = 10.0, colors: str | None = None) -> MeshLevel:
    positions, indices = uv_sphere(radius, CENTRE, n_lat=n_lat, n_lon=2 * n_lat)
    if colors is None:
        return MeshLevel(positions=positions, indices=indices)
    n = len(positions) if colors == "vertex" else len(indices)
    rgba = np.linspace(0.0, 1.0, 4 * n, dtype=np.float32).reshape(n, 4)
    return MeshLevel(
        positions=positions, indices=indices, colors=rgba, colors_layout=colors
    )


def _store(*n_lats: int, **kwargs) -> MultiscaleMeshStore:
    return MultiscaleMeshStore(levels=[_level(n) for n in n_lats], **kwargs)


def _whole(level: int = 0, retained=(0, 1, 2)) -> MeshSliceRequest:
    sid = uuid4()
    return MeshSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=level,
        displayed_axes=tuple(retained),
        retained_axes=tuple(retained),
        region=data_region(3),
        output_axes=tuple(reversed(retained)),
    )


def _cut(level: int = 0, z: float = 16.0) -> MeshSliceRequest:
    sid = uuid4()
    return MeshSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=level,
        displayed_axes=(1, 2),
        retained_axes=(1, 2),
        region=data_region(3),
        output_axes=(2, 1),
        section=MeshSectionRequest(
            normal=(1.0, 0.0, 0.0), offsets=(z,), mode="cut", outline=True, fill=True
        ),
    )


# -- validation ---------------------------------------------------------------


def test_a_store_needs_a_level():
    with pytest.raises(ValidationError):
        MultiscaleMeshStore(levels=[])


def test_every_level_has_the_same_columns():
    flat = MeshLevel(positions=np.zeros((3, 2)), indices=[[0, 1, 2]])
    with pytest.raises(ValidationError, match="same number of position columns"):
        MultiscaleMeshStore(levels=[_level(6), flat])


def test_colours_are_on_every_level_or_none():
    with pytest.raises(ValidationError, match="on every level or on none"):
        MultiscaleMeshStore(levels=[_level(6, colors="vertex"), _level(4)])


def test_every_level_has_the_same_colour_layout():
    with pytest.raises(ValidationError, match="same colors_layout"):
        MultiscaleMeshStore(
            levels=[_level(6, colors="vertex"), _level(4, colors="face")]
        )


def test_a_level_declares_its_colour_layout():
    positions, indices = uv_sphere(1.0, n_lat=4, n_lon=8)
    with pytest.raises(ValidationError, match="explicit colors_layout"):
        MeshLevel(
            positions=positions, indices=indices, colors=np.ones((len(positions), 4))
        )
    with pytest.raises(ValidationError, match="without colors"):
        MeshLevel(positions=positions, indices=indices, colors_layout="vertex")
    with pytest.raises(ValidationError, match="expects"):
        MeshLevel(
            positions=positions,
            indices=indices,
            colors=np.ones((3, 4)),
            colors_layout="vertex",
        )


def test_a_level_checks_its_array_shapes():
    with pytest.raises(ValidationError, match="n_vertices, N"):
        MeshLevel(positions=np.zeros(6), indices=[[0, 1, 2]])
    with pytest.raises(ValidationError, match="n_faces, 3"):
        MeshLevel(positions=np.zeros((4, 3)), indices=[[0, 1, 2, 3]])


def test_a_level_coerces_its_arrays_and_is_frozen():
    level = MeshLevel(
        positions=np.zeros((3, 3), dtype=np.float64),
        indices=np.array([[0, 1, 2]], dtype=np.int64),
    )
    assert level.positions.dtype == np.float32
    assert level.indices.dtype == np.int32
    with pytest.raises(ValidationError):
        level.positions = np.ones((3, 3))


# -- what a store reports -----------------------------------------------------


def test_counts_and_extent_are_the_finest_levels():
    fine, coarse = _level(12, radius=10.0), _level(4, radius=14.0)
    store = MultiscaleMeshStore(levels=[fine, coarse])

    assert store.level_count == 2
    assert store.ndim == 3
    assert (store.n_vertices, store.n_faces) == (fine.n_vertices, fine.n_faces)
    low = fine.positions.min(axis=0)
    high = fine.positions.max(axis=0)
    np.testing.assert_allclose(store.axis_extents, list(zip(low, high)))
    assert store.colors_mode == "none"


def test_dataset_info_lists_every_level():
    store = _store(12, 4)
    rows = dict(store.dataset_info().sections[0].rows)
    assert rows["Levels"] == "2"
    assert rows["Level 1"].startswith(f"{store.levels[0].n_vertices} vertices")
    assert rows["Level 2"].startswith(f"{store.levels[1].n_vertices} vertices")


async def test_a_levels_closure_is_reported_once_it_has_been_cut():
    store = _store(12, 4)
    assert "not computed" in dict(store.dataset_info().sections[0].rows)["Level 2"]

    await store.get_data(_cut(level=1))

    rows = dict(store.dataset_info().sections[0].rows)
    assert "closed: yes" in rows["Level 2"]
    assert "not computed" in rows["Level 1"]


# -- reads --------------------------------------------------------------------


async def test_a_request_reads_the_level_it_names():
    store = _store(12, 4)

    fine = await store.get_data(_whole(level=0))
    coarse = await store.get_data(_whole(level=1))

    assert (fine.level, coarse.level) == (0, 1)
    assert len(fine.indices) == store.levels[0].n_faces
    assert len(coarse.indices) == store.levels[1].n_faces


async def test_a_level_the_store_does_not_have_is_refused():
    store = _store(12, 4)
    with pytest.raises(ValueError, match="levels 0 to 1"):
        await store.get_data(_whole(level=2))


@pytest.mark.parametrize("colors", [None, "vertex", "face"])
async def test_one_level_reads_as_a_mesh_memory_store(colors):
    """The same arrays in either store give the same result, whole and cut."""
    level = _level(10, colors=colors)
    multiscale = MultiscaleMeshStore(levels=[level])
    memory = MeshMemoryStore(
        positions=level.positions,
        indices=level.indices,
        colors=level.colors,
        colors_layout=level.colors_layout,
    )
    slab = _whole()._replace(region=data_region(3, {0: (16.0, 4.0)}))

    for request in (_whole(), slab, _cut(), _cut(z=20.5)):
        ours = await multiscale.get_data(request)
        theirs = await memory.get_data(request)
        assert type(ours) is type(theirs)
        if isinstance(ours, MeshSectionData):
            fields = (
                "fill_positions",
                "fill_indices",
                "fill_colors",
                "fill_face_ids",
                "outline_positions",
                "outline_colors",
                "outline_face_ids",
            )
        else:
            fields = (
                "positions",
                "indices",
                "normals",
                "colors",
                "original_face_indices",
            )
        for name in fields:
            a, b = getattr(ours, name), getattr(theirs, name)
            if a is None or b is None:
                assert a is None and b is None, name
            else:
                np.testing.assert_array_equal(a, b, err_msg=name)
        assert ours.color_mode == theirs.color_mode
        assert ours.is_empty == theirs.is_empty


async def test_each_level_has_its_own_cache():
    store = _store(12, 4)
    await store.get_data(_cut(level=0))

    assert store.level_cache(0).peek(("closure",)) is not None
    assert store.level_cache(1).peek(("closure",)) is None
    assert store.level_cache(0) is not store.level_cache(1)


async def test_replacing_the_levels_announces_a_change_and_drops_the_caches():
    store = _store(12, 4)
    await store.get_data(_cut(level=0))
    seen = []
    store.data_changed.connect(seen.append)
    before = store.axis_extents

    store.levels = [_level(8, radius=5.0), _level(4, radius=5.0)]

    assert [event.kind for event in seen] == ["extent"]
    assert store.level_cache(0).peek(("closure",)) is None
    assert store.axis_extents != before


def test_a_store_round_trips_through_its_dump():
    store = _store(6, 4, name="cell")
    again = MultiscaleMeshStore.model_validate(store.model_dump())

    assert again.store_type == "mesh_multiscale"
    assert again.id == store.id
    for ours, theirs in zip(again.levels, store.levels):
        np.testing.assert_array_equal(ours.positions, theirs.positions)
        np.testing.assert_array_equal(ours.indices, theirs.indices)


# -- the LOD settings ---------------------------------------------------------


def test_the_coarse_level_is_one_based_and_never_the_finest():
    with pytest.raises(ValidationError):
        GeometryLodConfig(coarse_level=1)
    assert GeometryLodConfig().coarse_scale_index(4) == 3
    assert GeometryLodConfig(coarse_level=2).coarse_scale_index(4) == 1
    assert GeometryLodConfig(coarse_level=4).coarse_scale_index(4) == 3


def test_a_coarse_level_the_store_lacks_is_refused():
    with pytest.raises(ValueError, match="has 3 levels"):
        GeometryLodConfig(coarse_level=4).coarse_scale_index(3)
    with pytest.raises(ValueError, match=r"1 level\(s\)"):
        GeometryLodConfig(coarse_level=2).coarse_scale_index(1)


def test_one_level_has_no_coarse_level():
    assert GeometryLodConfig().coarse_scale_index(1) is None


def test_the_lod_settings_are_frozen():
    config = GeometryLodConfig()
    assert (config.dims_drag, config.dims_drag_draw, config.camera_motion) == (
        "coarse",
        "coarse",
        "coarse",
    )
    with pytest.raises(ValidationError):
        config.dims_drag = "full"
