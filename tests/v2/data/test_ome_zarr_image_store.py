"""Tests for OMEZarrImageDataStore."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pytest

if TYPE_CHECKING:
    import pathlib

from cellier.data.image._image_requests import ChunkRequest
from cellier.data.image._ome_zarr_image_store import OMEZarrImageDataStore

# ---------------------------------------------------------------------------
# Synthetic OME-Zarr v0.5 fixture
# ---------------------------------------------------------------------------

# 5D: t, c, z, y, x — 3 pyramid levels with anisotropic z.
# Level 0: (2,1,16,32,32) s=[1,1,2,1,1] t=[.5,0,.5,.5,.5]
# Level 1: (2,1, 8,16,16) s=[1,1,4,2,2] t=[.5,0,1.5,1,1]
# Level 2: (2,1, 4, 8, 8) s=[1,1,8,4,4] t=[.5,0,3.5,3,3]

_LEVEL_SHAPES = [
    (2, 1, 16, 32, 32),
    (2, 1, 8, 16, 16),
    (2, 1, 4, 8, 8),
]

_AXES = [
    {"name": "t", "type": "time", "unit": "second"},
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]

_DATASETS = [
    {
        "path": "0",
        "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 1.0, 2.0, 1.0, 1.0]},
            {"type": "translation", "translation": [0.5, 0.0, 0.5, 0.5, 0.5]},
        ],
    },
    {
        "path": "1",
        "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 1.0, 4.0, 2.0, 2.0]},
            {"type": "translation", "translation": [0.5, 0.0, 1.5, 1.0, 1.0]},
        ],
    },
    {
        "path": "2",
        "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 1.0, 8.0, 4.0, 4.0]},
            {"type": "translation", "translation": [0.5, 0.0, 3.5, 3.0, 3.0]},
        ],
    },
]


def _write_zarr3_array(path: pathlib.Path, shape: tuple[int, ...]) -> None:
    """Write a minimal zarr v3 array with float32 data filled with 1s."""
    import zarr

    store = zarr.storage.LocalStore(str(path))
    arr = zarr.create(
        store=store,
        shape=shape,
        dtype="float32",
        chunks=tuple(min(s, 16) for s in shape),
        zarr_format=3,
    )
    arr[...] = np.ones(shape, dtype=np.float32)


def _write_synthetic_ome_zarr(root: pathlib.Path) -> None:
    """Build a minimal 5D OME-Zarr v0.5 store on disk."""
    root.mkdir(parents=True, exist_ok=True)

    # Root zarr.json (group metadata).
    root_meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "name": "test",
                        "axes": _AXES,
                        "datasets": _DATASETS,
                        "version": "0.5",
                    }
                ],
            }
        },
    }
    (root / "zarr.json").write_text(json.dumps(root_meta))

    # Write each level as a zarr v3 array.
    for ds, shape in zip(_DATASETS, _LEVEL_SHAPES):
        _write_zarr3_array(root / ds["path"], shape)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ome_zarr_5d(tmp_path: pathlib.Path) -> str:
    """Build a synthetic 5D OME-Zarr v0.5 store and return file:// URI."""
    store_path = tmp_path / "test.ome.zarr"
    _write_synthetic_ome_zarr(store_path)
    return f"file://{store_path}"


def _req(scale_index: int, *axis_selections) -> ChunkRequest:
    return ChunkRequest(
        chunk_request_id=uuid4(),
        slice_request_id=uuid4(),
        scale_index=scale_index,
        axis_selections=axis_selections,
    )


# ---------------------------------------------------------------------------
# Tests: from_path construction
# ---------------------------------------------------------------------------


def test_from_path_returns_store(ome_zarr_5d: str) -> None:
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    assert isinstance(store, OMEZarrImageDataStore)


def test_scale_names(ome_zarr_5d: str) -> None:
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    assert store.scale_names == ["0", "1", "2"]


def test_axis_names_all_axes(ome_zarr_5d: str) -> None:
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    assert store.data_coordinate_system.axis_names() == ("t", "c", "z", "y", "x")


def test_axis_types(ome_zarr_5d: str) -> None:
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    assert [axis.axis_type for axis in store.data_coordinate_system.axes] == [
        "time",
        "channel",
        "space",
        "space",
        "space",
    ]


def test_axis_units(ome_zarr_5d: str) -> None:
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    assert [axis.unit for axis in store.data_coordinate_system.axes] == [
        "second",
        None,
        "micrometer",
        "micrometer",
        "micrometer",
    ]


def test_ndim(ome_zarr_5d: str) -> None:
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    assert store.ndim == 5


# ---------------------------------------------------------------------------
# Tests: level_scales / level_translations
# ---------------------------------------------------------------------------


def test_level_geometry_level0_is_the_identity(ome_zarr_5d: str) -> None:
    """The store states the pyramid as numbers.  What makes them a
    transform is the pair of level coordinate systems, which
    ``install_level_transforms`` attaches once the axes are known."""
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    np.testing.assert_allclose(store.level_scales[0], [1.0] * 5, atol=1e-12)
    np.testing.assert_allclose(store.level_translations[0], [0.0] * 5, atol=1e-12)
    t0 = store.level_transforms[0]
    ndim = t0.input_ndim
    np.testing.assert_allclose(t0.matrix, np.eye(ndim + 1), atol=1e-12)


def test_level_geometry_level1(ome_zarr_5d: str) -> None:
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    scale = np.asarray(store.level_scales[1])
    trans = np.asarray(store.level_translations[1])
    # t=1, c=1, z=2, y=2, x=2
    np.testing.assert_allclose(scale, [1.0, 1.0, 2.0, 2.0, 2.0], atol=1e-12)
    # t=0, c=0, z=0.5, y=0.5, x=0.5
    np.testing.assert_allclose(trans, [0.0, 0.0, 0.5, 0.5, 0.5], atol=1e-12)


def test_level_geometry_level2(ome_zarr_5d: str) -> None:
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    scale = np.asarray(store.level_scales[2])
    trans = np.asarray(store.level_translations[2])
    # t=1, c=1, z=4, y=4, x=4
    np.testing.assert_allclose(scale, [1.0, 1.0, 4.0, 4.0, 4.0], atol=1e-12)
    # t=0, c=0, z=1.5, y=2.5, x=2.5
    # Translation: (tr_k - tr_0) / s0  for z: (3.5 - 0.5) / 2.0 = 1.5
    #                                   for y: (3.0 - 0.5) / 1.0 = 2.5
    #                                   for x: (3.0 - 0.5) / 1.0 = 2.5
    np.testing.assert_allclose(trans, [0.0, 0.0, 1.5, 2.5, 2.5], atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: level_shapes
# ---------------------------------------------------------------------------


def test_level_shapes_full_rank(ome_zarr_5d: str) -> None:
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    assert store.level_shapes[0] == (2, 1, 16, 32, 32)
    assert store.level_shapes[1] == (2, 1, 8, 16, 16)
    assert store.level_shapes[2] == (2, 1, 4, 8, 8)


def test_n_levels(ome_zarr_5d: str) -> None:
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    assert store.n_levels == 3


# ---------------------------------------------------------------------------
# Tests: error cases
# ---------------------------------------------------------------------------


def test_invalid_scheme_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported URI scheme"):
        OMEZarrImageDataStore.from_path("bad_scheme:///some/path")


def test_wrong_type_raises(tmp_path: pathlib.Path) -> None:
    """A Plate group should raise TypeError."""
    root = tmp_path / "plate.ome.zarr"
    root.mkdir()
    plate_meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "plate": {
                    "name": "test-plate",
                    "columns": [{"name": "0"}],
                    "rows": [{"name": "A"}],
                    "wells": [{"path": "A/0", "rowIndex": 0, "columnIndex": 0}],
                },
            }
        },
    }
    (root / "zarr.json").write_text(json.dumps(plate_meta))
    with pytest.raises(TypeError, match="Plates and other types are not supported"):
        OMEZarrImageDataStore.from_path(f"file://{root}")


# ---------------------------------------------------------------------------
# Tests: serialisation round-trip
# ---------------------------------------------------------------------------


def test_serialisation_roundtrip(ome_zarr_5d: str) -> None:
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    json_str = store.model_dump_json()
    restored = OMEZarrImageDataStore.model_validate_json(json_str)
    assert restored.zarr_path == store.zarr_path
    assert restored.scale_names == store.scale_names
    assert (
        restored.data_coordinate_system.axis_names()
        == store.data_coordinate_system.axis_names()
    )
    assert restored.n_levels == store.n_levels
    assert restored.level_shapes == store.level_shapes


# ---------------------------------------------------------------------------
# Tests: get_data
# ---------------------------------------------------------------------------


async def test_get_data_3d(ome_zarr_5d: str) -> None:
    """Read a full spatial block from level 0 at t=0, c=0."""
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    req = _req(0, 0, 0, (0, 16), (0, 32), (0, 32))
    result = await store.get_data(req)
    assert result.shape == (16, 32, 32)
    assert result.dtype == np.float32
    # Data was filled with 1s.
    np.testing.assert_array_equal(result, 1.0)


async def test_get_data_2d_slice(ome_zarr_5d: str) -> None:
    """Read a single z-slice from level 0 at t=0, c=0, z=5."""
    store = OMEZarrImageDataStore.from_path(ome_zarr_5d)
    req = _req(0, 0, 0, 5, (0, 32), (0, 32))
    result = await store.get_data(req)
    assert result.shape == (32, 32)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, 1.0)


# ---------------------------------------------------------------------------
# Bf2Raw (bioformats2raw) multi-series container support
# ---------------------------------------------------------------------------

# Simpler 3D layout for Bf2Raw child images.
_BF2RAW_LEVEL_SHAPES = [
    (8, 16, 16),
    (4, 8, 8),
]

_BF2RAW_AXES = [
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]

_BF2RAW_DATASETS = [
    {
        "path": "0",
        "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 1.0, 1.0]},
        ],
    },
    {
        "path": "1",
        "coordinateTransformations": [
            {"type": "scale", "scale": [2.0, 2.0, 2.0]},
            {"type": "translation", "translation": [0.5, 0.5, 0.5]},
        ],
    },
]


def _write_bf2raw_image_group(
    root: pathlib.Path,
    *,
    axes: list[dict] | None = None,
    datasets: list[dict] | None = None,
    level_shapes: list[tuple[int, ...]] | None = None,
    fill_value: float = 1.0,
) -> None:
    """Write a standard Image group as a child of a Bf2Raw container."""
    if axes is None:
        axes = _BF2RAW_AXES
    if datasets is None:
        datasets = _BF2RAW_DATASETS
    if level_shapes is None:
        level_shapes = _BF2RAW_LEVEL_SHAPES

    root.mkdir(parents=True, exist_ok=True)
    image_meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "name": "test",
                        "axes": axes,
                        "datasets": datasets,
                        "version": "0.5",
                    }
                ],
            }
        },
    }
    (root / "zarr.json").write_text(json.dumps(image_meta))

    for ds, shape in zip(datasets, level_shapes):
        _write_zarr3_array(root / ds["path"], shape)


def _write_synthetic_bf2raw(
    root: pathlib.Path,
    n_series: int = 2,
) -> None:
    """Build a minimal Bf2Raw container with *n_series* child image groups."""
    root.mkdir(parents=True, exist_ok=True)

    # Root zarr.json — Bf2Raw marker.
    bf2raw_meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "bioformats2raw.layout": 3,
            }
        },
    }
    (root / "zarr.json").write_text(json.dumps(bf2raw_meta))

    # OME subgroup with Series metadata.
    ome_dir = root / "OME"
    ome_dir.mkdir()
    series_paths = [str(i) for i in range(n_series)]
    series_meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "series": series_paths,
            }
        },
    }
    (ome_dir / "zarr.json").write_text(json.dumps(series_meta))

    # Write child image groups.
    for i in range(n_series):
        _write_bf2raw_image_group(
            root / str(i),
            fill_value=float(i + 1),
        )


@pytest.fixture
def bf2raw_store(tmp_path: pathlib.Path) -> str:
    """Build a synthetic Bf2Raw container and return file:// URI."""
    store_path = tmp_path / "bf2raw.ome.zarr"
    _write_synthetic_bf2raw(store_path, n_series=2)
    return f"file://{store_path}"


# ---------------------------------------------------------------------------
# Tests: Bf2Raw construction
# ---------------------------------------------------------------------------


def test_bf2raw_from_path_returns_store(bf2raw_store: str) -> None:
    store = OMEZarrImageDataStore.from_path(bf2raw_store)
    assert isinstance(store, OMEZarrImageDataStore)


def test_bf2raw_axis_names(bf2raw_store: str) -> None:
    store = OMEZarrImageDataStore.from_path(bf2raw_store)
    assert store.data_coordinate_system.axis_names() == ("z", "y", "x")


def test_bf2raw_level_shapes(bf2raw_store: str) -> None:
    store = OMEZarrImageDataStore.from_path(bf2raw_store)
    assert store.level_shapes[0] == (8, 16, 16)
    assert store.level_shapes[1] == (4, 8, 8)


def test_bf2raw_n_levels(bf2raw_store: str) -> None:
    store = OMEZarrImageDataStore.from_path(bf2raw_store)
    assert store.n_levels == 2


def test_bf2raw_zarr_path_points_to_child(bf2raw_store: str) -> None:
    """zarr_path should be resolved to the child image group."""
    store = OMEZarrImageDataStore.from_path(bf2raw_store)
    assert store.zarr_path.endswith("/0")


def test_bf2raw_series_index_selects_child(bf2raw_store: str) -> None:
    store = OMEZarrImageDataStore.from_path(bf2raw_store, series_index=1)
    assert store.zarr_path.endswith("/1")


def test_bf2raw_series_index_out_of_range(bf2raw_store: str) -> None:
    with pytest.raises(ValueError, match="series_index=5 out of range"):
        OMEZarrImageDataStore.from_path(bf2raw_store, series_index=5)


def test_bf2raw_level_transforms_identity_at_0(bf2raw_store: str) -> None:
    store = OMEZarrImageDataStore.from_path(bf2raw_store)
    t0 = store.level_transforms[0]
    np.testing.assert_allclose(t0.matrix, np.eye(t0.input_ndim + 1), atol=1e-12)


async def test_bf2raw_get_data(bf2raw_store: str) -> None:
    """Read data from the first series of a Bf2Raw container."""
    store = OMEZarrImageDataStore.from_path(bf2raw_store)
    req = _req(0, (0, 8), (0, 16), (0, 16))
    result = await store.get_data(req)
    assert result.shape == (8, 16, 16)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Bf2Raw without OME/Series metadata (numeric fallback)
# ---------------------------------------------------------------------------


def test_bf2raw_without_series_metadata(tmp_path: pathlib.Path) -> None:
    """Bf2Raw without an OME/zarr.json should fall back to numeric paths."""
    root = tmp_path / "noseries.ome.zarr"
    root.mkdir(parents=True)

    bf2raw_meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "bioformats2raw.layout": 3,
            }
        },
    }
    (root / "zarr.json").write_text(json.dumps(bf2raw_meta))

    # Write a single child image group at "0/" (no OME/ subgroup).
    _write_bf2raw_image_group(root / "0")

    store = OMEZarrImageDataStore.from_path(f"file://{root}")
    assert isinstance(store, OMEZarrImageDataStore)
    assert store.zarr_path.endswith("/0")
    assert store.n_levels == 2


def test_bf2raw_serialisation_roundtrip(bf2raw_store: str) -> None:
    store = OMEZarrImageDataStore.from_path(bf2raw_store)
    json_str = store.model_dump_json()
    restored = OMEZarrImageDataStore.model_validate_json(json_str)
    assert restored.zarr_path == store.zarr_path
    assert restored.scale_names == store.scale_names
    assert (
        restored.data_coordinate_system.axis_names()
        == store.data_coordinate_system.axis_names()
    )
    assert restored.n_levels == store.n_levels
    assert restored.level_shapes == store.level_shapes
