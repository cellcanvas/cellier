"""Shared store fixtures for the cellier.convenience tests.

In-memory stores (image/labels/mesh/points/lines/multichannel) plus a few
on-disk multiscale zarr stores used by the ``add_*_multiscale`` convenience
methods. The single-channel float32 multiscale image reuses the shared
``small_zarr_store`` fixture from ``tests/conftest.py``; the label and
multichannel multiscale stores are written inline here (int32 / 4-D CZYX).
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorstore as ts

from cellier.data.graph._graph_memory_store import GraphMemoryStore
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------


@pytest.fixture
def image_store() -> ImageMemoryStore:
    """An (8, 16, 24) float32 image volume."""
    data = np.zeros((8, 16, 24), dtype=np.float32)
    return ImageMemoryStore(data=data, name="image")


@pytest.fixture
def labels_store() -> LabelMemoryStore:
    """An (8, 16, 16) int32 label volume with one labelled block."""
    data = np.zeros((8, 16, 16), dtype=np.int32)
    data[:, 4:10, 4:10] = 5
    return LabelMemoryStore(data=data, name="labels")


@pytest.fixture
def mesh_store() -> MeshMemoryStore:
    """A single tetrahedron mesh."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return MeshMemoryStore(positions=positions, indices=indices)


@pytest.fixture
def points_store() -> PointsMemoryStore:
    """Four 3-D points."""
    positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
    return PointsMemoryStore(positions=positions)


@pytest.fixture
def graph_store() -> GraphMemoryStore:
    """Four 3-D nodes in a chain."""
    positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    return GraphMemoryStore.from_arrays(positions, edges, name="graph")


@pytest.fixture
def lines_store() -> LinesMemoryStore:
    """Two 3-D line segments (positions are vertex pairs)."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    return LinesMemoryStore(positions=positions)


@pytest.fixture
def multichannel_store() -> ImageMemoryStore:
    """A (3, 2, 16, 16) float32 ZCYX volume (channel axis = 1)."""
    data = np.random.default_rng(0).random((3, 2, 16, 16)).astype(np.float32)
    return ImageMemoryStore(data=data, name="multichannel")


# ---------------------------------------------------------------------------
# On-disk multiscale stores
# ---------------------------------------------------------------------------


def _write_multiscale_zarr(root, levels, fill, dtype: str = "float32") -> None:
    """Write per-level zarr v3 arrays under *root*, each filled via *fill*."""
    for name, shape in levels:
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(root / name)},
            "metadata": {
                "shape": list(shape),
                "data_type": dtype,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [max(1, s // 2) for s in shape]},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        store = ts.open(spec).result()
        arr = np.zeros(shape, dtype=dtype)
        fill(arr)
        store[...].write(arr).result()


@pytest.fixture
def multiscale_image_store(small_zarr_store) -> MultiscaleZarrDataStore:
    """A 2-level float32 multiscale image built on the shared zarr pyramid."""
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(small_zarr_store),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        name="multiscale_image",
    )


@pytest.fixture
def multiscale_labels_store(tmp_path) -> MultiscaleZarrDataStore:
    """A 2-level (s0=16^3, s1=8^3) int32 multiscale label store."""

    def _fill(arr: np.ndarray) -> None:
        _d, h, w = arr.shape
        arr[:, h // 4 : h // 2, w // 4 : w // 2] = 3

    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (16, 16, 16)), ("s1", (8, 8, 8))],
        fill=_fill,
        dtype="int32",
    )
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        name="multiscale_labels",
    )


@pytest.fixture
def multichannel_multiscale_store(tmp_path) -> MultiscaleZarrDataStore:
    """A 2-level 4-D CZYX float32 multiscale store (2 channels, axis 0)."""

    def _fill(arr: np.ndarray) -> None:
        _c, _d, _h, w = arr.shape
        arr[0, :, :, : w // 2] = 1.0
        arr[1, :, :, w // 2 :] = 1.0

    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (2, 16, 16, 16)), ("s1", (2, 8, 8, 8))],
        fill=_fill,
    )
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0, 1.0), (1.0, 2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0, 0.0), (0.0, 0.5, 0.5, 0.5)],
        name="multichannel_multiscale",
    )
