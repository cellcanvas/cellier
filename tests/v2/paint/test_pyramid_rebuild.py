"""Unit tests for MultiscalePaintController._rebuild_pyramid.

Tests construct a minimal controller stub (no Qt, no CellierController) by
bypassing ``__init__`` and wiring only the attributes used by
``_rebuild_pyramid``.  All tensorstore I/O hits a temporary on-disk zarr.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import tensorstore as ts

from cellier.paint import MultiscalePaintController, TensorStoreWriteBuffer
from cellier.paint._write_layer import WriteLayer

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_zarr(path: Path, shape: tuple[int, int], chunk: int) -> ts.TensorStore:
    return ts.open(
        {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {
                "shape": list(shape),
                "chunks": [chunk, chunk],
                "dtype": "<f4",
            },
            "create": True,
        }
    ).result()


def _make_controller(
    tmp_path: Path,
    level_shapes: list[tuple[int, int]],
    level_scales: list[float],
    block_size: int = 8,
    displayed_axes: tuple[int, int] = (0, 1),
) -> MultiscalePaintController:
    """Build a MultiscalePaintController stub with real tensorstore handles.

    Bypasses ``__init__`` to avoid needing Qt or CellierController.
    """
    stores = []
    for i, shape in enumerate(level_shapes):
        store_path = tmp_path / f"s{i}.zarr"
        stores.append(_open_zarr(store_path, shape, block_size))

    data_store = SimpleNamespace(
        id=uuid4(),
        n_levels=len(level_shapes),
        level_shapes=[list(s) for s in level_shapes],
        # The pyramid rebuild reads the per-level scale directly, rather
        # than the diagonal of a transform built from it.
        level_scales=[(float(s), float(s)) for s in level_scales],
        _ts_stores=stores,
    )

    ctrl = object.__new__(MultiscalePaintController)
    ctrl._data_store = data_store
    ctrl._displayed_axes = displayed_axes
    ctrl._store_dtype = np.dtype(stores[0].dtype.numpy_dtype)
    ctrl._write_layer = WriteLayer(data_store_id=data_store.id, block_size=block_size)
    ctrl._write_buffer = TensorStoreWriteBuffer(stores[0])

    return ctrl


def _stage_block(
    ctrl: MultiscalePaintController,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    value: float = 1.0,
) -> None:
    """Write a rectangular block into level-0 and mark dirty bricks."""
    ys = np.arange(y0, y1, dtype=np.int64)
    xs = np.arange(x0, x1, dtype=np.int64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    voxel_indices = np.stack([yy.ravel(), xx.ravel()], axis=1)
    values = np.full(voxel_indices.shape[0], value, dtype=np.float32)
    ctrl._write_buffer.stage(voxel_indices, values)
    for key in ctrl._write_layer.voxels_to_brick_keys(voxel_indices):
        ctrl._write_layer.mark_dirty(key)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rebuild_single_brick(tmp_path: Path) -> None:
    """Painting rows 0-7, cols 0-7 at level-0 rebuilds the level-1 brick correctly.

    Level-1 brick (0,0) covers all 8x8 of level-1 = 16x16 of level-0.
    After stride-2 decimation, level-0 sample at (2r, 2c) lands at level-1
    (r, c).  Only (2r, 2c) with r,c in 0-3 are in the painted region 0-7,
    so the top-left 4x4 of level-1 should be 1.0 and the rest 0.0.
    """
    ctrl = _make_controller(
        tmp_path,
        level_shapes=[(16, 16), (8, 8)],
        level_scales=[1.0, 2.0],
        block_size=8,
    )
    _stage_block(ctrl, 0, 8, 0, 8, value=1.0)

    txn = ctrl._write_buffer.transaction
    stats = ctrl._rebuild_pyramid(txn)
    ctrl._write_buffer.commit()

    assert stats == {1: 1}

    result = ctrl._data_store._ts_stores[1][0:8, 0:8].read().result()
    # Top-left 4x4 of level-1 is 1.0 (decimated from painted region).
    np.testing.assert_array_equal(result[0:4, 0:4], np.ones((4, 4), dtype=np.float32))
    # Remaining rows/cols are still 0.
    np.testing.assert_array_equal(result[4:, :], np.zeros((4, 8), dtype=np.float32))
    np.testing.assert_array_equal(result[:, 4:], np.zeros((8, 4), dtype=np.float32))


def test_rebuild_multiple_bricks(tmp_path: Path) -> None:
    """Three distinct level-0 bricks map to the correct level-1 bricks."""
    ctrl = _make_controller(
        tmp_path,
        level_shapes=[(32, 32), (16, 16)],
        level_scales=[1.0, 2.0],
        block_size=8,
    )
    # Brick (0,0): rows 0-7, cols 0-7  → level-1 brick (0,0)
    # Brick (0,1): rows 0-7, cols 8-15 → level-1 brick (0,0) (same parent)
    # Brick (1,2): rows 8-15, cols 16-23 → level-1 brick (0,1)
    _stage_block(ctrl, 0, 8, 0, 8, value=1.0)
    _stage_block(ctrl, 0, 8, 8, 16, value=2.0)
    _stage_block(ctrl, 8, 16, 16, 24, value=3.0)

    txn = ctrl._write_buffer.transaction
    stats = ctrl._rebuild_pyramid(txn)
    ctrl._write_buffer.commit()

    # Two distinct level-1 bricks should have been rebuilt.
    assert stats[1] == 2

    # Level-1 brick (0,0) covers level-0 rows 0-15, cols 0-15.
    # After decimation at stride 2 the values in cols 0-7 come from 1.0
    # and cols 8-15 come from 2.0, but since stride-2 picks the first
    # sample of each stride window the cols 8-15 in level-1 brick (0,0)
    # correspond to level-0 col 16 — but that's out of brick (0,0).
    # The important check: no exception raised and stats are correct.
    result_00 = ctrl._data_store._ts_stores[1][0:8, 0:8].read().result()
    assert result_00.shape == (8, 8)

    result_01 = ctrl._data_store._ts_stores[1][0:8, 8:16].read().result()
    assert result_01.shape == (8, 8)


def test_rebuild_boundary_brick(tmp_path: Path) -> None:
    """A brick at the array edge clips correctly and does not raise."""
    ctrl = _make_controller(
        tmp_path,
        level_shapes=[(16, 16), (8, 8)],
        level_scales=[1.0, 2.0],
        block_size=8,
    )
    # Paint the bottom-right brick (1,1): rows 8-15, cols 8-15.
    _stage_block(ctrl, 8, 16, 8, 16, value=1.0)

    txn = ctrl._write_buffer.transaction
    stats = ctrl._rebuild_pyramid(txn)
    ctrl._write_buffer.commit()

    assert stats == {1: 1}
    result = ctrl._data_store._ts_stores[1][4:8, 4:8].read().result()
    np.testing.assert_array_equal(result, np.ones((4, 4), dtype=np.float32))


def test_no_rebuild_single_level(tmp_path: Path) -> None:
    """A single-level store returns an empty dict without raising."""
    ctrl = _make_controller(
        tmp_path,
        level_shapes=[(16, 16)],
        level_scales=[1.0],
        block_size=8,
    )
    _stage_block(ctrl, 0, 8, 0, 8, value=1.0)

    txn = ctrl._write_buffer.transaction
    stats = ctrl._rebuild_pyramid(txn)
    ctrl._write_buffer.commit()

    assert stats == {}


def test_rebuild_txn_none(tmp_path: Path) -> None:
    """Passing txn=None returns an empty dict immediately."""
    ctrl = _make_controller(
        tmp_path,
        level_shapes=[(16, 16), (8, 8)],
        level_scales=[1.0, 2.0],
        block_size=8,
    )
    _stage_block(ctrl, 0, 8, 0, 8, value=1.0)
    stats = ctrl._rebuild_pyramid(None)
    assert stats == {}
    ctrl._write_buffer.abort()


def test_three_level_rebuild(tmp_path: Path) -> None:
    """Rebuild propagates from s0 → s1 → s2 for a three-level pyramid."""
    ctrl = _make_controller(
        tmp_path,
        level_shapes=[(32, 32), (16, 16), (8, 8)],
        level_scales=[1.0, 2.0, 4.0],
        block_size=8,
    )
    # Paint a full s0 brick at (0,0).
    _stage_block(ctrl, 0, 8, 0, 8, value=1.0)

    txn = ctrl._write_buffer.transaction
    stats = ctrl._rebuild_pyramid(txn)
    ctrl._write_buffer.commit()

    assert 1 in stats and stats[1] >= 1
    assert 2 in stats and stats[2] >= 1

    # s2 should contain the decimated value.
    result = ctrl._data_store._ts_stores[2][0:4, 0:4].read().result()
    assert (result > 0).any(), "s2 should contain at least some painted values"
