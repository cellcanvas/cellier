# tests/v2/data/points/test_points_memory_store.py
"""Tests for PointsMemoryStore construction and get_data."""

import asyncio
from uuid import uuid4

import numpy as np

from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.data.points._points_requests import PointsSliceRequest
from tests._v2 import data_region


def _simple_store() -> PointsMemoryStore:
    """10 points uniformly spread in a 10x10x10 cube."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 10, size=(10, 3)).astype(np.float32)
    return PointsMemoryStore(positions=positions, name="test")


def _req(
    displayed: tuple[int, ...] = (0, 1, 2),
    sliced: dict[int, float] | None = None,
    thickness: float = 0.5,
) -> PointsSliceRequest:
    """One request, with the filter as the region the store now takes.

    ``sliced`` maps axis to a **data**-space position; each becomes a slab of
    half-width *thickness*, which is exactly what ``slice_indices`` plus
    ``thickness`` meant before Phase 8 deleted them.
    """
    if sliced is None:
        sliced = {}
    sid = uuid4()
    return PointsSliceRequest(
        slice_request_id=sid,
        chunk_request_id=sid,
        scale_index=0,
        displayed_axes=displayed,
        retained_axes=tuple(sorted(displayed)),
        region=data_region(
            3, {axis: (position, thickness) for axis, position in sliced.items()}
        ),
    )


# ── Construction ──────────────────────────────────────────────────────────────


def test_construction_defaults():
    store = _simple_store()
    assert store.n_points == 10
    assert store.ndim == 3
    assert store.colors is None
    assert store.color_mode == "uniform"


def test_colors_coerced_to_float32():
    positions = np.zeros((5, 3), dtype=np.float64)
    colors = np.ones((5, 4), dtype=np.float64)
    store = PointsMemoryStore(positions=positions, colors=colors)
    assert store.positions.dtype == np.float32
    assert store.colors.dtype == np.float32


def test_color_mode_vertex_when_colors_present():
    store = _simple_store()
    store.colors = np.ones((10, 4), dtype=np.float32)
    assert store.color_mode == "vertex"


# ── get_data — 3D (all axes displayed) ───────────────────────────────────────


def test_get_data_3d_returns_all_points():
    store = _simple_store()
    result = asyncio.run(store.get_data(_req(displayed=(0, 1, 2))))
    assert result.is_empty is False
    assert result.positions.shape == (10, 3)


def test_get_data_3d_positions_projected_to_3_columns():
    store = _simple_store()
    result = asyncio.run(store.get_data(_req(displayed=(0, 1, 2))))
    assert result.positions.shape[1] == 3


# ── get_data — 2D (proximity filter) ─────────────────────────────────────────


def test_get_data_2d_empty_slab():
    store = _simple_store()
    # Slice at z=1000, far outside the cube.
    result = asyncio.run(store.get_data(_req(displayed=(1, 2), sliced={0: 1000})))
    assert result.is_empty is True
    # Placeholder has shape (1, n_displayed_dims).
    assert result.positions.shape == (1, 2)


def test_get_data_2d_positions_projected():
    store = _simple_store()
    # All 10 points have z in [0, 10].  Slice at z=5, thickness=5 → all survive.
    result = asyncio.run(
        store.get_data(_req(displayed=(1, 2), sliced={0: 5}, thickness=5.0))
    )
    assert not result.is_empty
    # Projected positions must have only 2 columns (y, x).
    assert result.positions.shape[1] == 2


def test_get_data_2d_tight_filter():
    """Points outside the slab must be excluded."""
    # Place two points at z=0 and z=10.
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    store = PointsMemoryStore(positions=positions)
    result = asyncio.run(
        store.get_data(_req(displayed=(1, 2), sliced={0: 0}, thickness=0.5))
    )
    assert not result.is_empty
    assert result.positions.shape[0] == 1  # only the z=0 point


def test_get_data_2d_colors_gathered():
    positions = np.zeros((5, 3), dtype=np.float32)
    colors = np.eye(5, 4, dtype=np.float32)
    store = PointsMemoryStore(positions=positions, colors=colors)
    # All 5 points are at z=0; slice with thick slab → all survive.
    result = asyncio.run(
        store.get_data(_req(displayed=(1, 2), sliced={0: 0}, thickness=0.5))
    )
    assert not result.is_empty
    assert result.colors is not None
    assert result.colors.shape[0] == result.positions.shape[0]


# ── original_indices (pick index mapping) ─────────────────────────────────────


def test_original_indices_identity_in_3d():
    """A full 3-D view applies no filter, so the map is the identity arange."""
    positions = np.zeros((5, 3), dtype=np.float32)
    store = PointsMemoryStore(positions=positions)
    result = asyncio.run(store.get_data(_req()))
    assert list(result.original_indices) == list(range(5))


def test_original_indices_track_surviving_subset_in_2d():
    """A 2-D slice keeping one point reports that point's original index.

    This is the invariant element-level picking depends on: pygfx reports the
    row in the rendered subset, and ``original_indices`` maps it back so the
    correct point (here original index 3) is identified — not row 0.
    """
    positions = np.array([[z, 1.0, 1.0] for z in (0, 10, 20, 30, 40)], dtype=np.float32)
    store = PointsMemoryStore(positions=positions)
    result = asyncio.run(
        store.get_data(_req(displayed=(1, 2), sliced={0: 30}, thickness=0.5))
    )
    assert result.positions.shape[0] == 1
    assert list(result.original_indices) == [3]


# ── Checkpoint cancellation ───────────────────────────────────────────────────


def test_get_data_cancellable():
    """CancelledError fires at checkpoint A before the gather begins."""
    n = 200_000
    rng = np.random.default_rng(1)
    positions = rng.uniform(0, 100, size=(n, 3)).astype(np.float32)
    store = PointsMemoryStore(positions=positions)
    req = _req(displayed=(1, 2), sliced={0: 50})

    async def _run():
        task = asyncio.create_task(store.get_data(req))
        task.cancel()
        try:
            await task
            return "completed"
        except asyncio.CancelledError:
            return "cancelled"

    result = asyncio.run(_run())
    assert result == "cancelled"
