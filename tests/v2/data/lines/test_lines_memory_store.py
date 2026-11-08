"""Tests for LinesMemoryStore construction and get_data."""

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.lines._lines_requests import LinesSliceRequest
from tests._v2 import data_region


def _simple_store() -> LinesMemoryStore:
    """Four segments connecting the corners of a unit cube diagonal."""
    positions = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],  # segment 0
            [0, 0, 1],
            [1, 1, 0],  # segment 1
            [0, 1, 0],
            [1, 0, 1],  # segment 2
            [0, 1, 1],
            [1, 0, 0],  # segment 3
        ],
        dtype=np.float32,
    )
    return LinesMemoryStore(positions=positions, name="test")


def _req(
    displayed: tuple[int, ...] = (0, 1, 2),
    sliced: dict[int, float] | None = None,
    thickness: float = 0.5,
) -> LinesSliceRequest:
    """One request, with the filter as the region the store now takes.

    ``sliced`` maps axis to a **data**-space position; each becomes a slab of
    half-width *thickness*.
    """
    if sliced is None:
        sliced = {}
    sid = uuid4()
    return LinesSliceRequest(
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
    assert store.n_segments == 4
    assert store.ndim == 3
    assert store.colors is None
    assert store.color_mode == "uniform"


def test_odd_row_count_raises():
    with pytest.raises(ValueError, match="even"):
        LinesMemoryStore(positions=np.zeros((3, 3), dtype=np.float32))


def test_positions_coerced_to_float32():
    positions = np.zeros((4, 3), dtype=np.float64)
    store = LinesMemoryStore(positions=positions)
    assert store.positions.dtype == np.float32


def test_color_mode_vertex_when_colors_present():
    store = _simple_store()
    store.colors = np.ones((8, 4), dtype=np.float32)
    assert store.color_mode == "vertex"


# ── get_data — 3D (all axes displayed) ───────────────────────────────────────


def test_get_data_3d_returns_all_vertices():
    store = _simple_store()
    result = asyncio.run(store.get_data(_req(displayed=(0, 1, 2))))
    assert result.is_empty is False
    assert result.positions.shape == (8, 3)


def test_get_data_3d_projected_to_3_columns():
    store = _simple_store()
    result = asyncio.run(store.get_data(_req(displayed=(0, 1, 2))))
    assert result.positions.shape[1] == 3


# ── get_data — 2D (both-endpoint slab filter) ────────────────────────────────


def test_get_data_2d_empty_slab():
    store = _simple_store()
    # Slice far outside all data.
    result = asyncio.run(store.get_data(_req(displayed=(1, 2), sliced={0: 1000})))
    assert result.is_empty is True
    # Placeholder has 2 vertices in (n_displayed=2) columns.
    assert result.positions.shape == (2, 2)


def test_get_data_2d_projected_to_2_columns():
    store = _simple_store()
    result = asyncio.run(store.get_data(_req(displayed=(1, 2), sliced={0: 0})))
    if not result.is_empty:
        assert result.positions.shape[1] == 2


def test_get_data_2d_both_endpoints_required():
    """A segment that straddles the slice plane must be excluded."""
    # One segment: z=0 to z=10.  Slice at z=0, thickness=0.5 → end at z=10 fails.
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    store = LinesMemoryStore(positions=positions)
    result = asyncio.run(
        store.get_data(_req(displayed=(1, 2), sliced={0: 0}, thickness=0.5))
    )
    assert result.is_empty is True


def test_get_data_2d_both_endpoints_in_slab():
    """A segment with both endpoints in the slab must be included."""
    # One segment: both vertices at z=0.
    positions = np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]], dtype=np.float32)
    store = LinesMemoryStore(positions=positions)
    result = asyncio.run(
        store.get_data(_req(displayed=(1, 2), sliced={0: 0}, thickness=0.5))
    )
    assert not result.is_empty
    # One segment → 2 vertices.
    assert result.positions.shape[0] == 2


def test_get_data_2d_result_has_even_vertex_count():
    """Surviving vertex count must always be even (complete segment pairs)."""
    store = _simple_store()
    result = asyncio.run(
        store.get_data(_req(displayed=(1, 2), sliced={0: 0}, thickness=0.6))
    )
    assert result.positions.shape[0] % 2 == 0


def test_get_data_2d_colors_gathered():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],  # segment 0: both at z=0
            [5.0, 0.0, 0.0],
            [5.0, 1.0, 1.0],  # segment 1: both at z=5
        ],
        dtype=np.float32,
    )
    colors = np.eye(4, dtype=np.float32)
    store = LinesMemoryStore(positions=positions, colors=colors)
    result = asyncio.run(
        store.get_data(_req(displayed=(1, 2), sliced={0: 0}, thickness=0.5))
    )
    assert not result.is_empty
    assert result.colors is not None
    assert result.colors.shape[0] == result.positions.shape[0]


# ── original_edge_indices (pick index mapping) ────────────────────────────────


def test_original_edge_indices_identity_in_3d():
    """A full 3-D view keeps every segment, so the map is the identity arange."""
    positions = np.zeros((8, 3), dtype=np.float32)  # 4 segments
    store = LinesMemoryStore(positions=positions)
    result = asyncio.run(store.get_data(_req()))
    assert list(result.original_edge_indices) == [0, 1, 2, 3]


def test_original_edge_indices_track_surviving_subset_in_2d():
    """A 2-D slab keeping one segment reports that segment's original index.

    The pick reports the rendered edge index (here 0, the only survivor);
    ``original_edge_indices`` maps it back to original segment 2.
    """
    positions = np.empty((8, 3), dtype=np.float32)
    for k, z in enumerate((0, 10, 20, 30)):
        positions[2 * k] = (z, 0.0, 0.0)
        positions[2 * k + 1] = (z, 1.0, 1.0)
    store = LinesMemoryStore(positions=positions)
    result = asyncio.run(
        store.get_data(_req(displayed=(1, 2), sliced={0: 20}, thickness=0.5))
    )
    assert result.positions.shape[0] == 2  # one segment (two vertices)
    assert list(result.original_edge_indices) == [2]


# ── Checkpoint cancellation ───────────────────────────────────────────────────


def test_get_data_cancellable():
    """CancelledError fires at checkpoint A before the gather begins."""
    n_segments = 200_000
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 100, size=(n_segments * 2, 3)).astype(np.float32)
    store = LinesMemoryStore(positions=positions)
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
