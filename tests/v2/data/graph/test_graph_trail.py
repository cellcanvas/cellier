"""Asymmetric trail windows and the fade alpha (D11, D12, D13)."""

from __future__ import annotations

import numpy as np

from cellier.data.graph import GraphMemoryStore


def _line_graph(n: int = 12) -> GraphMemoryStore:
    """A 4-D tzyx chain whose axis-0 coordinate is the row index."""
    positions = np.zeros((n, 4), dtype=np.float32)
    positions[:, 0] = np.arange(n)
    positions[:, 1] = 5.0
    positions[:, 2] = np.arange(n)
    positions[:, 3] = 0.0
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int32)
    return GraphMemoryStore.from_arrays(positions, edges)


async def test_trail_window_asymmetric(make_request):
    """before=3, after=1 admits exactly [t-3, t+1] and is not symmetric."""
    store = _line_graph()
    data = await store.get_data(
        make_request(displayed=(2, 3), sliced={0: 6, 1: 5}, extents={0: (3.0, 1.0)})
    )
    assert np.array_equal(data.original_node_rows, np.array([3, 4, 5, 6, 7]))


async def test_trail_window_default_is_the_points_slab(make_request):
    """A sliced axis with no extent entry gets the (0.5, 0.5) slab."""
    store = _line_graph()
    data = await store.get_data(make_request(displayed=(2, 3), sliced={0: 6, 1: 5}))
    assert np.array_equal(data.original_node_rows, np.array([6]))


async def test_trail_fade_alpha(make_request):
    """Alpha is 1.0 at the index, clamps to min_alpha, and is 0.0 outside (D13)."""
    store = _line_graph()
    data = await store.get_data(
        make_request(
            displayed=(2, 3),
            sliced={0: 6, 1: 5},
            extents={0: (3.0, 0.0)},
            fades={0: (3.0, 3.0, 0.2)},
        )
    )
    rows = data.original_node_rows
    alpha = dict(zip(rows.tolist(), data.node_alpha.tolist()))
    assert alpha[6] == 1.0
    # 1 - 1/3 and 1 - 2/3, both above the 0.2 floor.
    assert np.isclose(alpha[5], 2 / 3)
    assert np.isclose(alpha[4], 1 / 3)
    # At the window edge the ramp reaches 0 but is clamped up to min_alpha.
    assert np.isclose(alpha[3], 0.2)

    # The dangling endpoint one step past the window is exactly 0.0.
    vertex_rows = data.edge_endpoint_rows.reshape(-1)
    out_of_window = data.edge_alpha[vertex_rows == 7]
    assert out_of_window.size and np.all(out_of_window == 0.0)


async def test_trail_fade_min_alpha_zero_reaches_zero(make_request):
    store = _line_graph()
    data = await store.get_data(
        make_request(
            displayed=(2, 3),
            sliced={0: 6, 1: 5},
            extents={0: (2.0, 0.0)},
            fades={0: (2.0, 2.0, 0.0)},
        )
    )
    alpha = dict(zip(data.original_node_rows.tolist(), data.node_alpha.tolist()))
    assert alpha[4] == 0.0


async def test_fade_before_and_after_are_independent(make_request):
    """A short forward falloff fades faster than a long backward one."""
    store = _line_graph()
    data = await store.get_data(
        make_request(
            displayed=(2, 3),
            sliced={0: 6, 1: 5},
            extents={0: (4.0, 4.0)},
            fades={0: (4.0, 1.0, 0.0)},
        )
    )
    alpha = dict(zip(data.original_node_rows.tolist(), data.node_alpha.tolist()))
    assert np.isclose(alpha[5], 0.75)  # one step back, falloff 4
    assert np.isclose(alpha[7], 0.0)  # one step forward, falloff 1


async def test_multiple_trail_axes_multiply(make_request):
    """Alphas multiply across axes."""
    positions = np.array([[6.0, 5.0, 0.0, 0.0], [5.0, 4.0, 1.0, 1.0]], dtype=np.float32)
    store = GraphMemoryStore.from_arrays(positions, np.zeros((0, 2), dtype=np.int32))

    data = await store.get_data(
        make_request(
            displayed=(2, 3),
            sliced={0: 6, 1: 5},
            extents={0: (2.0, 2.0), 1: (2.0, 2.0)},
            fades={0: (2.0, 2.0, 0.0), 1: (2.0, 2.0, 0.0)},
        )
    )
    alpha = dict(zip(data.original_node_rows.tolist(), data.node_alpha.tolist()))
    assert alpha[0] == 1.0
    # One step off on each axis: 0.5 * 0.5.
    assert np.isclose(alpha[1], 0.25)


async def test_no_fade_leaves_alpha_none(make_request):
    """An extent without a fade entry produces no alpha buffers."""
    store = _line_graph()
    data = await store.get_data(
        make_request(displayed=(2, 3), sliced={0: 6, 1: 5}, extents={0: (3.0, 1.0)})
    )
    assert data.node_alpha is None
    assert data.edge_alpha is None


async def test_edge_alpha_is_row_aligned_with_positions(make_request):
    """edge_alpha[i] belongs to edge_positions[i], by construction."""
    store = _line_graph()
    data = await store.get_data(
        make_request(
            displayed=(2, 3),
            sliced={0: 6, 1: 5},
            extents={0: (2.0, 0.0)},
            fades={0: (2.0, 2.0, 0.2)},
        )
    )
    assert data.edge_alpha.shape[0] == data.edge_positions.shape[0]
    vertex_rows = data.edge_endpoint_rows.reshape(-1)
    for i, row in enumerate(vertex_rows.tolist()):
        # min_alpha floors every in-window vertex above zero; out-of-window
        # vertices -- the dangling ends of D13 -- are forced to exactly 0.0.
        in_window = 4 <= row <= 6
        assert (data.edge_alpha[i] > 0.0) == in_window
