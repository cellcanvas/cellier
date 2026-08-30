"""The mask slice path: node window, D5's either-endpoint rule, gathers."""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data.graph import GraphMemoryStore
from cellier.data.points import PointsMemoryStore
from cellier.data.points._points_requests import PointsSliceRequest


async def test_slab_equivalence_with_points_store(make_request):
    """A trail-free graph selects the same node set as the points store.

    The regression anchor for D5's generalization of the scalar
    ``thickness`` into a per-axis ``(before, after)`` extent: with no trail
    configured the extents default to (0.5, 0.5), which is exactly the slab
    points and lines use.
    """
    rng = np.random.default_rng(0)
    positions = (rng.random((200, 3)) * 10).astype(np.float32)

    graph = GraphMemoryStore.from_arrays(positions, np.zeros((0, 2), dtype=np.int32))
    points = PointsMemoryStore(positions=positions)

    request = make_request(displayed=(1, 2), sliced={0: 5})
    graph_data = await graph.get_data(request)

    points_request = PointsSliceRequest(
        slice_request_id=request.slice_request_id,
        chunk_request_id=request.chunk_request_id,
        scale_index=0,
        displayed_axes=(1, 2),
        slice_indices={0: 5},
        thickness=0.5,
    )
    points_data = await points.get_data(points_request)

    assert np.array_equal(graph_data.original_node_rows, points_data.original_indices)
    assert np.allclose(graph_data.node_positions, points_data.positions)


async def test_either_endpoint_rule(make_request):
    """One endpoint in the slab renders the edge; neither does not (D5)."""
    positions = np.array(
        [[5.0, 0, 0], [9.0, 1, 1], [20.0, 2, 2], [21.0, 3, 3]], dtype=np.float32
    )
    edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
    store = GraphMemoryStore.from_arrays(positions, edges)

    data = await store.get_data(make_request(displayed=(1, 2), sliced={0: 5}))
    assert np.array_equal(data.original_edge_rows, np.array([0]))
    assert not data.edges_empty
    # The far-away edge contributes nothing.
    assert data.edge_positions.shape[0] == 2


async def test_far_endpoint_draws_no_node(make_request):
    """A dangling endpoint is a line vertex, never a node marker (D6)."""
    positions = np.array([[5.0, 0, 0], [9.0, 1, 1]], dtype=np.float32)
    store = GraphMemoryStore.from_arrays(positions, np.array([[0, 1]], dtype=np.int32))

    data = await store.get_data(make_request(displayed=(1, 2), sliced={0: 5}))
    # Node 1 is out of the slab: one node row, but two edge vertices.
    assert np.array_equal(data.original_node_rows, np.array([0]))
    assert data.node_positions.shape[0] == 1
    assert data.edge_positions.shape[0] == 2
    # The second edge vertex sits at node 1's projected position.
    assert np.allclose(data.edge_positions[1], positions[1, [1, 2]])


async def test_tracking_case_renders_edges(make_request, tracking_lineage):
    """A tzyx lineage sliced on t yields edges. Fails under both-endpoints."""
    positions, edges = tracking_lineage
    store = GraphMemoryStore.from_arrays(positions, edges)

    data = await store.get_data(make_request(displayed=(2, 3), sliced={0: 5, 1: 12}))
    assert data.original_edge_rows.shape[0] > 0
    assert not data.edges_empty


async def test_vertex_pair_layout(make_request):
    """edges[rows].reshape(-1) is the interleaved layout the material wants."""
    positions = np.array(
        [[0.0, 1, 2], [0.0, 3, 4], [0.0, 5, 6], [0.0, 7, 8]], dtype=np.float32
    )
    edges = np.array([[0, 2], [1, 3]], dtype=np.int32)
    store = GraphMemoryStore.from_arrays(positions, edges)

    data = await store.get_data(make_request(displayed=(1, 2), sliced={0: 0}))
    expected = positions[np.array([0, 2, 1, 3])][:, [1, 2]]
    assert np.allclose(data.edge_positions, expected)
    assert np.array_equal(data.edge_endpoint_rows, edges)


async def test_empty_slice_placeholder(make_request):
    """Placeholder geometry and empty flags, independently for each sub-node."""
    positions = np.array([[0.0, 1, 2], [0.0, 3, 4]], dtype=np.float32)
    store = GraphMemoryStore.from_arrays(positions, np.array([[0, 1]], dtype=np.int32))

    data = await store.get_data(make_request(displayed=(1, 2), sliced={0: 50}))
    assert data.nodes_empty and data.edges_empty
    assert data.node_positions.shape == (1, 2)
    assert data.edge_positions.shape == (2, 2)
    assert data.original_node_rows is None
    assert data.original_edge_rows is None


async def test_isolated_nodes_give_empty_edges_only(make_request):
    """Nodes present with zero edges is common; the flags are independent."""
    positions = np.array([[0.0, 1, 2], [0.0, 3, 4]], dtype=np.float32)
    store = GraphMemoryStore.from_arrays(positions, np.zeros((0, 2), dtype=np.int32))

    data = await store.get_data(make_request(displayed=(1, 2), sliced={0: 0}))
    assert not data.nodes_empty
    assert data.edges_empty


async def test_per_edge_colors_expand_to_two_vertices(make_request):
    """A store's per-edge RGBA becomes per-vertex at slice time."""
    positions = np.array(
        [[0.0, 1, 2], [0.0, 3, 4], [0.0, 5, 6], [0.0, 7, 8]], dtype=np.float32
    )
    edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
    edge_colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1]], dtype=np.float32)
    store = GraphMemoryStore.from_arrays(positions, edges, edge_colors=edge_colors)

    data = await store.get_data(make_request(displayed=(1, 2), sliced={0: 0}))
    assert data.edge_colors.shape == (4, 4)
    assert np.allclose(data.edge_colors, np.repeat(edge_colors, 2, axis=0))
    assert data.edge_color_mode == "vertex"


async def test_node_colors_and_sizes_are_gathered(make_request):
    positions = np.array([[0.0, 1, 2], [5.0, 3, 4]], dtype=np.float32)
    colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1]], dtype=np.float32)
    sizes = np.array([2.0, 8.0], dtype=np.float32)
    store = GraphMemoryStore.from_arrays(
        positions,
        np.zeros((0, 2), dtype=np.int32),
        node_colors=colors,
        node_sizes=sizes,
    )

    data = await store.get_data(make_request(displayed=(1, 2), sliced={0: 5}))
    assert np.allclose(data.node_colors, colors[[1]])
    assert np.allclose(data.node_sizes, sizes[[1]])
    assert data.node_color_mode == "vertex"
    assert data.node_size_mode == "vertex"


async def test_full_3d_view_keeps_everything(make_request, tracking_lineage):
    """With no sliced axes every node and edge survives."""
    positions, edges = tracking_lineage
    store = GraphMemoryStore.from_arrays(positions, edges)

    data = await store.get_data(make_request(displayed=(1, 2, 3), sliced={}))
    assert data.node_positions.shape[0] == positions.shape[0]
    assert data.edge_positions.shape[0] == 2 * edges.shape[0]


@pytest.mark.parametrize("strategy", ["mask"])
async def test_roi_strategy_is_not_inferred(make_request, strategy):
    """The store never switches strategy on its own (D17)."""
    store = GraphMemoryStore.from_arrays(
        np.zeros((4, 3), dtype=np.float32),
        np.array([[0, 1]], dtype=np.int32),
        slice_strategy=strategy,
    )
    await store.get_data(make_request(displayed=(1, 2), sliced={0: 0}))
    assert store.slice_strategy == strategy
    assert store._graph is None
