"""The two slice strategies must agree exactly (D17).

This is the highest-value module in the graph suite: ``mask`` is the
correctness reference, ``roi`` is the fast path, and the only thing that
makes the ROI path's ``query_edges_in_roi`` refinement trustworthy is that
the two are held to identical output on a matrix of shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data.graph import GraphMemoryStore

try:  # pragma: no cover - import probe
    import spatial_graph as _spatial_graph
except ImportError:  # pragma: no cover
    _spatial_graph = None

pytestmark = pytest.mark.skipif(
    _spatial_graph is None, reason="spatial-graph not installed"
)


def _lineage(
    n_tracks: int = 12,
    n_time: int = 20,
    edge_span: int = 1,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """A 4-D tzyx lineage whose edges span ``edge_span`` timepoints.

    ``edge_span`` is the knob that decides whether the ROI path wins: at 1
    frame the edge AABBs are tiny and the index is 5-6x; at half the axis
    they cover the whole slab and the index returns nearly everything.
    """
    rng = np.random.default_rng(7)
    positions = np.zeros((n_tracks * n_time, 4), dtype=dtype)
    edges = []
    for track in range(n_tracks):
        base = rng.random(3) * 50.0
        for t in range(n_time):
            row = track * n_time + t
            positions[row] = (t, *(base + t * 0.3))
            if t >= edge_span:
                edges.append((row - edge_span, row))
    return positions, np.asarray(edges, dtype=np.int32)


def _edge_pair_set(data) -> set[tuple[int, int]]:
    """Endpoint-pair set, order-insensitive within a pair and across edges."""
    if data.edge_endpoint_rows is None:
        return set()
    return {
        (int(min(u, v)), int(max(u, v))) for u, v in data.edge_endpoint_rows.tolist()
    }


@pytest.mark.parametrize("extent", [(0.5, 0.5), (5.0, 5.0)])
@pytest.mark.parametrize("n_sliced_axes", [1, 2])
@pytest.mark.parametrize("edge_span", [1, 8])
async def test_mask_and_roi_agree(make_request, extent, n_sliced_axes, edge_span):
    """Identical node rows and identical edge endpoint-pair sets.

    Parametrized over trail widths, sliced-axis counts, and both local and
    non-local edge topologies -- the matrix the design calls for, because a
    mistake in the ROI dtype, the unbounded axes, or the refinement shows up
    in exactly one cell of it.
    """
    positions, edges = _lineage(edge_span=edge_span)
    sliced = {0: 9} if n_sliced_axes == 1 else {0: 9, 1: 25}
    # The parametrized width belongs to the trail axis; a second sliced axis
    # gets a wide window on purpose, so the cell exercises two bounded axes
    # rather than degenerating into an empty slice that agrees trivially.
    extents = {0: extent}
    if n_sliced_axes == 2:
        extents[1] = (30.0, 30.0)
    displayed = (2, 3) if n_sliced_axes == 2 else (1, 2, 3)

    request = make_request(displayed=displayed, sliced=sliced, extents=extents)

    mask_store = GraphMemoryStore.from_arrays(positions, edges, slice_strategy="mask")
    roi_store = GraphMemoryStore.from_arrays(positions, edges, slice_strategy="roi")
    mask_data = await mask_store.get_data(request)
    roi_data = await roi_store.get_data(request)

    assert not mask_data.nodes_empty, "the cell must select something to compare"
    assert np.array_equal(
        np.sort(mask_data.original_node_rows), np.sort(roi_data.original_node_rows)
    )
    assert _edge_pair_set(mask_data) == _edge_pair_set(roi_data)
    assert mask_data.nodes_empty == roi_data.nodes_empty
    assert mask_data.edges_empty == roi_data.edges_empty


async def test_mask_and_roi_agree_with_fade(make_request):
    """The fade is computed over whichever rows the strategy returned."""
    positions, edges = _lineage()
    request = make_request(
        displayed=(2, 3),
        sliced={0: 9, 1: 25},
        extents={0: (4.0, 4.0), 1: (60.0, 60.0)},
        fades={0: (4.0, 4.0, 0.1)},
    )
    mask_store = GraphMemoryStore.from_arrays(positions, edges, slice_strategy="mask")
    roi_store = GraphMemoryStore.from_arrays(positions, edges, slice_strategy="roi")
    mask_data = await mask_store.get_data(request)
    roi_data = await roi_store.get_data(request)

    mask_alpha = dict(
        zip(mask_data.original_node_rows.tolist(), mask_data.node_alpha.tolist())
    )
    roi_alpha = dict(
        zip(roi_data.original_node_rows.tolist(), roi_data.node_alpha.tolist())
    )
    assert mask_alpha == roi_alpha


async def test_roi_refinement_is_not_redundant(make_request):
    """An edge crossing the slab with neither endpoint inside must be dropped.

    ``query_edges_in_roi`` indexes each edge by its segment AABB, so it
    reports this edge; D5's rule does not.  Asserting the raw query returns
    it and the refined result does not is what stops the refinement being
    mistaken for a no-op and deleted.
    """
    positions = np.array(
        [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [10.0, 5.0, 5.0]], dtype=np.float32
    )
    edges = np.array([[0, 1]], dtype=np.int32)
    store = GraphMemoryStore.from_arrays(positions, edges, slice_strategy="roi")

    # The raw index call: the AABB of edge 0 spans t in [0, 20] and crosses
    # the slab at t = 10, even though both endpoints are 10 units outside it.
    roi = np.stack(
        [
            np.array([9.5, -np.inf, -np.inf], dtype=np.float32),
            np.array([10.5, np.inf, np.inf], dtype=np.float32),
        ]
    )
    raw = np.asarray(store.graph.query_edges_in_roi(roi))
    assert raw.shape[0] == 1, "the index should report the crossing edge"

    data = await store.get_data(make_request(displayed=(1, 2), sliced={0: 10}))
    assert data.edges_empty, "the refinement should drop it"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
async def test_roi_dtype_mismatch_guarded(make_request, dtype):
    """The ROI is built from positions.dtype, so both widths work.

    A float64 ROI against float32 positions raises
    ``ValueError: Buffer dtype mismatch`` -- the binding does no coercion.
    """
    positions, edges = _lineage(n_tracks=4, n_time=6, dtype=dtype)
    store = GraphMemoryStore.from_arrays(positions, edges, slice_strategy="roi")
    data = await store.get_data(make_request(displayed=(1, 2, 3), sliced={0: 3}))
    assert data.original_node_rows.shape[0] > 0


async def test_roi_leaves_original_edge_rows_none(make_request):
    """The ROI path yields endpoint pairs, not edge rows.

    Translating the whole candidate set every reslice is the per-frame
    ``searchsorted`` D18 exists to remove; a pick resolves its single edge
    through ``edge_row_for_endpoints`` instead.
    """
    positions, edges = _lineage(n_tracks=4, n_time=6)
    store = GraphMemoryStore.from_arrays(positions, edges, slice_strategy="roi")
    data = await store.get_data(make_request(displayed=(1, 2, 3), sliced={0: 3}))
    assert data.original_edge_rows is None
    assert data.edge_endpoint_rows is not None


async def test_edge_row_for_endpoints_recovers_the_mask_row(make_request):
    """The on-pick translation returns exactly what the mask path would."""
    positions, edges = _lineage(n_tracks=4, n_time=6)
    mask_store = GraphMemoryStore.from_arrays(positions, edges, slice_strategy="mask")
    roi_store = GraphMemoryStore.from_arrays(positions, edges, slice_strategy="roi")
    request = make_request(displayed=(1, 2, 3), sliced={0: 3})
    mask_data = await mask_store.get_data(request)
    roi_data = await roi_store.get_data(request)

    mask_rows = {
        (int(min(u, v)), int(max(u, v))): int(row)
        for (u, v), row in zip(
            mask_data.edge_endpoint_rows.tolist(),
            mask_data.original_edge_rows.tolist(),
        )
    }
    for u, v in roi_data.edge_endpoint_rows.tolist():
        recovered = roi_store.edge_row_for_endpoints(int(u), int(v))
        assert recovered == mask_rows[(min(u, v), max(u, v))]


def test_edge_row_for_endpoints_missing_edge_returns_minus_one():
    positions, edges = _lineage(n_tracks=3, n_time=4)
    store = GraphMemoryStore.from_arrays(positions, edges)
    assert store.edge_row_for_endpoints(0, 11) == -1


async def test_roi_empty_slice_matches_mask(make_request):
    """Both strategies agree on the empty-placeholder path too."""
    positions, edges = _lineage(n_tracks=3, n_time=4)
    request = make_request(displayed=(1, 2, 3), sliced={0: 500})
    mask_store = GraphMemoryStore.from_arrays(positions, edges, slice_strategy="mask")
    roi_store = GraphMemoryStore.from_arrays(positions, edges, slice_strategy="roi")
    mask_data = await mask_store.get_data(request)
    roi_data = await roi_store.get_data(request)
    assert mask_data.nodes_empty and roi_data.nodes_empty
    assert mask_data.edges_empty and roi_data.edges_empty
    assert mask_data.node_positions.shape == roi_data.node_positions.shape
