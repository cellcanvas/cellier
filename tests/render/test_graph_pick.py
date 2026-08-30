"""Pick decoding for the graph compound visual (D14, D18).

Picks report the store's **original node id**, never the render-buffer row.
The distinction is invisible in a full 3-D view of a contiguously-numbered
graph and load-bearing everywhere else, which is what
``test_pick_through_a_sliced_view`` exists to prove.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pygfx as gfx
import pytest

from cellier.data.graph import GraphMemoryStore, GraphSliceRequest
from cellier.events._events import GraphEdgePickInfo, GraphNodePickInfo
from cellier.render.visuals._graph_memory import GFXGraphMemoryVisual
from cellier.transform import AffineTransform
from cellier.visuals import GraphAppearance, GraphVisual

try:  # pragma: no cover - import probe
    import spatial_graph as _spatial_graph
except ImportError:  # pragma: no cover
    _spatial_graph = None


def _store(**kwargs) -> GraphMemoryStore:
    """Four nodes with deliberately non-contiguous ids, three edges."""
    positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    kwargs.setdefault("node_ids", np.array([10, 20, 45, 900], dtype=np.uint64))
    return GraphMemoryStore.from_arrays(positions, edges, **kwargs)


def _visual(store) -> GFXGraphMemoryVisual:
    model = GraphVisual(
        name="test", data_store_id=str(store.id), appearance=GraphAppearance()
    )
    return GFXGraphMemoryVisual(
        visual_model=model,
        render_modes={"2d", "3d"},
        transform=AffineTransform.identity(ndim=store.ndim),
    )


def _commit(visual, store, displayed=(0, 1, 2), sliced=None, extents=None):
    shared = uuid4()
    request = GraphSliceRequest(
        slice_request_id=shared,
        chunk_request_id=shared,
        scale_index=0,
        displayed_axes=displayed,
        slice_indices=dict(sliced or {}),
        extents=dict(extents or {}),
        fades={},
    )
    data = asyncio.run(store.get_data(request))
    visual.on_data_ready([(request, data)])
    return data


def test_node_pick_returns_original_id():
    """A points hit yields the original id, not the row."""
    store = _store()
    visual = _visual(store)
    _commit(visual, store)

    result = visual.decode_pick(visual.node_points, {"vertex_index": 2}, store)
    assert isinstance(result, GraphNodePickInfo)
    assert result.node_id == 45
    assert result.node_row == 2


def test_edge_pick_returns_endpoint_ids():
    """A line hit yields the edge row plus both endpoint ids."""
    store = _store()
    visual = _visual(store)
    _commit(visual, store)

    # Vertex 2 is the first vertex of rendered edge 1 -> store edge (1, 2).
    result = visual.decode_pick(visual.node_edges, {"vertex_index": 2}, store)
    assert isinstance(result, GraphEdgePickInfo)
    assert result.edge_index == 1
    assert result.source_node_id == 20
    assert result.target_node_id == 45


def test_edge_pick_second_vertex_maps_to_the_same_edge():
    """Both vertices of a segment resolve to one edge -- the integer half."""
    store = _store()
    visual = _visual(store)
    _commit(visual, store)

    first = visual.decode_pick(visual.node_edges, {"vertex_index": 4}, store)
    second = visual.decode_pick(visual.node_edges, {"vertex_index": 5}, store)
    assert first == second
    assert first.edge_index == 2


def test_pick_through_a_sliced_view():
    """In a filtered view the rendered index is not the store index.

    This is the bug the whole map exists to prevent: slice away the first
    two nodes and rendered vertex 0 is store row 2, id 45 -- not row 0,
    id 10.
    """
    store = _store()
    visual = _visual(store)
    data = _commit(
        visual, store, displayed=(1, 2), sliced={0: 3}, extents={0: (1.0, 1.0)}
    )

    # Rows 2 and 3 survive the slab on axis 0.
    assert np.array_equal(data.original_node_rows, np.array([2, 3]))

    result = visual.decode_pick(visual.node_points, {"vertex_index": 0}, store)
    assert result.node_row == 2
    assert result.node_id == 45
    assert result.node_id != store.node_ids[0]


def test_edge_pick_through_a_sliced_view():
    store = _store()
    visual = _visual(store)
    data = _commit(
        visual, store, displayed=(1, 2), sliced={0: 3}, extents={0: (1.0, 1.0)}
    )

    # Edges 1 and 2 survive under D5 (either endpoint in the slab).
    assert np.array_equal(data.original_edge_rows, np.array([1, 2]))

    result = visual.decode_pick(visual.node_edges, {"vertex_index": 0}, store)
    assert result.edge_index == 1
    assert result.source_node_id == 20
    assert result.target_node_id == 45


@pytest.mark.skipif(_spatial_graph is None, reason="spatial-graph not installed")
def test_roi_edge_pick_resolves_the_row_on_demand():
    """Under 'roi' the row is recovered from the endpoints, at pick time.

    The ROI index returns endpoint pairs rather than edge rows; translating
    the whole candidate set every reslice is the per-frame ``searchsorted``
    D18 removes.  The payload must still carry the same ``edge_index`` a
    mask-sliced pick would.
    """
    mask_store = _store(slice_strategy="mask")
    roi_store = _store(slice_strategy="roi")

    mask_visual, roi_visual = _visual(mask_store), _visual(roi_store)
    mask_data = _commit(mask_visual, mask_store)
    roi_data = _commit(roi_visual, roi_store)

    assert mask_data.original_edge_rows is not None
    assert roi_data.original_edge_rows is None

    mask_pick = mask_visual.decode_pick(
        mask_visual.node_edges, {"vertex_index": 2}, mask_store
    )
    roi_pick = roi_visual.decode_pick(
        roi_visual.node_edges, {"vertex_index": 2}, roi_store
    )
    assert roi_pick.edge_index >= 0
    assert {roi_pick.source_node_id, roi_pick.target_node_id} == {
        mask_pick.source_node_id,
        mask_pick.target_node_id,
    }


def test_pick_on_neither_child_returns_none():
    """Defensive: an unrelated object is not this visual's problem."""
    store = _store()
    visual = _visual(store)
    _commit(visual, store)

    stranger = gfx.Points(
        gfx.Geometry(positions=np.zeros((1, 3), dtype=np.float32)),
        gfx.PointsMaterial(),
    )
    assert visual.decode_pick(stranger, {"vertex_index": 0}, store) is None


def test_pick_without_vertex_index_returns_none():
    store = _store()
    visual = _visual(store)
    _commit(visual, store)
    assert visual.decode_pick(visual.node_points, {}, store) is None


def test_pick_falls_back_to_rows_without_a_store():
    """A visual driven directly, with no store, still returns a payload."""
    store = _store()
    visual = _visual(store)
    _commit(visual, store)

    result = visual.decode_pick(visual.node_points, {"vertex_index": 1})
    assert result.node_id == 1
    assert result.node_row == 1


def test_render_manager_routes_graph_picks(qtbot):
    """The pick-details branch reaches the visual through the scene manager."""
    from cellier.controller import CellierController
    from cellier.scene.dims import CoordinateSystem

    controller = CellierController()
    controller.camera_reslice_enabled = False
    scene = controller.add_scene(
        dim="3d",
        coordinate_system=CoordinateSystem(name="world", axis_labels=("z", "y", "x")),
        name="main",
    )
    controller.add_canvas(scene_id=scene.id)
    store = _store()
    visual = controller.add_graph(store, scene.id, name="graph")

    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    _commit(gfx_visual, store)

    result = controller._render_manager._extract_pick_details(
        scene.id, gfx_visual.node_points, {"vertex_index": 3}
    )
    assert isinstance(result, GraphNodePickInfo)
    assert result.node_id == 900
