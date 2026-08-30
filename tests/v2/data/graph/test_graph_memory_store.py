"""GraphMemoryStore construction, laziness and serialization."""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data.graph import GraphMemoryStore

try:  # pragma: no cover - import probe
    import spatial_graph as _spatial_graph
except ImportError:  # pragma: no cover
    _spatial_graph = None


def _store(**kwargs) -> GraphMemoryStore:
    positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    return GraphMemoryStore.from_arrays(positions, edges, **kwargs)


def test_from_arrays_basic_shape():
    store = _store()
    assert store.ndim == 3
    assert store.n_nodes == 4
    assert store.n_edges == 3
    assert store.node_color_mode == "uniform"
    assert store.node_size_mode == "uniform"
    assert store.edge_color_mode == "uniform"


def test_edges_out_of_range_raises():
    positions = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="outside"):
        GraphMemoryStore.from_arrays(positions, np.array([[0, 5]], dtype=np.int32))


def test_empty_edges_allowed():
    store = GraphMemoryStore.from_arrays(
        np.zeros((3, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
    )
    assert store.n_edges == 0
    assert store.edges.shape == (0, 2)


def test_id_row_round_trip():
    """Non-contiguous ids survive ids_for_rows / rows_for_ids."""
    ids = np.array([10, 20, 45, 900], dtype=np.uint64)
    store = _store(node_ids=ids)
    rows = np.array([0, 2, 3])
    assert np.array_equal(store.ids_for_rows(rows), ids[rows])
    assert np.array_equal(store.rows_for_ids(ids[rows]), rows)


def test_id_row_round_trip_string_ids():
    """String ids round-trip too -- geff does not guarantee integers."""
    ids = np.array(["a", "bb", "ccc", "dddd"])
    store = _store(node_ids=ids)
    assert list(store.ids_for_rows([1, 3])) == ["bb", "dddd"]
    assert np.array_equal(store.rows_for_ids(["bb", "dddd"]), np.array([1, 3]))


def test_node_ids_length_mismatch_raises():
    with pytest.raises(ValueError, match="node_ids"):
        _store(node_ids=np.array([1, 2], dtype=np.uint64))


def test_edge_span_is_lazy():
    """Constructing does not compute it; first access does; second is cached."""
    store = _store()
    assert store._edge_span is None
    first = store.edge_span
    assert store._edge_span is not None
    assert store.edge_span is first
    assert np.allclose(first, [1.0, 1.0, 1.0])


def test_edge_span_empty_graph():
    store = GraphMemoryStore.from_arrays(
        np.zeros((3, 4), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
    )
    assert np.array_equal(store.edge_span, np.zeros(4, dtype=np.float32))


def test_store_json_roundtrip():
    """The direct regression for Phase 0's P4 finding.

    v1's layout held the SpatialGraph as a field and raised
    ``PydanticSerializationError`` here.
    """
    store = _store(
        node_ids=np.array([5, 6, 7, 8], dtype=np.uint64),
        node_colors=np.tile(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32), (4, 1)),
    )
    payload = store.model_dump_json()
    restored = GraphMemoryStore.model_validate_json(payload)
    assert np.array_equal(restored.positions, store.positions)
    assert np.array_equal(restored.edges, store.edges)
    assert np.array_equal(restored.node_ids, store.node_ids)
    assert np.array_equal(restored.node_colors, store.node_colors)


def test_store_json_roundtrip_through_data_store_union():
    """The store resolves through the DataStoreType discriminated union."""
    from pydantic import TypeAdapter

    from cellier.data import DataStoreType

    adapter = TypeAdapter(DataStoreType)
    restored = adapter.validate_json(_store().model_dump_json())
    assert isinstance(restored, GraphMemoryStore)


@pytest.mark.skipif(_spatial_graph is None, reason="spatial-graph not installed")
def test_index_is_lazy():
    """No SpatialGraph is built at construction (A1)."""
    store = _store()
    assert store._graph is None
    graph = store.graph
    assert store._graph is graph


@pytest.mark.skipif(_spatial_graph is None, reason="spatial-graph not installed")
def test_spatial_graph_ids_are_rows():
    """The index's node ids are arange(N), not the caller's ids (D18)."""
    store = _store(node_ids=np.array([10, 20, 45, 900], dtype=np.uint64))
    node_ids = np.sort(np.asarray(store.graph.nodes).astype(np.int64))
    assert np.array_equal(node_ids, np.arange(store.n_nodes))


def test_roi_without_spatial_graph_raises(monkeypatch):
    """slice_strategy='roi' with the extra absent raises, naming it (D17)."""
    import cellier.data.graph._graph_memory_store as module

    monkeypatch.setattr(module, "_spatial_graph_available", lambda: False)
    with pytest.raises(ValueError, match=r"cellier\[graph\]"):
        _store(slice_strategy="roi")
