"""Stores announce their own changes (``plans/store_change_events.md``)."""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data import (
    GraphMemoryStore,
    ImageMemoryStore,
    LinesMemoryStore,
    PointsMemoryStore,
)
from cellier.data._changes import StoreChange
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore


def _record(store) -> list[StoreChange]:
    changes: list[StoreChange] = []
    store.data_changed.connect(changes.append)
    return changes


def _points(n: int = 2) -> PointsMemoryStore:
    return PointsMemoryStore(positions=np.arange(3 * n, dtype=np.float32).reshape(n, 3))


def _graph() -> GraphMemoryStore:
    return GraphMemoryStore(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32),
        edges=np.array([[0, 1]]),
    )


def _mesh() -> MeshMemoryStore:
    return MeshMemoryStore(
        positions=np.eye(3, dtype=np.float32), indices=np.array([[0, 1, 2]])
    )


@pytest.mark.parametrize(
    "factory, field, value, kind",
    [
        (_points, "positions", np.zeros((2, 3), dtype=np.float32), "extent"),
        (_points, "colors", np.ones((2, 4), dtype=np.float32), "contents"),
        (_points, "sizes", np.ones(2, dtype=np.float32), "contents"),
        (
            lambda: LinesMemoryStore(positions=np.zeros((2, 3), dtype=np.float32)),
            "positions",
            np.ones((2, 3), dtype=np.float32),
            "extent",
        ),
        (_mesh, "positions", np.ones((3, 3), dtype=np.float32), "extent"),
        (_mesh, "indices", np.array([[2, 1, 0]]), "contents"),
        (_graph, "positions", np.zeros((2, 3), dtype=np.float32), "extent"),
        (_graph, "edges", np.array([[1, 0]]), "contents"),
        (_graph, "node_colors", np.ones((2, 4), dtype=np.float32), "contents"),
    ],
)
def test_reassigning_a_data_field_announces_its_kind(factory, field, value, kind):
    store = factory()
    changes = _record(store)

    setattr(store, field, value)

    assert changes == [StoreChange(kind)]


def test_a_non_data_field_announces_nothing():
    store = _points()
    changes = _record(store)

    store.name = "renamed"

    assert changes == []


@pytest.mark.parametrize("store_cls", [ImageMemoryStore, LabelMemoryStore])
def test_image_data_is_extent_only_when_the_shape_changes(store_cls):
    dtype = np.int32 if store_cls is LabelMemoryStore else np.float32
    store = store_cls(data=np.zeros((2, 3), dtype=dtype))
    changes = _record(store)

    store.data = np.ones((2, 3), dtype=dtype)
    store.data = np.ones((4, 3), dtype=dtype)

    assert [c.kind for c in changes] == ["contents", "extent"]
    assert store.axis_extents == ((-0.5, 3.5), (-0.5, 2.5))


def test_notify_changed_normalizes_regions():
    store = _points()
    changes = _record(store)

    store.notify_changed("contents", regions=[((0, 1), (2, 5), (0, 3))])

    assert changes == [StoreChange("contents", (((0.0, 1.0), (2.0, 5.0), (0.0, 3.0)),))]


def test_notify_changed_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="Unknown store change kind"):
        _points().notify_changed("shape")


def test_geometry_extents_are_cached_until_an_extent_change():
    store = PointsMemoryStore(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    )
    assert store.axis_extents[0] == (0.0, 1.0)

    store.positions[1, 0] = 9.0
    assert store.axis_extents[0] == (0.0, 1.0)  # in place: unseen, by contract

    store.notify_changed("contents")
    assert store.axis_extents[0] == (0.0, 1.0)  # contents keeps the extent

    store.notify_changed("extent")
    assert store.axis_extents[0] == (0.0, 9.0)

    store.positions = np.array([[-4.0, 0.0, 0.0]], dtype=np.float32)
    assert store.axis_extents[0] == (-4.0, -4.0)


def test_listeners_see_fresh_extents():
    """Caches are cleared before the signal fires."""
    store = _points()
    seen: list = []
    store.data_changed.connect(lambda _change: seen.append(store.axis_extents))
    _ = store.axis_extents  # prime the cache

    store.positions = np.full((1, 3), 7.0, dtype=np.float32)

    assert seen == [((7.0, 7.0), (7.0, 7.0), (7.0, 7.0))]


def test_graph_derived_caches_follow_the_data():
    store = _graph()
    assert np.allclose(store.edge_span, (1.0, 2.0, 3.0))

    store.positions = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float32)

    assert np.allclose(store.edge_span, (5.0, 5.0, 5.0))


def test_a_deep_copy_announces_and_keeps_no_listeners():
    """``to_model`` deep-copies stores; the copy must still announce."""
    store = _points()
    original = _record(store)
    copy = store.model_copy(deep=True)
    copied = _record(copy)

    copy.positions = np.zeros((1, 3), dtype=np.float32)

    assert copied == [StoreChange("extent")]
    assert original == []
