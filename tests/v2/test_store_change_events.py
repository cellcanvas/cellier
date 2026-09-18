"""The controller relays store changes (``plans/store_change_events.md``).

A store announces a change on ``data_changed``; the controller refreshes
extent-derived state for an ``extent`` change, emits the matching bus event,
and reslices every visual reading the store.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.events import DataStoreContentsChangedEvent, DataStoreMetadataChangedEvent
from cellier.scene.dims import spatial_axes
from cellier.visuals._image_memory import (
    InMemoryImageAppearance,
    InMemoryImageSingleAppearance,
)


@pytest.fixture
def controller():
    return CellierController(gui="offscreen")


@pytest.fixture
def scene(controller):
    return controller.add_scene(coordinate_system=spatial_axes("z", "y", "x"))


def _points_store() -> PointsMemoryStore:
    return PointsMemoryStore(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    )


def _record_bus(controller) -> list:
    events: list = []
    for event_type in (DataStoreMetadataChangedEvent, DataStoreContentsChangedEvent):
        controller._outgoing_events.subscribe(event_type, events.append)
    return events


def _record_reslices(controller, monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(controller, "reslice_visual", calls.append)
    return calls


def test_an_extent_change_is_announced_and_reslices_the_readers(
    controller, scene, monkeypatch
):
    store = _points_store()
    first = controller.add_points(store, scene.id)
    second = controller.add_points(store, scene.id)
    other = controller.add_points(_points_store(), scene.id)
    events = _record_bus(controller)
    reslices = _record_reslices(controller, monkeypatch)

    store.positions = np.zeros((3, 3), dtype=np.float32)

    assert events == [
        DataStoreMetadataChangedEvent(source_id=controller._id, data_store_id=store.id)
    ]
    assert sorted(map(str, reslices)) == sorted(map(str, (first.id, second.id)))
    assert other.id not in reslices


def test_a_contents_change_carries_its_regions(controller, scene, monkeypatch):
    store = _points_store()
    visual = controller.add_points(store, scene.id)
    events = _record_bus(controller)
    reslices = _record_reslices(controller, monkeypatch)

    store.notify_changed("contents", regions=[((0, 1), (0, 1), (0, 1))])

    assert events == [
        DataStoreContentsChangedEvent(
            source_id=controller._id,
            data_store_id=store.id,
            regions=(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),),
        )
    ]
    assert reslices == [visual.id]


def test_a_contents_change_leaves_extent_state_alone(controller, scene, monkeypatch):
    store = _points_store()
    controller.add_points(store, scene.id)
    refreshed: list = []
    monkeypatch.setattr(
        controller._render_manager, "refresh_visual_axis_extents", refreshed.append
    )
    _record_reslices(controller, monkeypatch)

    store.colors = np.ones((2, 4), dtype=np.float32)

    assert refreshed == []


def test_an_image_shape_change_refreshes_the_slice_check_extents(
    controller, scene, monkeypatch
):
    store = ImageMemoryStore(data=np.zeros((4, 6, 8), dtype=np.float32))
    visual = controller.add_image(
        store,
        scene.id,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(clim=(0.0, 1.0)),
    )
    # The render visual sizes its texture at construction and is not rebuilt
    # for a new shape (out of scope); keep the reslice from reaching it.
    _record_reslices(controller, monkeypatch)
    scene_manager = controller._render_manager._scenes[scene.id]

    store.data = np.zeros((10, 6, 8), dtype=np.float32)

    assert scene_manager._axis_extents[visual.id][0] == (-0.5, 9.5)


def test_a_store_registered_through_add_visual_is_relayed(controller, scene):
    """``add_points`` registers the store implicitly; it is wired all the same."""
    store = _points_store()
    controller.add_points(store, scene.id)

    assert store.id in controller._store_psygnal_handlers


def test_add_data_store_is_idempotent(controller, scene, monkeypatch):
    store = _points_store()
    controller.add_data_store(store)
    controller.add_data_store(store)
    controller.add_points(store, scene.id)
    events = _record_bus(controller)
    _record_reslices(controller, monkeypatch)

    store.notify_changed("contents")

    assert len(events) == 1


def test_remove_data_store_stops_relaying(controller, scene):
    store = _points_store()
    visual = controller.add_points(store, scene.id)
    controller.remove_visual(visual.id)
    controller.remove_data_store(store.id)
    events = _record_bus(controller)

    store.notify_changed("extent")

    assert events == []
    assert store.id not in controller._store_psygnal_handlers


def test_a_restored_store_is_relayed():
    """``from_model`` registers deep copies; their announcements still arrive."""
    source = CellierController(gui="offscreen")
    scene = source.add_scene(coordinate_system=spatial_axes("z", "y", "x"))
    source.add_points(_points_store(), scene.id)

    restored = CellierController.from_model(source.to_model())
    (store,) = restored._model.data.stores.values()
    events = _record_bus(restored)

    store.positions = np.zeros((1, 3), dtype=np.float32)

    assert [type(e) for e in events] == [DataStoreMetadataChangedEvent]
    assert events[0].data_store_id == UUID(str(store.id))


def test_close_disconnects_stores_and_overlays(controller, scene):
    from cellier.visuals import SceneBoundingBox

    store = _points_store()
    controller.add_points(store, scene.id)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))

    controller.close()

    assert controller._store_psygnal_handlers == {}
    assert controller._overlays == {}
    store.notify_changed("extent")  # nothing left listening, nothing raises
    box.visible = False
