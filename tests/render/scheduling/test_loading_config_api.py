"""The public loading API: the controller setter and the viewer wrappers.

``CellierController.set_loading_config`` merges fields into a multiscale
visual's ``render_config.loading``; ``Viewer`` and ``OrthoViewer`` mirror it
and the progress hooks.  An invalid combination raises and changes nothing.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cellier.events import (
    BackstopCompleteEvent,
    LoadingConfigChangedEvent,
    LoadingConfigUpdateEvent,
    LoadingProgress,
    ResliceProgressEvent,
)
from cellier.visuals import ProgressiveLoadingConfig
from tests.render.conftest import drain_loading
from tests.render.scheduling.test_backstop_integration import _add


def _record(controller, visual_id) -> list:
    events: list = []
    controller._outgoing_events.subscribe(
        LoadingConfigChangedEvent, events.append, entity_id=visual_id
    )
    return events


# -- the controller -------------------------------------------------------------------


def test_fields_merge_into_the_current_config(controller, multiscale_image_store):
    _scene, visual, _gfx = _add(controller, multiscale_image_store)
    events = _record(controller, visual.id)

    result = controller.set_loading_config(visual.id, dims_drag="backstop")

    assert result == ProgressiveLoadingConfig(dims_drag="backstop")
    assert visual.render_config.loading == result
    [event] = events
    assert event.loading == result
    assert event.source_id == controller._id
    # A second field keeps the first.
    controller.set_loading_config(visual.id, backstop_max_slot_fraction=0.3)
    assert visual.render_config.loading.dims_drag == "backstop"
    assert visual.render_config.loading.backstop_max_slot_fraction == 0.3


def test_the_caller_source_id_is_stamped(controller, multiscale_image_store):
    _scene, visual, _gfx = _add(controller, multiscale_image_store)
    events = _record(controller, visual.id)
    widget_id = uuid4()
    controller.set_loading_config(visual.id, source_id=widget_id, backstop_level=1)
    assert events[-1].source_id == widget_id


def test_an_invalid_combination_raises_and_changes_nothing(
    controller, multiscale_image_store
):
    _scene, visual, _gfx = _add(controller, multiscale_image_store)
    controller.set_loading_config(visual.id, dims_drag="backstop")
    before = visual.render_config.loading
    events = _record(controller, visual.id)

    with pytest.raises(ValueError, match="needs backstop=True"):
        controller.set_loading_config(visual.id, backstop=False)

    assert visual.render_config.loading == before
    assert events == []


def test_an_unknown_field_raises_with_a_suggestion(controller, multiscale_image_store):
    _scene, visual, _gfx = _add(controller, multiscale_image_store)
    with pytest.raises(ValueError, match="Did you mean 'dims_drag'"):
        controller.set_loading_config(visual.id, dims_dragg="backstop")


def test_a_visual_that_is_not_multiscale_raises(controller, image_volume):
    scene = controller.add_scene(dim="3d", name="scene")
    visual = controller.add_image(image_volume, scene.id)
    with pytest.raises(TypeError, match="multiscale"):
        controller.set_loading_config(visual.id, dims_drag="backstop")


def test_no_change_is_a_no_op(controller, multiscale_image_store):
    _scene, visual, _gfx = _add(controller, multiscale_image_store)
    events = _record(controller, visual.id)
    assert controller.set_loading_config(visual.id, dims_drag="eager") == (
        visual.render_config.loading
    )
    assert events == []


def test_assigning_render_config_directly_announces_the_change(
    controller, multiscale_image_store
):
    _scene, visual, _gfx = _add(controller, multiscale_image_store)
    events = _record(controller, visual.id)
    visual.render_config = visual.render_config.model_copy(
        update={"loading": ProgressiveLoadingConfig(backstop=False)}
    )
    [event] = events
    assert event.loading.backstop is False
    assert event.source_id == controller._id


def test_an_update_event_on_the_incoming_bus_applies(
    controller, multiscale_image_store
):
    _scene, visual, _gfx = _add(controller, multiscale_image_store)
    controller._incoming_events.emit(
        LoadingConfigUpdateEvent(
            source_id=uuid4(), visual_id=visual.id, field="dims_drag", value="backstop"
        )
    )
    assert visual.render_config.loading.dims_drag == "backstop"


# -- the Viewer -----------------------------------------------------------------------


@pytest.fixture
def viewer_with_image(qtbot, multiscale_image_store):
    from cellier.convenience import Viewer, spatial_axes
    from cellier.visuals._image import (
        MultiscaleImageAppearance,
        MultiscaleImageRenderConfig,
    )

    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="offscreen")
    visual = viewer.add_image_multiscale(
        multiscale_image_store,
        MultiscaleImageAppearance(force_level=1),
        render_config=MultiscaleImageRenderConfig(block_size=8),
    )
    viewer.add_canvas()
    yield viewer, visual
    viewer.controller.close()


async def test_viewer_mirrors_the_controller(viewer_with_image):
    viewer, visual = viewer_with_image
    progress: list = []
    backstops: list = []
    viewer.on_reslice_progress(visual, progress.append)
    viewer.on_backstop_complete(visual.id, backstops.append)
    assert viewer.loading_progress(visual) is None

    viewer.controller.fit_camera(viewer.scene.id)
    viewer.controller.reslice_all()
    await drain_loading(viewer.controller)
    await asyncio.sleep(0)

    assert progress and isinstance(progress[-1], ResliceProgressEvent)
    assert [type(e) for e in backstops] == [BackstopCompleteEvent]
    assert viewer.loading_progress(visual).complete
    assert viewer.set_loading(visual, dims_drag="backstop").dims_drag == "backstop"
    assert visual.render_config.loading.dims_drag == "backstop"


async def test_removing_the_visual_removes_its_subscriptions(viewer_with_image):
    from cellier.events import ResliceProgressEvent as Event

    viewer, visual = viewer_with_image
    viewer.on_reslice_progress(visual, lambda event: None)
    bus = viewer.controller._outgoing_events
    assert bus.get_subscribers(Event, entity_id=visual.id)
    viewer.controller.remove_visual(visual.id)
    assert not [s for s in bus.get_subscribers(Event) if s.entity_id == visual.id]


# -- the OrthoViewer, over a panel group ----------------------------------------------


@pytest.fixture
def ortho_with_image(qtbot, multiscale_image_store):
    from cellier.convenience import OrthoViewer, spatial_axes

    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui="offscreen")
    visuals = ortho.add_image_multiscale(multiscale_image_store)
    yield ortho, visuals
    ortho.controller.close()


def test_ortho_set_loading_writes_every_panel(ortho_with_image):
    ortho, visuals = ortho_with_image
    ortho.set_loading(visuals, dims_drag="backstop")
    assert {v.render_config.loading.dims_drag for v in visuals.values()} == {"backstop"}
    with pytest.raises(ValueError, match="needs backstop=True"):
        ortho.set_loading(next(iter(visuals.values())), backstop=False)
    assert all(v.render_config.loading.backstop for v in visuals.values())


def _fake_progress(ortho, monkeypatch, table):
    monkeypatch.setattr(ortho.controller, "loading_progress", table.get)


def test_ortho_progress_is_summed_over_the_panels(ortho_with_image, monkeypatch):
    ortho, visuals = ortho_with_image
    ids = [v.id for v in visuals.values()]
    table = {
        vid: LoadingProgress(needed_target=4, resident_target=i, complete=i == 4)
        for i, vid in enumerate(ids, start=1)
    }
    _fake_progress(ortho, monkeypatch, table)

    total = ortho.loading_progress(visuals)
    assert (total.needed_target, total.resident_target) == (16, 10)
    assert not total.complete

    received: list = []
    ortho.on_reslice_progress(visuals, received.append)
    ortho.controller._outgoing_events.emit(
        ResliceProgressEvent(uuid4(), uuid4(), ids[2], table[ids[2]])
    )
    [event] = received
    assert event.visual_id == ids[2]
    assert event.progress == total


def test_ortho_backstop_fires_once_every_panel_has_it(ortho_with_image, monkeypatch):
    ortho, visuals = ortho_with_image
    ids = [v.id for v in visuals.values()]
    table = {vid: LoadingProgress(backstop_complete=False) for vid in ids}
    _fake_progress(ortho, monkeypatch, table)
    received: list = []
    ortho.on_backstop_complete(visuals, received.append)

    bus = ortho.controller._outgoing_events
    for vid in ids:
        table[vid] = LoadingProgress(backstop_complete=True)
        bus.emit(BackstopCompleteEvent(uuid4(), uuid4(), vid))
    # Only the event that completed the group got through.
    assert [e.visual_id for e in received] == [ids[-1]]
