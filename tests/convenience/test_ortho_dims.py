"""``OrthoDimsController``: one world point across the four panels (design 3.6)."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
from cmap import Colormap

from cellier.convenience import OrthoViewer
from cellier.convenience._ortho_dims import OrthoDimsController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.events import DimsChangedEvent, DimsUpdateEvent, SliderAxesChangedEvent
from cellier.scene.dims import spatial_axes
from cellier.visuals import InMemoryImageAppearance, InMemoryImageSingleAppearance

#: ``(t, z, y, x)``: spatial axes 1-3, one extra axis.
_WORLD = [("t", "time"), *spatial_axes("z", "y", "x")]


def _positions(viewer: OrthoViewer) -> list[dict[int, float]]:
    return [
        dict(scene.dims.selection.slice_indices) for scene in viewer.scenes.values()
    ]


def _record(viewer: OrthoViewer, event_type=DimsChangedEvent) -> list:
    events: list = []
    for scene in viewer.scenes.values():
        viewer.controller._outgoing_events.subscribe(
            event_type, events.append, entity_id=scene.id
        )
    return events


def _add_image(viewer: OrthoViewer) -> None:
    store = ImageMemoryStore(data=np.zeros((8, 8, 8), dtype=np.float32))
    viewer.add_image(
        store,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map=Colormap("gray")),
    )


def test_viewer_exposes_its_dims_controller():
    viewer = OrthoViewer(_WORLD)
    assert isinstance(viewer.dims_controller, OrthoDimsController)
    assert viewer.dims_controller.scene_ids == tuple(
        viewer.scenes[key].id for key in ("xy", "xz", "yz", "vol")
    )


def test_set_slice_position_updates_every_panel_with_one_source_id():
    viewer = OrthoViewer(_WORLD)
    events = _record(viewer)

    viewer.dims_controller.set_slice_position(2, 4.5)

    assert all(position[2] == 4.5 for position in _positions(viewer))
    assert len(events) == 4
    assert {event.source_id for event in events} == {viewer.dims_controller.id}


def test_set_slider_override_updates_every_panel():
    viewer = OrthoViewer(_WORLD)
    _add_image(viewer)
    events = _record(viewer, SliderAxesChangedEvent)

    viewer.dims_controller.set_slider_override(0, True)

    for scene in viewer.scenes.values():
        assert scene.dims.slider_overrides == {0: True}
        assert scene.slider_axes == (0, 1, 2, 3)
    assert len(events) == 4
    assert {event.source_id for event in events} == {viewer.dims_controller.id}


def test_moving_the_xy_z_slider_moves_z_everywhere():
    """The XY panel's z slider is the stored z of XZ, YZ and the volume."""
    viewer = OrthoViewer(_WORLD)
    xy = viewer.scenes["xy"]
    widget_id = object()  # stands in for the widget's own id

    viewer.controller.incoming_events.emit(
        DimsUpdateEvent(
            source_id=widget_id,
            scene_id=xy.id,
            slice_indices={1: 6.0},
            displayed_axes=None,
        )
    )

    positions = _positions(viewer)
    assert all(position == positions[0] for position in positions)
    assert positions[0][1] == 6.0


def test_a_direct_model_edit_is_mirrored_without_looping():
    viewer = OrthoViewer(_WORLD)
    xz = viewer.scenes["xz"]
    events = _record(viewer)

    xz.dims.selection.slice_indices = {**xz.dims.selection.slice_indices, 0: 5.0}

    assert all(position[0] == 5.0 for position in _positions(viewer))
    # One change on the edited panel, one mirrored change on each other.
    assert len(events) == 4


def test_thickness_is_mirrored():
    viewer = OrthoViewer(_WORLD)
    viewer.controller.update_thickness(viewer.scenes["yz"].id, {0: 1.5})
    for scene in viewer.scenes.values():
        assert scene.dims.selection.thickness == {0: 1.5}


def test_displayed_axes_are_never_mirrored():
    viewer = OrthoViewer(_WORLD)
    before = {
        key: scene.dims.selection.displayed_axes for key, scene in viewer.scenes.items()
    }

    viewer.controller.set_displayed_axes(viewer.scenes["xy"].id, (1, 3))

    assert viewer.scenes["xy"].dims.selection.displayed_axes == (1, 3)
    for key in ("xz", "yz", "vol"):
        assert viewer.scenes[key].dims.selection.displayed_axes == before[key]


def test_construction_copies_the_first_panel_when_panels_disagree():
    viewer = OrthoViewer(_WORLD, link_axes=False)
    controller = viewer.controller
    controller.update_slice_indices(viewer.scenes["xy"].id, {0: 2.0, 2: 3.0})
    controller.update_slice_indices(viewer.scenes["yz"].id, {0: 9.0})
    controller.set_slider_override(viewer.scenes["vol"].id, 0, False)

    OrthoDimsController(
        controller, [viewer.scenes[k] for k in ("xy", "xz", "yz", "vol")]
    )

    positions = _positions(viewer)
    assert all(position == positions[0] for position in positions)
    assert positions[0][0] == 2.0
    assert all(scene.dims.slider_overrides == {} for scene in viewer.scenes.values())


def test_unlinked_panels_are_not_mirrored():
    viewer = OrthoViewer(_WORLD, link_axes=False)
    assert viewer.dims_controller is None
    assert not viewer.axis_sync_enabled

    viewer.controller.update_slice_indices(viewer.scenes["xy"].id, {0: 4.0})

    assert viewer.scenes["xz"].dims.selection.slice_indices[0] == 0.0


def test_re_enabling_sync_brings_the_panels_back_into_agreement():
    viewer = OrthoViewer(_WORLD)
    viewer.axis_sync_enabled = False
    viewer.controller.update_slice_indices(viewer.scenes["xy"].id, {0: 4.0})
    assert viewer.scenes["xz"].dims.selection.slice_indices[0] == 0.0

    viewer.axis_sync_enabled = True

    assert all(position[0] == 4.0 for position in _positions(viewer))


def test_close_stops_mirroring():
    viewer = OrthoViewer(_WORLD)
    viewer.dims_controller.close()
    viewer.controller.update_slice_indices(viewer.scenes["xy"].id, {0: 4.0})
    assert viewer.scenes["xz"].dims.selection.slice_indices[0] == 0.0


def test_center_slices_sets_all_four_panels_once(image_store):
    viewer = OrthoViewer(spatial_axes("z", "y", "x"))
    viewer.controller.add_data_store(image_store)
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    events = _record(viewer)

    viewer.center_slices()

    positions = _positions(viewer)
    assert all(position == positions[0] for position in positions)
    # One write per panel, all from the dims controller: no mirrored echoes.
    assert len(events) == 4
    assert {event.source_id for event in events} == {viewer.dims_controller.id}


def test_save_load_keeps_the_panels_in_agreement(tmp_path):
    viewer = OrthoViewer(_WORLD)
    _add_image(viewer)
    viewer.dims_controller.set_slice_positions({0: 1.0, 1: 3.0, 2: 4.0, 3: 5.0})
    viewer.dims_controller.set_slider_override(0, True)
    path = tmp_path / "ortho.json"
    viewer.to_file(path)

    loaded = OrthoViewer.from_file(path)

    positions = _positions(loaded)
    assert all(position == positions[0] for position in positions)
    assert positions[0] == {0: 1.0, 1: 3.0, 2: 4.0, 3: 5.0}
    slider_axes = {scene.slider_axes for scene in loaded.scenes.values()}
    assert slider_axes == {(0, 1, 2, 3)}
    assert loaded.axis_sync_enabled


# -- scrub forwarding (interaction tracker design 4.8) ---------------------------


def _record_interactions(viewer: OrthoViewer) -> list[tuple]:
    """``(panel key, phase, reason)`` for every ``DimsInteractionEvent``."""
    events: list[tuple] = []
    for key, scene in viewer.scenes.items():
        viewer.controller.on_dims_interaction(
            scene.id,
            lambda event, key=key: events.append((key, event.phase, event.reason)),
            owner_id=viewer.controller._id,
        )
    return events


def _states(viewer: OrthoViewer) -> dict[str, str]:
    return {
        key: viewer.controller.dims_interaction_state(scene.id)
        for key, scene in viewer.scenes.items()
    }


def _count_scopes(viewer: OrthoViewer, monkeypatch) -> list[tuple]:
    """Record every scope the mirror opens or closes."""
    controller = viewer.controller
    calls: list[tuple] = []
    keys = {scene.id: key for key, scene in viewer.scenes.items()}
    begin, end = controller.begin_dims_interaction, controller.end_dims_interaction

    def spy_begin(scene_id, *, source_id):
        calls.append(("begin", keys[scene_id]))
        begin(scene_id, source_id=source_id)

    def spy_end(scene_id, *, source_id):
        calls.append(("end", keys[scene_id]))
        end(scene_id, source_id=source_id)

    monkeypatch.setattr(controller, "begin_dims_interaction", spy_begin)
    monkeypatch.setattr(controller, "end_dims_interaction", spy_end)
    return calls


async def test_a_t_scrub_in_one_panel_scrubs_and_ends_all_four(monkeypatch):
    """``t`` is sliced in every panel, so a ``t`` tick ticks all four."""
    viewer = OrthoViewer(_WORLD)
    controller = viewer.controller
    xy = viewer.scenes["xy"]
    events = _record_interactions(viewer)
    dims_events = _record(viewer)
    widget = uuid4()

    controller.begin_dims_interaction(xy.id, source_id=widget)
    scopes = _count_scopes(viewer, monkeypatch)
    for t in (1.0, 2.0, 3.0):
        controller.update_slice_indices(
            xy.id, {0: t}, source_id=widget, interactive=True
        )
    assert all(position[0] == 3.0 for position in _positions(viewer))
    assert _states(viewer) == dict.fromkeys(("xy", "xz", "yz", "vol"), "active")
    # The origin starts first, and its start opens the others' scopes before
    # the mirrored ticks arrive: every mirrored tick is a scrub tick.
    assert events[0] == ("xy", "start", None)
    assert sorted(events) == sorted((key, "start", None) for key in viewer.scenes)
    assert len(dims_events) == 12
    assert all(event.interactive for event in dims_events)
    # The echo guard: three scopes for the whole scrub, no recursion.
    assert scopes == [("begin", "xz"), ("begin", "yz"), ("begin", "vol")]

    del events[:]
    controller.end_dims_interaction(xy.id, source_id=widget)
    assert sorted(events) == sorted((key, "end", "release") for key in viewer.scenes)
    assert _states(viewer) == dict.fromkeys(("xy", "xz", "yz", "vol"), "idle")
    assert not controller._dims_driver.tasks()


async def test_a_z_scrub_in_xy_scrubs_only_xy(monkeypatch):
    """The other panels display ``z``: their region does not change."""
    viewer = OrthoViewer(_WORLD)
    controller = viewer.controller
    xy = viewer.scenes["xy"]
    events = _record_interactions(viewer)
    widget = uuid4()

    with controller.dims_interaction(xy.id):
        controller.update_slice_indices(xy.id, {1: 5.0}, source_id=widget)
        controller.update_slice_indices(xy.id, {1: 6.0}, source_id=widget)
        assert all(position[1] == 6.0 for position in _positions(viewer))
        assert _states(viewer) == {
            "xy": "active",
            "xz": "idle",
            "yz": "idle",
            "vol": "idle",
        }
    # The forwarded scopes opened and closed with nothing emitted.
    assert events == [("xy", "start", None), ("xy", "end", "release")]


async def test_a_scrub_that_settles_releases_the_other_panels():
    viewer = OrthoViewer(_WORLD)
    controller = viewer.controller
    controller._render_manager.config.scheduler.dims_settle_s = 0.02
    events = _record_interactions(viewer)

    # A keyboard or wheel step: interactive, no scope.
    controller.update_slice_indices(
        viewer.scenes["xy"].id, {0: 2.0}, source_id=uuid4(), interactive=True
    )
    await asyncio.sleep(0.1)
    ends = {key: reason for key, phase, reason in events if phase == "end"}
    assert ends == {
        "xy": "settle",
        "xz": "release",
        "yz": "release",
        "vol": "release",
    }
    assert not controller._dims_driver.tasks()


async def test_without_forwarding_a_mirrored_tick_is_a_jump():
    """What the forwarding is for: unlinked, the mirror's writes plan in full."""
    viewer = OrthoViewer(_WORLD)
    viewer.dims_controller._enabled = True
    viewer.controller.unsubscribe_owner(viewer.dims_controller.id)  # no forwarding
    events = _record_interactions(viewer)
    dims_events = _record(viewer)
    viewer.controller.update_slice_indices(
        viewer.scenes["xy"].id, {0: 2.0}, source_id=uuid4(), interactive=True
    )
    assert events == [("xy", "start", None)]
    assert [event.interactive for event in dims_events].count(True) == 1


async def test_with_forwarding_no_mirrored_panel_plans_in_full(monkeypatch):
    from cellier.render.scheduling import PlanMode
    from cellier.visuals._image_memory import ImageVisual

    viewer = OrthoViewer(_WORLD)
    _add_image(viewer)
    controller = viewer.controller
    # Every visual opts in, so the mode each panel would plan is visible.
    monkeypatch.setattr(ImageVisual, "plans_coarse_while_moving", lambda self, n: True)
    keys = {scene.id: key for key, scene in viewer.scenes.items()}
    modes: dict[str, list] = {key: [] for key in viewer.scenes}
    render_manager = controller._render_manager
    original = render_manager.reslice_scene

    def spy(scene_id, dims_state, visual_configs, **kwargs):
        modes[keys[scene_id]].extend(c.plan_mode for c in visual_configs.values())
        return original(scene_id, dims_state, visual_configs, **kwargs)

    monkeypatch.setattr(render_manager, "reslice_scene", spy)
    widget = uuid4()
    xy = viewer.scenes["xy"]
    controller.begin_dims_interaction(xy.id, source_id=widget)
    for t in (1.0, 2.0):
        controller.update_slice_indices(
            xy.id, {0: t}, source_id=widget, interactive=True
        )
    for key in viewer.scenes:
        assert modes[key] == [PlanMode.BACKSTOP_ONLY] * 2, key

    controller.end_dims_interaction(xy.id, source_id=widget)
    for key in viewer.scenes:
        assert modes[key][2:] == [PlanMode.FULL], key
    await asyncio.sleep(0)


async def test_two_panels_scrubbed_at_once_end_on_stillness():
    """Two origins hold a forwarded scope on each other: the timer ends them.

    Each origin's forwarded scope is its own, so one release does not end the
    other's scrub.  But neither origin can end by release while the other's
    forwarded scope is open on it, so the pair ends on stillness.  Bounded by
    ``dims_settle_s``, like a leaked scope (design 4.3).
    """
    viewer = OrthoViewer(_WORLD)
    controller = viewer.controller
    controller._render_manager.config.scheduler.dims_settle_s = 0.02
    xy, xz = viewer.scenes["xy"], viewer.scenes["xz"]
    first, second = uuid4(), uuid4()
    controller.begin_dims_interaction(xy.id, source_id=first)
    controller.begin_dims_interaction(xz.id, source_id=second)
    # z ticks only xy and y only xz, so both panels start as origins.
    controller.update_slice_indices(xy.id, {1: 5.0}, source_id=first)
    controller.update_slice_indices(xz.id, {2: 5.0}, source_id=second)
    controller.update_slice_indices(xy.id, {0: 1.0}, source_id=first)  # all four
    assert set(_states(viewer).values()) == {"active"}

    controller.end_dims_interaction(xy.id, source_id=first)
    # xz's forwarded scopes are still open on the other three, xy included.
    assert set(_states(viewer).values()) == {"active"}
    controller.end_dims_interaction(xz.id, source_id=second)
    await asyncio.sleep(0.1)
    assert set(_states(viewer).values()) == {"idle"}
    assert not viewer.dims_controller._forwarded
    assert not controller._dims_driver.tasks()


async def test_closing_the_mirror_closes_its_forwarded_scopes():
    viewer = OrthoViewer(_WORLD)
    controller = viewer.controller
    xy = viewer.scenes["xy"]
    widget = uuid4()
    controller.begin_dims_interaction(xy.id, source_id=widget)
    controller.update_slice_indices(xy.id, {0: 1.0}, source_id=widget)
    assert _states(viewer)["vol"] == "active"
    viewer.dims_controller.close()
    assert _states(viewer) == {
        "xy": "active",
        "xz": "idle",
        "yz": "idle",
        "vol": "idle",
    }
    controller.end_dims_interaction(xy.id, source_id=widget)
