"""The loading indicators on both toolkits (design v3 5.13, Phase 6).

Lives with the scheduler tests because the interesting cases need a real
load: the indicator is fed by ``ResliceProgressEvent`` from the controller.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cellier.events import LoadingProgress, ResliceProgressEvent
from cellier.gui._loading import indicator_state, sum_progress
from tests.render.conftest import drain_loading
from tests.render.scheduling.test_backstop_integration import _add, _Gate, _until

# -- the state, without a toolkit ---------------------------------------------------


def test_the_text_follows_the_phase():
    assert indicator_state(None).text == "Not loaded"
    assert indicator_state(None).value == 0
    overview = LoadingProgress(
        needed_backstop=8,
        resident_backstop=3,
        needed_target=40,
        backstop_complete=False,
        complete=False,
    )
    assert indicator_state(overview).text == "Overview: 3 / 8"
    detail = overview._replace(resident_backstop=8, backstop_complete=True)
    detail = detail._replace(resident_target=12)
    state = indicator_state(detail)
    assert state.text == "Detail: 12 / 40"
    assert (state.maximum, state.value, state.busy) == (40, 12, True)
    done = detail._replace(resident_target=38, failed=2, complete=True)
    assert indicator_state(done).text == "Loaded, 2 failed"
    assert not indicator_state(done).busy
    over = done._replace(failed=0, truncated_target=5)
    assert indicator_state(over).text == "Loaded, 5 over budget"


def test_a_backstop_only_plan_says_the_detail_waits():
    drag = LoadingProgress(needed_backstop=8, resident_backstop=8, target_deferred=True)
    state = indicator_state(drag)
    assert state.text == "Overview ready. Detail on stop."
    assert state.busy
    assert state.value == 0


def test_an_empty_plan_is_drawn_full_once_complete():
    empty = LoadingProgress()  # nothing wanted, complete
    state = indicator_state(empty)
    assert (state.maximum, state.value, state.text) == (1, 1, "Loaded")


def test_a_group_is_summed():
    a = LoadingProgress(needed_target=10, resident_target=10)
    b = LoadingProgress(needed_target=6, resident_target=2, complete=False)
    total = sum_progress([a, b])
    assert (total.needed_target, total.resident_target) == (16, 12)
    assert total.backstop_complete and not total.complete
    assert sum_progress([]) is None


# -- the widgets, on a real load ------------------------------------------------------


def _make(toolkit, visual_ids, controller):
    initial = {vid: controller.loading_progress(vid) for vid in visual_ids}
    if toolkit == "qt":
        from cellier.gui.qt.visuals import QtLoadingIndicator

        return QtLoadingIndicator(visual_ids, initial=initial)
    from cellier.gui.anywidget.visuals import AnywidgetLoadingIndicator

    return AnywidgetLoadingIndicator(visual_ids, initial=initial)


def _shown(widget) -> str:
    return widget.text


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
async def test_the_indicator_follows_a_load(
    controller, multiscale_image_store, monkeypatch, toolkit
):
    gate = _Gate(monkeypatch, multiscale_image_store)
    scene, visual, _gfx = _add(controller, multiscale_image_store)
    widget = _make(toolkit, [visual.id], controller)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    assert _shown(widget) == "Not loaded"

    controller.fit_camera(scene.id)
    controller.reslice_all()
    scheduler = controller._render_manager.scheduler

    def detail_loading() -> bool:
        scheduler.commit_round()
        return _shown(widget).startswith("Detail")

    await _until(detail_loading)
    assert widget.state.busy
    assert widget.state.value == 0

    gate.open.set()
    await drain_loading(controller)
    await asyncio.sleep(0)
    assert _shown(widget) == "Loaded"
    assert widget.state.value == widget.state.maximum > 0

    # Closing unsubscribes: a later load does not reach it.
    widget.close()
    controller.update_slice_indices(scene.id, {0: 6.0})
    assert _shown(widget) == "Loaded"
    await drain_loading(controller)


async def test_an_indicator_built_after_the_load_starts_where_it_is(
    controller, multiscale_image_store
):
    scene, visual, _gfx = _add(controller, multiscale_image_store)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    widget = _make("qt", [visual.id], controller)
    assert widget.text == "Loaded"


def test_a_group_indicator_sums_its_visuals(qtbot):
    from cellier.gui.qt.visuals import QtLoadingIndicator

    a, b = uuid4(), uuid4()
    widget = QtLoadingIndicator([a, b])
    specs = widget.subscription_specs()
    assert {s.entity_id for s in specs} == {a, b}
    assert all(s.event_type is ResliceProgressEvent for s in specs)
    handler = specs[0].handler
    handler(
        ResliceProgressEvent(
            uuid4(),
            uuid4(),
            a,
            LoadingProgress(needed_target=4, resident_target=4),
        )
    )
    assert widget.text == "Loaded"
    handler(
        ResliceProgressEvent(
            uuid4(),
            uuid4(),
            b,
            LoadingProgress(needed_target=4, resident_target=1, complete=False),
        )
    )
    assert widget.text == "Detail: 5 / 8"


# -- the group around the bar -------------------------------------------------------


def test_the_qt_indicator_is_a_titled_group_without_a_side_label(qtbot):
    """Framed like the image and bounding-box controls; no label by the bar."""
    from qtpy.QtWidgets import QGroupBox, QLabel

    from cellier.gui.qt.visuals import QtLoadingIndicator

    widget = QtLoadingIndicator(uuid4())

    assert isinstance(widget.widget, QGroupBox)
    assert widget.widget.title() == "Data fetch status"
    texts = [label.text() for label in widget.widget.findChildren(QLabel)]
    assert "Data fetch status" not in texts
    assert "Loading" not in texts


def test_the_anywidget_indicator_has_the_same_title():
    pytest.importorskip("anywidget")
    from cellier.gui.anywidget.visuals import AnywidgetLoadingIndicator

    assert AnywidgetLoadingIndicator(uuid4()).title == "Data fetch status"
