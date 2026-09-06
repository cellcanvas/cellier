"""Tests for the Qt appearance-field base and ``QtVisibleToggle``.

The reference widget of ``plans/convenience_cleanup.md`` section 6.4.  The base
contract (layer 1) is tested once here against a stub subclass, so the 47
layer-3 classes stage 4 adds need only the thin tests at the bottom.
"""

from __future__ import annotations

from typing import Any, ClassVar
from uuid import uuid4

import numpy as np
import pytest

pytest.importorskip("qtpy")

from cellier.convenience import Viewer
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.events import (
    AppearanceChangedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.gui.qt.visuals import QtToggle, QtVisibleToggle
from cellier.visuals._image_memory import InMemoryImageAppearance


class _StubToggle(QtToggle):
    """A layer-3 class on the ordinary ``AppearanceChangedEvent`` path.

    ``visible`` is the one field routed to a different bus event, so the base
    contract is exercised against a field that is *not* special.
    """

    _field: ClassVar[str] = "wireframe"
    _label: ClassVar[str] = "Wireframe"
    _default_value: ClassVar[Any] = False


def _row_label_text(widget) -> str:
    """The name a control shows for itself, read off its label row.

    The row's first layout item is its label -- see
    ``LABELLED_ROW_OBJECT_NAME`` in ``cellier.gui.qt.visuals._chrome``.
    """
    return widget.widget.layout().itemAt(0).widget().text()


def _appearance_event(visual_id, field_name, new_value, source_id=None):
    return AppearanceChangedEvent(
        source_id=source_id or uuid4(),
        visual_id=visual_id,
        field_name=field_name,
        new_value=new_value,
        requires_reslice=False,
    )


def _two_image_viewer():
    viewer = Viewer(("z", "y", "x"), gui="qt")
    visuals = [
        viewer.add_image(
            ImageMemoryStore(data=np.zeros((8, 16, 24), dtype=np.float32)),
            appearance=InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0)),
        )
        for _ in range(2)
    ]
    return viewer, visuals


# ---------------------------------------------------------------------------
# Layer 1: the bus contract
# ---------------------------------------------------------------------------


def test_single_visual_id_is_normalised(qtbot):
    visual_id = uuid4()
    widget = _StubToggle(visual_id)
    qtbot.addWidget(widget.widget)

    assert widget.visual_ids == (visual_id,)
    assert widget.field == "wireframe"


def test_subscription_spec_per_visual_id(qtbot):
    ids = [uuid4(), uuid4(), uuid4()]
    widget = _StubToggle(ids)
    qtbot.addWidget(widget.widget)

    specs = widget.subscription_specs()
    assert [spec.entity_id for spec in specs] == ids
    assert {spec.event_type for spec in specs} == {AppearanceChangedEvent}
    assert all(spec.handler == widget._on_inbound_event for spec in specs)


def test_edit_emits_one_update_event_per_visual_id(qtbot):
    ids = [uuid4(), uuid4()]
    widget = _StubToggle(ids, initial_value=False)
    qtbot.addWidget(widget.widget)
    emitted: list = []
    widget.changed.connect(emitted.append)

    widget.control.setChecked(True)

    assert [event.visual_id for event in emitted] == ids
    assert {event.field for event in emitted} == {"wireframe"}
    assert {event.value for event in emitted} == {True}
    assert {event.source_id for event in emitted} == {widget._id}


def test_inbound_change_applies_without_reemitting(qtbot):
    visual_id = uuid4()
    widget = _StubToggle(visual_id, initial_value=False)
    qtbot.addWidget(widget.widget)
    emitted: list = []
    widget.changed.connect(emitted.append)

    widget._on_inbound_event(_appearance_event(visual_id, "wireframe", True))

    assert widget.value is True
    assert emitted == []  # blockSignals guard held


def test_inbound_echo_filtered_by_source_id(qtbot):
    visual_id = uuid4()
    widget = _StubToggle(visual_id, initial_value=False)
    qtbot.addWidget(widget.widget)

    widget._on_inbound_event(
        _appearance_event(visual_id, "wireframe", True, source_id=widget._id)
    )

    assert widget.value is False  # unchanged


def test_inbound_unrelated_field_ignored(qtbot):
    visual_id = uuid4()
    widget = _StubToggle(visual_id, initial_value=False)
    qtbot.addWidget(widget.widget)

    widget._on_inbound_event(_appearance_event(visual_id, "flat_shading", True))

    assert widget.value is False


def test_close_emits_closed(qtbot):
    widget = _StubToggle(uuid4())
    qtbot.addWidget(widget.widget)
    closed: list = []
    widget.closed.connect(lambda: closed.append(True))

    widget.close()

    assert closed == [True]


# ---------------------------------------------------------------------------
# Layer 3: QtVisibleToggle
# ---------------------------------------------------------------------------


def test_visible_toggle_binds_field_and_label(qtbot):
    widget = QtVisibleToggle(uuid4())
    qtbot.addWidget(widget.widget)

    assert widget.field == "visible"
    # The row's label carries the name; the checkbox itself has no text (see
    # ``plans/label_ownership_unification.md``).
    assert _row_label_text(widget) == "Visible"
    assert widget.control.text() == ""
    assert widget.value is True  # matches the model default


def test_visible_toggle_subscribes_to_the_visibility_event(qtbot):
    """The reason ``visible`` needs a spec of its own.

    A ``visible`` change is emitted as ``VisualVisibilityChangedEvent``, so a
    widget subscribed to ``AppearanceChangedEvent`` would never update.
    """
    widget = QtVisibleToggle(uuid4())
    qtbot.addWidget(widget.widget)

    specs = widget.subscription_specs()
    assert [spec.event_type for spec in specs] == [VisualVisibilityChangedEvent]


def test_visible_toggle_applies_the_visibility_event(qtbot):
    visual_id = uuid4()
    widget = QtVisibleToggle(visual_id, initial_value=True)
    qtbot.addWidget(widget.widget)
    emitted: list = []
    widget.changed.connect(emitted.append)

    widget._on_inbound_event(
        VisualVisibilityChangedEvent(
            source_id=uuid4(), visual_id=visual_id, visible=False
        )
    )

    assert widget.value is False
    assert emitted == []


# ---------------------------------------------------------------------------
# End to end through a real controller
# ---------------------------------------------------------------------------


def test_edit_reaches_every_visual_in_the_group(qtbot):
    """One toggle drives N visuals in lock-step -- the ortho fan-out shape."""
    viewer, visuals = _two_image_viewer()
    widget = QtVisibleToggle([v.id for v in visuals], initial_value=True)
    qtbot.addWidget(widget.widget)
    viewer.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )

    widget.control.setChecked(False)

    assert [v.appearance.visible for v in visuals] == [False, False]


def test_group_write_echoes_do_not_reemit(qtbot):
    """N echoes from the widget's own group write are all dropped."""
    viewer, visuals = _two_image_viewer()
    widget = QtVisibleToggle([v.id for v in visuals], initial_value=True)
    qtbot.addWidget(widget.widget)
    viewer.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )
    emitted: list = []
    widget.changed.connect(emitted.append)

    widget.control.setChecked(False)

    # Exactly the N the edit itself produced -- no echo-driven extras.
    assert len(emitted) == len(visuals)


def test_foreign_write_reaches_the_widget(qtbot):
    viewer, visuals = _two_image_viewer()
    widget = QtVisibleToggle([v.id for v in visuals], initial_value=True)
    qtbot.addWidget(widget.widget)
    viewer.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )

    viewer.controller.set_visual_visible(visuals[0].id, False)

    assert widget.value is False
