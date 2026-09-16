"""Tests for the anywidget appearance-field base and ``AnywidgetVisibleToggle``.

The anywidget half of ``tests/gui/qt/test_qt_visible_toggle.py``; same
structure, same assertions, so the two toolkits stay in step.  Only the trait
layer is covered here -- the ``toggle.js`` half is verified through the browser
procedure in section 6.1.1 of ``plans/convenience_cleanup.md``.
"""

from __future__ import annotations

from typing import Any, ClassVar
from uuid import uuid4

import numpy as np
import pytest

pytest.importorskip("anywidget")

from cellier.convenience import Viewer
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.events import (
    AppearanceChangedEvent,
    VisualVisibilityChangedEvent,
)
from cellier.gui.anywidget.visuals import (
    AnywidgetToggle,
    AnywidgetVisibleToggle,
)
from cellier.scene.dims import spatial_axes
from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance


class _StubToggle(AnywidgetToggle):
    """A layer-3 class on the ordinary ``AppearanceChangedEvent`` path."""

    _field: ClassVar[str] = "wireframe"
    _label: ClassVar[str] = "Wireframe"
    _default_value: ClassVar[Any] = False


def _appearance_event(visual_id, field_name, new_value, source_id=None):
    return AppearanceChangedEvent(
        source_id=source_id or uuid4(),
        visual_id=visual_id,
        field_name=field_name,
        new_value=new_value,
        requires_reslice=False,
    )


def _two_image_viewer():
    viewer = Viewer(spatial_axes("z", "y", "x"), gui="anywidget")
    visuals = [
        viewer.add_image(
            ImageMemoryStore(data=np.zeros((8, 16, 24), dtype=np.float32)),
            appearance=InMemoryImageAppearance(),
            single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
        )
        for _ in range(2)
    ]
    return viewer, visuals


# ---------------------------------------------------------------------------
# Layer 1: the bus contract
# ---------------------------------------------------------------------------


def test_single_visual_id_is_normalised():
    visual_id = uuid4()
    widget = _StubToggle(visual_id)

    assert widget.visual_ids == (visual_id,)
    assert widget.field == "wireframe"
    assert widget.widget is widget


def test_subscription_spec_per_visual_id():
    ids = [uuid4(), uuid4(), uuid4()]
    widget = _StubToggle(ids)

    specs = widget.subscription_specs()
    assert [spec.entity_id for spec in specs] == ids
    assert {spec.event_type for spec in specs} == {AppearanceChangedEvent}


def test_edit_emits_one_update_event_per_visual_id():
    ids = [uuid4(), uuid4()]
    widget = _StubToggle(ids, initial_value=False)
    emitted: list = []
    widget.changed.connect(emitted.append)

    widget.value = True

    assert [event.visual_id for event in emitted] == ids
    assert {event.field for event in emitted} == {"wireframe"}
    assert {event.value for event in emitted} == {True}
    assert {event.source_id for event in emitted} == {widget._id}


def test_inbound_change_applies_without_reemitting():
    visual_id = uuid4()
    widget = _StubToggle(visual_id, initial_value=False)
    emitted: list = []
    widget.changed.connect(emitted.append)

    widget._on_inbound_event(_appearance_event(visual_id, "wireframe", True))

    assert widget.value is True
    assert emitted == []  # _applying guard held


def test_inbound_echo_filtered_by_source_id():
    visual_id = uuid4()
    widget = _StubToggle(visual_id, initial_value=False)

    widget._on_inbound_event(
        _appearance_event(visual_id, "wireframe", True, source_id=widget._id)
    )

    assert widget.value is False


def test_inbound_unrelated_field_ignored():
    visual_id = uuid4()
    widget = _StubToggle(visual_id, initial_value=False)

    widget._on_inbound_event(_appearance_event(visual_id, "flat_shading", True))

    assert widget.value is False


def test_close_emits_closed():
    widget = _StubToggle(uuid4())
    closed: list = []
    widget.closed.connect(lambda: closed.append(True))

    widget.close()

    assert closed == [True]


# ---------------------------------------------------------------------------
# Assets: one .js per control type, shared by every field class
# ---------------------------------------------------------------------------


def test_field_classes_share_the_control_types_asset():
    """Layer 2 declares ``_esm``; layer 3 inherits the same ``FileContents``."""
    assert AnywidgetVisibleToggle._esm is AnywidgetToggle._esm
    assert _StubToggle._esm is AnywidgetToggle._esm
    assert "_esm" not in AnywidgetVisibleToggle.__dict__


def test_the_js_reads_generic_traits_not_field_names():
    """The shared asset must not reference a field-specific trait name."""
    source = str(AnywidgetToggle._esm)
    assert 'model.get("value")' in source
    assert 'model.get("label")' in source
    assert "visible" not in source
    assert "wireframe" not in source


def test_label_is_synced_so_the_js_can_render_it():
    widget = AnywidgetVisibleToggle(uuid4())
    assert widget.label == "Visible"
    assert widget.trait_metadata("label", "sync") is True
    assert widget.trait_metadata("value", "sync") is True


# ---------------------------------------------------------------------------
# Layer 3: AnywidgetVisibleToggle
# ---------------------------------------------------------------------------


def test_visible_toggle_binds_field_and_label():
    widget = AnywidgetVisibleToggle(uuid4())

    assert widget.field == "visible"
    assert widget.label == "Visible"
    assert widget.value is True  # matches the model default


def test_visible_toggle_subscribes_to_the_visibility_event():
    widget = AnywidgetVisibleToggle(uuid4())

    specs = widget.subscription_specs()
    assert [spec.event_type for spec in specs] == [VisualVisibilityChangedEvent]


def test_visible_toggle_applies_the_visibility_event():
    visual_id = uuid4()
    widget = AnywidgetVisibleToggle(visual_id, initial_value=True)
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


def test_edit_reaches_every_visual_in_the_group():
    viewer, visuals = _two_image_viewer()
    widget = AnywidgetVisibleToggle([v.id for v in visuals], initial_value=True)
    viewer.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )

    widget.value = False

    assert [v.appearance.visible for v in visuals] == [False, False]


def test_group_write_echoes_do_not_reemit():
    viewer, visuals = _two_image_viewer()
    widget = AnywidgetVisibleToggle([v.id for v in visuals], initial_value=True)
    viewer.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )
    emitted: list = []
    widget.changed.connect(emitted.append)

    widget.value = False

    assert len(emitted) == len(visuals)


def test_foreign_write_reaches_the_widget():
    viewer, visuals = _two_image_viewer()
    widget = AnywidgetVisibleToggle([v.id for v in visuals], initial_value=True)
    viewer.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )

    viewer.controller.set_visual_visible(visuals[0].id, False)

    assert widget.value is False
