"""Tests for the anywidget "Level of detail" control (plane rendering 9.3)."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("anywidget")

from cellier.events import AppearanceChangedEvent
from cellier.gui._level_of_detail import (
    COARSEST_LABEL,
    LEVEL_OF_DETAIL_FIELDS,
    SETTLED_BIAS_LABEL,
    level_of_detail_values,
    moving_field,
)
from cellier.gui.anywidget.visuals._level_of_detail import (
    AnywidgetLevelOfDetailControls,
)
from cellier.visuals import MultiscaleImageAppearance, MultiscaleLabelsAppearance


def _make(values=None, **kwargs):
    visual_id = uuid4()
    return AnywidgetLevelOfDetailControls(visual_id, values, **kwargs), visual_id


def _changed(widget, visual_id, field, value, source_id=None):
    widget._on_appearance_changed(
        AppearanceChangedEvent(
            source_id=source_id or uuid4(),
            visual_id=visual_id,
            field_name=field,
            new_value=value,
            requires_reslice=field == "settled_lod_bias",
        )
    )


def test_instantiate_smoke():
    widget, _ = _make(
        {
            "settled_lod_bias": 2.5,
            "coarsest_while_moving_3d": False,
            "coarsest_while_moving_2d": True,
        },
        n_displayed_dimensions=2,
    )
    assert widget.settled_lod_bias == pytest.approx(2.5)
    assert widget.coarsest_while_moving_3d is False
    assert widget.coarsest_while_moving_2d is True
    assert widget.moving_field == "coarsest_while_moving_2d"
    assert widget.title == "Level of detail"
    assert widget.labels == {"bias": SETTLED_BIAS_LABEL, "moving": COARSEST_LABEL}


@pytest.mark.parametrize(
    "appearance_cls", [MultiscaleImageAppearance, MultiscaleLabelsAppearance]
)
def test_the_trait_defaults_are_the_model_s(appearance_cls):
    widget, _ = _make()
    for field, value in level_of_detail_values(appearance_cls()).items():
        assert getattr(widget, field) == value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("settled_lod_bias", 3.5),
        ("coarsest_while_moving_3d", False),
        ("coarsest_while_moving_2d", True),
    ],
)
def test_a_trait_edit_emits_one_update_event(field, value):
    widget, visual_id = _make()
    emitted = []
    widget.changed.connect(emitted.append)
    setattr(widget, field, value)
    assert len(emitted) == 1
    event = emitted[0]
    assert event.source_id == widget._id
    assert event.visual_id == visual_id
    assert event.field == field
    assert event.value == value
    assert type(event.value) is type(value)


@pytest.mark.parametrize("field", LEVEL_OF_DETAIL_FIELDS)
def test_an_inbound_change_updates_the_trait_without_reemitting(field):
    widget, visual_id = _make()
    emitted = []
    widget.changed.connect(emitted.append)
    value = 4.0 if field == "settled_lod_bias" else not getattr(widget, field)
    _changed(widget, visual_id, field, value)
    assert getattr(widget, field) == value
    assert emitted == []


def test_the_echo_of_its_own_write_is_ignored():
    widget, visual_id = _make()
    _changed(widget, visual_id, "settled_lod_bias", 9.0, source_id=widget._id)
    _changed(widget, visual_id, "coarsest_while_moving_3d", False, widget._id)
    assert widget.settled_lod_bias == pytest.approx(1.0)
    assert widget.coarsest_while_moving_3d is True


def test_an_unrelated_field_is_ignored():
    widget, visual_id = _make()
    _changed(widget, visual_id, "force_level", 2)
    assert widget.settled_lod_bias == pytest.approx(1.0)


def test_following_the_view_is_not_an_edit():
    widget, _ = _make()
    emitted = []
    widget.changed.connect(emitted.append)
    assert widget.moving_field == moving_field(3)
    widget.n_displayed_dimensions = 2
    assert widget.moving_field == moving_field(2)
    assert emitted == []
    with pytest.raises(Exception, match="2 or 3"):
        widget.n_displayed_dimensions = 4


def test_one_dims_subscription_per_followed_scene():
    scenes = (uuid4(), uuid4())
    widget, visual_id = _make(scene_ids=scenes)
    specs = widget.subscription_specs()
    assert [spec.entity_id for spec in specs] == [visual_id, *scenes]


def test_the_front_end_binds_the_checkbox_to_the_view_s_setting():
    """The JS names both fields and picks by ``n_displayed_dimensions``."""
    import cellier.gui.anywidget.visuals as visuals

    source = (
        Path(visuals.__file__).parent / "static" / "level_of_detail.js"
    ).read_text()
    for name in (*LEVEL_OF_DETAIL_FIELDS, "n_displayed_dimensions"):
        assert name in source
    # The bias is sent when the slider settles, not while it is dragged.
    assert 'inp.addEventListener("change"' in source
