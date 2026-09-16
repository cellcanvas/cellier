"""Appearance controls on an ``OrthoViewer``.

Stage 2 of ``plans/convenience_cleanup.md`` (section 8).  Before it,
``AppearanceControls()`` in an ortho ``Layout`` was a **silent no-op**: both
renderers read ``viewer.scene``, an ``OrthoViewer`` exposes only ``scenes``,
and the dock came back ``None`` with no error (section 4.1).  The fix
generalises the channel path's fan-out rather than making the no-op loud, so
one widget drives all four panel visuals in lock-step.

Mirrors ``test_channel_controls.py``, which is the template section 8.5 names.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience import AppearanceControls, OrthoViewer, Viewer
from cellier.convenience._backend import ANYWIDGET_BACKEND
from cellier.convenience._hosts import QtLayoutHost
from cellier.convenience.gui._controls_config import (
    InMemoryImageControlsConfig,
    MultiscaleImageControlsConfig,
)
from cellier.convenience.layout._shared import appearance_targets
from cellier.convenience.layout._walk import build_appearance_widgets, render_dock
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    InMemoryImageSingleAppearance,
    MultiscaleImageSingleAppearance,
)
from cellier.visuals._image_memory import InMemoryImageAppearance

_PANELS = ("xy", "xz", "yz", "vol")


def _store() -> ImageMemoryStore:
    data = np.random.default_rng(0).random((8, 16, 16)).astype(np.float32)
    return ImageMemoryStore(data=data)


def _appearance() -> InMemoryImageAppearance:
    return InMemoryImageAppearance()


def _single() -> InMemoryImageSingleAppearance:
    return InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0))


def _ortho_with_controls(**config_kwargs):
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    visuals = ortho.add_image(
        _store(),
        appearance=_appearance(),
        controls=InMemoryImageControlsConfig(
            appearance=config_kwargs.pop("appearance", ["clim"]), **config_kwargs
        ),
        single=_single(),
    )
    return ortho, visuals


# ---------------------------------------------------------------------------
# Recording the config and the panel group
# ---------------------------------------------------------------------------


def test_add_image_records_the_config_and_the_group():
    ortho, visuals = _ortho_with_controls()

    panel_ids = [visuals[key].id for key in _PANELS]
    rep_id = panel_ids[0]
    assert list(ortho._controls_configs) == [rep_id]
    assert ortho._visual_groups[rep_id] == panel_ids


def test_add_image_multiscale_records_the_config_and_the_group(multiscale_image_store):
    from cellier.visuals._image import MultiscaleImageAppearance

    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    visuals = ortho.add_image_multiscale(
        multiscale_image_store,
        appearance=MultiscaleImageAppearance(),
        controls=MultiscaleImageControlsConfig(appearance=["lod_bias"]),
        single=MultiscaleImageSingleAppearance(color_map="viridis"),
    )

    panel_ids = [visuals[key].id for key in _PANELS]
    assert ortho._visual_groups[panel_ids[0]] == panel_ids


def test_controls_none_records_nothing():
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    ortho.add_image(_store(), appearance=_appearance())

    assert ortho._controls_configs == {}
    assert ortho._visual_groups == {}


# ---------------------------------------------------------------------------
# appearance_targets: the ortho cases (pure, no toolkit fixtures)
# ---------------------------------------------------------------------------


def test_target_expands_to_all_four_panels():
    ortho, visuals = _ortho_with_controls()

    (target,) = appearance_targets(ortho)

    # The representative is the first panel; the controls are seeded from it
    # and written to all four.
    assert target.visual is visuals["xy"]
    assert target.visual_ids == [visuals[key].id for key in _PANELS]


def test_target_on_a_single_scene_viewer_is_one_id():
    viewer = Viewer(spatial_axes("z", "y", "x"))
    visual = viewer.add_image(
        _store(),
        appearance=_appearance(),
        controls=InMemoryImageControlsConfig(appearance=["clim"]),
    )

    (target,) = appearance_targets(viewer)

    assert target.visual is visual
    assert target.visual_ids == [visual.id]


def test_no_targets_on_an_unconfigured_ortho():
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    ortho.add_image(_store(), appearance=_appearance())

    assert appearance_targets(ortho) == []


# ---------------------------------------------------------------------------
# One edit reaches every panel
# ---------------------------------------------------------------------------


def _qt_image_control(ortho):
    from cellier.gui._image_controls import image_control_values
    from cellier.gui.qt.visuals import QtImageControls

    (target,) = appearance_targets(ortho)
    widget = QtImageControls(
        target.visual_ids, image_control_values(target.visual, fields=["clim"])
    )
    ortho.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )
    return widget


def test_qt_edit_reaches_all_four_panels(qtbot):
    """The regression test for section 4.1, driven through a real controller.

    ``clim`` rather than ``color_map`` deliberately: writing ``color_map``
    poisons ``==`` on the appearance class for the rest of the process
    (section 6.2.2), which would break unrelated round-trip tests.
    """
    ortho, visuals = _ortho_with_controls()
    widget = _qt_image_control(ortho)

    widget._controls[("single", None, "clim")].setValue((0.25, 0.75))

    for key in _PANELS:
        assert visuals[key].single.clim == pytest.approx((0.25, 0.75))


def test_aabb_edit_reaches_all_four_panels(qtbot):
    """AABB is not an appearance field, so it fans out on its own event."""
    from cellier.gui.qt.visuals import QtAABBWidget

    ortho, visuals = _ortho_with_controls()
    (target,) = appearance_targets(ortho)

    widget = QtAABBWidget(target.visual_ids)
    ortho.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )

    widget._enabled_check.setChecked(True)
    widget._line_width_spin.setValue(4.5)

    for key in _PANELS:
        assert visuals[key].aabb.enabled is True
        assert visuals[key].aabb.line_width == pytest.approx(4.5)


def test_a_foreign_write_to_one_panel_reaches_the_widget(qtbot):
    """Subscribe-to-all, not subscribe-to-first (section 8.1 part 2).

    A sibling written by something other than the widget -- here panel 2,
    never the representative -- must still update the control.
    """
    ortho, visuals = _ortho_with_controls()
    widget = _qt_image_control(ortho)

    ortho.controller.update_single_appearance_field(
        visuals["yz"].id, "clim", (0.1, 0.6)
    )

    value = widget._controls[("single", None, "clim")].value()
    assert tuple(value) == pytest.approx((0.1, 0.6))


def test_the_widgets_own_echoes_are_all_dropped(qtbot):
    """A group edit produces N echoes, not one; every one must be filtered."""
    ortho, _visuals = _ortho_with_controls()
    widget = _qt_image_control(ortho)

    applied: list = []
    original = widget._appliers[("single", None, "clim")]
    widget._appliers[("single", None, "clim")] = lambda value: (
        applied.append(value),
        original(value),
    )

    widget._controls[("single", None, "clim")].setValue((0.25, 0.75))

    assert applied == []


def test_ortho_composite_and_channel_edits_mirror_to_all_four_panels():
    """Mode and settings are mirrored through the group methods (D3)."""
    from cellier.visuals import InMemoryImageChannelAppearance

    ortho = OrthoViewer([("c", "channel"), *spatial_axes("z", "y", "x")])
    data = np.random.default_rng(0).random((2, 8, 16, 16)).astype(np.float32)
    visuals = ortho.add_image(
        ImageMemoryStore(data=data),
        channel_axis=0,
        channels={
            0: InMemoryImageChannelAppearance(color_map="green"),
            1: InMemoryImageChannelAppearance(color_map="magenta"),
        },
    )

    ortho.set_image_composite(visuals["xz"], True)
    assert all(v.composite for v in visuals.values())

    ortho.update_image_single_field(visuals["yz"].id, "clim", (0.2, 0.4))
    assert all(v.single.clim == (0.2, 0.4) for v in visuals.values())

    ortho.update_image_channel_field(visuals, 1, "opacity", 0.5)
    assert all(v.channels[1].opacity == 0.5 for v in visuals.values())

    ortho.set_image_composite(visuals, False)
    assert not any(v.composite for v in visuals.values())


def test_ortho_cannot_composite_a_spatial_axis():
    """Some panel always displays a spatial axis (design 3.4)."""
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    data = np.zeros((8, 16, 16), dtype=np.float32)
    visuals = ortho.add_image(ImageMemoryStore(data=data), channel_axis=0)

    with pytest.raises(ValueError, match="displayed"):
        ortho.set_image_composite(visuals, True)
    assert not any(v.composite for v in visuals.values())


# ---------------------------------------------------------------------------
# The dock itself: AppearanceControls() in an ortho Layout
# ---------------------------------------------------------------------------


def test_appearance_dock_renders_on_an_ortho_viewer_qt(qtbot):
    """Was ``None`` before stage 2 -- no dock, no error (section 4.1)."""
    from tests.convenience._qt_acceptance import assert_panel_renders, control_labels

    ortho, _visuals = _ortho_with_controls(appearance=["color_map", "clim"])

    container = render_dock(AppearanceControls(), ortho, QtLayoutHost(), [])

    assert container is not None
    assert control_labels(container) == ["Image", "Bounding box"]
    assert_panel_renders(container)


def test_the_rendered_ortho_dock_drives_every_panel(qtbot):
    """End to end: build the dock from a Layout spec, then edit it."""
    from cellier.convenience.layout._spec import AppearanceControls

    ortho, visuals = _ortho_with_controls(appearance=["clim"])

    container = render_dock(AppearanceControls(), ortho, QtLayoutHost(), [])
    assert container is not None

    from superqt import QLabeledDoubleRangeSlider

    slider = container.findChild(QLabeledDoubleRangeSlider)
    slider.setValue((0.3, 0.7))

    for key in _PANELS:
        assert visuals[key].single.clim == pytest.approx((0.3, 0.7))


def test_appearance_dock_renders_on_an_ortho_viewer_anywidget():
    """The same fix reaches the anywidget renderer, which shares the resolver."""
    from tests.convenience._qt_acceptance import control_labels_anywidget

    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui="anywidget")
    visuals = ortho.add_image(
        _store(),
        appearance=_appearance(),
        controls=InMemoryImageControlsConfig(appearance=["clim"]),
        single=_single(),
    )
    (target,) = appearance_targets(ortho)

    built = build_appearance_widgets(
        target.visual,
        target.config,
        ortho.controller,
        target.visual_ids,
        backend=ANYWIDGET_BACKEND,
    )

    assert control_labels_anywidget(built) == ["Image", "Bounding box"]
    for widget in built:
        assert widget.visual_ids == tuple(visuals[key].id for key in _PANELS)


# ---------------------------------------------------------------------------
# The controller's write-side companions (design section 8.3 step 2)
# ---------------------------------------------------------------------------


def test_update_appearance_group_field_writes_every_visual():
    """The programmatic companion to the widget's subscribe-to-all read side."""
    from cellier.events import AppearanceChangedEvent

    ortho, visuals = _ortho_with_controls()
    panel_ids = [visuals[key].id for key in _PANELS]

    received: list = []
    ortho.controller._outgoing_events.subscribe(AppearanceChangedEvent, received.append)

    ortho.controller.update_appearance_group_field(panel_ids, "interpolation", "linear")

    for key in _PANELS:
        assert visuals[key].appearance.interpolation == "linear"
    assert len(received) == len(panel_ids)


def test_update_aabb_group_field_writes_every_visual():
    """AABB needs its own helper: different model, different event."""
    from cellier.events import AABBChangedEvent

    ortho, visuals = _ortho_with_controls()
    panel_ids = [visuals[key].id for key in _PANELS]

    received: list = []
    ortho.controller._outgoing_events.subscribe(AABBChangedEvent, received.append)

    ortho.controller.update_aabb_group_field(panel_ids, "enabled", True)

    for key in _PANELS:
        assert visuals[key].aabb.enabled is True
    assert len(received) == len(panel_ids)


def test_the_group_helpers_stamp_the_given_source_id():
    """So a widget writing through them still filters its own echoes."""
    from uuid import uuid4

    from cellier.events import AppearanceChangedEvent

    ortho, visuals = _ortho_with_controls()
    panel_ids = [visuals[key].id for key in _PANELS]
    source_id = uuid4()

    received: list = []
    ortho.controller._outgoing_events.subscribe(AppearanceChangedEvent, received.append)

    ortho.controller.update_appearance_group_field(
        panel_ids, "interpolation", "linear", source_id=source_id
    )

    assert {event.source_id for event in received} == {source_id}


def test_the_renderer_warns_about_a_field_the_model_does_not_have(qtbot):
    """The residual drop stage 3's validation cannot catch, on ortho too."""

    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    ortho.add_image(
        _store(),
        appearance=_appearance(),
        # Valid for the multiscale config class; absent from the in-memory model.
        controls=MultiscaleImageControlsConfig(appearance=["clim", "lod_bias"]),
    )

    with pytest.warns(UserWarning, match="lod_bias"):
        container = render_dock(AppearanceControls(), ortho, QtLayoutHost(), [])

    assert container is not None
