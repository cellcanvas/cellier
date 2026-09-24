"""Appearance controls on an ``OrthoViewer``.

Stage 2 of ``plans/convenience_cleanup.md`` (section 8).  Before it,
``AppearanceControls()`` in an ortho ``Layout`` was a **silent no-op**: both
renderers read ``viewer.scene``, an ``OrthoViewer`` exposes only ``scenes``,
and the dock came back ``None`` with no error (section 4.1).  The fix
generalises the channel path's fan-out rather than making the no-op loud.

Each fanned-out add records two controls groups: one widget drives the three
2D panel visuals in lock-step, another the 3D panel's visual.  The dock's
selector offers both, as ``"{name} (2D views)"`` and ``"{name} (3D view)"``.

Mirrors ``test_channel_controls.py``, which is the template section 8.5 names.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience import AppearanceControls, OrthoViewer, Viewer
from cellier.convenience._backend import ANYWIDGET_BACKEND, QT_BACKEND
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
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
)
from cellier.visuals._image_memory import InMemoryImageAppearance
from tests._gpu_budget import SMALL_BUDGETS

_PANELS = ("xy", "xz", "yz", "vol")
_2D = ("xy", "xz", "yz")
_3D = ("vol",)


def _store() -> ImageMemoryStore:
    data = np.random.default_rng(0).random((8, 16, 16)).astype(np.float32)
    return ImageMemoryStore(data=data)


def _appearance() -> InMemoryImageAppearance:
    return InMemoryImageAppearance()


def _single() -> InMemoryImageSingleAppearance:
    return InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0))


def _target(ortho, label):
    """The dock target the selector calls *label*."""
    (target,) = [t for t in appearance_targets(ortho) if t.label == label]
    return target


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


def test_add_image_records_a_2d_group_and_a_3d_group():
    ortho, visuals = _ortho_with_controls()

    ids_2d = [visuals[key].id for key in _2D]
    vol_id = visuals["vol"].id
    assert list(ortho._controls_configs) == [ids_2d[0], vol_id]
    assert ortho._visual_groups == {ids_2d[0]: ids_2d, vol_id: [vol_id]}
    # One config serves both.
    assert ortho._controls_configs[ids_2d[0]] is ortho._controls_configs[vol_id]
    assert ortho._controls_labels == {
        ids_2d[0]: "image (2D views)",
        vol_id: "image (3D view)",
    }


def test_add_image_multiscale_records_the_config_and_the_group(multiscale_image_store):
    from cellier.visuals._image import MultiscaleImageAppearance

    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    visuals = ortho.add_image_multiscale(
        multiscale_image_store,
        appearance=MultiscaleImageAppearance(),
        controls=MultiscaleImageControlsConfig(appearance=["lod_bias"]),
        single=MultiscaleImageSingleAppearance(color_map="viridis"),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS),
    )

    ids_2d = [visuals[key].id for key in _2D]
    vol_id = visuals["vol"].id
    assert ortho._visual_groups == {ids_2d[0]: ids_2d, vol_id: [vol_id]}


def test_controls_none_records_nothing():
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    ortho.add_image(_store(), appearance=_appearance())

    assert ortho._controls_configs == {}
    assert ortho._visual_groups == {}


# ---------------------------------------------------------------------------
# appearance_targets: the ortho cases (pure, no toolkit fixtures)
# ---------------------------------------------------------------------------


def test_targets_are_the_2d_views_then_the_3d_view():
    ortho, visuals = _ortho_with_controls()

    views_2d, view_3d = appearance_targets(ortho)

    # Each group is seeded from its first visual and writes to its own.
    assert views_2d.label == "image (2D views)"
    assert views_2d.visual is visuals["xy"]
    assert views_2d.visual_ids == [visuals[key].id for key in _2D]
    assert view_3d.label == "image (3D view)"
    assert view_3d.visual is visuals["vol"]
    assert view_3d.visual_ids == [visuals["vol"].id]


def test_the_labels_follow_the_name():
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    ortho.add_image(
        _store(),
        name="nuclei",
        controls=InMemoryImageControlsConfig(appearance=["clim"]),
    )

    assert [t.label for t in appearance_targets(ortho)] == [
        "nuclei (2D views)",
        "nuclei (3D view)",
    ]


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


def _qt_image_control(ortho, label="image (2D views)"):
    from cellier.gui._image_controls import image_control_values
    from cellier.gui.qt.visuals import QtImageControls

    target = _target(ortho, label)
    widget = QtImageControls(
        target.visual_ids, image_control_values(target.visual, fields=["clim"])
    )
    ortho.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )
    return widget


def test_qt_2d_edit_reaches_the_2d_panels_only(qtbot):
    """The regression test for section 4.1, driven through a real controller.

    ``clim`` rather than ``color_map`` deliberately: writing ``color_map``
    poisons ``==`` on the appearance class for the rest of the process
    (section 6.2.2), which would break unrelated round-trip tests.
    """
    ortho, visuals = _ortho_with_controls()
    widget = _qt_image_control(ortho)

    widget._controls[("single", None, "clim")].setValue((0.25, 0.75))

    for key in _2D:
        assert visuals[key].single.clim == pytest.approx((0.25, 0.75))
    assert visuals["vol"].single.clim == pytest.approx((0.0, 1.0))


def test_qt_3d_edit_reaches_the_3d_panel_only(qtbot):
    ortho, visuals = _ortho_with_controls()
    widget = _qt_image_control(ortho, "image (3D view)")

    widget._controls[("single", None, "clim")].setValue((0.2, 0.9))

    assert visuals["vol"].single.clim == pytest.approx((0.2, 0.9))
    for key in _2D:
        assert visuals[key].single.clim == pytest.approx((0.0, 1.0))


def test_aabb_edit_reaches_the_2d_panels(qtbot):
    """AABB is not an appearance field, so it fans out on its own event."""
    from cellier.gui.qt.visuals import QtAABBWidget

    ortho, visuals = _ortho_with_controls()
    target = _target(ortho, "image (2D views)")

    widget = QtAABBWidget(target.visual_ids)
    ortho.controller.connect_widget(
        widget, subscription_specs=widget.subscription_specs()
    )

    widget._enabled_check.setChecked(True)
    widget._line_width_spin.setValue(4.5)

    for key in _2D:
        assert visuals[key].aabb.enabled is True
        assert visuals[key].aabb.line_width == pytest.approx(4.5)
    assert visuals["vol"].aabb.enabled is False


def test_a_foreign_write_to_one_panel_reaches_the_widget(qtbot):
    """Subscribe-to-all, not subscribe-to-first (section 8.1 part 2).

    A sibling written by something other than the widget -- here the yz
    panel, never the representative -- must still update the control.
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


def test_the_2d_views_control_hides_the_3d_only_rows(qtbot):
    """The dock seeds each control from its panels' scenes: 2 and 3."""
    ortho, _visuals = _ortho_with_controls(
        appearance=["clim", "render_mode", "iso_threshold"]
    )
    by_label = {}
    for target in appearance_targets(ortho):
        (image, *_rest) = build_appearance_widgets(
            target.visual,
            target.config,
            ortho.controller,
            target.visual_ids,
            backend=QT_BACKEND,
        )
        by_label[target.label] = image

    views_2d = by_label["image (2D views)"]
    view_3d = by_label["image (3D view)"]
    assert views_2d.n_displayed_dimensions == 2
    assert view_3d.n_displayed_dimensions == 3
    assert views_2d._controls[("single", None, "render_mode")].isHidden()
    assert not view_3d._controls[("single", None, "render_mode")].isHidden()


def test_appearance_dock_renders_on_an_ortho_viewer_qt(qtbot):
    """Was ``None`` before stage 2 -- no dock, no error (section 4.1)."""
    from tests.convenience._qt_acceptance import assert_panel_renders, control_labels

    ortho, _visuals = _ortho_with_controls(appearance=["color_map", "clim"])

    container = render_dock(AppearanceControls(), ortho, QtLayoutHost(), [])

    assert container is not None
    # Two groups, so the selector ("Visual") heads the dock.
    assert control_labels(container) == ["Visual", "Image", "Bounding box"]
    assert_panel_renders(container)


def test_the_rendered_ortho_dock_drives_the_selected_group(qtbot):
    """End to end: build the dock from a Layout spec, then edit it.

    The selector starts on the first group, the 2D views.
    """
    from cellier.convenience.layout._spec import AppearanceControls

    ortho, visuals = _ortho_with_controls(appearance=["clim"])

    container = render_dock(AppearanceControls(), ortho, QtLayoutHost(), [])
    assert container is not None

    from superqt import QLabeledDoubleRangeSlider

    slider = container.findChild(QLabeledDoubleRangeSlider)
    slider.setValue((0.3, 0.7))

    for key in _2D:
        assert visuals[key].single.clim == pytest.approx((0.3, 0.7))
    assert visuals["vol"].single.clim == pytest.approx((0.0, 1.0))


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
    for label, keys, n_displayed in (
        ("image (2D views)", _2D, 2),
        ("image (3D view)", _3D, 3),
    ):
        target = _target(ortho, label)
        built = build_appearance_widgets(
            target.visual,
            target.config,
            ortho.controller,
            target.visual_ids,
            backend=ANYWIDGET_BACKEND,
        )

        assert control_labels_anywidget(built) == ["Image", "Bounding box"]
        for widget in built:
            assert widget.visual_ids == tuple(visuals[key].id for key in keys)
        assert built[0].n_displayed_dimensions == n_displayed


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
