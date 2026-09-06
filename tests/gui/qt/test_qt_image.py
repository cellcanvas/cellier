"""Tests for the Qt image/volume-render control widgets.

Covers ``QtRenderModeCombo``, ``QtIsoThresholdSlider``, and the composite
``QtVolumeRenderControls``. All three mirror fields on
``MultiscaleImageAppearance`` (whose ``render_mode`` literal --
``"iso"``/``"mip"``/``"smooth_iso"``/``"attenuated_mip"`` -- matches the
combo box items; ``InMemoryImageAppearance`` uses a different literal and
lacks ``attenuation`` entirely), so visuals are built via
``add_image_multiscale``.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier.controller import CellierController
from cellier.data.image._zarr_multiscale_store import (
    MultiscaleZarrDataStore,
)
from cellier.gui.qt.visuals._image import (
    QtIsoThresholdSlider,
    QtRenderModeCombo,
    QtVolumeRenderControls,
)
from cellier.scene.dims import CoordinateSystem
from cellier.transform import AffineTransform
from cellier.visuals import MultiscaleImageAppearance


def _make_multiscale_store(small_zarr_store, **kwargs) -> MultiscaleZarrDataStore:
    defaults = {
        "zarr_path": str(small_zarr_store),
        "scale_names": ["s0", "s1"],
        "level_transforms": [
            AffineTransform.identity(ndim=3),
            AffineTransform.from_scale_and_translation(
                (2.0, 2.0, 2.0), (0.5, 0.5, 0.5)
            ),
        ],
    }
    defaults.update(kwargs)
    return MultiscaleZarrDataStore(**defaults)


def _make_volume_visual(small_zarr_store, **appearance_kwargs):
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    store = _make_multiscale_store(small_zarr_store)
    defaults = {"color_map": "viridis", "render_mode": "mip"}
    defaults.update(appearance_kwargs)
    appearance = MultiscaleImageAppearance(**defaults)
    visual = controller.add_image_multiscale(
        data=store, scene_id=scene.id, appearance=appearance, name="vol"
    )
    return controller, visual


# ── QtRenderModeCombo / QtIsoThresholdSlider ──────────────────────────────
#
# Both are layer-3 field classes now, so their behaviour is the shared base's
# and is covered once in ``tests/gui/test_appearance_control_types.py``.  What
# is worth pinning here is that they bind the right field and that the
# standalone controls reach the model -- the composite below draws its own
# selector and does not use them.


def test_render_mode_combo_edit_reaches_model(qtbot, small_zarr_store):
    controller, visual = _make_volume_visual(small_zarr_store, render_mode="mip")
    combo = QtRenderModeCombo(visual.id, initial_value="mip", choices=("mip", "iso"))
    qtbot.addWidget(combo.widget)
    controller.connect_widget(combo, subscription_specs=combo.subscription_specs())

    combo.control.setCurrentText("iso")

    assert visual.appearance.render_mode == "iso"
    assert combo.value == "iso"


def test_render_mode_combo_model_push_without_reemit(qtbot, small_zarr_store):
    controller, visual = _make_volume_visual(small_zarr_store, render_mode="mip")
    combo = QtRenderModeCombo(
        visual.id, initial_value="mip", choices=("mip", "smooth_iso")
    )
    qtbot.addWidget(combo.widget)
    controller.connect_widget(combo, subscription_specs=combo.subscription_specs())

    emitted = []
    combo.changed.connect(emitted.append)

    controller.update_appearance_field(visual.id, "render_mode", "smooth_iso")

    assert combo.value == "smooth_iso"
    assert emitted == []


def test_iso_threshold_slider_edit_reaches_model(qtbot, small_zarr_store):
    controller, visual = _make_volume_visual(small_zarr_store, iso_threshold=0.5)
    slider = QtIsoThresholdSlider(visual.id, initial_value=0.5)
    qtbot.addWidget(slider.widget)
    controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    slider.value = 0.8

    assert visual.appearance.iso_threshold == pytest.approx(0.8)


def test_iso_threshold_slider_model_push_without_reemit(qtbot, small_zarr_store):
    controller, visual = _make_volume_visual(small_zarr_store, iso_threshold=0.5)
    slider = QtIsoThresholdSlider(visual.id, initial_value=0.5)
    qtbot.addWidget(slider.widget)
    controller.connect_widget(slider, subscription_specs=slider.subscription_specs())

    emitted = []
    slider.changed.connect(emitted.append)

    controller.update_appearance_field(visual.id, "iso_threshold", 0.9)

    assert slider.value == pytest.approx(0.9)
    assert emitted == []


# ── QtVolumeRenderControls ────────────────────────────────────────────────
# `isHidden()` (own explicit hidden flag) is used instead of `isVisible()`
# (which also factors in ancestor on-screen visibility, always False here
# since the container is never shown).


def test_volume_controls_instantiate_smoke(qtbot):
    controls = QtVolumeRenderControls(
        visual_id=object(),
        dtype_max=1.0,
        initial_render_mode="iso",
        initial_threshold=0.4,
        initial_attenuation=2.0,
    )
    qtbot.addWidget(controls.widget)
    assert controls._combo.currentText() == "iso"
    assert controls._slider.value() == pytest.approx(0.4)
    assert controls._attenuation_slider.value() == pytest.approx(2.0)
    # ISO mode -> threshold visible, attenuation hidden.
    assert controls._slider.isHidden() is False
    assert controls._attenuation_slider.isHidden() is True


def test_volume_controls_mode_switch_toggles_visibility(qtbot):
    controls = QtVolumeRenderControls(
        visual_id=object(),
        dtype_max=1.0,
        initial_render_mode="mip",
        initial_threshold=0.4,
    )
    qtbot.addWidget(controls.widget)
    assert controls._slider.isHidden() is True
    assert controls._attenuation_slider.isHidden() is True

    controls._combo.setCurrentText("attenuated_mip")

    assert controls._slider.isHidden() is True
    assert controls._attenuation_slider.isHidden() is False


def test_volume_controls_attenuation_edit_reaches_model(qtbot, small_zarr_store):
    controller, visual = _make_volume_visual(
        small_zarr_store, render_mode="attenuated_mip", attenuation=1.0
    )
    controls = QtVolumeRenderControls(
        visual_id=visual.id,
        dtype_max=1.0,
        initial_render_mode="attenuated_mip",
        initial_threshold=0.5,
        initial_attenuation=1.0,
    )
    qtbot.addWidget(controls.widget)
    controller.connect_widget(
        controls, subscription_specs=controls.subscription_specs()
    )

    controls._attenuation_slider.setValue(3.5)

    assert visual.appearance.attenuation == pytest.approx(3.5)


def test_volume_controls_model_push_updates_without_reemit(qtbot, small_zarr_store):
    controller, visual = _make_volume_visual(
        small_zarr_store, render_mode="iso", iso_threshold=0.5
    )
    controls = QtVolumeRenderControls(
        visual_id=visual.id,
        dtype_max=1.0,
        initial_render_mode="iso",
        initial_threshold=0.5,
    )
    qtbot.addWidget(controls.widget)
    controller.connect_widget(
        controls, subscription_specs=controls.subscription_specs()
    )

    emitted = []
    controls.changed.connect(emitted.append)

    controller.update_appearance_field(visual.id, "render_mode", "attenuated_mip")

    assert controls._combo.currentText() == "attenuated_mip"
    assert controls._attenuation_slider.isHidden() is False
    assert controls._slider.isHidden() is True
    assert emitted == []


def test_volume_controls_inbound_echo_filtered_by_source_id(qtbot, small_zarr_store):
    """A widget must ignore the bus echo of its own write.

    The core of the bus contract, and the one thing that stops a control and
    its model oscillating.  Covered on every anywidget module and, until this,
    on only three Qt ones -- the systematic half of D14
    (``plans/gui_backend_unification.md``).

    Driven through the handler rather than the controller, because the
    controller stamps its own ``source_id``; the case under test is an event
    carrying *this widget's* id.
    """
    from cellier.events import AppearanceChangedEvent

    controller, visual = _make_volume_visual(
        small_zarr_store, render_mode="iso", iso_threshold=0.5
    )
    controls = QtVolumeRenderControls(
        visual_id=visual.id,
        dtype_max=1.0,
        initial_render_mode="iso",
        initial_threshold=0.5,
    )
    qtbot.addWidget(controls.widget)
    controller.connect_widget(
        controls, subscription_specs=controls.subscription_specs()
    )

    controls._on_visual_changed(
        AppearanceChangedEvent(
            source_id=controls._id,  # our own echo -> ignored
            visual_id=visual.id,
            field_name="render_mode",
            new_value="mip",
            requires_reslice=False,
        )
    )

    assert controls._combo.currentText() == "iso"


def test_volume_controls_inbound_unrelated_field_ignored(qtbot, small_zarr_store):
    """A field this panel does not drive must leave it alone."""
    from cellier.events import AppearanceChangedEvent

    controller, visual = _make_volume_visual(
        small_zarr_store, render_mode="iso", iso_threshold=0.5
    )
    controls = QtVolumeRenderControls(
        visual_id=visual.id,
        dtype_max=1.0,
        initial_render_mode="iso",
        initial_threshold=0.5,
    )
    qtbot.addWidget(controls.widget)
    controller.connect_widget(
        controls, subscription_specs=controls.subscription_specs()
    )

    controls._on_visual_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=visual.id,
            field_name="clim",
            new_value=(0.0, 500.0),
            requires_reslice=False,
        )
    )

    assert controls._combo.currentText() == "iso"
