"""The loading-settings control on both toolkits, and the scheduler log category.

The control sends ``LoadingConfigUpdateEvent`` and follows
``LoadingConfigChangedEvent``.  An invalid combination is refused by the
controller: the control shows the last settings and the reason, and
corrects nothing.
"""

from __future__ import annotations

import itertools
import logging

import pytest

from cellier.visuals import ProgressiveLoadingConfig
from tests.render.conftest import drain_loading
from tests.render.scheduling.test_backstop_integration import _add

_SERIALS = itertools.count()


def _make(toolkit, visual_ids, loading, n_levels=2):
    if toolkit == "qt":
        from cellier.gui.qt.visuals import QtLoadingConfigControls

        return QtLoadingConfigControls(
            visual_ids, loading=loading.model_dump(), n_levels=n_levels
        )
    from cellier.gui.anywidget.visuals import AnywidgetLoadingConfigControls

    return AnywidgetLoadingConfigControls(
        visual_ids, loading=loading.model_dump(), n_levels=n_levels
    )


def _user_edit(widget, field, value) -> None:
    """Change one control as a user would."""
    if hasattr(widget, "input"):  # Qt
        control = widget.input(field)
        if hasattr(control, "setChecked"):
            control.setChecked(value)
        elif hasattr(control, "setCurrentText"):
            control.setCurrentText(value)
        else:
            control.setValue(value)
    else:  # anywidget: what the front end sets, a fresh serial per edit
        serial = next(_SERIALS)
        widget.edit = {"field": field, "value": value, "serial": serial}


def _shown(widget) -> dict:
    """The values the control shows, as control values."""
    if hasattr(widget, "input"):
        out = {}
        for name in ProgressiveLoadingConfig.model_fields:
            control = widget.input(name)
            if hasattr(control, "isChecked"):
                out[name] = control.isChecked()
            elif hasattr(control, "currentText"):
                out[name] = control.currentText()
            else:
                out[name] = control.value()
        return out
    return dict(widget.config)


def _error(widget) -> str:
    return widget.error


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_an_edit_applies_and_an_invalid_one_is_refused(
    controller, multiscale_image_store, toolkit
):
    _scene, visual, _gfx = _add(controller, multiscale_image_store)
    widget = _make(toolkit, [visual.id], visual.render_config.loading)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    _user_edit(widget, "dims_drag", "backstop")
    assert visual.render_config.loading.dims_drag == "backstop"
    assert _shown(widget)["dims_drag"] == "backstop"
    assert _error(widget) == ""

    _user_edit(widget, "backstop", False)
    # Refused: the model is unchanged, the control shows it again, and why.
    assert visual.render_config.loading.backstop is True
    assert _shown(widget)["backstop"] is True
    assert "needs backstop=True" in _error(widget)

    # A change from elsewhere is shown, and clears the message.
    controller.set_loading_config(visual.id, dims_drag="eager")
    assert _shown(widget)["dims_drag"] == "eager"
    assert _error(widget) == ""
    _user_edit(widget, "backstop", False)
    assert visual.render_config.loading.backstop is False

    # Level 0 is "coarsest", i.e. None.
    _user_edit(widget, "backstop", True)
    _user_edit(widget, "backstop_level", 1)
    assert visual.render_config.loading.backstop_level == 1
    _user_edit(widget, "backstop_level", 0)
    assert visual.render_config.loading.backstop_level is None

    widget.close()


def test_a_group_control_edits_every_visual(controller, multiscale_image_store, qtbot):
    _scene, first, _g1 = _add(controller, multiscale_image_store)
    scene2 = controller.add_scene(dim="3d", name="other")
    second = controller.add_image_multiscale(multiscale_image_store, scene2.id)
    widget = _make("qt", [first.id, second.id], first.render_config.loading)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    _user_edit(widget, "dims_drag", "backstop")
    assert first.render_config.loading.dims_drag == "backstop"
    assert second.render_config.loading.dims_drag == "backstop"


def test_the_level_control_offers_the_stores_levels(qtbot):
    from uuid import uuid4

    widget = _make("qt", [uuid4()], ProgressiveLoadingConfig(), n_levels=5)
    level = widget.input("backstop_level")
    assert (level.minimum(), level.maximum()) == (0, 5)
    assert level.specialValueText() == "coarsest"


# -- the panel ------------------------------------------------------------------------


def test_the_panel_offers_it_only_when_asked(multiscale_image_store):
    from cellier.convenience import MultiscaleImageControlsConfig
    from cellier.convenience.layout._shared import appearance_specs
    from cellier.visuals import MultiscaleImageVisual
    from tests._v2 import level_transforms

    visual = MultiscaleImageVisual(
        name="image",
        data_store_id="store",
        level_transforms=level_transforms([[1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]]),
    )
    off = appearance_specs(visual, MultiscaleImageControlsConfig(appearance=True))
    assert "loading_config" not in [spec.kind for spec in off.specs]

    on = appearance_specs(
        visual,
        MultiscaleImageControlsConfig(appearance=True, loading_controls=True),
        multiscale_image_store,
    )
    kinds = [spec.kind for spec in on.specs]
    assert kinds.index("loading_config") == kinds.index("loading") + 1
    spec = on.specs[kinds.index("loading_config")]
    assert spec.title == "Progressive loading"
    assert spec.values["n_levels"] == 2
    assert spec.values["loading"] == ProgressiveLoadingConfig().model_dump()


# -- the "scheduler" log category -----------------------------------------------------


async def test_the_scheduler_category_logs_passes_and_rounds(
    controller, multiscale_image_store, caplog
):
    from cellier.logging import disable_debug_logging, enable_debug_logging

    enable_debug_logging(categories=("scheduler",), use_rich=False, level=logging.INFO)
    try:
        with caplog.at_level(logging.INFO, logger="cellier.render.scheduler"):
            scene, _visual, _gfx = _add(controller, multiscale_image_store)
            controller.fit_camera(scene.id)
            controller.reslice_all()
            await drain_loading(controller)
    finally:
        disable_debug_logging()
    messages = [r.getMessage() for r in caplog.records if r.name.endswith("scheduler")]
    assert any(m.startswith("pass  cache=") for m in messages)
    assert any(m.startswith("commit  cache=") for m in messages)
